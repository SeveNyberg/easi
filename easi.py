# Electron Acceleration SImulation (EASI)
# Main simulation file
# Code author: Seve Nyberg
# Affiliation: Department of Physics and Astronomy, University of Turku, Finland

# Github URL: https://github.com/SeveNyberg/easi

import numpy as np # For array and mathematical operations
import shutil # For parameter file copying to results path
from time import time # To measure simulation wall time
from numba import jit # To make simulation calculations faster
import multiprocessing as mp # For multiprocessing

# This try-excepts looks horrible, but it ensures that the multiprocessing works on windows without compromising plotting parameter importing.
try:
    if __name__ == "__main__":
        from prms import * # Input parameter file
    else:
        from .prms import * # Input parameter file in the directory the easi.py-script itself is located (plotting)
except ImportError:
        from prms import * # Input parameter file
        
from simulation_functions import * # Additional functions shared between this script and other project files

# Set the RNG seed for injection
np.random.seed(999999999)

# Initialize starting values from prms.py
if continuity_testing:
    #remove the shock
    ux1 = V_A1
    downstream_mfp_scaler = 1.0
    theta_1 = 0.0
theta_1 = np.deg2rad(theta_1) # Shock obliquity in radians
tan_theta_1 = np.tan(theta_1) # Tangent of shock obliquity
ion_inert = eval_ion_inert(n_part, e, m_i, eps0, c) # ion inertial length for natural unit conversion in meters
T_max = T_max * c / ion_inert # Unitless maximum simulation time
if L_1 == "inf": # Check whether no global focusing is wished for
    inv_L_1 = 0 # Set inverse of L_1 to 0 at the limit of infinity
else:
    inv_L_1 = 1/(L_1 * 1000 / ion_inert) # Unitless ambient focusing length
if mfp_1 == "inf": # Check whether no scattering is wished for
    inv_mfp_1 = 0 # Set invers to 0 at the limit of infinity
else:   
    inv_mfp_1 = 1/(mfp_1 * 1000 / ion_inert)
M_A = ux1/(np.cos(theta_1)*V_A1) # The Alfvén Mach number in the de Hoffmann-Teller Frame (HTF)
V_A1 = V_A1*1000/c # unitless Alfvén speed
rg = eval_rg(M_A, theta_1, beta) # Gas compression ratio computed from parameters
m_i = m_i/m_e # Unitless ion mass
alpha = m_i * V_A1 # With units corresponds to m_i/m_e * V_A1/c
a_ratio = a_prm/alpha # Mean free path scaling term, alpha is solved from physics, a used as scaler
rb = eval_rb(M_A, rg, theta_1) # Magnetic compression ratio
ux1 = ux1*1000/c # Unitless upstream flow speed in HTF
v_L = E_to_v(eval_E_natural(E_L * 1000, m_e, c, e)) * c # The speed for which the focusing length needed for the box size is calculated for
box_size = eval_box_size(mfp_1 * 1000, ux1 * c, theta_1, v_L) / ion_inert # The focusing length controlled box size, this can be multiplied with x_bound_scalers to increase or decrease box size.


def inject():
    """
    Creates an array of particles with randomly sampled starting times, positioned close to the shock (x_start), speeds based on a given injection energy, and random pitch angle cosines (See part about pitch angle cosines below for details). Particles are injected into the plasma rest frame and then moved to HTF after injection.

    Parameters:
        -
        
    Returns:
        particles (numpy array [ N * [x, v, mu, t, t_start, seed]]): An array of all particle data.
    """
    
    # Creating a particle array in the form [ N x [x, v, mu, t, t_start, seed] ]
    particles = np.zeros((N_p,6))
    
    # Add seed number for each particle for multiprocessing
    particles[:,5] = [i for i in range(N_p, 2*N_p)]
    
    # Sample a starting time for the particle and save it separately to a progressing time and a starting time column
    start_time = np.random.random(size=N_p) * T_max * T_inj_scaler
    particles[:,3] = np.copy(start_time)
    particles[:,4] = np.copy(start_time)
    
    # Set particle starting position
    particles[:,0] = x_min_edge
    
    # Evaluate the flow profile at the edge of the box
    u, _ = eval_flow_profile(x_min_edge, particles[:,3])
    
    if spec_inj:
        # Set the speeds and pitch angle cosines to all particles by sampling from the defined spectrum
        particles[:,1], particles[:,2] = spec_func(u)
        
    else:
        # Set the monoenergetic injection speed for all particles
        particles[:,1] = v_inj_0

        # Sample a random pitch angle cosine for the particles from a population that ends towards the shock when injecting from the edge of the simulation box
        R_min = (u - particles[:,1])/(u + particles[:,1])
        R_min[R_min < 0] = 0
        R_min = R_min**2
        R = np.random.random(size=N_p) * (1 - R_min) + R_min
        particles[:,2] = (-u + (particles[:,1] + u) * np.sqrt(R)) / (particles[:,1])
        
    # Change the frame of reference from plasma rest frame to HTF
    particles[:,1], particles[:,2] = change_frame(particles[:,1], particles[:,2], -u)
    
    return particles


if only_ambient_mfp:
    @jit(nopython=True)
    def eval_mfp_inv(x, v, t, tan_theta):
        """
        Evaluates the inverse mean free path as a hyperbolic tangent profile. Sets the maximum mean free path as the in upstream as the upstream ambient mean free path mfp_1 and in the downstream downstream_mfp_scaler * mfp_1.

        Parameters:
            x (numpy array (float)): The positions of the particles.
            v (numpy array (float)): The speeds of the particles. (overloaded)
            t (numpy array (float)): Current simulation time for each particle separately. 
            tan_theta (numpy array (float)): The local tangents of the shock obliquities. (overloaded)

        Returns:
            mfp_inv (numpy array (float)): The inverses of the mean free paths.
        """
        
        
        
        # Set a maximum mean free path scaled with the magnetic field profile
        inv_mfp_ambient_prof = 2 * inv_mfp_1 / (mfp_prof(x, t) * (1 - downstream_mfp_scaler) + 1 + downstream_mfp_scaler)
        
        mfp_inv = inv_mfp_ambient_prof
        
        return mfp_inv
    
    
else:
    @jit(nopython=True)
    def eval_mfp_inv(x, v, t, tan_theta):
        """
        Evaluates the inverse mean free path as a combination of hyperbolic tangent and a hyperbolic cosine at the shock. Sets the maximum mean free path as the ambient focusing length.

        Parameters:
            x (numpy array (float)): The positions of the particles.
            v (numpy array (float)): The speeds of the particles.
            t (numpy array (float)): Current simulation time for each particle separately. 
            tan_theta (numpy array (float)): The local tangents of the shock obliquities. 

        Returns:
            mfp_inv (numpy array (float)): The inverses of the mean free paths.
        """
        
        # Calculate gamma factor needed for mean free path
        gamma = eval_gamma(v)
        
        # Set a maximum mean free path scaled with the magnetic field profile
        inv_mfp_ambient_prof = 2 * inv_mfp_1 / (mfp_prof(x, t) * (1 - downstream_mfp_scaler) + 1 + downstream_mfp_scaler)
        
        # Calculate the inverse of the mean free path
        mfp_inv = 1/mfp_scaler * mfp_matcher * np.sqrt( ( 1 + tan_theta**2 ) / ( 1 + tan_theta_1**2 )) / (a_ratio * gamma * v ) * B_prof_der(x,t) + inv_mfp_ambient_prof
        
        return mfp_inv


@jit(nopython=True)
def eval_L_inv(x, t, tan_theta):
    '''
    Evaluates the inverse focusing length in the de Hoffmann-Teller frame (HTF).
    
    Parameters:
        x (numpy array (float)): Positions of particles.
        t (numpy array (float)): Current simulation time for each particle separately. 
        tan_theta (numpy array (float)): Tangents of local magnetic field obliquities of particles.
        
    Returns:
        L_inv (numpy array (float)): The inverse focusing lengths for all particles.
    '''
    
    # Calculate the scale length
    L_inv = -( tan_theta / (1 + tan_theta**2 )**(3/2) * (rb - 1) * tan_theta_1 * B_prof_der(x,t)) 
    
    return L_inv


@jit(nopython=True)
def eval_flow_profile(x, t):
    """
    Calculates the local values of flow speed and tangent of shock obliquity.
    
    Parameters:
        x (numpy array (float)): The positions of the particles.
        t (numpy array (float)): Current simulation time for each particle separately. 
        
    Returns:
        u (numpy array (float)): The local flow speeds.
        tan_theta (numpy array (float)): The local tangents of the shock obliquities.
    """
    
    # Calculate the tangent of the shock obliquity and the flow profile u
    tan_theta = tan_theta_1 * (1 + (rb - 1) * B_prof(x,t) ) 
    u = ux1 / np.cos(theta_1) * (1 - (1 - rb/rg) * u_prof(x,t) )
    
    return u, tan_theta


@jit(nopython=True)
def eval_timestep(x, v, mu, t, tan_theta, u, L_inv):
    """
    Calculates the largest plausible time steps for given particles taking into account the spatial histogram grid size. Also calculates the focusing scale length, with a minimum scale length of the ambient scale length from the parameters.

    Parameters:
        x (numpy array (float)): The positions of the particles.
        v (numpy array (float)): The speeds of the particles.
        mu (numpy array (float)): The pitch angle cosines of the particles.
        t (numpy array (float)): Current simulation time for each particle separately. 
        tan_theta (numpy array (float)): The tangents of the shock obliquities.
        u (numpy array (float)): The speed of the flow relative to the shock.
        L_inv (numpy array (float)): The local inverse of the scale length for each particle separately.

    Returns:
        dt (numpy array (float)): The time step sizes for particles.
        
    """
    
    # Set an initial timestep size relative to the histogram bin length or if distance is less to the shock than the bin length, then to abs(x) with minimum value of small_dt, set to dx/c with units. 
    
    # Get the histogram bin for the particles
    hist_loc = spatial_indices(x)
    
    # Get the movement direction of the particle
    direction = np.sign(mu)
    
    # Get the bin edge index by checking the propagation direction (assuming mu = -1 or = 1), index solved by 0.5 + 1.6*direction, 1.6 to mitigate float errors. Results always in -1 or 2 depending on direction
    rel_index = 0.5 + direction*(1.6)
    rel_index2 = rel_index.astype(np.int64)
    true_index = np.minimum(hist_loc + rel_index2, len(bin_steps)-1)
    
    # Get the distance to the specified bin edge
    x_dist = bin_steps[true_index] - x
    
    # Check for out of bounds particles in upstream
    x_dist[(true_index < 0)] = direction[true_index < 0] * dx[0]
    x_dist[(true_index > len(bin_steps)-1)] = direction[(true_index > len(bin_steps)-1)] * dx[-1]

    # Solve the timestep
    dt = direction * (x_dist) * np.sqrt(1 + tan_theta**2) / (v) * 0.5

    # Change to the plasma rest frame of reference to calculate scattering mean free path
    v, mu = change_frame(v, mu, u)
    
    # Calculate the mean free path
    mfp_inv = eval_mfp_inv(x, v, t, tan_theta)
    
    # Change back to de Hoffmann-Teller frame
    v, mu = change_frame(v, mu, -u)

    # Check that time step is not too large compared to the mean free path, otherwise divide by 2 until small enough
    while len(dt[dt * v * mfp_inv > 0.02]) > 0: 
        dt[dt * v * mfp_inv > 0.02] /= 2    
    
    # Limit the change in mean free path to 3 % of the original value
    change = 0.03
    x_temp = move_particles(x, v, mu, dt, tan_theta)
    u_temp, tan_theta_temp = eval_flow_profile(x_temp, t)
    v_temp, mu_temp = change_frame(v, mu, u_temp)
    mfp_inv_diff = eval_mfp_inv(x_temp, v_temp, t, tan_theta_temp) - mfp_inv
    v_temp, mu_temp = change_frame(v_temp, mu_temp, -u_temp)
    while len(mfp_inv[np.abs(mfp_inv_diff) > change * mfp_inv]) > 0:
        clause = np.abs(mfp_inv_diff) > change * mfp_inv
        dt[clause] /= 2
        x_temp[clause] = move_particles(x[clause], v_temp[clause], mu_temp[clause], dt[clause], tan_theta[clause])
        u_temp[clause], tan_theta_temp[clause] = eval_flow_profile(x_temp[clause], t[clause])
        v_temp[clause], mu_temp[clause] = change_frame(v_temp[clause], mu_temp[clause], u_temp[clause])
        mfp_inv_diff[clause] = eval_mfp_inv(x_temp[clause], v_temp[clause], t[clause], tan_theta_temp[clause]) - mfp_inv[clause]
        v_temp[clause], mu_temp[clause] = change_frame(v_temp[clause], mu_temp[clause], -u_temp[clause])
        
    # Check that the time step is not too large compared to the scale length
    while len(dt[dt * v * L_inv > 0.02]) > 0:
        dt[dt * v * L_inv > 0.02] /= 2
        
    return dt
    
    
@jit(nopython=True)
def split_timestep(x, v, mu, dt, tan_theta):
    """
    Splits the timestep to two parts ofr accurate histogram logging. Each timestep is logged to the respective histogram cell it belongs to. If the particle does not mobe from one histogram cell to another, second timestep will be zero and thus not logged.
    
    Parameters: 
        x (numpy array (float)): The positions of the particles.
        v (numpy array (float)): The speed of the particles.
        mu (numpy array (float)): The pitch angle cosines of the particles.
        dt (numpy array (float)): The time step sizes. 
        tan_theta (numpy array (float)): The tangents of the shock obliquities. 

    Returns: 
        dt1 (numpy array (float)): The time spent in the first cell for each particle.
        dt2 (numpy array (float)): The time spent, if any spent, in the second cell for each particle.
    """
    
    # Calculate the changes in x-coordinates.
    x_diff = v * mu * dt / np.sqrt(1 + tan_theta**2)
    
    # Get the histogram bin for the particles
    hist_loc = spatial_indices(x)
    
    # Get the movement direction of the particle
    direction = np.sign(mu)
    
    # Get the bin edge index by checking the propagation direction (assuming mu = -1 or = 1), index solved by 0.5 + 0.6*direction, 0.6 to mitigate float errors. Results always in 0 or 1 depending on direction
    rel_index = 0.5 + direction*(0.6)
    rel_index = rel_index.astype(np.int64)
    true_index = hist_loc + rel_index
    
    # Get the distance to the specified bin edge
    x_diff1 = (bin_steps[true_index] - x)
    
    # Solve the timesteps spent in each bin.
    dt1 = np.minimum(dt * x_diff1 / x_diff, dt)
    dt2 = dt - dt1
    
    return dt1, dt2


@jit(nopython=True)
def move_particles(x, v, mu, dt, tan_theta):
    """
    Moves the particles on the x-axis in HTF.

    Parameters: 
        x (numpy array (float)): The positions of the particles.
        v (numpy array (float)): The speed of the particles.
        mu (numpy array (float)): The pitch angle cosines of the particles.
        dt (numpy array (float)): The time step sizes. 
        tan_theta (numpy array (float)): The tangents of the shock obliquities. 

    Returns: 
        x + x_diff (numpy array (float)): The new locations along the x-axis after movement is applied. 
    """
    
    # Calculate the changes in x-coordinates.
    x_diff = v * mu * dt / np.sqrt(1 + tan_theta**2)
    
    return x + x_diff


@jit(nopython=True)
def scatter_particles(x, v, mu, t, dt, u, tan_theta, R1, R2):
    """
    Apply scattering to the particles in the plasma rest frame.

    Parameters:
        x (numpy array (float)): The positions of the particles.
        v (numpy array (float)): The speed of the particles.
        mu (numpy array (float)): The pitch angle cosines of the particles.
        t (numpy array (float)): Current simulation time for each particle separately. 
        dt (numpy array (float)): The time step sizes. 
        u (numpy array (float)): The local flow speed for each particle.
        tan_theta (numpy array (float)): The tangents of the shock obliquities.
        R1 (numpy array (float)): Random numbers generated out of function in order to keep seeding due to numba.
        R2 (numpy array (float)): Random numbers generated out of function in order to keep seeding due to numba.

    Returns:
        mu (numpy array (float)): Scattered pitch angle cosines of the particles.
    """

    # Calculate inverse mean free paths and product of time steps, particle speeds, and inverse mean free paths
    mfp_inv = eval_mfp_inv(x, v, t, tan_theta)
    b = 2.0 * dt * v * mfp_inv
    
    # Calculate theta and phi
    theta = np.sqrt( - b * np.log( 1.0 - R1 ))
    phi = 2.0 * np.pi * R2

    # Calculate the pitch angle cosine of the particles after scattering
    mu_sqrt = 1 - mu**2
    mu = mu * np.cos(theta) + np.sqrt(mu_sqrt) * np.sin(theta) * np.cos(phi)

    return mu


@jit(nopython=True)
def focus_particles(v, mu, dt, L_inv):
    """
    Calculates the focused pitch angle cosine of the particles in HTF.

    Parameters:
        v (numpy array (float)): The speed of the particles.
        mu (numpy array (float)): The pitch angle cosines of the particles.
        dt (numpy array (float)): The time step sizes. 
        L_inv (numpy array (float)): The inverses of the scale lengths.

    Returns:
        mu (numpy array (float)): The focused pitch angle cosines of the particles.
    """
    
     # Calculate the movement compared to length scale
    a = v * dt * L_inv
    clause_big = np.abs(a) > 1e-6
    clause_small = ~clause_big

    #mu = focus_big_a(mu, a)
    # Calculate the focused pitch angle cosine
    mu[clause_big] = focus_big_a(mu[clause_big], a[clause_big])
    mu[clause_small] = focus_small_a(mu[clause_small], a[clause_small])
    
    return mu


@jit(nopython=True)
def focus_big_a(mu, a):
    """
    The focusing equation for particles that move more than 1e-6 of the focusing length scale during a timestep (abs(a) > 1e-6).
    
     Parameters:
        mu (numpy array (float)): The pitch angle cosines of the particles.
        a (numpy array (float)): The moved distance in relation to the focusing scale length (a = v * dt / L)

    Returns:
        mu (numpy array (float)): The focused pitch angle cosines of the particles.
    """
    
    return 1 / a * (-1 + np.sqrt( 1 + a**2 * (1 + 2*mu/a)))


@jit(nopython=True)
def focus_small_a(mu, a):
    """
    The focusing equation for particles that move less than or equal to 1e-6 of the focusing length scale during a timestep (abs(a) <= 1e-6).
    
     Parameters:
        mu (numpy array (float)): The pitch angle cosines of the particles.
        a (numpy array (float)): The moved distance in relation to the focusing scale length (a = v * dt / L)

    Returns:
        mu (numpy array (float)): The focused pitch angle cosines of the particles.
    """
    
    # Taylor expansion of the focus_big_a() equation to for small values of a.
    mu = mu + 0.5 * (1 - mu**2) * (1 - a * mu) * a
     
    # Check whether the Taylor expansion resulted in a mu < -1
    if np.any(mu < -1):
        mu[mu < -1] = - 2 - mu[mu < 1]
    
    return mu


if continuity_testing:
    def downstream_return(x, v, mu, u):
        """
        Provides particles escaping through the downstream boundary with a chance to return based on their speed. Reinjects the particle back to the box with a pitch angle cosine that ends up back in the box depending on their speed, as speed is conserved in the plasma rest frame.

        Parameters: 
            x (numpy array (float)): The positions of the particles.
            v (numpy array (float)): The speed of the particles.
            mu (numpy array (float)): The pitch angle cosines of the particles.
            u (numpy array (float)): The local flow speed for each particle.

        Returns: 
            x (numpy array (float)): The updated positions of the particles, either out of the box or at the downstream boundary.
            v (numpy array (float)): The updated speed of the particles through scatterin for particles that go the chance to return.
            mu (numpy array (float)): The reinjected pitch angle cosines of the particles for particles that got the chance to return. 
        """
        
        # Change to plasma rest frame 
        v_prime, mu_prime = change_frame(v, mu, u)

        # Sample a random pitch angle cosine for the particles from a population that ends towards the shock when injecting from the edge of the simulation box
        R_max = (v_prime - u)/(u + v_prime)
        R_max[R_max < 0] = -999
        R_max[~(R_max < 0)] = R_max[~(R_max < 0)]**2
        R1 = np.random.random(size=len(v_prime)) 
        clause2 = R1 < R_max
        x[clause2] = x_max
        mu_prime[clause2] = - (u[clause2] + (v_prime[clause2] - u[clause2]) * np.sqrt(R1[clause2])) / (v_prime[clause2])
        
        # Change to HTF
        v, mu = change_frame(v_prime, mu_prime, -u)
    
        return x, v, mu
    
    
    def upstream_return(x, v, mu, u):
        """
        Return particles escaping from the upstream boundary. Reinjects the particle back to the box with a pitch angle cosine that ends up back in the box depending on their speed, as speed is conserved in the plasma rest frame.

        Parameters: 
            x (numpy array (float)): The positions of the particles.
            v (numpy array (float)): The speed of the particles.
            mu (numpy array (float)): The pitch angle cosines of the particles.
            u (numpy array (float)): The local flow speed for each particle.

        Returns: 
            x (numpy array (float)): The updated positions of the particles, either out of the box or at the downstream boundary.
            v (numpy array (float)): The updated speed of the particles through scatterin for particles that go the chance to return.
            mu (numpy array (float)): The reinjected pitch angle cosines of the particles for particles that got the chance to return. 
        """
        
        # Change to plasma rest frame 
        v_prime, mu_prime = change_frame(v, mu, u)
        
         # Sample a random pitch angle cosine for the particles from a population that ends towards the shock when injecting from the edge of the simulation box
        R_min = (u - v_prime)/(u + v_prime)
        R_min[R_min < 0] = 0
        R_min = R_min**2
        R = np.random.random(size=len(x)) * (1 - R_min) + R_min
        mu_prime = (-u + (v_prime + u) * np.sqrt(R)) / (v_prime)
        x[:] = x_min
        
        # Change to HTF
        v, mu = change_frame(v_prime, mu_prime, -u)        
    
        return x, v, mu
    
    
else:
    def downstream_return(x, v, mu, u):
        """
        Overloaded function. Does nothing when continuity_testing is set to False.
        """
        
        return x, v, mu 
    
    
    def upstream_return(x, v, mu, u):
        """
        Overloaded function. Does nothing when continuity_testing is set to False.
        """
        
        return x, v, mu 
        

@jit(nopython=True)
def spatial_indices(x):
    '''
    Calculates and returns the corresponding x index of each particle in the time integration histogram. Due to how np.digitize works, subtract 1.
    
    Parameters:
        x (numpy array (float)): The position of each particle to be added to the histogram.
        
    Returns:
        x_ind (numpy array (float)): The indices corresponding to the positions of the particles in the spatial axis of the time integration histogram.
    '''
    
    return np.digitize(x, bin_edges)-1
    

if histogram_frame == "plasma rest":
    print("Logging the histogram in plasma rest frame.")
    
    @jit(nopython=True)
    def integration_indices(x, v, mu, u, tan_theta, dt):
        """
        Calculate the corresponding indices to the time integration histogram for each particle in the plasma rest frame.
        
        Parameters:
            x (numpy array (float)): The positions of the particles.
            v (numpy array (float)): The speed of the particles.
            mu (numpy array (float)): The pitch angle cosines of the particles.
            u (numpy array (float)): The local fluid frame speed for each particle.
            tan_theta (numpy array (float)): (overloaded)
            
        Returns:
            xi (numpy array (float)): The spatial indices of the particles in the histogram.
            pi (numpy array (float)): The momentum indices of the particles in the histogram.
            mui (numpy array (float)): The pitch angle cosine indices of the particles in the histogram.
            pparai (numpy array (int)): The momentum indices of the particles in the momentum histogram.
            pperpi (numpy array (int)): The pitch angle cosine indices of the particles in the momentum histogram.
            dt_prime (numpy array (float)): The timestep size to be used in the logging to the (x,v,mu) histogram, lorentzian transformation taken into account.
            dt_prime (numpy array (float)): The timestep size to be used in the logging to the momentum histogram, lorentzian transformation taken into account. (overloaded)
        """
        
        # Calculate the x index
        xi = spatial_indices(x)

        # Change to plasma rest frame 
        v_prime, mu_prime, dt_prime = change_frame_dt(v, mu, u, dt)
        
        # Calculate the mu index in the plasma rest frame
        mui = ((mu_prime + 1) / 2 * mu_N) # mu_min = -1, mu_max - mu_min = 2
        
        # Calculate the particle momentum and then the momentum index in the plasma rest frame
        p = p_func(v_prime)
        pi = ((np.log10(p) - p_min) / (p_max - p_min) * p_N)   
        
        # Calculate the particle parallel and perpendicular momentum indices in the plasma rest frame
        ppara = p * mu_prime
        mu_sqrt = 1 - mu_prime**2
        pperp = p * np.sqrt(mu_sqrt)
        
        pparai = np.zeros(len(ppara))
        pperpi = np.zeros(len(pperp))
        
        within = (np.log10(np.abs(ppara)) > p_pmin)
        pparai[within] = (np.sign(ppara[within])*(np.log10(np.abs(ppara[within])) - p_pmin) / (p_pmax - p_pmin) * p_N/2)
        pparai[~within] = -999999
        pparai[pparai < 0] -= 1
            
        within = (np.log10(np.abs(pperp)) > p_pmin)
        pperpi[within] = (np.sign(pperp[within])*(np.log10(np.abs(pperp[within])) - p_pmin) / (p_pmax - p_pmin) * p_N/2)
        pperpi[~within] = -999999

        return xi, pi, mui, pparai, pperpi, dt_prime, dt_prime
    
    
elif histogram_frame == "normal incidence":
    print("Logging the histogram in normal incidence frame.")
    
    @jit(nopython=True)
    def integration_indices(x, v, mu, u, tan_theta, dt):
        """
        Calculate the corresponding indices to the time integration histogram for each particle in the normal incidence frame for the (x, v, mu) histogram and in the plasma rest frame for the momentum histogram.
        
        Parameters:
            x (numpy array (float)): The positions of the particles.
            v (numpy array (float)): The speed of the particles.
            mu (numpy array (float)): The pitch angle cosines of the particles.
            u (numpy array (float)): The local fluid frame speed for each particle.
            tan_theta (numpy array (float)): The local tangent of magnetic field obliquity for each particles' location.
            
        Returns:
            xi (numpy array (float)): The spatial indices of the particles in the histogram.
            pi (numpy array (float)): The momentum indices of the particles in the histogram.
            mui (numpy array (float)): The pitch angle cosine indices of the particles in the histogram.
            pparai (numpy array (int)): The momentum indices of the particles in the momentum histogram.
            pperpi (numpy array (int)): The pitch angle cosine indices of the particles in the momentum histogram.
            dt_prime (numpy array (float)): The timestep size to be used in the logging to the (x,v,mu) histogram, lorentzian transformation taken into account.
            dt_prf (numpy array (float)): The timestep size to be used in the logging to the momentum histogram, lorentzian transformation taken into account. 
        """
        
        # Calculate the x index
        xi = spatial_indices(x)
        
        # Sample a phase angle assuming gyrotropic medium in HTF.
        phi = np.random.random()*2*np.pi 
        
        # Change to normal incidence frame
        theta_temp = np.arctan(tan_theta)
        v_prime, mu_prime, dt_prime = htf_to_nif(v, mu, u, theta_temp, phi, dt)
         
        # Calculate the mu index in normal incidence frame
        mui = ((mu_prime + 1) / 2 * mu_N) # mu_min = -1, mu_max - mu_min = 2
        
        # Calculate the particle momentum and then the momentum index in normal incidence frame
        p = p_func(v_prime)
        pi = ((np.log10(p) - p_min) / (p_max - p_min) * p_N)  
        
        # Calculate the particle parallel and perpendicular momentum indices in plasma rest frame
        v_prime, mu_prime, dt_prf = change_frame_dt(v, mu, u, dt)
        p_prime = p_func(v)
        ppara = p_prime * mu_prime
        mu_sqrt = 1 - mu_prime**2
        pperp = p_prime * np.sqrt(mu_sqrt)
        
        pparai = np.zeros(len(ppara))
        pperpi = np.zeros(len(pperp))
        
        within = (np.log10(np.abs(ppara)) > p_pmin)
        pparai[within] = (np.sign(ppara[within])*(np.log10(np.abs(ppara[within])) - p_pmin) / (p_pmax - p_pmin) * p_N/2)
        pparai[~within] = -999999
        pparai[pparai < 0] -= 1
            
        within = (np.log10(np.abs(pperp)) > p_pmin)
        pperpi[within] = (np.sign(pperp[within])*(np.log10(np.abs(pperp[within])) - p_pmin) / (p_pmax - p_pmin) * p_N/2)
        pperpi[~within] = -999999

        return xi, pi, mui, pparai, pperpi, dt_prime, dt_prf
    

else:
    if histogram_frame != "HTF":
        print("Typo in histogram_frame!")
    print("Logging the histogram in HTF.")        
        
    @jit(nopython=True)
    def integration_indices(x, v, mu, u, tan_theta, dt):
        """
        Calculate the corresponding indices to the time integration histogram for each particle in the de Hoffmann-Teller frame for the (x,p,mu) histogram and in the plasma rest frame for the momentum histogram.
        
        Parameters:
            x (numpy array (float)): The positions of the particles.
            v (numpy array (float)): The speed of the particles.
            mu (numpy array (float)): The pitch angle cosines of the particles.
            u (numpy array (float)): For overloading of the function, not used here.
            tan_theta (numpy array (float)): For overloading of the function, not used here.
            
        Returns:
            xi (numpy array (float)): The spatial indices of the particles in the histogram.
            pi (numpy array (float)): The momentum indices of the particles in the histogram.
            mui (numpy array (float)): The pitch angle cosine indices of the particles in the histogram.
            pparai (numpy array (int)): The momentum indices of the particles in the momentum histogram.
            pperpi (numpy array (int)): The pitch angle cosine indices of the particles in the momentum histogram.
            dt (numpy array (float)): The timestep size to be used in the logging to the (x,p,mu) histogram. (overloaded)
            dt_prime (numpy array (float)): The timestep size to be used in the logging to the momentum histogram, lorentzian transformation taken into account. 
        """
        
        # Calculate the x index
        xi = spatial_indices(x)

        # Calculate the mu index in HTF
        mui = ((mu + 1) / 2 * mu_N) # mu_min = -1, mu_max - mu_min = 2
        
        # Calculate the particle momentum and then the momentum index in HTF
        p = p_func(v)
        pi = ((np.log10(p) - p_min) / (p_max - p_min) * p_N)   
        
        # Calculate the particle parallel and perpendicular momentum indices in the plasma rest frame
        v_prime, mu_prime, dt_prf = change_frame_dt(v, mu, u, dt)
        p_prime = p_func(v_prime) 
        ppara = p_prime * mu_prime
        mu_sqrt = 1 - mu_prime**2
        pperp = p_prime * np.sqrt(mu_sqrt)
        
        pparai = np.zeros(len(ppara))
        pperpi = np.zeros(len(pperp))
        
        within = (np.log10(np.abs(ppara)) > p_pmin)
        pparai[within] = (np.sign(ppara[within])*(np.log10(np.abs(ppara[within])) - p_pmin) / (p_pmax - p_pmin) * p_N/2)
        pparai[~within] = -999999
        pparai[pparai < 0] -= 1
            
        within = (np.log10(np.abs(pperp)) > p_pmin)
        pperpi[within] = (np.sign(pperp[within])*(np.log10(np.abs(pperp[within])) - p_pmin) / (p_pmax - p_pmin) * p_N/2)
        pperpi[~within] = -999999

        return xi, pi, mui, pparai, pperpi, dt, dt_prf
    

@jit(nopython=True)
def integrate_particles(xi, pi, mui, pparai, pperpi, dt, dt_prf, int_hist, int_v_para_hist, int_p_hist, v, mu):
    """
    Add the time step data to the corresponding location in the time integration matrix. Ensure that index array have been changed to integers before this function. Histograms will be altered in-place, so no returns.
    
    Parameters:
        xi (numpy array (int)): The spatial indices of the particles in the histogram.
        pi (numpy array (int)): The momentum indices of the particles in the histogram.
        mui (numpy array (int)): The pitch angle cosine indices of the particles in the histogram.
        pparai (numpy array (int)): The momentum indices of the particles in the momentum histogram.
        pperpi (numpy array (int)): The pitch angle cosine indices of the particles in the momentum histogram.
        dt (numpy array (float)): The timestep sizes of the particles.
        dt_prf (numpy array (float)): The lorentzian transformed timestep sizes of the particles in the plasma rest frame.
        int_hist (numpy array [x_N x p_N x mu_N]): The time integration histogram.
        int_v_para_hist (numpy array [x_N x v_para_N x mu_N]): The time integration histogram with v_para instead of momentum.
        int_p_hist (numpy array [x_N x p_N x p_N]): The time integration momentum histogram.
        v (numpy array (float)): The speed of the particles.
        mu (numpy array (float)): The pitch angle cosines of the particles.
        
    Returns:
        -
    """
    
    # Add corresponding timesteps to specified indices
    hist_count =  v * mu * dt 
    for i in range(len(dt)):
        if xi[i] >= 0 and xi[i] <= x_N and pi[i] >= 0 and pi[i] <= p_N and mui[i] >= 0 and mui[i] <= mu_N:
            int_hist[xi[i], pi[i], mui[i]] += dt[i]
        if xi[i] >= 0 and xi[i] <= x_N and pi[i] >= 0 and pi[i] <= p_N and mui[i] >= 0 and mui[i] <= mu_N:
            int_v_para_hist[xi[i], pi[i], mui[i]] += hist_count[i]
        if xi[i] >= 0 and xi[i] <= x_N and pparai[i] <= p_N//2 and pparai[i] >= -p_N//2 and pperpi[i] <= p_N//2 and pperpi[i] >= -p_N//2:
            int_p_hist[xi[i], pparai[i], pperpi[i]] += dt_prf[i]

                

def simulate_particles(particles, i, t_end):
    """
    Simulates an array of particles with focusing and scattering. If a particle exits the simulation box, freeze it.

    Parameters:
        particles (numpy array [ N * [x, v, mu, t, t_start, seed]]): The array of all particle data.
        i (int): Snapshot index for seeding.
        t_end (float): Maximum physical time for simulation (or maximum time for current snapshot). (UNUSED, currently, but can be uncommented at the end of the loop if needed)

    Returns:
        particles (numpy array [x, v, mu, t, t_start, seed]): The updated array of all particle data.
        int_hist (numpy array [x_N x p_N x mu_N]): The time integration histogram.
        int_v_para_hist (numpy array [x_N x p_N x mu_N]): The time integration histogram with v_para instead of momentum.
        int_p_hist (numpy array [x_N x p_N x p_N]): The time integration momentum histogram.
    """
    
    # Random seed again for multiprocessing
    np.random.seed(int(i*particles[0,5]))
    
    # Create the integration histograms
    int_hist = np.zeros((x_N+1, p_N+1, mu_N+1))
    int_v_para_hist = np.zeros((x_N+1, p_N+1, mu_N+1))
    int_p_hist = np.zeros((x_N+1, p_N+1, p_N+1))
    
    # Create a boolean array to determine which particles are still in the box and simulated
    in_box = np.ones(len(particles)).astype(bool)
    
    # Operate while there are particles remaining within the box
    while len(in_box[in_box == True]) > 0:
        
        # Set x, v, mu, and t as aliases to make written code clearer
        x = particles[in_box,0]
        v = particles[in_box,1]
        mu = particles[in_box,2]
        t = particles[in_box,3]
        
        # Calculate the tangents of the shock obliquities and the flow profile u
        u, tan_theta = eval_flow_profile(x, t)

        # Calculate the inverse focusing length
        L_inv = eval_L_inv(x, t, tan_theta)
        
        # Calculate the timesteps and inverse scale lengths in HTF
        dt = eval_timestep(x, v, mu, t, tan_theta, u, L_inv)

        # Apply focusing if boolean switch allows
        mu = focus_particles(v, mu, dt, L_inv)
        
        # Split the timestep to the portions in the first cell and possible second cell
        dt1, dt2 = split_timestep(x, v, mu, dt, tan_theta)
        
        # Add the first timesteps to the integration histogram
        xi, pi, mui, pparai, pperpi, dti, dt_prf = integration_indices(x, v, mu, u, tan_theta, dt1)     
              
        # Add the first timestep to the time integration histograms
        integrate_particles(xi.astype(int), pi.astype(int), mui.astype(int), 
                            pparai.astype(int), pperpi.astype(int),
                            dti, dt_prf, int_hist, int_v_para_hist, int_p_hist, v, mu)
        
        # Move the particles
        x = move_particles(x, v, mu, dt, tan_theta)
        
        # Progress time for all particles
        t += dt
        
        # Calculate the tangents of the shock obliquities and the flow profile u in the new position
        u, tan_theta = eval_flow_profile(x, t)
        
        # If upstream return is defined then return from the downstream also (see downstream_return())
        out_clause1 = (x > x_max)
        x[out_clause1], v[out_clause1], mu[out_clause1] = downstream_return(x[out_clause1], v[out_clause1], mu[out_clause1], u[out_clause1])

        out_clause2 = (x < x_min)
        x[out_clause2], v[out_clause2], mu[out_clause2] = upstream_return(x[out_clause2], v[out_clause2], mu[out_clause2], u[out_clause2])
        
        # Check that particles are in the simulation box after movement, the particles that are returned to the box during this timestep should not be considerd for the rest of the loop
        in_loop =  np.array((x <= x_max + 1e-10) & (x >= x_min - 1e-10) & ~out_clause1 & ~out_clause2)
        in_loop2 = np.array((x <= x_max + 1e-10) & (x >= x_min - 1e-10))
        
        # Propagate particles with the velocity it exited the box with till the maximum simulation time
        #x[~in_loop] = move_particles(x[~in_loop], v[~in_loop], mu[~in_loop], T_max - t[~in_loop], tan_theta[~in_loop])

        # Add the second timesteps to the integration histogram
        xi, pi, mui, pparai, pperpi, dti, dt_prf = integration_indices(x[in_loop], v[in_loop], mu[in_loop], u[in_loop], tan_theta[in_loop], dt2[in_loop])       
        integrate_particles(xi.astype(int), pi.astype(int), mui.astype(int), 
                            pparai.astype(int), pperpi.astype(int),
                            dti, dt_prf, int_hist, int_v_para_hist, int_p_hist, v[in_loop], mu[in_loop])
        
        # Change to the plasma rest frame of reference for scattering
        v[in_loop], mu[in_loop], dt_prime = change_frame_dt(v[in_loop], mu[in_loop], u[in_loop], dt[in_loop])

        # Scatter the particles in the plasma rest frame
        R1 = np.random.random(size = len(x[in_loop]))
        R2 = np.random.random(size = len(x[in_loop]))
        mu[in_loop] = scatter_particles(x[in_loop], v[in_loop], mu[in_loop], t[in_loop], dt_prime, u[in_loop], tan_theta[in_loop], R1, R2)

        # Change back to de Hoffmann-Teller frame
        v[in_loop], mu[in_loop] = change_frame(v[in_loop], mu[in_loop], -u[in_loop])
        
        # Update the particles array with the simulated data
        particles[in_box,0] = x
        particles[in_box,1] = v
        particles[in_box,2] = mu
        particles[in_box,3] = t
        
        # Check which particles to simulate on next round based on the in_loop estimated earlier and the maximum time of the snapshot
        in_box[in_box == True] *= in_loop2

        # If particles exceed maximum defined simulation time, remove them from the simulation
        #in_box[particles[:,3] > t_end] = False

    return particles, int_hist, int_v_para_hist, int_p_hist


#--------------------------------------------------------------------------------------
# Injection parameters
E_inj_0 = eval_E_natural(E_inj_0 * 1000, m_e, c, e) # Unitless minimum injection energy of particles
v_inj_0 = E_to_v(E_inj_0) # Unitless minimum injection velocity, With units corresponds to c * np.sqrt(E_inj * (E_inj + 2*m_e*c**2)) / (E_inj + m_e*c**2) and divided by c to use units of c

# Time integration histogram parameters
p_inj_0 = v_to_p(v_inj_0) # Unitless minimum injection momentum of particles, with units corresponds to gamma * m_e * v_inj
p_pmin = np.log10(p_min_f*p_inj_0) # Unitless minimum of time integral momentum grid
p_pmax = np.log10(p_max_f*p_inj_0) # Unitless maximum of time integral momentum grid
x_max = down_bound_scaler * box_size
x_min_edge = -up_bound_scaler * box_size
x_min = x_min_edge
u, _ = eval_flow_profile(x_min_edge, 0) # Uses t = 0 currently, as uses cases for temporally evolving profiles have not been mapped out yet
v_inj_HTF, _ = change_frame(v_inj_0, 1, -u)
p_inj_HTF = eval_gamma(v_inj_HTF) * v_inj_HTF
p_min = np.log10(p_min_f * p_inj_HTF)
p_max = np.log10(p_max_f * p_inj_HTF)
@jit(nopython=True)
def p_func(v):
    return v_to_p(v)
bin_edges = spatial_bins()
dx = bin_edges[1:]-bin_edges[:-1]
x_N = len(bin_edges)-1
bin_steps = np.append(bin_edges, bin_edges[-1] + dx[-1]) # Add final bin for timestepping

if not only_ambient_mfp:
    # Scale the cosh and tanh profiles with mfp_matcher.
    t = 0
    x = np.linspace(-10,10, 100)
    _, tan_theta = eval_flow_profile(x, 0) # Uses t = 0 currently, as uses cases for temporally evolving profiles have not been mapped out yet

    inv_mfp_ambient_prof = 2 * inv_mfp_1 / (mfp_prof(x, t) * (1 - downstream_mfp_scaler) + 1 + downstream_mfp_scaler)
    mfp_inv = np.sqrt( ( 1 + tan_theta**2 ) / ( 1 + tan_theta_1**2 )) / (a_ratio * eval_gamma(v_inj_0) * v_inj_0 ) * B_prof_der(x,t) 

    mfp_matcher = max(inv_mfp_ambient_prof)/max(mfp_inv)

#--------------------------------------------------------------------------------------

def main():
    
    # Create the path to save data in
    path = write_init()

    # Copy the parameter file to the snapshot directory to keep track of parameters in the analysis
    shutil.copy(f"prms.py", f"{path}prms.py")
    save_prm(path)
    
    # Copy the simulation file to the snapshot directory for repeatability
    shutil.copy(f"easi.py", f"{path}easi.py")
    
    # Add __init__.py to import corresponding version when plotting
    open(f"{path}__init__.py", "w").close()

    # Print parameters to see
    print("box_size:", up_bound_scaler * box_size, "d_i")
    print("r_g:", rg)
    print("r_b:", rb)

    # Create particle array and inject particles using the function inject()
    print("Starting injection...")
    particles = inject()

    if save_injection:
        np.save(f"{path}inj_spec.npy", particles)

    # Split the particle list to equal parts based on amount of CPUs for multiprocessing
    cpu_count = mp.cpu_count()
    mp_particles = []
    for i in range(cpu_count):
        start = int(i * N_p/cpu_count)
        end = int((i+1) * N_p/cpu_count)
        mp_particles.append(particles[start : end])

    # Create the snapshot times for saving particle data into files
    snapshots = np.linspace(T_max/N_snapshots, T_max, N_snapshots)

    start = time()
    print(f"Starting {path[-7:-1]}")
    print("Number of CPUs:", cpu_count)

    # Go through all snapshots
    for i, t_end in enumerate(snapshots):

        print("Working on snapshot:", str(i+1)+"/"+str(len(snapshots))+"...")
        
        # Parallel processing with multiprocessing.Pool()
        args = [(parts, i+1, t_end) 
                for parts in mp_particles]
        with mp.Pool() as pool:
            results = pool.starmap(simulate_particles, args)
        
        # Combine particle data to the original particles array
        for j in range(cpu_count):
            particles[int(j*(N_p/(cpu_count))) : int((j+1) * N_p/(cpu_count))] = results[j][0] 
        
        # Calculate the histogram by summing all the histograms of different CPU processes
        int_hist = np.sum(np.array(results, dtype=object)[:,1], axis = 0)
        int_v_para_hist = np.sum(np.array(results, dtype=object)[:,2], axis = 0)
        int_p_hist = np.sum(np.array(results, dtype=object)[:,3], axis = 0)
        
        # Save the particle data to the location defined by write_init()
        np.save(f"{path}snapshot_{i}", particles) 

        # Save the time integration histogram to the location defined by write_init()
        np.save(f"{path}int_hist_{i}", int_hist)
        np.save(f"{path}int_v_para_hist_{i}", int_v_para_hist)
        np.save(f"{path}int_p_hist_{i}", int_p_hist)

    # Print the simulation wall time after all loops
    print(f"Wall time: {time()-start:.2f} s = {(time()-start)/60:.2f} min")

    
if __name__ == "__main__":
    main()
    print("DONE!")
