# Electron Acceleration SImulation (EASI)
# Main simulation file
# Code author: Seve Nyberg

# Github URL: https://github.com/SeveNyberg/easi
# Model article DOI: (in prep.)

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
theta_1 = np.deg2rad(theta_1) # Shock obliquity in radians
tan_theta_1 = np.tan(theta_1) # Tangent of shock obliquity
ion_inert = eval_ion_inert(n_part, e, m_i, eps0, c) # ion inertial length for natural unit conversion in meters
T_max = T_max * c / ion_inert # Unitless maximum simulation time
inv_L_ambient = 1/(L_ambient * 1000 / ion_inert) # Unitless ambient focusing length
M_A = ux1/(np.cos(theta_1)*V_A1) # The Alfvén Mach number in the de Hoffmann-Teller Frame (HTF)
V_A1 = V_A1*1000/c # unitless Alfvén speed
rg = eval_rg(M_A, theta_1, beta) # Gas compression ratio computed from parameters
m_i = m_i/m_e # Unitless ion mass
alpha = m_i * V_A1 # With units corresponds to m_i/m_e * V_A1/c
a_ratio = a_prm/alpha # Mean free path scaling term, alpha is solved from physics, a used as scaler
rb = eval_rb(M_A, rg, theta_1) # Magnetic compression ratio
ux1 = ux1*1000/c # Unitless upstream flow speed in HTF
v_L = E_to_v(eval_E_natural(E_L * 1000, m_e, c, e)) * c # The speed for which the focusing length needed for the box size is calculated for
box_size = eval_box_size(L_ambient * 1000, ux1 * c, theta_1, v_L) / ion_inert # The focusing length cotnrolled box size, this can be multiplied with x_bound_scalers to increase or decrease box size.


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
    particles[:,0] = x_min
    
    # Evaluate the flow profile at the edge of the box
    u, _ = eval_flow_profile(x_min)
    
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
    def eval_mfp_inv(x, v, tan_theta):
        """
        Evaluates the inverse mean free path as a hyperbolic tangent profile. Sets the maximum mean free path as the ambient focusing length.

        Parameters:
            x (numpy array (float)): The positions of the particles.
            v (numpy array (float)): The speeds of the particles.
            tan_theta (numpy array (float)): The local tangents of the shock obliquities. 

        Returns:
            mfp_inv (numpy array (float)): The inverses of the mean free paths.
        """
        
        t = "not implemented"
        
        # Set a maximum mean free path scaled with the magnetic field profile
        inv_L_ambient_prof = 2 * inv_L_ambient / (mfp_prof(x, t) * (1 - downstream_mfp_scaler) + 1 + downstream_mfp_scaler)
        
        mfp_inv = inv_L_ambient_prof
                   
        
        return mfp_inv
    
    
else:
    @jit(nopython=True)
    def eval_mfp_inv(x, v, tan_theta):
        """
        Evaluates the inverse mean free path as a combination of hyperbolic tangent and a hyperbolic cosine at the shock. Sets the maximum mean free path as the ambient focusing length.

        Parameters:
            x (numpy array (float)): The positions of the particles.
            v (numpy array (float)): The speeds of the particles.
            tan_theta (numpy array (float)): The local tangents of the shock obliquities. 

        Returns:
            mfp_inv (numpy array (float)): The inverses of the mean free paths.
        """
        
        t = "not implemented"
        
        # Calculate gamma factor needed for mean free path
        gamma = eval_gamma(v)
        
        # Set a maximum mean free path scaled with the magnetic field profile
        inv_L_ambient_prof = 2 * inv_L_ambient / (mfp_prof(x, t) * (1 - downstream_mfp_scaler) + 1 + downstream_mfp_scaler)
        
        # Calculate the inverse of the mean free path
        mfp_inv = 1/mfp_scaler * mfp_matcher * np.sqrt( ( 1 + tan_theta**2 ) / ( 1 + tan_theta_1**2 )) / (a_ratio * gamma * v ) * B_prof_der(x,t) + inv_L_ambient_prof
        
        return mfp_inv


@jit(nopython=True)
def eval_L_inv(x, t, tan_theta):
    '''
    Evaluates the inverse focusing length in the de Hoffmann-Teller frame (HTF).
    
    Parameters:
        x (numpy array (float)): Positions of particles.
        t (numpy array (float)): Current time of particles.
        tan_theta (numpy array (float)): Tangents of local magnetic field obliquities of particles.
        
    Returns:
        L_inv (numpy array (float)): The inverse focusing lengths for all particles.
    '''
    
    # Calculate the scale length
    t = "not implemented"
    L_inv = -( tan_theta / (1 + tan_theta**2 )**(3/2) * (rb - 1) * tan_theta_1 * B_prof_der(x,t) + inv_L_ambient) 
    
    return L_inv


@jit(nopython=True)
def eval_flow_profile(x):
    """
    Calculates the local values of flow speed and tangent of shock obliquity.
    
    Parameters:
        x (numpy array (float)): The positions of the particles.
        
    Returns:
        u (numpy array (float)): The local flow speeds.
        tan_theta (numpy array (float)): The local tangents of the shock obliquities.
    """
    
    # Calculate the tangent of the shock obliquity and the flow profile u
    t = "not implemented"
    tan_theta = tan_theta_1 * (1 + (rb - 1) * B_prof(x,t) ) 
    u = ux1 / np.cos(theta_1) * (1 - (1 - rb/rg) * u_prof(x,t) )
    
    return u, tan_theta


@jit(nopython=True)
def eval_timestep(x, v, mu, tan_theta, u, L_inv):
    """
    Calculates the largest plausible time steps for given particles taking into account the spatial histogram grid size. Also calculates the focusing scale length, with a minimum scale length of the ambient scale length from the parameters.

    Parameters:
        x (numpy array (float)): The positions of the particles.
        v (numpy array (float)): The speeds of the particles.
        mu (numpy array (float)): The pitch angle cosines of the particles.
        tan_theta (numpy array (float)): The tangents of the shock obliquities.
        u (numpy array (float)): The speed of the flow relative to the shock.

    Returns:
        dt (numpy array (float)): The time step sizes for particles.
        L_inv (numpy array (float)): The inverses of the scale lengths.
    """
    
    # Set an initial timestep size relative to the histogram bin length or if distance is less to the shock than the bin length, then to abs(x) with minimum value of small_dt, set to dx/c with units. 
    if custom_bins:
        # Get the histogram bin for the particles
        hist_loc = spatial_indices(x)
        # Get the movement direction of the particle
        direction = np.sign(mu)
        # Get the bin edge index by checking the propagation direction (assuming mu = -1 or = 1), index solved by 0.5 + 1.6*direction, 1.6 to mitigate float errors. Results always in -1 or 2 depending on direction
        rel_index = 0.5 + direction*1.6
        rel_index = rel_index.astype(np.int64)
        true_index = hist_loc + rel_index
        # Solve the timestep
        # Need to check if dir -1 and index 0 then use dx[0]
        dt_hist = np.minimum( np.abs(bin_steps[hist_loc - 1] - x) / (v) * 0.9, np.abs(bin_steps[hist_loc + 2] - x) / (v) * 0.9 ) 
        dt_hist[true_index == -1] = (bin_steps[0] - dx[0] - x[true_index == -1]) / (direction[true_index == -1] * v[true_index == -1]) * 0.9
    else:
        dt_hist = dx * 0.9
        
    dt = np.zeros(len(x)) + np.minimum(dt_hist, np.maximum(np.abs(x), small_dt))
    
    # Change to the plasma rest frame of reference to calculate scattering mean free path
    v, mu = change_frame(v, mu, u)
    
    # Calculate the mean free path
    mfp_inv = eval_mfp_inv(x, v, tan_theta)
    
    # Change back to de Hoffmann-Teller frame
    v, mu = change_frame(v, mu, -u)

    # Check that time step is not too large compared to the mean free path, otherwise divide by 2 until small enough
    while len(dt[dt * v * mfp_inv > 0.02]) > 0: 
        dt[dt * v * mfp_inv > 0.02] /= 2    
    
    # Limit the change in mean free path to 10 % of the original value
    change = 0.1
    x_temp = move_particles(x, v, mu, dt, tan_theta)
    u_temp, tan_theta_temp = eval_flow_profile(x_temp)
    v_temp, mu_temp = change_frame(v, mu, u_temp)
    mfp_inv_diff = eval_mfp_inv(x_temp, v_temp, tan_theta_temp) - mfp_inv
    v_temp, mu_temp = change_frame(v_temp, mu_temp, -u_temp)
    while len(mfp_inv[np.abs(mfp_inv_diff) > change * mfp_inv]) > 0:
        clause = np.abs(mfp_inv_diff) > change * mfp_inv
        dt[clause] /= 2
        x_temp[clause] = move_particles(x[clause], v_temp[clause], mu_temp[clause], dt[clause], tan_theta[clause])
        u_temp[clause], tan_theta_temp[clause] = eval_flow_profile(x_temp[clause])
        v_temp[clause], mu_temp[clause] = change_frame(v_temp[clause], mu_temp[clause], u_temp[clause])
        mfp_inv_diff[clause] = eval_mfp_inv(x_temp[clause], v_temp[clause], tan_theta_temp[clause]) - mfp_inv[clause]
        v_temp[clause], mu_temp[clause] = change_frame(v_temp[clause], mu_temp[clause], -u_temp[clause])
        
    # Check that the time step is not too large compared to the scale length
    while len(dt[dt * v * L_inv > 0.02]) > 0:
        dt[dt * v * L_inv > 0.02] /= 2
        
    return dt


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
        (numpy array (float)): The new locations along the x-axis after movement is applied. 
    """
    
    # Calculate the changes in x-coordinates.
    x_diff = v * mu * dt / np.sqrt(1 + tan_theta**2)
    
    return x + x_diff


@jit(nopython=True)
def scatter_particles(x, v, mu, dt, u, tan_theta, R1, R2, R3):
    """
    Apply scattering to the particles in the plasma rest frame.

    Parameters:
        x (numpy array (float)): The positions of the particles.
        v (numpy array (float)): The speed of the particles.
        mu (numpy array (float)): The pitch angle cosines of the particles.
        dt (numpy array (float)): The time step sizes. 
        tan_theta (numpy array (float)): The tangents of the shock obliquities.
        R1 (numpy array (float)): Random numbers generated out of function in order to keep seeding due to numba.
        R2 (numpy array (float)): Random numbers generated out of function in order to keep seeding due to numba.
        R3 (numpy array (float)): Passed due to overload of function, no function. 

    Returns:
        x (numpy array (float)): The original positions of the particles (returned due to overloading of function.)
        v (numpy array (float)): The original speeds of the particles (returned due to overloading of function.)
        mu (numpy array (float)): Scattered pitch angle cosines of the particles.
    """

    # Calculate inverse mean free paths and product of time steps, particle speeds, and inverse mean free paths
    mfp_inv = eval_mfp_inv(x, v, tan_theta)
    b = 2.0 * dt * v * mfp_inv * eval_gamma(u)
    
    # Calculate theta and phi
    theta = np.sqrt( - b * np.log( 1.0 - R1 ))
    phi = 2.0 * np.pi * R2

    # Calculate the pitch angle cosine of the particles after scattering
    mu = mu * np.cos(theta) + np.sqrt(1.0 - (mu**2)) * np.sin(theta) * np.cos(phi)

    return x, v, mu


@jit(nopython=True)
def focus_particles(v, mu, dt, L_inv):
    """
    Calculates the focused pitch angle cosine of the particles in HTF.

    Parameters:
        v (numpy array (float)): The speed of the particles.
        mu (numpy array (float)): The pitch angle cosines of the particles.
        dt (numpy array (float)): The time step sizes. 
        tan_theta (numpy array (float)): The tangents of the shock obliquities. (UNUSED)
        L_inv (numpy array (float)): The inverses of the scale lengths.

    Returns:
        mu (numpy array (float)): The focused pitch angle cosines of the particles.
    """
    
    # Calculate the movement compared to length scale
    a = v * dt * L_inv

    # Calculate the focused pitch angle cosine
    mu = 1 / a * (-1 + np.sqrt( 1 + a**2 * (1 + 2*mu/a)))
  
    return mu


if custom_bins:    
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
    
    
else:
    @jit(nopython=True)
    def spatial_indices(x):
        '''
        Calculates and returns the corresponding x index of each particle in the time integration histogram.
        
        Parameters:
            x (numpy array (float)): The position of each particle to be added to the histogram.
            
        Returns:
            x_ind (numpy array (float)): The floats corresponding to indices corresponding to the positions of the particles in the spatial axis of the time integration histogram.
        '''
        
        return ((x - x_min) / (x_max - x_min) * x_N)


@jit(nopython=True)
def integration_indices(x, v, mu, u):
    """
    Calculate the corresponding indices to the time integration histogram for each particle.
    
    Parameters:
        x (numpy array (float)): The positions of the particles.
        v (numpy array (float)): The speed of the particles.
        mu (numpy array (float)): The pitch angle cosines of the particles.
        u (numpy array (float)): The local fluid frame speed for each particle.
        
    Returns:
        xi (numpy array (float)): The spatial indices of the particles in the histogram.
        pi (numpy array (float)): The momentum indices of the particles in the histogram.
        mui (numpy array (float)): The pitch angle cosine indices of the particles in the histogram.
        pparai (numpy array (int)): The momentum indices of the particles in the momentum histogram.
        pperpi (numpy array (int)): The pitch angle cosine indices of the particles in the momentum histogram.
    """
    
    # Calculate the x index using either custom bins or default bins
    xi = spatial_indices(x)

    # Calculate the mu index in HTF
    mui = ((mu + 1) / 2 * mu_N) # mu_min = -1, mu_max - mu_min = 2
    
    # Calculate the particle momentum and then the momentum index in HTF
    p = eval_gamma(v) * v
    pi = ((np.log10(p) - p_min) / (p_max - p_min) * p_N)   
    
    # Calculate the particle parallel and perpendicular momentum indices in the fluid rest frame
    v_prime, mu_prime = change_frame(v, mu, u)
    p = eval_gamma(v_prime) * v_prime
    ppara = p * mu_prime
    pperp = p * np.sqrt(1 - mu_prime**2)
    
    pparai = np.zeros(len(ppara))
    pperpi = np.zeros(len(pperp))
    
    within = (np.log10(np.abs(ppara)) > p_pmin)
    pparai[within] = (np.sign(ppara[within])*(np.log10(np.abs(ppara[within])) - p_pmin) / (p_pmax - p_pmin) * p_N/2)
    pparai[~within] = -999999
    pparai[pparai < 0] -= 1
        
    within = (np.log10(np.abs(pperp)) > p_pmin)
    pperpi[within] = (np.sign(pperp[within])*(np.log10(np.abs(pperp[within])) - p_pmin) / (p_pmax - p_pmin) * p_N/2)
    pperpi[~within] = -999999

    return xi, pi, mui, pparai, pperpi


@jit(nopython=True)
def integrate_particles(xi, pi, mui, pparai, pperpi, dt, int_hist, int_p_hist):
    """
    Add the time step data to the corresponding location in the time integration matrix. Ensure that index array have been changed to integers before this function. Histograms will be altered in-place, so no returns.
    
    Parameters:
        xi (numpy array (int)): The spatial indices of the particles in the histogram.
        pi (numpy array (int)): The momentum indices of the particles in the histogram.
        mui (numpy array (int)): The pitch angle cosine indices of the particles in the histogram.
        pparai (numpy array (int)): The momentum indices of the particles in the momentum histogram.
        pperpi (numpy array (int)): The pitch angle cosine indices of the particles in the momentum histogram.
        dt (numpy array (float)): The timestep sizes of the particles.
        int_hist (numpy array [x_N x p_N x mu_N]): The time integration histogram.
        int_p_hist (numpy array [x_N x p_N x p_N]): The time integration momentum histogram.
    """
    
    # Add corresponding timesteps to specified indices
    for i in range(len(dt)):
        if xi[i] >= 0 and xi[i] <= x_N and pi[i] >= 0 and pi[i] <= p_N and mui[i] >= 0 and mui[i] <= mu_N:
            int_hist[xi[i], pi[i], mui[i]] += dt[i]
        if xi[i] >= 0 and xi[i] <= x_N and pparai[i] <= p_N//2 and pparai[i] >= -p_N//2 and pperpi[i] <= p_N//2 and pperpi[i] >= -p_N//2:
            int_p_hist[xi[i], pparai[i], pperpi[i]] += dt[i]


def simulate_particles(particles, i, t_end):
    """
    Simulates an array of particles with focusing and scattering. If a particle exits the simulation box, freeze it.

    Parameters:
        particles (numpy array [ N * [x, v, mu, t, t_start, seed]]): The array of all particle data.
        i (int): Snapshot index for seeding.
        t_end (float): Maximum time for simulation (or maximum time for current snapshot). (UNUSED, currently, but can be uncommented at the end of the loop if needed)

    Returns:
        particles (numpy array [x, v, mu, t, t_start, seed]): The updated array of all particle data.
        int_hist (numpy array [x_N x p_N x mu_N]): The time integration histogram.
        int_p_hist (numpy array [x_N x p_N x p_N]): The time integration momentum histogram.
    """
    
    # Random seed again for multiprocessing
    np.random.seed(int(i*particles[0,5]))
    
    # Create the integration histograms
    int_hist = np.zeros((x_N+1, p_N+1, mu_N+1))
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
        u, tan_theta = eval_flow_profile(x)

        # Calculate the inverse focusing length
        L_inv = eval_L_inv(x, t, tan_theta)
        
        # Calculate the timesteps and inverse scale lengths
        dt = eval_timestep(x, v, mu, tan_theta, u, L_inv)

        # Apply focusing if boolean switch allows
        mu = focus_particles(v, mu, dt, L_inv)

        # Move the particles
        x = move_particles(x, v, mu, dt, tan_theta)
        
        # Calculate the tangents of the shock obliquities and the flow profile u in the new position
        u, tan_theta = eval_flow_profile(x)
        
        # Check indices of particles in the simulation box after movement
        in_loop =  np.array((x < x_max + 1e-10) & (x > x_min - 1e-10))

        # Propagate particles with the velocity it exited the box with till the maximum simulation time
        #x[~in_loop] = move_particles(x[~in_loop], v[~in_loop], mu[~in_loop], T_max - t[~in_loop], tan_theta[~in_loop])

        # Add the timesteps to the integration histogram
        xi, pi, mui, pparai, pperpi = integration_indices(x[in_loop], v[in_loop], mu[in_loop], u[in_loop])       
        integrate_particles(xi.astype(int), pi.astype(int), mui.astype(int), 
                            pparai.astype(int), pperpi.astype(int),
                            dt[in_loop], int_hist, int_p_hist)
        
        # Change to the plasma rest frame of reference for scattering
        v[in_loop], mu[in_loop] = change_frame(v[in_loop], mu[in_loop], u[in_loop])

        # Scatter the particles in the plasma rest frame
        R1 = np.random.random(size = len(x[in_loop]))
        R2 = np.random.random(size = len(x[in_loop]))
        R3 = np.random.random(size = len(x[in_loop]))
        x[in_loop], v[in_loop], mu[in_loop] = scatter_particles(x[in_loop], v[in_loop], mu[in_loop], dt[in_loop], u[in_loop], tan_theta[in_loop], R1, R2, R3)

        # Change back to de Hoffmann-Teller frame
        v[in_loop], mu[in_loop] = change_frame(v[in_loop], mu[in_loop], -u[in_loop])

        # Progress time for all particles
        t += dt
        
        # Update the particles array with the simulated data
        particles[in_box,0] = x
        particles[in_box,1] = v
        particles[in_box,2] = mu
        particles[in_box,3] = t
        
        # Check which particles to simulate on next round based on the in_loop estimated earlier and the maximum time of the snapshot
        in_box[in_box == True] *= in_loop

        # If particles exceed maximum defined simulation time, remove them from the simulation
        #in_box[particles[:,3] > t_end] = False

    return particles, int_hist, int_p_hist


#--------------------------------------------------------------------------------------
# Injection parameters
E_inj_0 = eval_E_natural(E_inj_0 * 1000, m_e, c, e) # Unitless minimum injection energy of particles
v_inj_0 = E_to_v(E_inj_0) # Unitless minimum injection velocity, With units corresponds to c * np.sqrt(E_inj * (E_inj + 2*m_e*c**2)) / (E_inj + m_e*c**2) and divided by c to use units of c

# Time integration histogram parameters
p_inj_0 = v_to_p(v_inj_0) # Unitless minimum injection momentum of particles, with units corresponds to gamma * m_e * v_inj
p_pmin = np.log10(p_min_f*p_inj_0) # Unitless minimum of time integral momentum grid
p_pmax = np.log10(p_max_f*p_inj_0) # Unitless maximum of time integral momentum grid
x_max = down_bound_scaler * box_size
x_min = -up_bound_scaler * box_size    
u, _ = eval_flow_profile(x_min)
v_inj_HTF, _ = change_frame(v_inj_0, 1, -u)
p_inj_HTF = eval_gamma(v_inj_HTF) * v_inj_HTF
p_min = np.log10(p_min_f * p_inj_HTF)
p_max = np.log10(p_max_f * p_inj_HTF)
if custom_bins:
    # Define the left bin edges here if custom bins are defined
    bin_edges = spatial_bins()
    dx = bin_edges[1:]-bin_edges[:-1]
    x_N = len(bin_edges)-1
    bin_steps = np.append(bin_edges, bin_edges[-1] + dx[-1]) # Add final bin for timestepping
else:
    dx = (x_max - x_min)/ x_N # Unitless size of x cell in time integral grid

if not only_ambient_mfp:
    t = 0
    x = np.linspace(-10,10, 100)
    _, tan_theta = eval_flow_profile(x)

    inv_L_ambient_prof = 2 * inv_L_ambient / (mfp_prof(x, t) * (1 - downstream_mfp_scaler) + 1 + downstream_mfp_scaler)
    mfp_inv = np.sqrt( ( 1 + tan_theta**2 ) / ( 1 + tan_theta_1**2 )) / (a_ratio * eval_gamma(v_inj_0) * v_inj_0 ) * B_prof_der(x,t) 

    mfp_matcher = max(inv_L_ambient_prof)/max(mfp_inv)

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

    # Print the box size to see
    print("box_size:", box_size, "d_i")

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
        int_p_hist = np.sum(np.array(results, dtype=object)[:,2], axis = 0)
        
        # Save the particle data to the location defined by write_init()
        np.save(f"{path}snapshot_{i}", particles) 

        # Save the time integration histogram to the location defined by write_init()
        np.save(f"{path}int_hist_{i}", int_hist)
        np.save(f"{path}int_p_hist_{i}", int_p_hist)

    # Print the simulation wall time after all loops
    print(f"Wall time: {time()-start:.2f} s = {(time()-start)/60:.2f} min")

    
if __name__ == "__main__":
    main()
    print("DONE!")
