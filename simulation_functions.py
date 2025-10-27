# Electron Acceleration SImulation (EASI)
# Complementary simulation functions
# Code author: Seve Nyberg
# Affiliation: Department of Physics and Astronomy, University of Turku, Finland

# Github URL: https://github.com/SeveNyberg/easi

from numba import jit # To make simulation calculations faster
import numpy as np # For array and mathematical operations
import os # For directory paths

@jit(nopython=True)
def change_frame(v, mu, u):
    """
    Change particles' frame of reference relativistically.

    Parameters: 
        v (numpy array (float)): The original speeds of the particles. 
        mu (numpy array (float)): The pitch angle cosines of the particles.  
        u (numpy array (float)): The velocity of the new frame relative to the original one. 

    Returns: 
        v_prime (numpy array (float)): The speeds of the particles in the new frame. 
        mu_prime (numpy array (float)): The pitch angle cosine of the particles in the new frame.  
    """    

    # Gamma value for given speed v in original frame. 
    gamma = eval_gamma(v)
    # Gamma value for given speed u in original frame.
    gamma_u = eval_gamma(u)  

    # Momentum component parallel to x-axis in original frame. 
    # With units corresponds to gamma * m_e * v * mu
    p_para = gamma * v * mu  
  
    # Gamma value for given speed v in the new frame. 
    # With units corresponds to gamma_u * (gamma - u * p_para / (m_e * c**2))
    gamma_prime = gamma_u * (gamma - u * p_para)  
    # Momentum component parallel to x-axis in the new frame.
    # With units corresponds to gamma_u * (p_para - gamma * m_e * u)
    p_para_prime = gamma_u * (p_para - gamma * u)   

    # Final speed in the new frame. 
    # With units corresponds to c * np.sqrt(1 - 1/gamma_prime**2)
    v_prime = np.sqrt(1 - 1/gamma_prime**2)  
       
    # Final pitch angle cosine between direction of motion and magnetic field in the new frame. 
    # With units corresponds to p_para_prime / (gamma_prime * m_e * v_prime)
    mu_prime = p_para_prime / (gamma_prime * v_prime)   

    return v_prime, mu_prime


@jit(nopython=True)
def change_frame_dt(v, mu, u, dt):
    """
    Change particles' frame of reference relativistically, also transforming the current timestep of the particle to the respective frame.

    Parameters: 
        v (numpy array (float)): The original speeds of the particles. 
        mu (numpy array (float)): The pitch angle cosines of the particles.  
        u (numpy array (float)): The velocity of the new frame relative to the original one. 
        dt (numpy array (float)): The timestep size of the particle in the original one.

    Returns: 
        v_prime (numpy array (float)): The speeds of the particles in the new frame. 
        mu_prime (numpy array (float)): The pitch angle cosine of the particles in the new frame.  
        dt_prime (numpy array (float)): The timestep size in the new frame.
    """    

    # Gamma value for given speed v in original frame. 
    gamma = eval_gamma(v)
    # Gamma value for given speed u in original frame.
    gamma_u = eval_gamma(u)  

    # Momentum component parallel to x-axis in original frame. 
    # With units corresponds to gamma * m_e * v * mu
    p_para = gamma * v * mu  
  
    # Gamma value for given speed v in the new frame. 
    # With units corresponds to gamma_u * (gamma - u * p_para / (m_e * c**2))
    gamma_prime = gamma_u * (gamma - u * p_para)  
    # Momentum component parallel to x-axis in the new frame.
    # With units corresponds to gamma_u * (p_para - gamma * m_e * u)
    p_para_prime = gamma_u * (p_para - gamma * u)   

    # Final speed in the new frame. 
    # With units corresponds to c * np.sqrt(1 - 1/gamma_prime**2)
    v_prime = np.sqrt(1 - 1/gamma_prime**2)  
       
    # Final pitch angle cosine between direction of motion and magnetic field in the new frame. 
    # With units corresponds to p_para_prime / (gamma_prime * m_e * v_prime)
    mu_prime = p_para_prime / (gamma_prime * v_prime)   
    
    # Evaluate the new time step
    gamma_prime = eval_gamma(v_prime)
    dt_prime = gamma_prime/gamma*dt

    return v_prime, mu_prime, dt_prime


@jit(nopython=True)
def htf_to_nif(v, mu, u, theta, phi, dt):
    """
    Frame transformation from HTF to normal incidence frame (NIF). As NIF has a 2D velocity in relation to HTF considering the one dimension of the simulation, this function is different from the 1D change_frame() and change_frame_dt().
    
    Parameters: 
        v (numpy array (float)): The original speeds of the particles. 
        mu (numpy array (float)): The pitch angle cosines of the particles.  
        u (numpy array (float)): The velocity of the new frame relative to the original one. 
        theta (numpy array (float)): The local magnetic field obliquity for each particle separately in radians.
        phi (numpy array (float)): The phase angle of each particle.
        dt (numpy array (float)): The timestep size of the particle in the original one.

    Returns: 
        v_prime (numpy array (float)): The speeds of the particles in the new frame. 
        mu_prime (numpy array (float)): The pitch angle cosine of the particles in the new frame.  
        dt_prime (numpy array (float)): The timestep size in the new frame.
    """
    # Shorthands for repeated values
    gamma = eval_gamma(v)
    mu_sqrd = 1-mu**2
    mu_sqrd[mu_sqrd<0] = 0
    sqrt_mu = np.sqrt(mu_sqrd)
    cos_theta = np.cos(theta)
    cos_phi = np.cos(phi)
    sin_theta = np.sin(theta)
    
    # Solve the frame relative speed
    U_boost = u*np.sin(theta)
    
    # Solve the original velocity components of the particles in the coordinate system where the shock normal is the first component and two tangential components are chosen in relation to that
    v_n = v * ( mu*cos_theta - sqrt_mu * cos_phi * sin_theta)
    v_t = v * ( mu*sin_theta + sqrt_mu * cos_phi * cos_theta)
    v_q = v * sqrt_mu * np.sin(phi)
    
    # Another shorthand
    denom = 1 - v_t*U_boost
    
    # Solve the new velocity components
    v_prime_n = v_n /(eval_gamma(U_boost)*denom)
    v_prime_t = (v_t - U_boost) / denom
    v_prime_q = v_q /(eval_gamma(U_boost)*denom)
    
    # Solve the final speed and pitch angle cosine
    v_prime = np.sqrt(v_prime_n**2 + v_prime_t**2 + v_prime_q**2)
    mu_prime = (v_prime_n * cos_theta + v_prime_t * sin_theta) / v_prime
    
    # Evaluate the new time step
    gamma_prime = eval_gamma(v_prime)
    dt_prime = gamma_prime/gamma*dt
    
    return v_prime, mu_prime, dt_prime
    

@jit(nopython=True)
def eval_gamma(v):
    """
    Evaluates the Lorentz gamma factor.
    
    Parameters:
        v (numpy array (float)): The speeds of the particles.
        
    Returns:
        gamma (numpy array (float)): The Lorentz gamma factors for particles.
    """
    
    return 1/np.sqrt(1 - v**2) # With units corresponds to 1/np.sqrt(1 - v**2/c**2)


@jit(nopython=True)
def eval_rg(M_A, theta_1, beta):
    """
    Compute gas compression ratio of the shock iteratively with Newton's method. Borrowed from Alexandr Afanasiev's SOLPACS plotting code. Added functionality of if M_A = 1, rg = 1, only relevant when obliquity is also 0.
    
    Parameters:
        M_A (float): Alfven mach number in the de Hoffmann-Teller frame.
        theta_1 (float): Shock obliquity parameter in radians.
        beta (float): Plasma beta in the upstream.
        
    Returns: 
        r (float): The gas compression ratio of the shock.
    """

    if M_A == 1:
        return 1
    else:
        # Define aliases for used variables
        g = 5.0/3.0  # adiabatic index
        half_gb = 0.5 * g * beta
        cos_theta_sqr = np.cos(theta_1)**2
        sin_theta_sqr = 1.0 - cos_theta_sqr
        
        # Define gas compression ratio function to iterate over
        def funcs(z):
            a = z**2 * (g + 1.0) * cos_theta_sqr + (1.0 - g * z) * sin_theta_sqr
            a_prime = 2.0 * z * (g + 1.0) * cos_theta_sqr - g * sin_theta_sqr 
            
            A = (1.0 + z) * a - 2.0 * half_gb * z**2
            A_prime = a + (1.0 + z) * a_prime - 4.0 * half_gb * z
            
            B = (z**2 * (g - 1.0) * cos_theta_sqr + (1.0 + (2.0 - g) * z) * sin_theta_sqr)
            B_prime = 2.0 * z * (g - 1.0) * cos_theta_sqr + (2.0 - g) * sin_theta_sqr
            
            f = M_A**2 - A/B
            f_prime = -(A_prime * B - A * B_prime)/B**2
            
            return f, f_prime

        # Initial guess
        z0 = M_A**2 - 1.0

        # Iterate to find solution with Newton's method
        r_old = 4.0
        dr_abs = 1.0
        while dr_abs > 1.0e-8:
            r = M_A**2/(1.0 + z0)
            dr_abs = np.abs(r - r_old)
            r_old = r
            f, f_prime = funcs(z0)
            z0 -= f/f_prime

        return r


@jit(nopython=True)
def eval_rb(M_A, rg, theta_1):
    """
    Compute the magnetic compression ratio of the shock.  Added functionality of if M_A = 1, rb = 1, only relevant when obliquity is also 0.
    
    Parameters:
        M_A (float): The Alfven Mach number of the shock in the de Hoffman-Teller Frame.
        rg (float): The gas compression ratio of the shock.
        theta_1 (float): The shock obliquity parameter in radians.
        
    Returns:
        rb (float): The magnetic compression ratio of the shock.
    """

    if M_A == 1:
        return 1
    else:
        cos_theta_sqr = np.cos(theta_1)**2
        sin_theta_sqr = 1.0 - cos_theta_sqr
        return np.sqrt(cos_theta_sqr + ((M_A**2 - 1)/(M_A**2 - rg) * rg)**2 * sin_theta_sqr)


@jit(nopython=True)
def eval_box_size(mfp_ambient, ux1, theta_1, v):
    """
    Compute the upstream box size of the simulation based on the diffusion length of the chosen particle speed derived from the diffusion coefficient along the shock normal.
    
    Parameters:
        mfp_ambient (float): The ambient focusing length/mean free path of the simulation
        ux1 (float): The upstream flow speed along the shock normal parameter.
        theta_1 (float): The general shock obliquity parameter in radians.
        v (float): The speed of the particle for which the box size will be set.
        
    Returns:
        L_box (float): The upstream box size of the simulation.
    """

    return mfp_ambient * np.cos(theta_1)**2 * v / (3 * ux1)


@jit(nopython=True)
def eval_ion_inert(n_part, e, m_i, eps0, c):
    """
    Compute the ion inertial length in the upstream.
    
    Parameters:
        n_part (float): Particle density in the units of cm^-3.
        e (float): Elementary charge in the units of C.
        m_i (float): Proton mass in the units of kg.
        eps0 (float): Vacuum permittivity in the units of F m^-1.
        c (float): Speed of light in the units of m s^-1.
        
    Returns:
        ion_inert (float): The ion inertial length in the units of m.
    """
    
    return c / np.sqrt(n_part*1e6*e**2/(m_i*eps0))


@jit(nopython=True)
def eval_E_natural(E, m_e, c, e):
    """
    Evaluate energy in the natural units used in the simulation
    
    Parameters:
        E (float): The energy in the units of eV.
        m_e (float): Mass of electron in the units of kg.
        c (float): Speed of light in the units of m s^-1.
        e (float): Elementary charge in the units of C.
        
    Returns:
        E (float): Energy in the natural units used in the simulation (m_e c**2).
    """

    return E*e/(m_e*c**2)


@jit(nopython=True)
def E_to_v(E):
    """
    Compute speed of electron from energy with relativity in the units of speed of light (c).
    
    Parameters:
        E (float): The energy needed in the natural units (m_e c^2) of the simulation.
        
    Returns:
        v (float): Speed in the units of speed of light (c) for electrons.
    """
    
    return np.sqrt(E * (E + 2)) / (E + 1)


@jit(nopython=True)
def v_to_p(v):
    """
    Compute momentum of electron from speed with relativity in the units m_e * c.
    
    Parameters:
        v (float): The speed needed in the natural units (c) of the simulation.
        
    Returns:
        p (float): Momentum in the units of m_e c for electrons.
    """
    
    return eval_gamma(v) * v


@jit(nopython=True)
def p_to_E(p):
    """
    Compute energy of electron from momentum with relativity in the units m_e * c**2.
    
    Parameters:
        p (float): The momentum needed in the natural units (m_e * c) of the simulation.
        
    Returns:
        E (float): Energy in the units of m_e c^2 for electrons.
    """
    
    return np.sqrt(1 + p**2) - 1 


@jit(nopython=True)
def v_to_E(v):
    """
    Compute energy from speed.
    
    Parameters:
        v (float): The speed in the units of c.
        
    Returns:
        E (float): Energy in the units of m_e c^2 for electrons.
    """
    
    return p_to_E(v_to_p(v))


@jit(nopython=True)
def p_to_v(p):
    """
    Compute speed of particles from momentum in the units of c.
    
    Parameters:
        p (float): The momentum needed in the natural units (m_e * c) of the simulation.
        
    Returns:
        v (float): Speed in the units of c.
    """
    
    return E_to_v(p_to_E(p))

@jit(nopython=True)
def E_to_p(E):
    """
    Compute momentum from energy.
    
    Parameters:
        E (float): Energy in the units of m_e c^2 for electrons.
        
    Returns:
        p (float): The momentum needed in the natural units (m_e * c) of the simulation.
    """
    
    return v_to_p(E_to_v(E))


def write_init():
    """
    This function creates a directory in which to store the results of the simulation and returns the path to the directory relative to the simulation script's location.
    
    Returns:
        (string): The path to the run folder relative to the simulation script.
    """

    # Create a path to store all the results of all simulation runs in the current directory of the script
    path = f".{os.sep}results" 

    # If the path doesn't exist, make the directory
    if not os.path.exists(path):
        os.makedirs(path)
        open(f"{path}{os.sep}__init__.py", "w").close()

    # Check if the run directory exists under the results directory, if it does increment the run number
    run_no = 0
    while os.path.isdir(f"{path}{os.sep}run{str(0) * (3 - len(str(run_no))) + str(run_no)}"):
        run_no += 1
    
    # Create the run directory with three digit numbering starting from 000
    os.makedirs(f"{path}{os.sep}run{str(0) * (3 - len(str(run_no))) + str(run_no)}")
            
    # Return the path of the directory
    return f"{path}{os.sep}run{str(0) * (3 - len(str(run_no))) + str(run_no)}{os.sep}"


def save_prm(path):
    """
    Saves the parameters loaded from prms.py and additional parameters from the simulation file to the run folder.
    
    Parameters:
        path (str): The path to the run folder to which the parameter file should be saved to.
    """
    
    # Import the parameters as an object to fetch attributes and their values
    import prms
    
    # Create a dict to store attributes and values in
    p = {}
    
    # Go through all attribute names in the prms object
    for var_name in dir(prms):
        
        # Exclude redundant attributes, such as imported modules in the prms.py script
        exclude = ( var_name[0] != "_" and 
                    var_name != "np" and
                    var_name != "jit" and
                    var_name != "E_to_v" and
                    var_name != "eval_E_natural" and 
                    var_name != "eval_box_size" and
                    var_name != "eval_ion_inert" )
        if exclude:
            # Add the attribute and its value to the dict
            p[var_name] = getattr(prms, var_name)
        
    # Save the created dict as a .npy with pickling on to return it as a dict later
    np.save(f"{path}prms.npy", p, allow_pickle = True)
    
    # Delete the imported variables from the name space to not interfere with the simulation code
    del prms
    

def power_law_distr_inj(u, N_p, m_e, c, e, spec_index):
    '''
    A ready-made power law distribution injection routine. Samples a speed and pitch angle cosine for all particles.
    
    Parameters: 
        u (numpy array (float)): A list of local flow speed for each particle. If injecting all particles in the same position, u(x) should be the same for all,
        N_p (integer): The amount of Monte Carlo particles in the simulation.
        m_e (float): Electron mass in the units of kg.
        c (float): The speed of light in the units of m s^-1.
        e (float): Elementary charge in the units of C.
        spec_index (float): The spectral index chosen for the power-law.
        
    Returns:
        v (numpy array (float)): Sampled particle speeds in the units of c.
        mu (numpy array (float)): Sampled particle pitch angle cosines.
    '''
    
    # Distribution parameters
    E_inj_0 = eval_E_natural(E_inj_0 * 1000, m_e, c, e) # Unitless minimum injection energy of particles
    p_inj_0 = E_to_p(E_inj_0) # Unitless minimum injection momentum of particles, with units corresponds to gamma * m_e * v_inj
    E_inj_1 = eval_E_natural(E_inj_1 * 1000, m_e, c, e) # Unitless maximum injection energy of particles
    p_inj_1 = E_to_p(E_inj_1) # Unitless maximum injection momentum of particles, with units corresponds to gamma * m_e * v_inj
    
    # Power-law distribution of injection momentum
    R1 = np.random.random(size=N_p)
    p_inj = p_inj_0 *(1 -  R1 * (1 - (p_inj_1/p_inj_0)**(1 -  spec_index)))**(1/(1 -  spec_index))
    v = p_to_v(p_inj)
    
    # Sample a random pitch angle cosine for the particles from a population that ends towards the shock when injecting from the edge of the simulation box
    R_min = (u - v)/(u + v)
    R_min[R_min < 0] = 0
    R_min = R_min**2
    R2 = np.random.random(size=N_p) * (1 - R_min) + R_min
    mu = (-u + (v + u) * np.sqrt(R2)) / (v)
    
    return v, mu


def maxwellian_distr_inj(u, N_p, T, k_boltz, m_e, c):
    '''
    A ready-made Maxwellian distribution injection routine. Samples a speed and pitch angle cosine for all particles.
    
    Parameters: 
        u (numpy array(float)): A list of local flow speed for each particle. If injecting all particles in the same position, u(x) should be the same for all,
        N_p (integer): The amount of Monte Carlo particles in the simulation.
        T (float): The temperature of the Maxwellian distribution in the units of K.
        k_boltz (float): The Boltzmann constant in the units of J K^-1.
        c (float): The speed of light in the units of m s^-1.
        e (float): Elementary charge in the units of C.
        
    Returns:
        v (numpy array(float)): Sampled particle speeds in the units of c.
        mu (numpy array (float)): Sampled particle pitch angle cosines.
    '''
    
    # Set particle injection speed
    erf = lambda x: np.sign(x) * (1 - 1 /(1 + 0.278393 * (np.sign(x) * x) + 0.230389 * (np.sign(x) * x)**2 + 0.000972 * (np.sign(x) * x)**3 + 0.078108 * (np.sign(x) * x)**4)**4)
    # F = lambda x: u * erf(x) - np.sqrt(2 * k_boltz * T / (np.pi * m_e)) * np.exp(-x**2)
    v_min = -u * c
    v_max = 0.2 * c
    N = 10000 # Number of cells in interpolation table
    v_para_table = np.linspace(v_min, v_max, N)
    u = u*c
    
    # nj = F(v_para_table / np.sqrt(2 * k_boltz * T / (m_e))) - F(-u / np.sqrt(2 * k_boltz * T / (m_e))) 
    # ninf = u - F(u / np.sqrt(2 * k_boltz * T / (m_e))) 
    nj = (np.sqrt(np.pi * k_boltz * T / (2 * m_e)) * u * (erf(v_para_table / np.sqrt(2 * k_boltz * T / (m_e)))
          + erf(u / np.sqrt(2 * k_boltz * T / (m_e)))) + k_boltz * T / m_e * (np.exp(- u**2 / (2 * k_boltz * T / m_e)) 
        - np.exp(- v_para_table**2 / (2 * k_boltz * T / m_e))))
    ninf = np.sqrt(np.pi * k_boltz * T / (2 * m_e)) * u * (erf(u / np.sqrt(2 * k_boltz * T / (m_e))) + 1) + k_boltz * T / m_e * np.exp(- u**2 / (2 * k_boltz * T / m_e)) 
    P = nj/ninf
    P[P < 0] = 0

    R1 = np.random.random(size = N_p) * (P[-1])
    k_ind = np.digitize(R1, P)
    k_ind[k_ind == N - 1] = N - 2
    
    v_para = v_para_table[k_ind] + (R1 - P[k_ind]) * (v_para_table[k_ind + 1] - v_para_table[k_ind])/(P[k_ind + 1] - P[k_ind])
    
    R2_max = 1 - np.exp(- (v_max**2 - v_para**2)/(2 * k_boltz * T / m_e))
    R2 = np.random.random(size = N_p) * R2_max
    v_perp_squared = - 2 * k_boltz * T / m_e * np.log(1 - R2)
    v = np.sqrt(v_para**2 + v_perp_squared)
    mu = v_para/v
    
    return v/c, mu
