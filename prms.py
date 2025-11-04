# Electron Acceleration SImulation (EASI)
# Simulation input parameters
# Code author: Seve Nyberg
# Affiliation: Department of Physics and Astronomy, University of Turku, Finland

# Github URL: https://github.com/SeveNyberg/easi

import numpy as np
from numba import jit
from simulation_functions import eval_E_natural, eval_box_size, eval_ion_inert, v_to_p, p_to_v, E_to_p, E_to_v, power_law_distr_inj, maxwellian_distr_inj

#-------------------------------------
# Switches
#-------------------------------------
spec_inj = True # Toggles defined custom spectral injection vs monoenergetic injection with E_inj_0
save_injection = True # Toggles saving of injection particle data
only_ambient_mfp = True # Toggles off the cosh profiled mean free path at the shock and instead uses only the ambient tanh profile for the mean free path
sda_regime = False # Toggles SDA regime on, in practice sizes the simulation box to 20 d_i instead of the defined particle speed's diffusion length. Ignores up_bound_scaler and down_bound_scaler.
histogram_frame = "HTF" # Choose the frame in which the (x, p, mu)  and (x, v_para, mu) histograms are gathered, options are "HTF", "plasma rest", "normal incidence", defaults to "HTF" in case of typos
continuity_testing = False # Tests mass conservation of the simulation by injecting particles at the upstream boundary, removing the shock from the simulation (M_A = 1, theta_Bn = 0), and giving the particles a chance to return at the upstream and downstream boundary.

#-------------------------------------
# Physical constants
#-------------------------------------
m_e = 9.1093837015e-31 # [kg], electron mass from CODATA RECOMMENDED VALUES OF THE FUNDAMENTAL PHYSICAL CONSTANTS: 2018 NIST SP 961 (May 2019)
m_i = 1.67262192369e-27 # [kg], proton mass from CODATA RECOMMENDED VALUES OF THE FUNDAMENTAL PHYSICAL CONSTANTS: 2018 NIST SP 961 (May 2019)
c = 299792458 # [m/s], speed of light from CODATA RECOMMENDED VALUES OF THE FUNDAMENTAL PHYSICAL CONSTANTS: 2018 NIST SP 961 (May 2019)
e = 1.602176634e-19 # [C], elementary charge from CODATA RECOMMENDED VALUES OF THE FUNDAMENTAL PHYSICAL CONSTANTS: 2018 NIST SP 961 (May 2019)
eps0 = 8.8541878128e-12 # [F/m], vacuum permittivity from CODATA RECOMMENDED VALUES OF THE FUNDAMENTAL PHYSICAL CONSTANTS: 2018 NIST SP 961 (May 2019)
mu0 = 1.25663706212e-6 # [H/m], vacuum permeability from CODATA RECOMMENDED VALUES OF THE FUNDAMENTAL PHYSICAL CONSTANTS: 2018 NIST SP 961 (May 2019)
R_sun = 6.96e5 # [km], T. M. Brown & J. Christensen-Dalsgaard, Astrophys. J. 500, L195 (1998) Many values for the Solar radius have been published, most of which are consistent with this result.
AU = 1.4959e8 # [km], While A is approximately equal to the semi-major axis of the Earth’s orbit, it is not exactly so. Nor is it exactly the mean Earth-Sun distance. There are a number of reasons: a) the Earth’s orbit is not exactly Keplerian due to relativity and to perturbations from other planets; b) the adopted value for the Gaussian gravitational constant k is not exactly equal to the Earth’s mean motion; and c) the mean distance in a Keplerian orbit is not equal to the semi-major axis a: r = a(1 + e^2/2), where e is the eccentricity. (Discussion courtesy of Myles Standish, JPL).
k_boltz = 1.380649e-23 # [J/K], Botlzmann constant from CODATA RECOMMENDED VALUES OF THE FUNDAMENTAL PHYSICAL CONSTANTS: 2018 NIST SP 961 (May 2019)

#-------------------------------------
# Physical parameters
#-------------------------------------
ux1 = 1400.0 # [km/s], The upstream flow speed along the shock normal direction in the shock rest frame 
V_A1 = 150.0 # [km/s], The upstream Alfvén speed 
theta_1 = 70.0 # [deg], Shock obliquity
n_part = 9e7 # [1/cm^3], particle density
L_1 = "inf" # [km], KEEP "inf" FOR NOW AS GLOBAL FOCUSING IS NOT IMPLEMENTED. Ambient focusing length for parts of simulation where focusing length is very small.
mfp_1 = R_sun # [km], Upstream ambient mean free path for parts of the simulation where the provided mean free path profiles are very small.
T_max = 10.0 # [s], The maximum time of the simulation (UNUSED, though possible to use by uncommenting lines in simulate_particles() and inject_particles().)
E_inj_0 = 1.0 # [keV], The monoenergetic injection energy of particles.
T = 2e6 # [K], temperature for solving the plasma beta and the maxwellian injection

#-------------------------------------
# Numerical parameters
#-------------------------------------
N_p = 1000000 # The amount of Monte Carlo particles
mfp_scaler = 1e-4 # Scales the minimum value of the cosh-profile at the shock to match this factor times the downstream value of the ambient mean free path profile.
downstream_mfp_scaler = 0.1 # The scaler of the downstream mean free path based on the upstream mean free path. 0.1 would correspond to 10% of the upstream value.
up_bound_scaler = 1.0 # Scaler of the focusing length determining box size on the upstream side
down_bound_scaler = up_bound_scaler # Scaler of the focusing length determining box size on the downstream side
d = 1.0 # The shock thickness (measured in ion inertial lengths), tunes the obliquity and flow speed profile widths.
k = 1.0 # Mean free path cosh-profile width scaler when only_ambient_mfp == False.
a_prm = 1.0 # Mean free path value scaling parameter
N_snapshots = 1 # Number of snapshots to save from the simulation (UNUSED, KEEP AT 1, SEE T_max IF WANT SNAPSHOTS ARE TO BE USED)
E_L = E_inj_0 # [keV], energy for which the box size/focusing length will be set for
T_inj_scaler = 1.0 # Scaler for the injection time. Example: T_inj_scaler = 0.9 would result in the last particle being injected at 90% of the maximum time of the simulation.
# If SDA regime is to be used, set up_bound_scaler and down_bound_scaler to values such that the box size is 20 d_i both in the upstream and the downstream along x.
if sda_regime:
    ion_inert = eval_ion_inert(n_part, e, m_i, eps0, c)
    v_L = E_to_v(eval_E_natural(E_L * 1000, m_e, c, e)) * c
    box_size = eval_box_size(mfp_1 * 1000, ux1 * 1000, np.deg2rad(theta_1), v_L) / ion_inert
    up_bound_scaler = 20/box_size
    down_bound_scaler = up_bound_scaler
    
#-------------------------------------
# Time integration histogram parameters
#-------------------------------------
p_min_f = 1e-1 # Multiplier, p_inj * p_min_f is the lower bound of p in the time integral histogram
p_max_f = 1e2 # Multiplier, p_inj * p_max_f is the upper bound of p in the time integral histogram
x_N = 100 # Amount of x cells in the intergral histogram if custom spatial bins are not used
p_N = 150 # Amount of p cells in the time integral histogram
mu_N = 100 # Amount of mu cells in the time integral histogram

@jit(nopython=True)
def spatial_bins():
    '''
    Define the bin edges for the time integration histogram.
    
    Returns:
        bin_edges (numpy array (float)): The bin edges. Have the array contain all left edges of bins + as the last element the right edge of the last bin.
    '''
    
    # Evaluate the box size of the simulation
    ion_inert = eval_ion_inert(n_part, e, m_i, eps0, c)
    v_L = E_to_v(eval_E_natural(E_L * 1000, m_e, c, e)) * c
    box_size = eval_box_size(mfp_1 * 1000, ux1 * 1000, np.deg2rad(theta_1), v_L) / ion_inert
    
    # Results in 25 bins between upstream boundary and -10, 300 bins between -10 and 10, and 25 bins between 10 and downstream boundary. Last value is kept as the limiting boundary for the last bin, but not used for binning.
    bin_edges = np.linspace(-up_bound_scaler * box_size, -10, 26)[:-1]
    bin_edges = np.append(bin_edges, np.linspace(-10,10, 301)[:-1])
    bin_edges = np.append(bin_edges, np.linspace(10, down_bound_scaler * box_size, 26))
    
    # 50 equisized bins throughout the box
    # bin_edges = np.linspace(-up_bound_scaler * box_size, down_bound_scaler * box_size, 50)
    
    if np.any(bin_edges[1:] - bin_edges[:-1] < 1e-10):
        print("Bin size close to zero (<1e-10) detected, was this intended?")
    
    return bin_edges


#-------------------------------------
# Custom spectral injection speed function
#-------------------------------------
def spec_func(u):
    '''
    The function from which the injection speed of the particle should be sampled from. Format the equation so that an array of speeds for each particle is returned in the units of speed of light (c). The input is a list of random numbers in [0,1) for each particle separately, seeded in the simulation code.
    
    Parameters:
        u (numpy array (float)): An array of local flow speeds for each particle separately.
        
    Returns:
        v (numpy array (float)): An array of injection speeds for each particle in the units of speed of light (c).
        mu (numpy array (float)): An array of particle pitch angle cosines for each particle separately.
    '''
    # E_inj_1 = 100.0 # keV, the maximum injected energy from a power-law distribution
    # spec_index = 4 # The momentum power-law spectral index used for the power_law injection
    # return power_law_distr_inj(R1, u, N_p, m_e, c, e, spec_index)
    
    v, mu = maxwellian_distr_inj(u, N_p, T, k_boltz, m_e, c)
    
    return v, mu


#-------------------------------------
# Physical shock profiles
#-------------------------------------
# The profile of the magnetic field strength, take care that flow profile and magnetic field profile can form a de Hoffmann - Teller frame
@jit(nopython=True)
def B_prof(x,t):
    return (1 + np.tanh(x / (k * d))) / 2
    
# The derivative of the magnetic field strength profile
@jit(nopython=True)
def B_prof_der(x,t):
    return  1 / (2 * d * np.cosh(x / (k * d))**2)
    
# The flow speed profile, take care that flow profile and magnetic field profile can form a de Hoffmann - Teller frame 
@jit(nopython=True)
def u_prof(x,t):
    return (1 + np.tanh(x/d)) / 2
    
# The derivative of the flow speed profile
@jit(nopython=True)
def u_prof_der(x,t):
    return 1 / (2 * d * np.cosh(x / (d))**2)
    
# The ambient mean free path profile (evolution from upstream to downstream disregarding the magnetic field strength derivative)
@jit(nopython=True)
def mfp_prof(x,t):
    return np.tanh(-x / (k*d))