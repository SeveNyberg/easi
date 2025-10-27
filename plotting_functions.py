# Electron Acceleration SImulation (EASI)
# Plotting functions 
# Code author: Seve Nyberg
# Affiliation: Department of Physics and Astronomy, University of Turku, Finland

# Github URL: https://github.com/SeveNyberg/easi
 
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
import importlib
from simulation_functions import *

# Colorblind-friendly custom color map
custom_colors = np.array(["#000000","#00598C","#C30303","#E69F00","#F0E442","#98DAFF","#00D89E"])
mpl.rcParams["axes.prop_cycle"] = mpl.cycler('color', custom_colors)

def load_all_data(run_nos, snapshot):
    '''
    Loads the simulation data for the run numbers given.
    
    Parameters:
        run_nos (list): A list of simulation run numbers as integers.
        snapshot (int): An integer indicating the chosen snapshot. SNAPSHOTTING IS NOT IMPLEMENTED BY DEFAULT, SEE T_max IN prms.py.
        
    Returns:
        - 
    '''

    # Set path strings
    particles = []
    inthist = []
    vparahist = []
    pinthist = []
    paths = []
    inj_specs = []
    prms = []
    for run_no in run_nos:
        path = f".{os.sep}results{os.sep}run{str(0)*(3-len(str(run_no)))+str(run_no)}{os.sep}"
        path_snap = f"{path}snapshot_{snapshot}.npy"
        path_int = f"{path}int_hist_{snapshot}.npy"
        path_v_para_int = f"{path}int_v_para_hist_{snapshot}.npy"
        path_pint = f"{path}int_p_hist_{snapshot}.npy"
        path_inj_spec = f"{path}inj_spec.npy"
        path_prms = f"{path}prms.npy"

        # Load simulation result particle array and time integral histograms
        try:
            particles.append(np.load(path_snap))
        except: print("No particles array saved!")
        try:
            inthist.append(np.load(path_int)[:-1,:-1,:-1])
        except: print("No integration histogram saved!")
        try:
            vparahist.append(np.load(path_v_para_int)[:-1,:-1,:-1])
        except: print("No v_para integration histogram saved!")
        try:
            pinthist.append(np.load(path_pint)[:-1,:-1,:-1])
        except: print("No momentum integration histogram saved!")
        try:
            paths.append(path)
        except: print("No paths determined!")
        try:
            inj_specs.append(np.load(path_inj_spec))
        except: print("No injection spectrum saved!")
        try: 
            prms.append(np.load(path_prms, allow_pickle=True).item())
        except: print("No prms.npy found!")
        
    # Create directory "figures" if it does not already exist
    if not os.path.exists(f".{os.sep}figures"):
        os.makedirs(f".{os.sep}figures")

    return particles, inthist, vparahist, pinthist, paths, inj_specs, prms
    
    
def plot_espec(particles_list, parameters_list, paths, xlims = [None, None], ylims = [None, None], plasmarestframe = True, upstream = True, downstream = True, theory = False,titles = None, savefigs = False, show = True, color_start = None):
    '''
    Plots the escaping particle energy spectrum from the list of Monte Carlo particle data.
    
    Parameters:
        particles_list (list): List of np arrays of particles (see load_all_data for structure).
        parameters_list (list): List of parameter files corresponding to the runs.
        paths (list): List of directory paths for the runs.
        xlims (list): List of the x-axis limits in the plot, index 0 corresponds to the lower boundary and index 1 to the higher boundary.
        ylims (list): List of the y-axis limits in the plot, index 0 corresponds to the lower boundary and index 1 to the higher boundary.
        plasmarestframe (boolean): Toggle whether to plot in the upstream plasma frame or the shock frame, True for plasma frame.
        upstream (boolean): Toggle whether to plot the upstream escaping particles.
        downstream (boolean): Toggle whether to plot the downstream escaping particles.
        theory (boolean): Toggle wheter to plot the DSA predicted theoretical curve.
        titles (list): List of legend titles per run.
        savefigs (boolean): Toggle whether to save figures or not.
        show (boolean): Toggle whether to display plots or not.
        color_start (integer): Choose the starting index for the custom color map, if color separation is needed for different plots.
        
    Returns:
        -
    '''
    
    # Set font size
    mpl.rcParams.update({'font.size': 9})
    
    # Choose color map starting point
    global custom_colors
    if color_start != None:
        custom_colors = np.concatenate((custom_colors[color_start:], custom_colors[:color_start]))
        mpl.rcParams["axes.prop_cycle"] = mpl.cycler('color', custom_colors)
    
    # Format the figure and axis artists
    fig, ax = plt.subplots(1, 1, figsize = (6, 3))
    
    # To plot the theoretical curve only once, set a toggle to plot it only once in the loop
    theory_toggle = False
    
    # Set the alpha values for all curves
    ALPHA_VAL = 0.9
    
    # Iterate through all simulation runs
    for title_i, (particles, p) in enumerate(zip(particles_list, parameters_list)):     
        
        # Set aliases for particle data
        x = particles[:,0]
        v = particles[:,1]
        mu = particles[:,2]
        
        # Injectiona and shock parameters
        E_inj = p["E_inj_0"]
        M_A = p["ux1"]/(np.cos(np.deg2rad(p["theta_1"]))*p["V_A1"]) # The Alfvén Mach number in the de Hoffmann-Teller Frame (HTF)
        rg = eval_rg(M_A, np.deg2rad(p["theta_1"]), p["beta"]) # Gas compression ratio computed from parameters
        
        # If changing from de Hoffmann - Teller frame to the upstream plasma rest frame, solve the upstream flow speed and change particle speed and pitch angle cosine 
        if plasmarestframe:
            rb = eval_rb(M_A, rg, np.deg2rad(p["theta_1"]))
            u = p["ux1"]/np.cos(np.deg2rad(p["theta_1"])) * (1 - (1-rb/rg) * (1 + np.tanh(x/p["d"]))/2)
            v, mu = change_frame(v, mu, u*1e3/p["c"])
        
        # Set legend labels
        if titles == None:
            label = ""
        else:
            label = titles[title_i]
            
        # Set plot limits in MeV
        E_min = E_inj * 1e-3 * 1e-1 if xlims[0] == None else xlims[0] * 1e-3 # MeV
        E_max = 1e1 if xlims[1] == None else xlims[1] * 1e-3 # MeV
            
        # Create the energy spectrum for upstream and downstream per toggles
        if upstream and not downstream:
            ax.set_title("Upstream escaping energy spectrum")
            energy_counts, intensity_counts = energy_spectrum(v[x < 0]*p["c"], E_min = E_min, E_max = E_max)
            energy_counts *= 1000 # Change from MeV to keV
            ax.plot(energy_counts, intensity_counts, label=f"{label}",  alpha = ALPHA_VAL)
        elif downstream and not upstream:
            ax.set_title("Downstream escaping energy spectrum")
            energy_counts, intensity_counts = energy_spectrum(v[x > 0]*p["c"], E_min = E_min, E_max = E_max)
            energy_counts *= 1000 # Change from MeV to keV
            ax.plot(energy_counts, intensity_counts, label=f"{label}", alpha = ALPHA_VAL)
        elif upstream:
            ax.set_title("Energy spectrum")
            energy_counts, intensity_counts = energy_spectrum(v[x < 0]*p["c"], E_min = E_min, E_max = E_max)
            energy_counts *= 1000 # Change from MeV to keV
            ax.plot(energy_counts, intensity_counts, label=f"Upstream{' ' if label != '' else ''}{label}",  alpha = ALPHA_VAL)
            if downstream:
                ax.set_title("Energy spectrum")
                energy_counts, intensity_counts = energy_spectrum(v[x > 0]*p["c"], E_min = E_min, E_max = E_max)
                energy_counts *= 1000 # Change from MeV to keV
                ax.plot(energy_counts, intensity_counts, label=f"Downstream{' ' if label != '' else ''}{label}", alpha = ALPHA_VAL)
        elif downstream:
            ax.set_title("Energy spectrum")
            energy_counts, intensity_counts = energy_spectrum(v[x > 0]*p["c"], E_min = E_min, E_max = E_max)
            energy_counts *= 1000 # Change from MeV to keV
            ax.plot(energy_counts, intensity_counts, label=f"Downstream{' ' if label != '' else ''}{label}", alpha = ALPHA_VAL)
        
        # Plot the theoretical curve if desired, tune variable 'norm' to adjust height of curve
        if theory:
            if not theory_toggle:
                spec_index = (rg+2)/(rg-1)
                norm = np.amax(intensity_counts)*1e1
                ax.plot(energy_counts[10:], norm * (energy_counts[10:])**-(spec_index), c="k", linestyle="--", alpha = ALPHA_VAL, label = "DSA Theory")
                print("Spec index:", spec_index)
                theory_toggle = True

    # Set plot properties
    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.set_ylim(ylims)
    ax.set_xlim(xlims)
    #ax[0].set_ylabel(r"Particle intensity [1/(cm$^2$ sr s MeV)]")
    ax.set_ylabel(r"Particle intensity [arb.]")
    ax.set_xlabel(r"$E$ [keV]")
    ax.legend(loc="upper right")


    # Save and show figures if so desired
    if savefigs:
        suffix = ""
        for path_str in paths:
            suffix += "_"+path_str[-4:-1]
        for path_str in paths:
            name = f"{path_str}_us{upstream}_ds{downstream}_espec.png" if len(paths) == 1 else f"{path_str}espec_us{upstream}_ds{downstream}_{suffix}.png"
            print("Saving to:", name)
            plt.savefig(name, facecolor = "white", bbox_inches = "tight", dpi = 300)
    if show:
        plt.show()
    else:
        plt.clf()
        
    # Reset color wheel
    if color_start != None:
        custom_colors = np.array(["#000000","#00598C","#C30303","#E69F00","#F0E442","#98DAFF","#00D89E"])
        mpl.rcParams["axes.prop_cycle"] = mpl.cycler('color', custom_colors)


def plot_hist(inthist_list, vparahist_list, parameters_list, paths, 
             zoomin = [None, None], E_slice = [None,None],
             spat_slice = [None,None], espec = False, energy_axis = False,
             upstream = True, downstream = True, v_para_hist = False, y_lim_first = [None, None], y_lim_second = [None,None], x_lim_second = [None,None], 
             split = True, titles = None, savefigs = False, show = True):   
     
    '''
    Plots the time integrated histogram data (position, momentum, pitch angle cosine)
    
    Parameters:
        inthist_list (list): List of np arrays containing the time integrated histograms (see load_all_data for structure).
        parameters_list (list): List of parameter files corresponding to the runs.
        paths (list): List of directory paths for the runs.
        zoomin (list): List of the position axis limits in the plot, index 0 corresponds to the upstream boundary and index 1 to the downstream boundary.
        E_slice (list): Set boundaries to the histogram in the momentum space in energy units. Index 0 corresponds to the lower boundary and index 1 to the upper boundary.
        upstream (boolean): Toggle whether to plot the upstream escaping particles.
        downstream (boolean): Toggle whether to plot the downstream escaping particles.
        split (boolean): Toggles whether to plot in subplots or separate figures. split = True results in separate figures.
        titles (list): List of legend titles per run.
        savefigs (boolean): Toggle whether to save figures or not.
        show (boolean): Toggle whether to display plots or not.
        
    Returns:
        -
    ''' 
     
    # Set font size
    mpl.rcParams.update({'font.size': 10})
    
    # If splitting figures, create several figures
    if split:
        _, ax0 = plt.subplots(1,1,figsize=(6,5))
        _, ax1 = plt.subplots(1,1,figsize=(6,5))
        _, ax2 = plt.subplots(1,1,figsize=(6,5))
        ax = [ax0, ax1, ax2]
    # Otherwise use subplots
    else:
        fig, ax = plt.subplots(1, 3, figsize = (18, 5))
    
    # Go through all provided simulation runs
    for title_i, (inthist, p, path) in enumerate(zip(inthist_list if not v_para_hist else vparahist_list, parameters_list, paths)):
        
        # Evaluate the simulation box size in physical units
        ion_inert = eval_ion_inert(p["n_part"], p["e"], p["m_i"], p["eps0"], p["c"])
        v_L = E_to_v(eval_E_natural(p["E_L"] * 1000, p["m_e"], p["c"], p["e"])) * p["c"]
        box_size = eval_box_size(p["mfp_1"] * 1000, p["ux1"] * 1000, np.deg2rad(p["theta_1"]), v_L) / ion_inert
        x_max = p["down_bound_scaler"] * box_size
        x_min = -p["up_bound_scaler"] * box_size 
        
        # Evaluate upstream flow speed along x in physical units
        # Load simulation code for functions and parameters, then solve for the upstream flow speed
        easi = importlib.import_module("results." + path[-7:-1] + ".easi")
        importlib.reload(easi)
        eval_flow_profile = easi.eval_flow_profile 
        eval_L_inv = easi.eval_L_inv   
        u, _ =  eval_flow_profile(x_min, 0) # Uses t = 0 currently, as uses cases for temporally evolving profiles have not been mapped out yet
        
        # Evaluate particle injection speed and momentum distribution edges
        E_inj = eval_E_natural(p["E_inj_0"] * 1000, p["m_e"], p["c"], p["e"])
        v_inj = E_to_v(E_inj)
        v_inj, _ = change_frame(v_inj, 1, -u)
        p_inj = eval_gamma(v_inj) * v_inj
        p_min = np.log10(p["p_min_f"] * p_inj)
        p_max = np.log10(p["p_max_f"] * p_inj)
        
        # If custom spatial bins are used, solve those
        x_edges = p["spatial_bins"]()
        x_grid = x_edges[:-1] + (x_edges[1:] - x_edges[:-1])/2
            
        # Slice in the spatial direction if so desired
        if spat_slice[0] != None:
            inthist = inthist[np.argmax(spat_slice[0] < x_edges):np.argmax(x_edges > spat_slice[1])]
            x_grid = x_grid[np.argmax(spat_slice[0] < x_edges):np.argmax(x_edges > spat_slice[1])]
            x_edges = x_edges[np.argmax(spat_slice[0] < x_edges):np.argmax(x_edges > spat_slice[1])]
            x_edges = np.append(x_edges, np.array([x_edges[-1] + (x_max-x_min)/p["x_N"]]))
            
        # Create the pitch angle cosine histogram and momentum histogram also in the units of energy
        mu_grid = np.linspace(-1, 1, p["mu_N"])
        p_grid = np.logspace(p_min, p_max, p["p_N"])
        E_grid = p_to_E(p_grid)* p["m_e"] * p["c"]**2 / p["e"] /1000
        
        # Check if plotting the histogram in an energy range and slice the histogram with that information
        if E_slice[0] != None:
            print("Sliced!")
            inthist = inthist[:,(E_grid > E_slice[0]) & (E_grid < E_slice[1])]
            p_grid = p_grid[(E_grid > E_slice[0]) & (E_grid < E_slice[1])]
        
        # If labels for the legend are provided, add to plots
        if titles == None:
            label = ""
        else:
            label = titles[title_i]
        
        # If zooming in to the shock, plot in units 1 * d_i, otherwise use units 1000 * d_i
        divider = 1000
        div_txt = r"1000$\cdot$"
        if zoomin[0] != None:
            divider = 1 
            div_txt = ""
        
        # Create the spatial histogram
        if v_para_hist:
            _, tan_theta = eval_flow_profile(x_grid, 0) # Uses t = 0 currently, as uses cases for temporally evolving profiles have not been mapped out yet
            spatial_count = np.sum(inthist, axis = (1,2))/(x_edges[1:]-x_edges[:-1]) / np.sqrt(1 + tan_theta**2) 
        else:
            spatial_count = np.sum(inthist, axis = (1,2))/(x_edges[1:]-x_edges[:-1])
        spatial_count[0] = np.nan
        ax[0].plot(x_grid/divider, spatial_count, label=label, alpha = 0.7)
        
        # Create the momentum histogram
        momentum_count = np.sum(inthist, axis = (0,2))/(x_max-x_min)
        momentum_count/=p_grid**3
        if espec == False:
            if energy_axis:
                E_corr_grid = p_to_E(p_grid)
                momentum_count = (momentum_count[:-1] / np.diff(E_corr_grid))
                E_corr_grid *= (p["m_e"] * p["c"]**2) / p["e"]
                ax[1].plot(E_corr_grid[:-1], momentum_count, 
                        label=f"{'Total' if upstream or downstream else ''}{' ' if label != '' else ''}{label}", alpha = 0.7)
                if p["sda_regime"]:
                    # electron speed from eV
                    v_inj = p["c"] * np.sqrt(1 - ( 1 / ( 1 + p["E_inj_0"]/511 ) )**2)
                    v2 = v_inj + 2*p["ux1"]*1000/np.cos(np.deg2rad(p["theta_1"]))
                    gamma = 1/np.sqrt(1 - (v2/p["c"])**2)
                    max_E = (gamma-1)*p['m_e']*(p['c'])**2/p['e']
                    ax[1].text(0.05,0.95,f"max. SDA energy: {np.round(max_E,1)} eV", transform=ax[1].transAxes)
                    ax[1].vlines(max_E, ymin = np.min(momentum_count), ymax = np.max(momentum_count), ls="--", colors="red")
            else:
                ax[1].plot(p_grid, momentum_count, 
                        label=f"{'Total' if upstream or downstream else ''}{' ' if label != '' else ''}{label}", alpha = 0.7)
        else:
            # Particle omni-directional intensity in units of 1/(cm^2 sr s MeV): 
            x_size = 1#x_max - x_min
            factor = 1e-4/(4*np.pi * x_size)
            intensity_counts = factor * momentum_count[:-1]/(p_grid[1:] - p_grid[0:-1]) 
            ax[1].plot(E_grid[:-1], intensity_counts, 
                    label=f"{'Total' if upstream or downstream else ''}{' ' if label != '' else ''}{label}", alpha = 0.7)
            #ax[1].set_xlim((1e-1,1e1))
        
        # Create the pitch angle cosine histogram
        mu_count = np.sum(inthist, axis = (0,1))/(x_max-x_min)
        ax[2].plot(mu_grid, mu_count, 
                   label=f"{'Total' if upstream or downstream else ''}{' ' if label != '' else ''}{label}", alpha = 0.7)
        
        # Plot the upstream and downstream portions of the histogram separately per function inputs
        if upstream:
            ax[1].plot(p_grid, np.sum(inthist[x_grid < 0], axis = (0,2))/(x_max-x_min), label=f"Upstream{' ' if label != '' else ''}{label}", alpha = 0.7)
            ax[2].plot(mu_grid, np.sum(inthist[x_grid < 0], axis=(0,1))/(x_max-x_min), label=f"Upstream{' ' if label != '' else ''}{label}", alpha = 0.7)
        if downstream:
            ax[1].plot(p_grid, np.sum(inthist[x_grid > 0], axis = (0,2))/(x_max-x_min), label=f"Downstream{' ' if label != '' else ''}{label}", alpha = 0.7)
            ax[2].plot(mu_grid, np.sum(inthist[x_grid > 0], axis=(0,1))/(x_max-x_min), label=f"Downstream{' ' if label != '' else ''}{label}", alpha = 0.7)
    
    # Set plot setting for the spatial histogram with the potential zoomin
    if v_para_hist:
        ax[0].set_ylabel(r"$B^{-1}(x) \cdot \Sigma$ $v\cdot \mu \cdot$d$t$")
    else:
        ax[0].set_ylabel(r"$\Sigma$ d$t$")
    ax[0].set_xlabel(r"$x$ ["+f"{div_txt}"+r"$d_i$]")
    ax[0].set_yscale("log")
    ax[0].set_ylim(y_lim_first)
    if zoomin[0] != None:
        ax[0].set_xlim((zoomin[0],zoomin[1]))
    if titles != None:
        ax[0].legend(loc="lower right")

    # Plot the momentum histogram
    ax[1].set_xscale("log")
    ax[1].set_yscale("log")
    ax[1].set_ylabel("PSD")
    ax[1].set_ylim(y_lim_second)
    ax[1].set_xlim(x_lim_second)
    if energy_axis:
        ax[1].set_xlabel(r"$E$ [$eV$]") #
    else:
        ax[1].set_xlabel(r"$p$ [$m_e c$]")
    if titles != None or upstream or downstream:
        ax[1].legend()
    
    # Plots the pitch angle cosine histogram
    ax[2].set_ylabel("count")
    ax[2].set_yscale("log")
    ax[2].set_xlabel(r"$\mu$")
    if titles != None or upstream or downstream:
        ax[2].legend()

    # Save all figures and take into account whether they're split or not
    
    for path_str in paths:
        np.savetxt(f"{path_str}PSD_X.csv", E_corr_grid[:-1], delimiter=',')
        np.savetxt(f"{path_str}PSD_Y.csv", momentum_count, delimiter=',')
    if savefigs:
        if split:
            for path_str in paths:
                plt.sca(ax[0])
                name = f"{path_str}boxint_x.png"
                print("Saving to:", name)
                plt.savefig(name, facecolor = "white", bbox_inches = "tight", dpi = 300)
                plt.sca(ax[1])
                name = f"{path_str}boxint_p.png"
                print("Saving to:", name)
                plt.savefig(name, facecolor = "white", bbox_inches = "tight", dpi = 300)
                plt.sca(ax[2])
                name = f"{path_str}boxint_mu.png"
                print("Saving to:", name)
                plt.savefig(name, facecolor = "white", bbox_inches = "tight", dpi = 300)
                
        else:
            suffix = ""
            for path_str in paths:
                suffix += "_"+path_str[-4:-1]
            for path_str in paths:
                name = f"{path_str}boxint.png" if len(paths) == 1 else f"{path_str}boxint{suffix}.png"
                print("Saving to:", name)
                plt.savefig(name, facecolor = "white", bbox_inches = "tight", dpi = 300)
    if show:
        plt.show()
    else:
        plt.clf()


def plot_2dhist(inthist_list, parameters_list, paths, 
             zoomin = [None, None], 
             scaler0 = 1, scaler1 = 1, scaler2 = 1,
             split = True, savefigs = False, show = True): 
    
    '''
    Plots the time integrated histogram data 2-dimensionally with a color intensity as the third dimension (position, momentum, pitch angle cosine)
    
    Parameters:
        inthist_list (list): List of np arrays containing the time integrated histograms (see load_all_data for structure).
        parameters_list (list): List of parameter files corresponding to the runs.
        paths (list): List of directory paths for the runs.
        zoomin (list): List of the position axis limits in the plot, index 0 corresponds to the upstream boundary and index 1 to the downstream boundary.
        scaler0 (float): Intensity value scaler for colorbar of the first plot (x vs. p), if one wishes for different ranges for different plots.
        scaler1 (float): Intensity value scaler for colorbar of the second plot (x vs. mu), if one wishes for different ranges for different plots.
        scaler2 (float): Intensity value scaler for colorbar of the third plot (mu vs. p), if one wishes for different ranges for different plots.
        split (boolean): Toggles whether to plot in subplots or separate figures. split = True results in separate figures.
        savefigs (boolean): Toggle whether to save figures or not.
        show (boolean): Toggle whether to display plots or not.
        
    Returns:
        -
    ''' 
    
    # Set the font size and chosen color map
    mpl.rcParams.update({'font.size': 10})
    color = "inferno" 
    
    # Go throgh all provided simulations and plot each separately
    for inthist, p, path in zip(inthist_list, parameters_list, paths):

        # If splitting, create several figures
        if split:
            fig, ax0 = plt.subplots(1,1,figsize=(7,5))
            _, ax1 = plt.subplots(1,1,figsize=(7,5))
            _, ax2 = plt.subplots(1,1,figsize=(7,5))
            ax = [ax0, ax1, ax2]
        # Else use subplots
        else:
            fig, ax = plt.subplots(1, 3, figsize = (22, 5))
        
        # Evaluate the simulation box size in physical units
        ion_inert = eval_ion_inert(p["n_part"], p["e"], p["m_i"], p["eps0"], p["c"])
        v_L = E_to_v(eval_E_natural(p["E_L"] * 1000, p["m_e"], p["c"], p["e"])) * p["c"]
        box_size = eval_box_size(p["mfp_1"] * 1000, p["ux1"] * 1000, np.deg2rad(p["theta_1"]), v_L) / ion_inert
        x_max = p["down_bound_scaler"] * box_size
        x_min = -p["up_bound_scaler"] * box_size        
        
        # Evaluate upstream flow speed along x in physical units
        # Load simulation code for functions and parameters, then solve for the upstream flow speed
        easi = importlib.import_module("results." + path[-7:-1] + ".easi")
        importlib.reload(easi)
        eval_flow_profile = easi.eval_flow_profile    
        u, _ =  eval_flow_profile(x_min, 0) # Uses t = 0 currently, as uses cases for temporally evolving profiles have not been mapped out yet 
        
        # Evaluate particle injection speed and the momentum histogram maximum and minimum momenta
        E_inj = eval_E_natural(p["E_inj_0"] * 1000, p["m_e"], p["c"], p["e"])
        v_inj = E_to_v(E_inj)
        v_inj, _ = change_frame(v_inj, 1, -u)
        p_inj = eval_gamma(v_inj) * v_inj
        p_min = np.log10(p["p_min_f"] * p_inj)
        p_max = np.log10(p["p_max_f"] * p_inj)
        
        # Evaluate the grid
        x_edges = p["spatial_bins"]()
        x_grid = x_edges[:-1]
        
        # Evalaute the pitch angle cosine grid
        mu_grid = np.linspace(-1, 1, p["mu_N"])
        
        # Evaluate the momentum grid
        p_grid = np.logspace(p_min, p_max, p["p_N"])
        
        # If using a zoomin, use the units of 1 * d_i, otherwise use 1000 * d_i
        div_txt = r"1000$\cdot$"
        if zoomin[0] != None:
            divider = 1 
            div_txt = ""
            ax[0].set_xlim((zoomin[0],zoomin[1]))
            ax[1].set_xlim((zoomin[0],zoomin[1]))
        else: 
            divider = 1000
            
        # Solve the meshed grid for x vs. p
        X,Y = np.meshgrid(x_grid, p_grid)
        
        # Evaluate the counts and the corresponding colorbar, then plot
        counts = np.sum(inthist, axis = 2).T/(x_edges[1:]-x_edges[:-1])
        norming = mpl.colors.SymLogNorm(np.amax(counts)*scaler0, vmin = 0, vmax = np.amax(counts), linscale = 1.0, base = 10)
        plot = ax[0].pcolormesh(X/divider, Y, counts, cmap = color, shading = 'auto', norm = norming)
        ax[0].set_yscale("log")
        ax[0].set_xlabel(r"$x$ ["+f"{div_txt}"+r"$d_i$]")
        ax[0].set_ylabel(r"$p$ [$m_e c$]")
        fig.colorbar(plot, ax = ax[0])
        
        # Solve the meshed grid for x vs. mu
        X,Y = np.meshgrid(x_grid, mu_grid)
        
        # Evaluate the counts and the corresponding colorbar, then plot
        counts = np.sum(inthist, axis = 1).T/(x_edges[1:]-x_edges[:-1])        
        norming = mpl.colors.SymLogNorm(np.amax(counts)*scaler1, vmin = 0, vmax = np.amax(counts), linscale = 1.0, base = 10)
        plot = ax[1].pcolormesh(X/divider, Y, counts, cmap = color, shading = 'auto', norm=norming)
        ax[1].set_xlabel(r"$x$ ["+f"{div_txt}"+r"$d_i$]")
        ax[1].set_ylabel(r"$\mu$")
        fig.colorbar(plot, ax = ax[1])
        
        # A code block to evaluate reflected population density vs. the injected density
        # int_xlim = [-10,-3]
        # inj_pop = counts[mu_grid > 0] 
        # inj_pop = np.sum(inj_pop[:,(x_grid > int_xlim[0]) & (x_grid < int_xlim[1])])
        # ref_pop = counts[mu_grid < 0]
        # ref_pop = np.sum(ref_pop[:,(x_grid > int_xlim[0]) & (x_grid < int_xlim[1])])
        # print(f"Ratio of reflected population from injected population (neg. mu / pos. mu, x \in [{int_xlim[0]}, {int_xlim[1]}]):", ref_pop/inj_pop)

        # Solve the meshed for mu vs. p
        X,Y = np.meshgrid(mu_grid, p_grid)
        
        # Evaluate the counts and the corresponding colorbar, then plot
        counts = np.sum(inthist, axis = 0)
        norming = mpl.colors.SymLogNorm(np.amax(counts)*scaler2, vmin = 0, vmax = np.amax(counts), linscale = 1.0, base = 10)
        plot = ax[2].pcolormesh(X, Y, counts, cmap = color, shading = 'auto', norm = norming)
        ax[2].set_yscale("log")
        ax[2].set_xlabel(r"$\mu$")
        ax[2].set_ylabel(r"$p$ [$m_e c$]")
        fig.colorbar(plot, ax = ax[2])
        
        # Save and show the figures per given settings
        if savefigs:
            if split:
                plt.sca(ax[0])
                name = f"{path}2dint_p.png"
                print("Saving to:", name)
                plt.savefig(name, facecolor = "white", bbox_inches = "tight", dpi = 300)
                plt.sca(ax[1])
                name = f"{path}2dint_mu.png"
                print("Saving to:", name)
                plt.savefig(name, facecolor = "white", bbox_inches = "tight", dpi = 300)
                plt.sca(ax[2])
                name = f"{path}2dint_mu_p.png"
                print("Saving to:", name)
                plt.savefig(name, facecolor = "white", bbox_inches = "tight", dpi = 300)
                
            else:
                name = f"{path}2dint.png"
                print("Saving to:", name)
                plt.savefig(name, facecolor = "white", bbox_inches = "tight", dpi = 300)
        if show:
            plt.show()
        else:
            plt.clf()
            
        
def plot_2dmomhist(pinthist_list, parameters_list, paths, 
                   zoomin = [None, None], arr_loc = [-1], units_E = False,
                   xlims = [None, None], ylims = [None, None],
                   bottom_row_only = False, upstream_slice = True,
                   savefigs = False, show = True): 
    '''
    Plots the time integrated parallel and perpendicular momentum histogram data 2-dimensionally with a color intensity as the third dimension (position, momentum, count)
    
    Parameters:
        pinthist_list (list): List of np arrays containing the time integrated parallel and perpendicular histograms (see load_all_data for structure).
        parameters_list (list): List of parameter files corresponding to the runs.
        paths (list): List of directory paths for the runs.
        arr_loc (list): Locations of arrows and slices to be taken from the parallel momentum towards the upstream in the units of d_i.
        units_E (boolean): Toggle for whether to plot in momentum in units of [m_e c] or energy in units of [keV]. units_E = True plots in energy units.
        xlims (list): List of the x-axis limits in the plot, index 0 corresponds to the lower boundary and index 1 to the higher boundary.
        ylims (list): List of the y-axis limits in the plot, index 0 corresponds to the lower boundary and index 1 to the higher boundary.
        savefigs (boolean): Toggle whether to save figures or not.
        show (boolean): Toggle whether to display plots or not.
        
    Returns:
        -
    ''' 
    
    
    # Set the colormap, 'fonts' for axis label font sizes and then the general font size, which is set to 15
    color = "inferno"
    fonts = 20
    mpl.rcParams.update({'font.size': 15})
    
    # Go through all the provided simulation runs
    for pinthist, p, path in zip(pinthist_list, parameters_list, paths):

        # Create the subplots
        fig, ax = plt.subplots(2, 2, figsize = (15, 10))
        
        # Evaluate the simulation box size in physical units
        ion_inert = eval_ion_inert(p["n_part"], p["e"], p["m_i"], p["eps0"], p["c"])
        v_L = E_to_v(eval_E_natural(p["E_L"] * 1000, p["m_e"], p["c"], p["e"])) * p["c"]
        box_size = eval_box_size(p["mfp_1"] * 1000, p["ux1"] * 1000, np.deg2rad(p["theta_1"]), v_L) / ion_inert
        x_max = p["down_bound_scaler"] * box_size
        x_min = -p["up_bound_scaler"] * box_size    
        
        # Evaluate particle injection speed and the momentum histogram minimum and maximum momenta
        E_inj = eval_E_natural(p["E_inj_0"] * 1000, p["m_e"], p["c"], p["e"])
        v_inj = E_to_v(E_inj)
        p_inj = eval_gamma(v_inj) * v_inj
        p_min = np.log10(p["p_min_f"] * p_inj)
        p_max = np.log10(p["p_max_f"] * p_inj)
        
        # Evaluate the grid
        x_edges = p["spatial_bins"]()
        x_grid = x_edges[:-1]
        
        # Solve for the momenta grid
        p_grid = np.logspace(p_min, p_max, p["p_N"]//2)

        # Solve the counts for all momenta plots in their respective grids
        counts_para_pos = (np.sum(pinthist, axis = 2).T)[:p["p_N"]//2-1] / ((p_grid[1:]-p_grid[:-1])[:,None])/(x_edges[1:]-x_edges[:-1])
        counts_para_neg = (np.sum(pinthist, axis = 2).T)[-1:-p["p_N"]//2:-1] / ((p_grid[1:]-p_grid[:-1])[:,None])/(x_edges[1:]-x_edges[:-1])
        counts_perp_pos = (np.sum(pinthist, axis = 1).T)[:p["p_N"]//2-1] / ((p_grid[1:]-p_grid[:-1])[:,None] * p_grid[:-1,None] * 2*np.pi)/(x_edges[1:]-x_edges[:-1])

        # A code block to evaluate the beam density vs. the rest of the particle population
        # E_lim = [2e0,13e0]
        # p_grid2 = p_grid
        # p_grid = p_grid[:-1]
        # int_xlim = [-10e-3,-3e-3]
        # box_arg = (x_grid > int_xlim[0]) & (x_grid < int_xlim[1])
        # inj_pop = (counts_para_neg[:-1]*(p_grid[1:]-p_grid[:-1])[:,None])[p_grid[:-1] < E_lim[0]]
        # inj_pop = np.sum(inj_pop[:,box_arg]) + np.sum(counts_para_pos[:-1,box_arg]*(p_grid[1:]-p_grid[:-1])[:,None])# + np.sum(counts_perp_pos[:,box_arg])
        # beam_pop = (counts_para_neg[:-1]*(p_grid[1:]-p_grid[:-1])[:,None])[(p_grid[:-1] > E_lim[0]) & (p_grid[:-1] < E_lim[1])]
        # beam_pop = np.sum(beam_pop[:,box_arg])
        # print(f"Ratio of beam integral vs. the rest (x \in [{int_xlim[0]}, {int_xlim[1]}]):", beam_pop/inj_pop)
        # p_grid = p_grid2

        # Change to energy units if desired
        if units_E:
            E_grid = p_to_E(p_grid[:-1])*p["m_e"]*p["c"]**2/p["e"]/1000
            X,Y = np.meshgrid(x_grid, E_grid)
        
        # Else use x vs. p
        else:
            X,Y = np.meshgrid(x_grid, p_grid[:-1])
            
        # If zooming in use units of 1 * d_i, else use 1000 * d_i
        divider = 1000
        div_txt = r"1000$\cdot$"
        if zoomin[0] != None:
            divider = 1
            div_txt = ""

        # Set the colorbar normalization per the set y-axis limits for the reduced momentum distribution
        norming_para = mpl.colors.SymLogNorm(ylims[0], vmin = 0, vmax = ylims[1], linscale = 1.0, base = 10)
        norming_perp = mpl.colors.SymLogNorm(ylims[0], vmin = 0, vmax = ylims[1], linscale = 1.0, base = 10)

        # Parallel direction plotting
        plot = ax[0,0].pcolormesh(X/divider, Y, counts_para_pos, cmap = color, shading = 'auto', norm = norming_para)
        ax[0,0].set_yscale("log")
        ax[0,0].set_ylim((xlims[0], xlims[1]))
        if zoomin[0] != None:
            ax[0,0].set_xlim((zoomin[0],zoomin[1]))
        if units_E:
            ax[0,0].set_ylabel(r"$p_{\parallel}c$ [keV]", fontsize = fonts)
        else:
            ax[0,0].set_ylabel(r"$p_\parallel$ [$m_e c$]", fontsize = fonts)
        if not bottom_row_only:
            clb = fig.colorbar(plot, ax = ax[0,0])
        #clb.ax.set_ylabel("count")
        
        # Anti-parallel direction plotting
        plot = ax[1,0].pcolormesh(X/divider, Y, counts_para_neg, cmap = color, shading = 'auto', norm = norming_para)
        ax[1,0].set_yscale("log")
        ax[1,0].set_xlabel(r"$x$ ["+f"{div_txt}"+r"$d_i$]", fontsize = fonts)
        
        ax[1,0].set_ylim((xlims[0], xlims[1]))
        if zoomin[0] != None:
            ax[1,0].set_xlim((zoomin[0],zoomin[1]))
        if units_E:
            ax[1,0].set_ylabel(r"$-p_{\parallel}c$ [keV]", fontsize = fonts)
        else:
            ax[1,0].set_ylabel(r"$-p_\parallel$ [$m_e c$]", fontsize = fonts)
        clb = fig.colorbar(plot, ax = ax[1,0])
        #clb.ax.set_ylabel("count")
        
        # Energy spectrum arrows
        if upstream_slice:
            ymax = ax[1,0].get_ylim()[1]
            ymin = ymax/4
            xmin = ax[1,0].get_xlim()[0]
            xmax = ax[1,0].get_xlim()[1]
            #colors = ["#0072B2", "#D55E00", "#32E2C4"]
            colors = custom_colors[1:]
            citer = iter(colors)
            for loc in arr_loc:
                arrcolor = next(citer)
                xloc = loc
                xloc0 = xloc - (xmax-xmin)/100*divider
                xloc1 = xloc + (xmax-xmin)/100*divider

                xarr = np.array([xloc0,xloc1,xloc,xloc0])/divider
                yarr = [ymin, ymin, ymin*0.8, ymin]
                
                ax[1,0].plot(xarr,yarr,color=arrcolor)#, lw = 3, ms = 3)
                ax[1,0].vlines(xloc/divider, ymin = ymin, ymax = ymax, color = arrcolor)
            
        if not upstream_slice:
            ymax = ax[0,0].get_ylim()[1]
            ymin = ymax/4
            xmin = ax[0,0].get_xlim()[0]
            xmax = ax[0,0].get_xlim()[1]
            #colors = ["#0072B2", "#D55E00", "#32E2C4"]
            colors = custom_colors[1:]
            citer = iter(colors)
            for loc in arr_loc:
                arrcolor = next(citer)
                xloc = loc
                xloc0 = xloc - (xmax-xmin)/100*divider
                xloc1 = xloc + (xmax-xmin)/100*divider

                xarr = np.array([xloc0,xloc1,xloc,xloc0])/divider
                yarr = [ymin, ymin, ymin*0.8, ymin]
                
                ax[0,0].plot(xarr,yarr,color=arrcolor)#, lw = 3, ms = 3)
                ax[0,0].vlines(xloc/divider, ymin = ymin, ymax = ymax, color = arrcolor)
            
        ax[1,0].invert_yaxis()
        
        # Perpendicular direction plotting
        plot = ax[0,1].pcolormesh(X/divider, Y, counts_perp_pos, cmap = color, shading = 'auto', norm = norming_perp)
        ax[0,1].set_yscale("log")
        
        ax[0,1].set_ylim((xlims[0], xlims[1]))
        if zoomin[0] != None:
            ax[0,1].set_xlim((zoomin[0],zoomin[1]))
        if units_E:
            ax[0,1].set_ylabel(r"$p_{\perp}c$ [keV]", fontsize = fonts)
        else:
            ax[0,1].set_ylabel(r"$p_\perp$ [$m_e c$]", fontsize = fonts)
        if not bottom_row_only:
            clb = fig.colorbar(plot, ax = ax[0,1])
        #clb.ax.set_ylabel("count")
        
        # Plot the slices along the specified arrow locations
        if units_E:
            x_vals =  p_to_E(p_grid[:-1])*p["m_e"]*p["c"]**2/p["e"]/1000
        else:
            x_vals = p_grid[:-1]
        for i, loc in enumerate(arr_loc):
            slicex = np.digitize(loc, x_grid)
            if upstream_slice:
                ax[1,1].plot(x_vals, counts_para_neg[:,slicex], 
                            label = f"$x = $ {x_grid[slicex]:.1f} $d_i$", 
                            color = colors[i], alpha = 0.85)
            if not upstream_slice:
                ax[1,1].plot(x_vals, counts_para_pos[:,slicex], 
                            label = f"$x = $ {x_grid[slicex]:.1f} $d_i$", 
                            color = colors[i], alpha = 0.85)
        # Set the slice plot properties
        ax[1,1].set_xscale("log")
        ax[1,1].set_yscale("log")
        ax[1,1].set_ylabel("Reduced momentum distribution [arb.]")  
        ax[1,1].set_ylim(ylims)
        ax[1,1].legend()
        if units_E:
            ax[1,1].set_xlim(xlims)
            ax[1,1].set_xlabel(r"-$p_{\parallel}c$ [keV]", fontsize = fonts)
        else:
            ax[1,1].set_xlim(xlims)
            ax[1,1].set_xlabel(r"$-p_\parallel$ [$m_e c$]", fontsize = fonts)
        plt.sca(ax[1,1])
        clb = plt.colorbar(mappable=mpl.cm.ScalarMappable(norm=norming_para, cmap=color),orientation="vertical", ax = ax[1,1])
        clb.remove()
        
        if bottom_row_only:
            ax[0,0].set_visible(False)
            ax[0,1].set_visible(False)
        plt.tight_layout()
        
        # Save and show figures per settings
        if savefigs:
            name = f"{path}2dpint.png"
            print("Saving to:", name)
            plt.savefig(name, facecolor = "white", bbox_inches = "tight", dpi = 300)
        if show:
            plt.show()
        else:
            plt.clf()


def plot_residence_t(particles_list, parameters_list, paths, 
                     E_slice = [None,None], xlims = [None, None], ylims = [None, None],
                     upstream = True, downstream = True,
                     savefigs = False, show = True):   
    '''
    Plot the residence time distributions and calculate mean residence times separately for the upstream and downstream escaping population.
    
    Parameters:
        particles_list (list): List of np arrays of particles (see load_all_data for structure).
        parameters_list (list): List of parameter files corresponding to the runs.
        paths (list): List of directory paths for the runs.
        E_slice (list): Set boundaries to the histogram in the momentum space in energy units. Index 0 corresponds to the lower boundary and index 1 to the upper boundary.
        xlims (list): List of the x-axis limits in the plot, index 0 corresponds to the lower boundary and index 1 to the higher boundary.
        ylims (list): List of the y-axis limits in the plot, index 0 corresponds to the lower boundary and index 1 to the higher boundary.
        upstream (boolean): Toggle whether to plot the upstream escaping particles.
        downstream (boolean): Toggle whether to plot the downstream escaping particles.
        savefigs (boolean): Toggle whether to save figures or not.
        show (boolean): Toggle whether to display plots or not.
        
    Returns:
        -
    ''' 
    
    # Create the subplots
    _, ax = plt.subplots(1, 1, figsize = (6, 5))
    
    # Create lists to store upstream and downstream mean escaping times for each provided simulation run separately
    u_means = []
    d_means = []

    # Go through all provided simulation runs
    for particles, p, path in zip(particles_list, parameters_list, paths):
        
        # If choosing an energy range, take only those Monte Carlo particles into account
        if E_slice[0] != None:
            print("Sliced!")
            E0 = eval_E_natural(E_slice[0]*1000, p["m_e"], p["c"], p["e"])
            E1 = eval_E_natural(E_slice[1]*1000, p["m_e"], p["c"], p["e"])
            particles = particles[(v_to_E(particles[:,1]) > E0) & (v_to_E(particles[:,1]) < E1)]

        # Evaluate the factor needed for physical time
        t_unit = np.sqrt(p["n_part"] * 1e6 * p["e"]**2 / (p["m_i"] * p["eps0"]))
        
        # Aliases for particle position, injection time, and escape time
        x = particles[:,0]
        t = particles[:,3]/t_unit
        t_start = particles[:,4]/t_unit
        
        # Evaluate residence time from the injection time and escape time
        t_res = t - t_start
        
        # Evaluate the histogram for the upstream escaping population, then solve the mean residence times and plot
        if upstream: 
            t_hist_up, bins_up = np.histogram((t_res[x < 0]), bins = 100)
            bins_up = bins_up[:-1] + (bins_up[1:]-bins_up[:-1])/2
            ax.plot(bins_up, t_hist_up, label=f"Exited upstream{' '+path if len(paths) > 1 else ''}", alpha = 0.7)
            u_means.append(np.round(np.mean(t_res[x < 0]),2))
            ax.text(0.1,0.1,f"Upstream mean: {u_means[0]} s", transform = ax.transAxes)

        # Evaluate the histogram for the downstream escaping population, then solve the mean residence times and plot
        if downstream:
            t_hist, bins = np.histogram(t_res[x > 0], bins = 100)
            bins = bins[:-1] + (bins[1:]-bins[:-1])/2
            ax.plot(bins, t_hist, label=f"Exited downstream{' '+path if len(paths) > 1 else ''}", alpha = 0.7)
            d_means.append(np.round(np.mean(t_res[x > 0]),2))
            ax.text(0.1,0.05,f"Downstream mean: {d_means[0]} s", transform = ax.transAxes)


    # Set plot properties
    ax.set_xlabel(f"$t$ [s]")
    ax.set_ylabel("Particle count")
    ax.set_yscale("log")
    ax.set_title("Residence Time")
    ax.set_xlim(xlims)
    ax.set_ylim(ylims)
    ax.legend()

    # Save and show the figure per settings
    if savefigs:
        suffix = ""
        for path_str in paths:
            suffix += "_"+path_str[-4:-1]
        for path_str in paths:
            name = f"{path_str}restime_single.png" if len(paths) == 1 else f"{path_str}restime{suffix}.png"
            print("Saving to:", name)
            plt.savefig(name, facecolor = "white", bbox_inches = "tight", dpi = 300)
    if show:
        plt.show()
    else:
        plt.clf()
        

def energy_spectrum(v, x_min = 0, x_max = 1, E_min = 0.001, E_max = 1.0, E_nbins = 50):
    """
    Calculate the differential energy spectrum for electrons based on velocity data.
    
    Parameters:
        v (array): Array of partices velocities.
        x_min (float): Integration box lower bound.
        x_max (float): Integration box upper bound.
        E_min (float): Energy grid min value.
        E_max (float): Energy grid max value.
        E_nbins (int): Number of energy bins.
        
    Returns:
        energy_bins (array): Bin centers for energy.
        intensity_counts (array): Normalized counts for each bin.
    """
    
    # Physical constants
    c = 299792458
    e = 1.602176634e-19
    m_e = 9.1093837015e-31
    
    # Rest energy of an electon
    E0 = m_e * c**2 / (e * 1e6)
    
    # Convert velocities to momenta (MeV * s/m)
    p = m_e * v / np.sqrt(1 - (v/c)**2) /(e * 1e6) 

    # Create the momentum grid
    p_min = np.sqrt(E_min**2 + 2 * E_min * E0)/c
    p_max = np.sqrt(E_max**2 + 2 * E_max * E0)/c
    num_points = E_nbins + 1
    p_grid = np.geomspace(p_min, p_max, num=num_points) 

    # Compute particle counts     
    part_counts, bin_edges = np.histogram(p, bins=p_grid)

    # Compute energy bins
    momentum_bin_centers = 0.5 * (bin_edges[0:-1] + bin_edges[1:])   
    energy_bins = np.sqrt(E0**2 + (momentum_bin_centers * c)**2) - E0
    
    # Particle omni-directional intensity in units of 1/(cm^2 sr s MeV): 
    x_size = x_max - x_min
    factor = 1e-4/(4*np.pi * x_size)
    intensity_counts = factor * part_counts[:]/(bin_edges[1:] - bin_edges[0:-1]) 
    
    return energy_bins, intensity_counts


def show_prms(run_no):
    
    # Set path strings
    path = f".{os.sep}results{os.sep}run{str(0)*(3-len(str(run_no)))+str(run_no)}{os.sep}"
            
    with open(f"{path}prms.py") as file:
        data = file.read()
     
    print(data)
    
    
def plot_profiles(parameters_list, path_list, zoomin = False, 
                  k = None, E = None, lambda_text = True, 
                  mfp_ylims = [None, None], titles = None, savefigs = False, show = True, legend = False):
    '''
    Plot the shock profiles of obliquity, flow speed and mean free path.
    
    Parameters:
        parameters_list (list): List of parameter files corresponding to the runs.
        paths (list): List of directory paths for the runs.
        zoomin (boolean): Toggle whether to zoomin close to the shock, [-10 d_i, 10 d_i].
        k (float): If a custom k value (check prms.py) is desired, set it here.
        E (float): The partice energy for which the mean free path will be plotted. When setting None will use injection energy of particles (E_inj_0 in prms.py).
        lambda_text (boolean): Toggles the mean free path equation text on top of the plot.
        mfp_ylims (list): Controls the y-axis limits in the mean free path plot. Index 0 is for the lower boundary and index 1 is for the upper boundary.
        titles (list): List of legend titles per run.
        savefigs (boolean): Toggle whether to save figures or not.
        show (boolean): Toggle whether to display plots or not.
        legend (boolean): Toggles the legend below the mean free path plot, uses labels set in 'titles' argument.
        
    Returns:
        -
    '''
    
    # Create the subplots
    _, ax = plt.subplots(3,1,figsize=(4,5+legend*2), sharex=True, constrained_layout=True)
    
    # Go through all provided runs
    for ind, (parameters, path) in enumerate(zip(parameters_list, path_list)):
        # parameters = parameters[run_no]
        # path = path[run_no]
        
        # Set the labels for the legend
        if titles == None:
            label = ""
        else:
            label = titles[ind]
        
        # Import the code and its parameters
        easi = importlib.import_module("results." + path[-7:-1] + ".easi")
        importlib.reload(easi)
        
        # Set aliases for the code's mean free path and flow speed profile functions
        eval_mfp_inv = easi.eval_mfp_inv
        eval_flow_profile = easi.eval_flow_profile
        
        # Set an alias for the parameters
        p = parameters

        # Evaluate the box size in physical units
        ion_inert = eval_ion_inert(p["n_part"], p["e"], p["m_i"], p["eps0"], p["c"])
        v_L = E_to_v(eval_E_natural(p["E_L"] * 1000, p["m_e"], p["c"], p["e"])) * p["c"]
        box_size = eval_box_size(p["mfp_1"] * 1000, p["ux1"] * 1000, np.deg2rad(p["theta_1"]), v_L) / ion_inert
        x_max = p["down_bound_scaler"] * box_size
        x_min = -p["up_bound_scaler"] * box_size
        
        print("box size:", box_size)   
        
        # Evaluate shock properties in physical units
        M_A = p["ux1"]/(np.cos(np.deg2rad(p["theta_1"]))*p["V_A1"]) # The Alfvén Mach number in the de Hoffmann-Teller Frame (HTF)
        rg = eval_rg(M_A, np.deg2rad(p["theta_1"]), p["beta"]) # Gas compression ratio computed from parameters
        rb = eval_rb(M_A, rg, np.deg2rad(p["theta_1"]))
        theta_1 = np.deg2rad(p["theta_1"]) # Shock obliquity in radians  
        tan_theta_1 = np.tan(theta_1)  
        ux1 = p["ux1"]*1e3/p["c"]
        V_A1 = p["V_A1"]*1000/p["c"] # unitless Alfvén speed
        m_i = p["m_i"]/p["m_e"] # Unitless ion mass
        alpha = m_i * V_A1 # With units corresponds to m_i/m_e * V_A1/c
        a_ratio = p["a_prm"]/alpha
        
        # Create the position axis grid and check whether zoomin is required
        x = np.linspace(x_min, x_max, 100000)
        if zoomin:
            x = np.linspace(-10, 10, 100000)
        
        # Set the k value and mean free path energy value per settings
        if k == None:
            k = p["k"]
        if E == None:
            E = p["E_inj_0"]
            
        # Solve for the mean free path profile with the chosen energy and the flow speed profile
        v = E_to_v(eval_E_natural(E*1000, p["m_e"], p["c"], p["e"]))
        u, tan_theta = eval_flow_profile(x, 0) # Uses t = 0 currently, as uses cases for temporally evolving profiles have not been mapped out yet    
        mfp_inv = eval_mfp_inv(x, v, 0, tan_theta) # Uses t = 0 currently, as uses cases for temporally evolving profiles have not been mapped out yet
        
        print("Ion inertial length d_i =", ion_inert, "m")
        print("Maximum flow speed =", max(u*p["c"]/1000), "km/s")
        
        # Plot the obliquity profile and the flow speed profile only for the first run in the provided set of runs
        if ind == 0:
            plt.sca(ax[0]) 
            #plt.xlabel(r"$x$ [$d_i$]")
            font = 13
            plt.ylabel(r"$\theta_{Bn}\ [^\circ]$", fontsize = font)
            plt.plot(x, np.rad2deg(np.arctan(tan_theta)))#, c="r")
            
            plt.sca(ax[1])
            #plt.xlabel(r"$x$ [$d_i$]")
            plt.ylabel("$u$ [km/s]", fontsize = font)
            plt.plot(x, u*p["c"]/1000) 
        
        # Plot the mean free path profile for all provided runs
        plt.sca(ax[2])
        plt.xlabel(r"$x$ [$d_i$]")
        if lambda_text:
            plt.text(0.02,0.15, 
                    r"$\lambda \propto\frac{r_\mathrm{L}}{J_z} \propto \frac{r_\mathrm{L}}{\partial B_y / \partial x}$", 
                    fontsize = 17, transform = plt.gca().transAxes)
        plt.plot(x, 1/mfp_inv*ion_inert/1e3, alpha = 0.7, label = label)
        plt.ylabel(r"$\lambda (E_{\rm inj})$ [km]", fontsize = font)
        plt.yscale("log")
        plt.ylim(mfp_ylims)
        if mfp_ylims[0] != None:
            nticks = np.log10(mfp_ylims[1]/mfp_ylims[0])+1
            maj_loc = mpl.ticker.LogLocator(numticks=nticks)
            ax[2].yaxis.set_major_locator(maj_loc)
            
        # Show the legend per settings
        if legend:
            plt.legend(loc = (0.2,-0.22*len(titles)))#, transform = plt.gca().transAxes)
    
    # Save and show figures per settings
    if savefigs:
        suffix = ""
        suffix += "_"+path[-4:-1]
        name = f"{path}profiles{suffix}.png"
        print("Saving to:", name)
        plt.savefig(name, facecolor = "white", bbox_inches = "tight", dpi = 300)
    if show:
        plt.show()
    else:
        plt.clf()
    
    
def plot_inj(parameters_list, paths, inj_specs = []):
    '''
    Plot the particle injection population properties (position, speed, pitch angle cosine, parallel speed).
    
    Parameters:
        parameters_list (list): List of parameter files corresponding to the runs.
        paths (list): List of directory paths for the runs.
        inj_specs (list): List of particle properties at injection moment of particles.
        
    Returns:
        -
    '''
    
    if inj_specs == []:
        print("No injection spectra!")
        
    else:       
        
        # Go through all provided simulation runs
        for inj_spec, p, path in zip(inj_specs, parameters_list, paths):
            
            # Load simulation code for functions and parameters
            easi = importlib.import_module("results." + path[-7:-1] + ".easi")
            importlib.reload(easi)
            
            # Set alias for flow speed profile
            eval_flow_profile = easi.eval_flow_profile
            
            # Evaluate box size in physical units
            ion_inert = eval_ion_inert(p["n_part"], p["e"], p["m_i"], p["eps0"], p["c"])
            v_L = E_to_v(eval_E_natural(p["E_L"] * 1000, p["m_e"], p["c"], p["e"])) * p["c"]
            box_size = eval_box_size(p["mfp_1"] * 1000, p["ux1"] * 1000, np.deg2rad(p["theta_1"]), v_L) / ion_inert
            x_min = -p["up_bound_scaler"] * box_size        
            
            # Evaluate shock parameters in physical units
            M_A = p["ux1"]/(np.cos(np.deg2rad(p["theta_1"]))*p["V_A1"]) # The Alfvén Mach number in the de Hoffmann-Teller Frame (HTF)
            rg = eval_rg(M_A, np.deg2rad(p["theta_1"]), p["beta"]) # Gas compression ratio computed from parameters
            rb = eval_rb(M_A, rg, np.deg2rad(p["theta_1"]))
            
            # Evaluate the flow speed profile in physical units
            theta_1 = np.deg2rad(p["theta_1"]) # Shock obliquity in radians  
            ux1 = p["ux1"]*1e3/p["c"]
            u, _ =  eval_flow_profile(x_min, 0) # Uses t = 0 currently, as uses cases for temporally evolving profiles have not been mapped out yet 
            u = u * np.ones(len(inj_spec)) 

            # Plot the different injection spectra in the plasma rest frame
            print("Plasma rest frame")
            _, axes = plt.subplots(1,4, figsize=(15,4)) 
            labels = ("x", "v", "mu", "vpara")
            nf_specs = np.copy(inj_spec)
            nf_specs[:,1], nf_specs[:,2] = change_frame(nf_specs[:,1], nf_specs[:,2], u)
            
            # Calculate beam limits as per Holman & Pesses 1983
            # print("mean injection speed", np.mean(nf_specs[:,1]) * c, "m/s")
            # print("mean injection parallel speed", np.mean(nf_specs[:,1] * nf_specs[:,2]) * c, "m/s")
            # print("shock speed", p["ux1"]*1e3)
            # # cos theta > v_
            # #print("minimum obliquity for mean", np.rad2deg(np.arccos(p["ux1"]*1e3/(np.mean(nf_specs[:,1] * nf_specs[:,2]) * c))))
            # rb1 = 3.8
            # print("minimum obliquity for mean", np.rad2deg(np.arccos(26000/1400)))
            # print("rb", rb)
            # print("maximum obliquity for mean", np.rad2deg(np.arccos((rb)**(-1/2)*(p["ux1"]*1e3/(np.mean(nf_specs[:,1]) * c)))))
            
            for i, ax in enumerate(axes[:-1]):
                plt.sca(ax)
                plt.xlabel(labels[i])
                y,x = np.histogram(nf_specs[:,i], bins=10)
                plt.plot(x[:-1],y)
                plt.yscale("log")
                if i==1:
                    plt.xlim((0,0.2))
                if i==2:
                    plt.xlim((-1.1,1.1))
            plt.sca(axes[3])
            plt.xlabel(labels[3])    
            y,x = np.histogram(nf_specs[:,1]*nf_specs[:,2], bins=100)
            plt.plot(x[:-1],y)
            plt.yscale("log")  
            plt.show()

            # Plot the different injection spectra in the de Hoffmann - Teller frame
            print("de Hoffmann-Teller Frame")
            _, axes = plt.subplots(1,4, figsize=(15,4)) 
            labels = ("x", "v", "mu","vpara")
            for i, ax in enumerate(axes[:-1]):
                plt.sca(ax)
                plt.xlabel(labels[i])
                y,x = np.histogram(inj_spec[:,i], bins=100)
                plt.plot(x[:-1],y)
                plt.yscale("log")
                if i==2:
                    plt.xlim((-1.1,1.1))
            plt.sca(axes[3])
            plt.xlabel(labels[3])     
            y,x = np.histogram(inj_spec[:,1]*inj_spec[:,2], bins=100)
            plt.plot(x[:-1],y)
            plt.yscale("log")        
            plt.show()