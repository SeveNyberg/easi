# Electron Acceleration SImulation (EASI)

When using the model or parts of the model and script in your work, please acknowledge the usage of the model and script and cite Nyberg et al. 2025 (in prep.).

## Introduction

A Monte Carlo test particle code to simulate electron acceleration in interplanetary shocks. The details of the model are written in Nyberg et al. 2025 (in prep.).

## Installation

The installation instructions use [Miniconda](https://www.anaconda.com/docs/getting-started/miniconda/install) to setup the needed python environment with packages. The installation assumes you are either using Linux/MacOS terminal or Anaconda Prompt in Windows (Anaconda Prompt was installed with your Miniconda/other Anaconda installation).

For more intricate instructions, [click here.](https://gitlab.utu.fi/lanjoh/easi-documentation/-/blob/main/Documentation.md). 

### Download the project

Download the project and setup the files so that your resulting file structure looks as follows:
```
EASI (main directory)
|--easi.py
|--simulation_functions.py
|--prms.py
|--plot_results.ipynb
|--plotting_functions.py
|--REQUIREMENTS.txt
|--__init__.py
```

### Creating a virtual environment

The has been tested on **python version 3.11.4**. Create a virtual python environment with 
```
conda create -n easi python=3.11.4
```

Activate the virtual environment for installation of packages and operation of the code. **Remember to always activate this virtual environment when operating any part of the code!**:
```
conda activate easi
```

### Installing packages

**Ensure that the virtual environment is active to not mess with your local packages!** 

Move to the directory you have set the project up in:
```
cd <path to main directory>
```

The required packages and their versions are listed in REQUIREMENTS.txt. The list of packets can be installed using pip:
```
pip install -r REQUIREMENTS.txt
```

The project should be operational now.

## Operation

### Files included in the project

- easi.py - The main simulation script with some physics functions. In general, you do not need to access this one.
- simulation\_functions.py - Some more generalistic functions used in plotting and simulating. In general, you do not need to access this one.
- prms.py - The input parameter file for the simulation. Access this when inputting your choice of parameters and profiles.
- plot\_results.ipynb - The simulation result plotting iPython notebook. Access this when plotting results of your simulation.
- plotting\_functions.py - Functions for plotting results. You do not need to access this one, unless you want to change resulting figures.
- REQUIREMENTS.txt - The required python packages with versions. A reduced library might work, though has not been tested.
- \_\_init\_\_.py - Needed for correct importing structures within python. Does not need to be interacted with, just leave it in the directory.

### Running simulation
- Set your simulation input parameters in **prms.py** and save.
- To run your code call 
```
python easi.py
```
In your terminal
- If you're running the code in an IDE instead of a terminal, the script might result in an import error. If this is the case, use the native terminal or Anaconda Prompt on your machine.
- The script will print into your terminal once it is done. Depending on the choice of parameters you can expect the simulation to run in a couple of minutes for \~10000 Monte Carlo particles. If the simulation takes longer, the choice of parameters is affecting the time the Monte Carlo particles take to escape the simulation box, and thus is not inherently a problem.

### Plotting results
- Once the simulation has finished, you can start plotting. **If you prefer to use your own iPython notebook setup, ensure that your iPython notebook also uses python version 3.11.4 by using the same virtual environment you created for the project!**
- Start your plotting notebook by typing 
```
jupyter-notebook plot_results.ipynb
```
- Within the first cells of the notebook, set the run number for which you want figures to be created (the run number is announced in the terminal when the simulation starts, you can also simply browse the directory names within the newly created folder "results" withing your simulation directory.)
- You can also set titles and legends for the figures, and plot multiple runs at the same time.
- Cells can be run by pressing Shift + Enter, or by using the "Run" menu at the top of the page. 

