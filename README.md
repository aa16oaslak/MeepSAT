# MEEPSAT: MEEP Simulation and Analysis for Telescopes

CMB telescopes observe the sky in a wavelength of roughly 10 to 1 mm, making diffraction from optical instrument a key factor to address. We introduce a publicly available FDTD Simulation package 
"MEEPSAT" (based on MEEP) intended to help the design and characterization of current and future generation telescopes, including Taurus and Simons Observatory. 
These FDTD simulation help to probe the potential systematic effects of the various instruments of the telescope. Current geometric optics (GRASP) and ray-tracing (ZEMAX) softwares 
cannot introduce features such as deformations, using varied shaped instruments in the optical system (for e.g., different types of Absorbers). With the help of MEEPSAT, we intend to shed light on 
optical systematics and detector loading questions. 

MEEPSAT (MEEP Simulation and Analysis for Telescopes) is a Python-based framework 
for simulating and analyzing optical systems using the MEEP FDTD (Finite-Difference Time-Domain) library. This repository contains various modules for initializing, running, and analyzing 2D simulations, 
as well as visualizing the results.

## Getting Started

### Prerequisites

- Python 3.8 or higher
- MEEP
- NumPy
- Matplotlib
- h5py

### 🛠️ Installation

Due to the complex dependencies of the MEEP FDTD software, it is highly recommended to install this package using the **Conda** package manager. More details are mentioned here in the official documentation of MEEP: https://meep.readthedocs.io/en/latest/Installation/

1.  **Create a new Conda environment and Install MEEP using Conda:**
    
    **pymeep:**
    ```bash
    conda create -n meep -c conda-forge pymeep
    ```

    **If you want to install parallel version of pymeep (recommended)**
    ```bash
    conda create -n parallel_meep -c conda-forge pymeep=*=mpi_mpich_*
    ```


2.  **Activate and check if the conda environment is installed properly or not:**
    
    **pymeep:**
    ```bash
    conda activate meep
    ```

    **Install parallel pymeep and other dependencies:**
    ```bash
    conda activate parallel_meep
    ```

    **Check whether everything is working or not:**
    ```bash
    python -c 'import meep'
    ```

3.  **Install this project from the repository:**
    
    **For public repository (once the project is public):**
    ```bash
    pip install git+https://github.com/aa16oaslak/MeepSAT.git
    ```
    
    **For private repository (requires authentication):**
    ```bash
    pip install git+https://'your_token'@github.com/aa16oaslak/MEEPSAT_WFH.git
    ```
    Replace ```'your_token' ``` with your PAT token generated in your github account
    
    **Alternative for private repository using SSH:**
    ```bash
    pip install git+ssh://git@github.com/aa16oaslak/MEEPSAT_WFH.git
    ```
    
    **For development installation (editable and recommended for collaborators to use this):**
    ```bash
    git clone https://github.com/aa16oaslak/MeepSAT.git
    cd MEEPSAT_WFH
    pip install -e .
    ```

4. **Check if everything is working with MEEPSAT import or not**
    ```bash
    python -c 'import meep as mp; import meepsat as mpsat; print("Yayy!")'
    ```

5. **For updating things to the latest version of MEEPSAT, just use git pull**
    ```bash
    cd /path/to/my-private-meepsat-repo-head-directory
    git pull
    ```

- For Additional Checks, you can use the following piece of code to see if everything is working correctly within MEEPSAT:

    ```bash
    conda activate your_meep_conda_environment
    python
    ```
    Within python environment, copy paste the following lines of code
    ```python
    import meepsat.analyse
    import meepsat.simulation_2D
    import meepsat.plot

    # Check what's available in each module
    print("Functions in meepsat.analyse:")
    print([name for name in dir(meepsat.analyse) if not name.startswith('_')])

    print("\nFunctions in meepsat.simulation_2D:")
    print([name for name in dir(meepsat.simulation_2D) if not name.startswith('_')])

    print("\nFunctions in meepsat.plot:")
    print([name for name in dir(meepsat.plot) if not name.startswith('_')])

    ```

    If you are getting an output with all the functions in the various modules of MEEPSAT, then you are all set to do some FDTD sims!!

### Repository Struture
    ```
    MEEPSAT/
    ├── README.md             # overview of the project
    ├── pyproject.toml        # For initial Installation through pip 
    ├── data/                 # data files used in the project
    │   ├── README.md         # describes data from different softwares
    │   └── sub-directory/    # may contain subdirectories
    ├── processed_data/       # important files/data resulted from the analysis of the sims
    ├── manuscript/           # manuscript of MEEPSAT in latex doc
    ├── results/              # results of the analysis (data, tables, figures)
    ├── info/                 # contains all code related information in the project
    │   ├── LICENSE           # license for your code
    │   ├── requirements.txt  # software requirements and dependencies
    │   └── ...
    └── meepsat
        ├── simulation_2D.py     # Contains all the base level codes for MEEPSAT
        └── ...
    --------------------------THIS WON'T BE AVAILABLE IN THE PUBLISHED VERSION OF THE CODE--------------------------
    └── Tutorials
        ├── sub-directory/     # Contains all the Tutorials/examples used till now for the development of the code
        └── ...                                        
    --------------------------THIS WON'T BE AVAILABLE IN THE PUBLISHED VERSION OF THE CODE--------------------------
    └── examples
        ├── sub-directory/     # Contains different example projects
                ├── sim_files/
                ├── output_files/
        ├── jupyter_notebooks     # Contains all the base level codes for MEEPSAT
            ├── TBD
                    ├── TBD
                    ├── TBD
    └── doc/                  # documentation for your project
        ├── index.rst
        └── ...
    ```