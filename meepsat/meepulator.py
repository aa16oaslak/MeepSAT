import sys
import os
import site
from pathlib import Path

# First check for MEEPSAT_PATH environment variable (highest priority)
meepsat_env_path = os.environ.get('MEEPSAT_PATH')
if meepsat_env_path and os.path.exists(os.path.join(meepsat_env_path, 'meepsat')):
    print(f"Using MEEPSAT from environment variable: {{meepsat_env_path}}")
    main_dir = meepsat_env_path
    meepsat_dir = os.path.join(main_dir, 'meepsat')
    sys.path.append(main_dir)
    sys.path.append(meepsat_dir)
    meepsat_found = True
else:
    # Method 1: Try to import directly if MEEPSAT is installed properly
    try:
        import meepsat
        print("MEEPSAT found in installed packages")
        meepsat_dir = os.path.dirname(meepsat.__file__)
        main_dir = os.path.dirname(meepsat_dir)
        sys.path.append(meepsat_dir)  # Add meepsat directory to path for submodule imports
        meepsat_found = True
    except ImportError:
        # Method 2: Try to find MEEPSAT in common locations
        possible_paths = [
            Path.cwd().parent,                     # Parent of current directory
            Path.cwd(),                           # Current directory
            Path(__file__).resolve().parent.parent,  # Parent of script directory
            Path.home() / "Phd_work/MEEPSAT_WFH",      # User's home directory + common path
            Path('/cfs/data/asab1238/MEEPSAT_WFH'),   # HPC specific path
            Path('/cfs/data/asab1238/MEEPSAT_WFH')    # Another HPC specific path
        ]
        
        meepsat_found = False
        for path in possible_paths:
            try:
                meepsat_dir = path / "meepsat"
                if meepsat_dir.exists():
                    main_dir = str(path)
                    meepsat_dir = str(meepsat_dir)
                    sys.path.append(main_dir)
                    sys.path.append(meepsat_dir)  # Critical: add meepsat directory for submodule imports
                    print(f"MEEPSAT found at: {{meepsat_dir}}")
                    meepsat_found = True
                    break
            except Exception as e:
                print(f"Error checking path {{path}}: {{e}}")
        
        if not meepsat_found:
            print("WARNING: MEEPSAT directory not found automatically.")
            print("Please set the MEEPSAT_PATH environment variable before running.")
            print("For example: export MEEPSAT_PATH=/path/to/MEEPSAT")
            main_dir = "."
            meepsat_dir = "./meepsat"
            sys.path.append(main_dir)
            sys.path.append(meepsat_dir)

import meep as mp
import numpy as np
import pandas as pd
import h5py
import meepsat.extra_functions as ef

class MEEPulator:
    """
    Data Manipulator class for post-processing of Meep simulations.

    Note: extract_dat_for_mp_at_every() can be used in mp.at_every() 
    method in Meep simulations to extract the data at each timestep.
    """    
    def __init__(self,
                 sim,
                 mpsat_sim,
                 required_data: list = None,
                 filename='data.h5',
                 format='h5'):
        
        """
        Initialises the Analyse class

        Parameters
        ----------
        filename : str
            Name of the file to save the data.

        sim : meep.Simulation
            Meep simulation object.

        mpsat_sim : MEEPSAT simulation object

        required_data : list
            List of required data to be extracted. Default is None.
        

        format : str
            Format of the file to save the data. Default is 'h5'.
        
        """
        self.sim = sim
        self.mpsat_sim = mpsat_sim

        # available data keys for both primary and secondary data
        self.avail_primary_data = ['name', 'resolution', 'cell', 'gridsize']
        self.avail_sec_data = ['time_steps', 'Ex', 'Ey', 'Ez', 'Hx', 'Hy', 'Hz', 
                          'Ex2_lin', 'Ex2_dB', 'Ex2_total_lin', 'Ex2_total_dB',
                          'Ey2_lin', 'Ey2_dB', 'Ey2_total_lin', 'Ey2_total_dB',
                          'Ez2_lin', 'Ez2_dB', 'Ez2_total_lin', 'Ez2_total_dB']

        # Check if the user has provided the correct data keys
        if required_data is not None:
            for data in required_data:
                if data not in self.avail_sec_data:
                    raise ValueError(f'{data} is not a valid data key. Please use one of {self.avail_sec_data}.')
        else:
            # Finding the common elements between avail_sec_data and required_data
            required_data = ef.common_elements(avail_sec_data, required_data)
    
        self.required_data = required_data

        self.filename = filename
        self.format = format
        # Create a dictionary to store the data
        self.data = {}
        # Create a category dictionary to store the categories of the data
        self.category_dict = {}

    def available_data(self):
        """
        Prints the available data keys for both primary and secondary data.
        """
        print('Available primary data keys:', self.avail_primary_data)
        print('Available secondary data keys:', self.avail_sec_data)

    #^ Functions for Data Management through the Dictionary
    def create_key(self, key):
        """
        Creates a key in the dictionary.

        Parameters
        ----------
        key : str
            Key to be created.

        """
        # Check if the key already exists
        if key in self.data.keys():
            raise ValueError('Key already exists in the dictionary.')
        else:
            self.data[key] = None

    def append_data(self, key, value):
        """
        Appends the data to the dictionary.

        Parameters
        ----------
        key : str
            Key to store the data.

        value : any
            Value to be stored.

        """
        self.data[key] = value

    def append_data_to_list(self, key, value):
        """
        Appends the data to the list in the dictionary.

        Parameters
        ----------
        key : str
            Key to store the data.

        value : any
            Value to be stored.

        """
        # Check if the key exists
        if key not in self.data.keys():
            raise ValueError('Key does not exist in the data dictionary.')
        else:
            self.data[key].append(value)

    def key_category(self, dict_key, category):
        """
        Categorises the keys in the dictionary.

        Parameters
        ----------
        key : str
            Key to be categorised.

        category : str
            Category of the key.

        """
        # Check if the key exists
        if dict_key not in self.data.keys():
            raise ValueError('Key does not exist in the data dictionary.')
        else:
            self.category_dict[category] = dict_key

    #~################################################################################################

    #^ 1) FUNCTIONS FOR the PRIMARY DATA EXTRACTION
    # ! These functions will be executed at the beginning of the simulation
    def name(self,
                return_dat: bool = False):
        """
        Extracts the name of the Meep simulation object.
        """
        name = self.mpsat_sim.sim_name
        if return_dat == True:
            return name


    def current_time_step(self,
                          return_dat: bool = False):
        """
        Extracts the current time step from the Meep simulation object.

        Returns
        -------
        time : float
            Time step of the simulation.

        """
        time = self.sim.round_time()
        if return_dat == True:
            return time
    
    def current_frequency(self,
                          return_dat: bool = False):
        """
        Extracts the current frequency from the Meep simulation object.

        Returns
        -------
        frequency : float
            Frequency of the simulation.

        """
        frequency = self.mpsat_sim.frequency # self.sim.meep_freq()
        if return_dat == True:
            return frequency
    
    def resolution(self,
                   return_dat: bool = False):
        """
        Extracts the resolution from the Meep simulation object.

        Returns
        -------
        resolution : float
            Resolution of the simulation.

        """
        resolution = self.mpsat_sim.resolution
        if return_dat == True:
            return resolution
    
    def cell_size(self,
                  return_dat: bool = False):
        """
        Extracts the cell size from the Meep simulation object.

        Returns
        -------
        cell_size : meep.Vector3
            Cell size of the simulation.

        """
        cell_size = self.mpsat_sim.cell_size
        if return_dat == True:
            return cell_size
            
    def cell(self,
             return_dat: bool = False):
        """
        Extracts the cell from the Meep simulation object.

        Returns
        -------
        cell : meep.Cell
            Cell of the simulation.

        """
        cell = self.mpsat_sim.cell
        if return_dat == True:
            return cell
    
    def coordinates(self,
                    return_dat: bool = False):
        """
        Extracts the coordinates from the Meep simulation object.

        Returns
        -------
        coordinates : meep.Coordinates
            Coordinates of the simulation in the form of [x, y, z, w].

        """
        x, y, z, w = self.sim.get_array_metadata()
        coordinates = [x, y, z, w]
        if return_dat == True:
            return coordinates
    
    def gridsize(self,
                    return_dat: bool = False):
        
        """
        Extracts the gridsize (resolution*cell) from the Meep simulation object.
        """
        gridsize = self.cell_size(return_dat= True) * self.resolution(return_dat= True)
        if return_dat == True:
            return gridsize

    # ! LARGE TIME SERIES DATA EXTRACTION (FOR EACH TIMESTEPS) FUNCTIONS
    #* Timesteps
    def time_steps(self,
                    return_dat: bool = False):
        """
        Extracts the time step from the Meep simulation object.
        """
        time_step = self.current_time_step(return_dat= True)
        if return_dat == True:
            return time_step

    #* E-field data in complex form
    def Ex(self,
           return_dat: bool = False):
        """
        Extracts the Ex data from the Meep simulation object.

        Returns
        -------
        Ex_data : array
            Ex data of the simulation.
        """
        Ex_data = self.sim.get_array(center=mp.Vector3(), 
                                     size=self.cell(return_dat= True), 
                                     component=mp.Ex,
                                     cmplx=True)
        
        if return_dat == True:
            return Ex_data

    def Ey(self,
           return_dat: bool = False):
        """
        Extracts the Ey data from the Meep simulation object.

        Returns
        -------
        Ey_data : array
            Ey data of the simulation.
        """
        Ey_data = self.sim.get_array(center=mp.Vector3(), 
                                     size=self.cell(return_dat= True), 
                                     component=mp.Ey,
                                     cmplx=True)
        
        if return_dat == True:
            return Ey_data
    
    def Ez(self,
           return_dat: bool = False):
        """
        Extracts the Ez data from the Meep simulation object.

        Returns
        -------
        Ez_data : array
            Ez data of the simulation.
        """
        Ez_data = self.sim.get_array(center=mp.Vector3(), 
                                     size=self.cell(return_dat= True), 
                                     component=mp.Ez,
                                     cmplx=True)

        if return_dat == True:
            return Ez_data
        
    #* B-field data
    def Hx(self,
           return_dat: bool = False):
        """
        Extracts the Hx data from the Meep simulation object.

        Returns
        -------
        Hx_data : array
            Hx data of the simulation.
        """
        Hx_data = self.sim.get_array(center=mp.Vector3(), 
                                     size=self.cell(return_dat= True), 
                                     component=mp.Hx,
                                     cmplx=True)

        if return_dat == True:
            return Hx_data

    def Hy(self,
           return_dat: bool = False):
        """
        Extracts the Hy data from the Meep simulation object.

        Returns
        -------
        Hy_data : array
            Hy data of the simulation.
        """
        Hy_data = self.sim.get_array(center=mp.Vector3(), 
                                     size=self.cell(return_dat= True), 
                                     component=mp.Hy,
                                     cmplx=True)

        if return_dat == True:
            return Hy_data

    def Hz(self,
           return_dat: bool = False):
        """
        Extracts the Hz data from the Meep simulation object.

        Returns
        -------
        Hz_data : array
            Hz data of the simulation.
        """
        Hz_data = self.sim.get_array(center=mp.Vector3(), 
                                     size=self.cell(return_dat= True), 
                                     component=mp.Hz,
                                     cmplx=True)
        
        if return_dat == True:
            return Hz_data
        
    #* Power data
    #Ex
    def Ex2_lin(self,
            return_dat: bool = False):
        """
        Squares the Ex data from the Meep simulation object.

        Returns
        -------
        Ex2_data : array
            Squared Ex data of the simulation.
        """
        Ex = self.Ex(return_dat = True)
        Ex2_data = Ex**2

        if return_dat == True:
            return Ex2_data
        
    def Ex2_dB(self,
            return_dat: bool = False):
        """
        Power in dB of the Ex data from the Meep simulation object.

        Returns
        -------
        Ex2_data : array
            Power in dB of the Ex data of the simulation.
        """
        Ex = self.Ex(return_dat = True)
        Ex2_data_dB = 10*np.log10(Ex**2)

        if return_dat == True:
            return Ex2_data_dB
    
    def Ex2_total_lin(self,
            return_dat: bool = False):
        
        """
        Total integrated power in Ex at a particular timestep.

        Returns
        -------
        Ex2_total_lin : float
            Total integrated power in Ex at a particular timestep.
        """
        
        Ex2_data = self.Ex2_lin(return_dat = True)
        Ex2_total_lin = np.sum(np.abs(Ex2_data))

        if return_dat == True:
            return Ex2_total_lin
        
    def Ex2_total_dB(self,
            return_dat: bool = False):
        
        """
        Total integrated power in Ex at a particular timestep in dB.

        Returns
        -------
        Ex2_total_dB : float
            Total integrated power in Ex at a particular timestep in dB.
        """
        
        Ex2_total_lin = self.Ex2_total_lin(return_dat = True)
        Ex2_total_dB = 10*np.log10(Ex2_total_lin)

        if return_dat == True:
            return Ex2_total_dB
        
    #Ey
    def Ey2_lin(self,
            return_dat: bool = False):
        """
        Squares the Ey data from the Meep simulation object.

        Returns
        -------
        Ey2_data : array
            Squared Ey data of the simulation.
        """
        Ey = self.Ey(return_dat = True)
        Ey2_data = Ey**2

        if return_dat == True:
            return Ey2_data
        
    def Ey2_dB(self,
            return_dat: bool = False):
        """
        Power in dB of the Ey data from the Meep simulation object.

        Returns
        -------
        Ey2_data : array
            Power in dB of the Ey data of the simulation.
        """
        Ey = self.Ey(return_dat = True)
        Ey2_data_dB = 10*np.log10(Ey**2)

        if return_dat == True:
            return Ey2_data_dB
    
    def Ey2_total_lin(self,
            return_dat: bool = False):
        """
        Total integrated power in Ey at a particular timestep.

        Returns
        -------
        Ey2_total_lin : float
            Total integrated power in Ey at a particular timestep.
        """
        Ey2_data = self.Ey2_lin(return_dat = True)
        Ey2_total_lin = np.sum(np.abs(Ey2_data))

        if return_dat == True:
            return Ey2_total_lin
        
    def Ey2_total_dB(self,
            return_dat: bool = False):
        """
        Total integrated power in Ey at a particular timestep in dB.

        Returns
        -------
        Ey2_total_dB : float
            Total integrated power in Ey at a particular timestep in dB.
        """
        Ey2_total_lin = self.Ey2_total_lin(return_dat = True)
        Ey2_total_dB = 10*np.log10(Ey2_total_lin)

        if return_dat == True:
            return Ey2_total_dB
        
    #Ez
    def Ez2_lin(self,
            return_dat: bool = False):
        """
        Squares the Ez data from the Meep simulation object.

        Returns
        -------
        Ez2_data : array
            Squared Ez data of the simulation.
        """
        Ez = self.Ez(return_dat = True)
        Ez2_data = Ez**2

        if return_dat == True:
            return Ez2_data
        
    def Ez2_dB(self,
            return_dat: bool = False):
        """
        Power in dB of the Ez data from the Meep simulation object.

        Returns
        -------
        Ez2_data : array
            Power in dB of the Ez data of the simulation.
        """
        Ez = self.Ez(return_dat = True)
        Ez2_data_dB = 10*np.log10(Ez**2)

        if return_dat == True:
            return Ez2_data_dB
    
    def Ez2_total_lin(self,
            return_dat: bool = False):
        """
        Total integrated power in Ez at a particular timestep.

        Returns
        -------
        Ez2_total_lin : float
            Total integrated power in Ez at a particular timestep.
        """
        Ez2_data = self.Ez2_lin(return_dat = True)
        Ez2_total_lin = np.sum(np.abs(Ez2_data))

        if return_dat == True:
            return Ez2_total_lin
        
    def Ez2_total_dB(self,
            return_dat: bool = False):
        """
        Total integrated power in Ez at a particular timestep in dB.

        Returns
        -------
        Ez2_total_dB : float
            Total integrated power in Ez at a particular timestep in dB.
        """
        Ez2_total_lin = self.Ez2_total_lin(return_dat = True)
        Ez2_total_dB = 10*np.log10(Ez2_total_lin)

        if return_dat == True:
            return Ez2_total_dB

    #~################################################################################################
    def init_primary_data_keys(self):
        """
        Creates the key of the above functions and appends the data to the dictionary in the beginning of the simulation.
        """
        for prim_data in self.avail_primary_data:
            print(f"Creating key for: {prim_data}")
            self.create_key(prim_data)
            self.append_data(prim_data, [])
            print('Key created for:', prim_data)

        print('Primary data initialised into the dictionary successfully!')

    def init_secondary_data_keys(self):
        """
        Creates the key of the above functions and appends the data to the dictionary in the beginning of the simulation.
        """
        for sec_data in self.required_data:
            print(f"Creating key for: {sec_data}")
            self.create_key(sec_data)
            self.append_data(sec_data, [])
            print('Key created for:', sec_data)

        print('Secondary data initialised into the dictionary successfully!')

    def store_primary_data(self):
        """
        Creates the key of the above functions and appends the data to the dictionary in the beginning of the simulation.
        """
        print("Starting to store primary data...")
        for prim_data in self.avail_primary_data:
            self.append_data_to_list(key=prim_data,
                                     value=getattr(self, prim_data)(return_dat=True))
            print(f"Data extracted for: {prim_data}")

            """print(f"Creating key for: {prim_data}")
            self.create_key(prim_data)
            print(f"Appending data for: {prim_data}")
            data = getattr(self, prim_data)(return_dat=True)
            print(f"Data for {prim_data}: {data}")
            self.append_data(prim_data, data)
            print(f"Key created for: {prim_data}")"""

        print('Primary data loaded into the dictionary successfully!')

    #^ MAIN FUNCTION THAT WILL BE USED inside sim.run(mp.at_every())
    """def store_secondary_data_mp_at_every(self):
        
        Uses all the required functions to calculate the required data at every timestep.

        Returns
        -------
        A callable function to be used in mp.at_every.
        
        for sec_data in self.required_data:
                self.append_data_to_list(key=sec_data, 
                                        value=getattr(self, sec_data)(return_dat=True))
        print('Data extracted successfully at timestep:', self.current_time_step())"""

    #~################################################################################################
    #^ FUNCTIONS FOR THE FINAL DATA MANAGEMENT PROCESS
    def get_raw_data(self):
        """
        Returns the raw data dictionary.

        Returns
        -------
        data : dict
            Raw data dictionary.

        """
        return self.data

    def save_data(self):
        """
        Saves the data in the specified format in a specific directory.

        Saving format:
        \sim_name
        \sim_parameter 1
        \sim_parameter 2
        ... (and so on)

        # Secondary data (timestep dependent)
        \efield_cmplx_arr
            array for timestep 1
            array for timestep 2
            ... (and so on)

        \power_arr
            array for timestep 1
            array for timestep 2
            ... (and so on)
        
        \total_power_val
            value for timestep 1
            value for timestep 2
            ... (and so on)


        # Similarly, for other data.
        """
        if self.format == 'h5':
            # Saving the Dictonary in the HDF5 format in the specified directory
            with h5py.File(self.filename, 'w') as f:
                # Save the main data
                for key, value in self.data.items():
                    f.create_dataset(key, data=value)
                    print('Dataset created for:', key)

            print('Data saved successfully in the HDF5 format with the filename:', self.filename)
            

        """sim_name = self.filename.split('.')[0]
        sim_total_timesteps = self.sim.meep_timesteps()

        # Check the format
        if self.format == 'h5':
            # Save the data in the HDF5 format
            with h5py.File(self.filename, 'w') as f:
                # Save the main data
                for key, value in self.data.items():
                    f.create_dataset(key, data=value)

        #^ Add other formats below #elif self.format == 'csv':
        else:
            raise ValueError('Invalid format. Please use h5.')"""
