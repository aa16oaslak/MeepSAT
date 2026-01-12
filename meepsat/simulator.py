"""
This module contains all the necessary functions for initialising, running and anlysing the simulation
"""

import os
import warnings
from typing import Callable
#import meep_testings as mp
import meep as mp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib import rc

import meepsat.meep_geometry as comp # Importing the components made using the MEEP functions
import meepsat.permittivity_components as comp_eps # Importing the components made using the epsilon functions
# import meep_visualization_meepsat_ver as mpsat_plt # Importing the plotting functions
import meepsat.helpers as exf # Importing the extra functions

def plot_and_save_epsilon(simulation, savepath, filename_prefix, epsilon_data_name, 
                          size_x, size_y, vmin=0.5, vmax=3, cmap='viridis', 
                          figsize=(8, 4), dpi=300, return_epsilon=False):
    """
    Plot and save the epsilon (permittivity) map from a MEEP simulation.
    
    Parameters:
    -----------
    simulation : mp.Simulation
        The MEEP simulation object
    savepath : str
        Directory path where files will be saved
    filename_prefix : str
        Prefix for the output filenames (e.g., "geometry_plot")
    epsilon_data_name : str
        Name for the epsilon dataset in the HDF5 file
    size_x : float
        Size of simulation cell in x direction (mm)
    size_y : float
        Size of simulation cell in y direction (mm)
    vmin : float, optional
        Minimum value for colormap scale (default: 0.5)
    vmax : float, optional
        Maximum value for colormap scale (default: 3)
    cmap : str, optional
        Matplotlib colormap name (default: 'viridis')
    figsize : tuple, optional
        Figure size (width, height) in inches (default: (8, 4))
    dpi : int, optional
        Resolution for saved figure (default: 300)
    
    Returns:
    --------
    epsilon : np.ndarray
        The extracted epsilon array
    """
    # Run simulation briefly to get epsilon
    simulation.run(until=0)
    epsilon = simulation.get_epsilon()
    
    # Plot the epsilon map geometry
    plt.figure(figsize=figsize)
    plt.imshow(epsilon.T, interpolation='spline36', cmap=cmap, origin='lower', 
               extent=[-size_x/2, size_x/2, -size_y/2, size_y/2],
               vmin=vmin, vmax=vmax)
    plt.colorbar(label='Permittivity (ε)')
    plt.xlabel('X (mm)')
    plt.ylabel('Y (mm)')
    plt.title('Epsilon Map')
    plt.savefig(os.path.join(savepath, f"{filename_prefix}.png"), dpi=dpi, bbox_inches='tight')
    plt.close()
    
    # Save the epsilon map to an HDF5 file
    h5_filename = os.path.join(savepath, f"{filename_prefix}.h5")
    with h5py.File(h5_filename, "w") as h5file:
        h5file.create_dataset(epsilon_data_name, data=epsilon)
    
    print(f"Epsilon plot saved to: {os.path.join(savepath, filename_prefix)}.png")
    print(f"Epsilon data saved to: {h5_filename}")
    
    if return_epsilon:
        return epsilon


def check_resolution_and_pml(data,
                             mpsat_sim,
                             meep_sim = None,
                             smallest_freq: float = None,
                             highest_n: float = None):
    """
    Function to check if the resolution and pml are sufficient for the simulation

    Arguments
    ---------
    data : dict
        Dictionary containing the simulation parameters (extracted from the json file)
    meep_sim: meep Simulation object
        MEEP simulation object
    highest_freq : float
        Highest frequency of the source in meep units
    highest_n : float
        Highest refractive index of the materials in the simulation box

    Returns
    -------
    data : dict
        Updated dictionary containing the simulation parameters (extracted from the json file)
    """
    if meep_sim is None and highest_n is None:
        raise ValueError("Either the MEEP simulation object or the highest refractive index must be provided to check the resolution and PML thickness.")


    #! 1ST step: Extract the highest refractive index from the epsilon data
    if highest_n is None:
        #! 0TH step: Run the simulation to extract the epsilon data
        print("No highest refractive index provided. Extracting the highest refractive index from the epsilon data...")
        print("Running a quick simulation to extract the epsilon data for checking the resolution and PML thickness...")
        meep_sim.run(until=0)
        print("Simulation run complete!")
        print("Extracting the epsilon data...")
        epsilon_map = meep_sim.get_epsilon() 
        highest_n = np.sqrt(np.max(epsilon_map))
        print("Highest refractive index in the simulation box: ", highest_n)
    else:
        print("Highest refractive index provided: ", highest_n)

    #! Checking the PML thickness and factor
    given_pml_thickness = data['boundary_layers']['boundary']['size']*data['boundary_layers']['boundary']['factor_dpml']
    
    # Print statements for old PML and resolution values
    print("Given PML thickness: ", given_pml_thickness)
    print("Given resolution: ", data["simulation"]['primary_params']['resolution'])

    #! Checking the source frequency and assuming it to be the largest frequency present in the system
    if "sources" in data:
        print("Assuming the wavelength of the Source to be the largest wavelength present in the system and doing a sanity check on the PML thickness.")
        for source in data["sources"]:
            wavelength_meep_source = 1 / data["sources"][source]["frequecy"]  # Wavelength in MEEP unit
            min_pml_thickness = 0.5*wavelength_meep_source # Source: https://meep-hr.readthedocs.io/en/latest/FAQ/#checking-convergence
            if given_pml_thickness < min_pml_thickness:
                print(f"PML thickness {given_pml_thickness} is less than the minimum required {min_pml_thickness}. Setting PML thickness to {min_pml_thickness}.")
                data['boundary_layers']['boundary']['size'] = min_pml_thickness

    if smallest_freq is not None:
        print("Smallest frequency provided: ", smallest_freq)
        wavelength_meep_largest = 1 / smallest_freq  # Wavelength in MEEP unit
        # It should at least 1/2 wavelength_meep_ of the source
        min_pml_thickness = 0.5*wavelength_meep_largest # Source: https://meep-hr.readthedocs.io/en/latest/FAQ/#checking-convergence
        if given_pml_thickness < min_pml_thickness:
            print(f"PML thickness {given_pml_thickness} is less than the minimum required {min_pml_thickness}. Setting PML thickness to {min_pml_thickness}.")
            data['boundary_layers']['boundary']['size'] = min_pml_thickness

    if highest_n is not None:
        print("Highest refractive index provided: ", highest_n)
        if "sources" in data:
            for source in data["sources"]:
                wavelength_meep_ = 1 / data["sources"][source]["frequecy"]  # Wavelength in MEEP unit
                wavelength_meep_inside_medium = wavelength_meep_ / highest_n  # Wavelength in MEEP unit inside the medium
                min_pml_thickness = 0.5*wavelength_meep_inside_medium # Source: https://meep-hr.readthedocs.io/en/latest/FAQ/#checking-convergence
                if given_pml_thickness < min_pml_thickness:
                    print(f"PML thickness {given_pml_thickness} is less than the minimum required {min_pml_thickness}. Setting PML thickness to {min_pml_thickness}.")
                    data['boundary_layers']['boundary']['size'] = min_pml_thickness

    

    #! Check if the resolution criteria is met or not for the highest refractive index
    if highest_n is not None:
        print("Highest refractive index provided: ", highest_n)
        wavelength_meep_ = 1 / data["sources"]['source1']["frequecy"]  # Wavelength in MEEP unit
        wavelength_meep_inside_medium = wavelength_meep_ / highest_n  # Wavelength in MEEP unit inside the medium
        freq_inside_medium = 1 / wavelength_meep_inside_medium  # Frequency inside the medium
        if data["simulation"]['primary_params']['resolution'] / freq_inside_medium < 8:
            print("Resolution criteria doesn't meet the criteria for the smallest frequency. Increasing the resolution to meet the criteria.")
            print("Wavelength Meep: ", wavelength_meep_)
            print("Wavelength Meep inside medium: ", wavelength_meep_inside_medium)
            data["simulation"]['primary_params']['resolution'] = int(freq_inside_medium * 8)

    #! Checking if the resolution criteria is met or not for the source's frequency
    # resolution/frequency ratio should be at least 10   
    if "sources" in data:
        print("Assuming the wavelength of the Source to be the largest wavelength present in the system and doing a sanity check on the PML thickness.")
        for source in data["sources"]:
            if data["simulation"]['primary_params']['resolution'] / data["sources"][source]["frequecy"] < 10:
                print("Resolution criteria doesn't meet the criteria for the provided source frequency. Increasing the resolution to meet the criteria.")
                data["simulation"]['primary_params']['resolution'] = int(data["sources"][source]["frequecy"] * 8)

    #! Checking if the resolution criteria is met or not for the smallest frequency
    if smallest_freq is not None:
        print("Smallest frequency provided: ", smallest_freq)
        if data["simulation"]['primary_params']['resolution'] / smallest_freq < 10:
            print("Resolution criteria doesn't meet the criteria for the smallest frequency. Increasing the resolution to meet the criteria.")
            data["simulation"]['primary_params']['resolution'] = int(smallest_freq * 8)

    #! Now assuming the smallest frequency present in the system is the ARC thickness (lambda/4 assumption)
    # check if data["lenses"] exists in data
    if "lenses" in data:
        #! Important: THE BELOW TWO LIST WILL CONTAIN THE VALUES FOR ALL THE LENGTH SCALES AND REFRACTIVE INDICES PRESEENT IN THE DIFFERENT LENSES 
        arc_length_arr = []; 
        arc_n_arr = []
        for lens in data["lenses"]:
            # Checking for #! Single ARC parameters
            if "AR_left" in data["lenses"][lens] or "AR_right" in data["lenses"][lens]:
                arc_length_arr.append(data["lenses"][lens]["AR_left"])
                arc_length_arr.append(data["lenses"][lens]["AR_right"])
                arc_n_arr.append(data["lenses"][lens]["AR_material"])

            # Checking for #! Multi-layer ARC parameters
            if "AR_left_layers" in data["lenses"][lens]:
                arc_length_arr.extend(data["lenses"][lens]["AR_left_layers"])
                arc_n_arr.extend(data["lenses"][lens]["AR_left_materials"])
            if "AR_right_layers" in data["lenses"][lens]:
                arc_length_arr.extend(data["lenses"][lens]["AR_right_layers"])
                arc_n_arr.extend(data["lenses"][lens]["AR_right_materials"])


            # Checking for #! Stepped pyramid ARC parameters
            if "ARC_type" in data["lenses"][lens]:
                if data["lenses"][lens]["ARC_type"] == "stepped_pyramid":
                    # Since pitch is a single float value and kerf, width are lists
                    arc_length_arr.append(data["lenses"][lens]["step_ARC_pitch"])  # Remove the brackets
                    arc_length_arr.extend(data["lenses"][lens]["step_ARC_kerf"])
                    arc_length_arr.extend(data["lenses"][lens]["step_ARC_depth"])
                    # Appending the materials
                    # Check if step_ARC_material_nref is a list or a single value
                    material_nref = data["lenses"][lens]["step_ARC_material_nref"]
                    if isinstance(material_nref, list):
                        arc_n_arr.extend(material_nref)
                    else:
                        # If it's a single float value, append it instead of extend
                        arc_n_arr.append(material_nref)

            # Checking for #! Delamination layer parameters
            if "delam_thick" in data["lenses"][lens]:
                if data["lenses"][lens]["delam_thick"] != 0:
                    arc_length_arr.append(data["lenses"][lens]["delam_thick"])
                    arc_length_arr.append(data["lenses"][lens]["delam_width"])

            # Checking for #! Surface error parameters
            #!! UPDATE THIS LATER WHEN YOU ARE USING SURFACE ERROR IN THE SIMULATION

        # ! Now checking the resolution criteria for all the arc_n_arr 
        for n in arc_n_arr:
            wavelength_meep_inside_medium = wavelength_meep_ / n  # Wavelength in MEEP unit inside the medium
            freq_inside_medium = 1 / wavelength_meep_inside_medium  # Frequency inside the medium
            if data["simulation"]['primary_params']['resolution'] / freq_inside_medium < 8:
                print("Resolution criteria doesn't meet the criteria for the ARC layers. Increasing the resolution to meet the criteria.")
                data["simulation"]['primary_params']['resolution'] = int(freq_inside_medium * 8)

        #! Checking the resolution criteria for all the arc_length_arr
        for wavelength_meep_arc in arc_length_arr:
            # Add validation to ensure wavelength_meep_arc is a scalar
            if isinstance(wavelength_meep_arc, (list, tuple)):
                # If it's accidentally a list, extract the first element or flatten
                if len(wavelength_meep_arc) > 0:
                    wavelength_meep_arc = wavelength_meep_arc[0]
                else:
                    continue
            
            if wavelength_meep_arc == 0:
                print(f"Warning: Found zero wavelength in arc_length_arr, skipping...")
                continue
                
            freq_arc = 1 / wavelength_meep_arc  # Frequency corresponding to the ARC layer thickness
            if data["simulation"]['primary_params']['resolution'] / freq_arc < 8:
                print("Resolution criteria doesn't meet the criteria for the ARC layers. Increasing the resolution to meet the criteria.")
                data["simulation"]['primary_params']['resolution'] = int(freq_arc * 8)



    mpsat_sim.resolution = data["simulation"]['primary_params']['resolution']
    
    print("All length scales of lenses in the simulation: ", arc_length_arr)
    print("All refractive indices of different components in the of lense in the simulation: ", arc_n_arr)
    print("Modified resolution: ", data["simulation"]['primary_params']['resolution'])
    print("Modified PML thickness: ", data['boundary_layers']['boundary']['size']*data['boundary_layers']['boundary']['factor_dpml'])

    return data, mpsat_sim

            

def convert_to_meep_units(self, value, unit_type, from_unit='um'):
    """
    Converts real-world units to MEEP simulation units.
    
    In MEEP, the simulation uses normalized units where c=1. This function helps
    convert from physical units to MEEP's normalized units based on a chosen
    length scale.
    
    Parameters:
    ----------
    value : float
        The numerical value to convert
    unit_type : str
        The type of unit being converted:
        - 'length': Length units (e.g., μm to MEEP units)
        - 'frequency': Frequency units (e.g., THz to MEEP units)
        - 'time': Time units (e.g., ps to MEEP units)
    from_unit : str, optional
        The physical unit to convert from. Default is 'um' (micrometers).
        Supported units:
        - Length: 'nm', 'um', 'mm', 'm'
        - Frequency: 'Hz', 'GHz', 'THz'
        - Time: 'fs', 'ps', 'ns', 's'
    
    Returns:
    -------
    float
        The converted value in MEEP units
    
    Examples:
    --------
    # Convert 1.55 μm wavelength to MEEP units
    wavelength_meep = convert_to_meep_units(1.55, 'length', 'um')
    
    # Convert 193 THz frequency to MEEP units
    freq_meep = convert_to_meep_units(193, 'frequency', 'THz')
    
    # Convert 100 fs time to MEEP units
    time_meep = convert_to_meep_units(100, 'time', 'fs')
    """
    # Speed of light in m/s
    c = 299792458

    # Define the base length unit (default is μm)
    length_scale = 1.0  # MEEP units

    # Length conversion factors to meters
    length_to_meters = {
        'nm': 1e-9,
        'um': 1e-6,
        'mm': 1e-3,
        'm': 1.0
    }
    
    # Frequency conversion factors to Hz
    freq_to_hz = {
        'Hz': 1.0,
        'GHz': 1e9,
        'THz': 1e12
    }
    
    # Time conversion factors to seconds
    time_to_seconds = {
        'fs': 1e-15,
        'ps': 1e-12,
        'ns': 1e-9,
        's': 1.0
    }
    
    if unit_type == 'length':
        if from_unit not in length_to_meters:
            raise ValueError(f"Unsupported length unit: {from_unit}. Use one of {list(length_to_meters.keys())}")
        # Convert length to MEEP units (normalized to length_scale)
        return value * length_to_meters[from_unit] / (length_to_meters['um'] * length_scale)
    
    elif unit_type == 'frequency':
        if from_unit not in freq_to_hz:
            raise ValueError(f"Unsupported frequency unit: {from_unit}. Use one of {list(freq_to_hz.keys())}")
        # Convert frequency to MEEP units
        frequency_hz = value * freq_to_hz[from_unit]
        wavelength_m = c / frequency_hz
        return length_to_meters['um'] * length_scale / wavelength_m
    
    elif unit_type == 'time':
        if from_unit not in time_to_seconds:
            raise ValueError(f"Unsupported time unit: {from_unit}. Use one of {list(time_to_seconds.keys())}")
        # Convert time to MEEP units
        time_s = value * time_to_seconds[from_unit]
        return time_s * c / (length_to_meters['um'] * length_scale)
    
    else:
        raise ValueError("Unit type must be 'length', 'frequency', or 'time'")

        




class sim_init():
    """
    For initialising the simulation parameters
    """

    def __init__(self,
                 sim_name: str = None,
                 cell_size: list = None,
                 smallest_freq: float = None,
                 smallest_wavelength: float = None,
                 resolution: float = None,
                 boundary_layer_type: str = None,
                 boundary_layer_size: float = None,
                 factor_dpml: float = None,
                 verbosity: int = 0):
        
        """ 
        Initialises the simulation parameters

        Arguments
        ---------
        sim_name : str
            Name of the simulation

        cell_size : list
            Size of the cell in the x, y and z directions
            For 2D simulations, the z-direction size is 0 and the cell size is [sx, sy, 0]
            For 3D simulations, the cell size is [sx, sy, sz]
            `Current supported options: Only 2D`

        freq : float
            Frequency of the source in meep units

        wavelength : float
            Wavelength of the source in meep units
            If the frequency is provided, the wavelength is calculated as 1/freq (c=1)

        resolution : float
            Resolution of the simulation: number of grid points per unit meep wavelength

        boundary_layer_type : str 
            Type of the boundary layer
            Three basic types of terminations are supported in Meep: 
            Bloch-periodic boundaries, metallic walls, and PML absorbing layers
            `Curent supported options: 'PML'

        boundary_layer_size : float
            Thickness of the boundary layer
            `Current supported options: Only PML (dpml)`

        factor_dpml : float
            Factor by which the boundary layer thickness is multiplied 
            `Current supported options available: Only for PML` 
        """        
        self.sim_name = sim_name
        #==================================
        if smallest_freq:
            self.freq = smallest_freq
            self.wavelength = 1/smallest_freq
        elif smallest_wavelength:
            self.wavelength = smallest_wavelength
            self.freq = 1/smallest_wavelength
        else:
            raise ValueError('Either frequency or wavelength should be provided')
        
        #==================================

        if resolution/self.freq < 8:
            raise ValueError('The resolution should be atleast 8 points per wavelength. The grid size should be small enough that it can accurately resolve the wavelength of the electromagnetic wave, but not too small to unnecessarily increase computational requirements.')
        else:
            self.resolution = resolution

        #==================================
        if boundary_layer_type == 'PML':
            self.boundary_layer_type = boundary_layer_type
            self.dpml = boundary_layer_size

            if factor_dpml:
                self.factor_dpml = factor_dpml
            else:
                warnings.warn('No factor provided for the PML boundary layer thickness. Assuming the default factor to be 2')
                self.factor_dpml = 2

            self.cell_size = [cell_size[0] + self.factor_dpml*self.dpml, cell_size[1] + self.factor_dpml*self.dpml, 0]
            self.cell = mp.Vector3(self.cell_size[0], self.cell_size[1], self.cell_size[2])
        #==================================
        ### Here we can add other boundary layer types in the future versions
        # elif boundary_layer_type == 'metallic':
        #================================== 
        else:
            raise ValueError('Only PML boundary layer is supported in the current MEEPSAT version')
        
        # ! REST IS IMP: BUT THESE TWO ARE VERY IMPORTANT PARAMETERS FOR THE SIMULATION
        self.meep_geometry = [] # List to store the optical components made using the MEEP functions
        self.eps_geometry = [] # List to store the optical components made using the epsilon functions


    def print_simulation_parameters(self):
        """
        Prints the simulation parameters including the simulation name, cell size, frequency, wavelength, resolution, 
        boundary layer type, boundary layer size, and the factor for PML boundary layer thickness.
        Parameters:
        None
        Returns:
        None
        """
        print(f"Simulation name: {self.sim_name}")
        print(f"Cell size: {self.cell_size}")
        print(f"Frequency: {self.freq}")
        print(f"Wavelength: {round(self.wavelength, 2)}")
        print(f"Resolution: {self.resolution}")
        print(f"Boundary layer type: {self.boundary_layer_type}")
        print(f"Boundary layer size: {self.dpml}")
        print(f"Factor for PML boundary layer thickness: {self.factor_dpml}")


    def list_components(self):
        """
        Prints the components of the MEEP and Epsilon geometries.
        This method prints the components of the `meep_geometry` and `eps_geometry`
        attributes of the class instance. It first prints the components of the 
        `meep_geometry` followed by the components of the `eps_geometry`.
        Output:
            Prints the components to the console.
        """
        
        print('---MEEP Function Components---')
        for component in self.meep_geometry:
            print(component)
        print('----------------\n')
        
        print('---Epsilon Function Components---')
        for component in self.eps_geometry:
            print(component)
        print('----------------')

    # ! ###########################################################################################################
    # ! ###########################################################################################################
    # ! FOR ADDING/CREATING THE OPTICAL OBJECTS/COMPONENTS USING THE DEFAULT MEEP OBJECTS
    
    def add_meep_geometry(self, object):
        """
        Adds the MEEP objects/components to the simulation

        Arguments
        ---------
        object : object
            Object created using the MEEP functions
            can be any of the following:
            list of MEEP objects, individual MEEP objects
        """
        self.meep_geometry.append(object)
        print("{} added to the list of components created using the MEEP functions!".format(object))

    # ! ###########################################################################################################
    # ! ###########################################################################################################
    # ! FOR ADDING/CREATING THE OPTICAL OBJECTS/COMPONENTS USING THE DEFAULT MEEP OBJECTS

    def add_eps_geometry(self,
                          component: object = None):
        """
        Adding the eps component to the component list

        Arguments
        ---------
        component : object
            Component created using the epsilon functions in the components_2D_eps.py file
        """
        self.eps_geometry.append(component)
        print("{} added to the list of components created using the epsilon functions!".format(component))


    # ! ###########################################################################################################
    # ! ###########################################################################################################
    # ! FOR PERFORMING THE VARIOUS FUNCTIONS OF THE SIMULATION


    def meep_sim_obj(self,
                    sources: list = None,
                    geometry: list = None,
                    epsilon_h5_file: str = None,
                    boundary_layers: list = None,
                    additional_kwargs: bool = False,
                    **kwargs: dict):
        """
        Creates the MEEP simulation object

        Arguments
        ---------
        geometry : list
            List containing the geometry of the simulation created using the MEEP functions/objects

        epsilon_func_obj : callable function for the permittivity map

        epsilon_h5_file : str
            Name of the h5 file containing the epsilon data

        additional_kwargs : bool
            If True, additional keyword arguments are provided

        kwargs : dict
            Additional keyword arguments for mp.Simulation() function as a dictionary
            The user need to carefully read the MEEP documentation for the additional arguments
            Remember, the default arguments used here are:
            cell_size, boundary_layers, 
        """
        if epsilon_h5_file:
            epsilon_func_obj =  epsilon_h5_file + '.h5:eps'
        #else:
        #    raise warnings('No epsilon file provided. Please provide the epsilon file for the simulation')

        print("Creating the MEEP simulation object with following parameters:\n")
        print("Cell size: ", self.cell_size, 
              "Boundary layer type: ", self.boundary_layer_type, 
              "Boundary layer size: ", self.dpml)
        
        self.sources = sources
        self.geometries = geometry
        self.boundary_layers = boundary_layers


        '''if additional_kwargs == True:
            filtered_mp_Simulation_kwargs = exf.filter_dict(kwargs, mp.Simulation)
            self.sim = mp.Simulation(cell_size = self.cell,
                                     boundary_layers = self.boundary_layer,
                                     geometry = geometry,
                                     epsilon_input_file = epsilon_func_obj,
                                     sources = self.source,
                                     resolution = self.resolution,
                                     **filtered_mp_Simulation_kwargs)
        else:
            self.sim = mp.Simulation(cell_size = self.cell,
                                     boundary_layers = self.boundary_layer,
                                     geometry = geometry,
                                     epsilon_input_file = epsilon_func_obj,
                                     sources = self.source,
                                     resolution = self.resolution)'''
            
        if additional_kwargs == True:
            filtered_mp_Simulation_kwargs = exf.filter_dict(kwargs, mp.Simulation)
            self.sim = mp.Simulation(cell_size = self.cell,
                                     sources = self.sources,
                                     geometry = self.geometries,
                                     epsilon_input_file = epsilon_func_obj,
                                     boundary_layers = self.boundary_layers,
                                     resolution = self.resolution,
                                     **filtered_mp_Simulation_kwargs)
        elif epsilon_h5_file:
            self.sim = mp.Simulation(cell_size = self.cell,
                                     sources = self.sources,
                                     geometry = self.geometries,
                                     epsilon_input_file = epsilon_func_obj,
                                     boundary_layers = self.boundary_layers,
                                     resolution = self.resolution)
            
        else:
            self.sim = mp.Simulation(cell_size = self.cell,
                                     sources = self.sources,
                                     geometry = self.geometries,
                                     boundary_layers = self.boundary_layers,
                                     resolution = self.resolution)

        print("MEEP simulation object created successfully! Here is the simulation object: ", self.sim)
        return self.sim


    # def run_simulation(self,
    #                    sim = None, 
    #                    mpsat_sim = None,
    #                    json_file_path: str = None,                       
    #                    runtime = None,
    #                    get_mp4: bool = False,
    #                    dpi: int = 150,
    #                    image_every: int = 5,
    #                    Nfps: int = 24,
    #                    savepath: str = None,
    #                    movie_name: str = 'movie',
    #                    requested_data: list = None,
    #                    save_data: bool = False,
    #                    verbose: bool = False,
    #                    kwargs: dict= None):

    # def run_simulation(self,
    #                 sim = None, 
    #                 mpsat_sim = None,
    #                 json_file_path: str = None,                       
    #                 runtime = None,
    #                 get_mp4: bool = False,
    #                 dpi: int = 150,
    #                 image_every: int = 5,
    #                 Nfps: int = 24,
    #                 savepath: str = None,
    #                 movie_name: str = 'movie',
    #                 requested_data: list = None,
    #                 monitor_list: list = None,  # Add this parameter
    #                 save_data: bool = False,
    #                 verbose: bool = False,
    #                 kwargs: dict= None):


    def run_refence_simulation(self,
                    sim = None, 
                    mpsat_sim = None,
                    json_file_path: str = None,                       
                    runtime = None,
                    get_mp4: bool = False,
                    dpi: int = 150,
                    image_every: int = 5,
                    flux_monitor_save_freq: int = 5,
                    Nfps: int = 24,
                    savepath: str = None,
                    movie_name: str = 'movie',
                    requested_data: list = None,
                    monitor_list: list = None, # Add this parameter
                    flux_monitor_list: list = None,
                    is_reference: bool = True, 
                    save_data: bool = False,
                    verbose: bool = False,
                    kwargs: dict= None):
        """
        Run the MEEP simulation with no geometries

        Arguments
        ---------
        sim : object
            MEEP simulation object
        """
        print("Running the MEEP reference simulation with runtime: ", runtime)
        self.runtime = runtime
        # Check if the simulation object is provided
        if sim is None:
            raise ValueError('No MEEP simulation object provided. Please provide the simulation object to run the simulation')
        

    def run_simulation(self,
                    sim = None, 
                    mpsat_sim = None,
                    json_file_path: str = None,                       
                    runtime = None,
                    get_mp4: bool = False,
                    dpi: int = 150,
                    image_every: int = 5,
                    flux_monitor_save_freq: int = 5,
                    Nfps: int = 24,
                    savepath: str = None,
                    movie_name: str = 'movie',
                    requested_data: list = None,
                    monitor_list: list = None,
                    flux_monitor_list: list = None,
                    is_reference: bool = False, 
                    save_data: bool = False,
                    verbose: bool = False,
                    kwargs: dict= None):
        
        # ~ !!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        print("Running the MEEP simulation with runtime: ", runtime)
        self.runtime = runtime
        # Check if the simulation object is provided
        if sim is None:
            raise ValueError('No MEEP simulation object provided. Please provide the simulation object to run the simulation')
        
        # Check if the user wants to animate the simulation
        if get_mp4 :
            f = plt.figure(dpi = dpi)
            
            #Animate object
            """
            field_func = lambda x: 20*np.log10(np.abs(x))
            
            def colorbar(ax):
                matplotlib.colorbar.ColorbarBase(ax=ax)
                return ax
            """

            animate = mp.Animate2D(self.sim,
                                   f = f,
                                   fields=mp.Ez,
                                   realtime=True)
            """realtime=True,
            field_parameters={'alpha':0.8,
                                'cmap':'RdBu',
                                'interpolation':'none'},
            boundary_parameters={'hatch':'o', 
                                'linewidth':1.5, 
                                'facecolor':'y', 
                                'edgecolor':'b', 
                                'alpha':0.3})"""
                                   #plot_modifiers = [colorbar])
            
            if kwargs:
                """filtered_mp_Simulation_run_kwargs = exf.filter_dict(kwargs, sim.run)
                self.sim.run(mp.at_every(image_every, calculate_energy_at_time),
                             mp.at_every(image_every, animate),
                             until = self.runtime, 
                             **filtered_mp_Simulation_run_kwargs)"""
            
            else:
                """
                self.sim.use_output_directory()
                self.sim.run(mp.at_beginning(mp.output_epsilon),
                             mp.at_every(image_every, animate),
                             mp.at_every(image_every, mp.output_png(compnt= mp.Ez, 
                                                                     options="-Zc dkbluered -C $EPS")),
                             mp.at_every(image_every, mp.output_png(compnt= mp.Hy,
                                                                    options="-Zc dkbluered -C $EPS")),  
                             mp.at_end(mp.output_efield_z),
                             until = self.runtime)
                """
                # Defining a null animation object            
                """self.sim.run(mp.at_every(image_every, animate),
                             mp.at_every(image_every, mp_sim_run.Ez),
                             mp.at_every(image_every, mp_sim_run.Hy),
                             mp.at_every(image_every, mp_sim_run.Ez2_dB),
                             until = self.runtime)"""
                
                # self.sim.run(mp.at_every(image_every, animate),
                #              mp.at_every(image_every, mp_sim_run.Ey2_dB),
                #              mp.at_end(mp.output_efield_y),
                #              until = self.runtime)
                #~ Working:
                """self.sim.use_output_directory(dname= savepath)
                self.sim.run(mp.at_every(image_every, animate),
                                mp.at_every(image_every, mp_sim_run.Ez2_dB),
                                mp.at_end(mp.output_efield_z),
                                until=self.runtime)"""
                import meepsat.stepfunctions as stepfunctions, json_to_script
                # if flux_monitor_list:
                #     print("Initialising the flux monitors in mp_sim_run.py:")
                #     print("====================================================================")
                #     print("mp_sim_run.set_flux_monitor_registry(flux_monitor_list, savepath, image_every)")
                #     mp_sim_run.set_flux_monitor_registry(flux_monitor_list, savepath, flux_monitor_save_freq)
                #     print("====================================================================")    

                # if flux_monitor_list:
                #         print("Initialising the flux monitors in mp_sim_run.py:")
                #         print("====================================================================")
                #         print(f"mp_sim_run.set_flux_monitor_registry(flux_monitor_list, savepath, {flux_monitor_save_freq}, is_reference={is_reference})")
                #         mp_sim_run.set_flux_monitor_registry(flux_monitor_list, savepath, flux_monitor_save_freq, is_reference=is_reference)
                #         print("==============================================================")
                #         print("Checking the status of Reference Simulation: ", is_reference)
                #         print("==============================================================")

                self.sim.use_output_directory(dname= savepath)
                stepfunctions.set_animation_params(anim_params= {'image_every': image_every, 
                                                              'Nfps': Nfps, 
                                                              'anim_file_name': savepath + movie_name}) 

                # At the start of run_simulation function
                print(f"Requested data configuration: {requested_data}")
                # Set the output directory
                # Check the data requested by the user, make the output an executable script
                # print("Simulation running with the followig MEEP script:")
                # print("====================================================================")
                # print(json_to_script.sims_data_requested(data_required= requested_data, 
                #                                          run_time= runtime))
                # print("====================================================================")
                # # Execute the script
                # exec(json_to_script.sims_data_requested(data_required= requested_data, 
                #                                          run_time= runtime)
                # print("Initialising the various Monitors in mp_sim_run.py:")
                # print("====================================================================")
                # print(json_to_script.init_monitors_in_mp_sim_run(monitor_list= monitor_list))
                # print("====================================================================")
                # exec(json_to_script.init_monitors_in_mp_sim_run(monitor_list= monitor_list))
                
                print("Initialising the various Monitors in mp_sim_run.py:")
                print("====================================================================")
                print("mp_sim_run.set_volume_monitor_registry(monitor_list)")
                stepfunctions.set_volume_monitor_registry(monitor_list, savepath, image_every)
                print("====================================================================")

                print("Simulation running with the following MEEP script:")
                print("====================================================================")
                print(json_to_script.sims_data_requested(data_required=requested_data, 
                                                        run_time=runtime,
                                                        monitor_list=monitor_list))

                print("====================================================================")

                # Execute the script
                exec(json_to_script.sims_data_requested(data_required=requested_data, 
                                                    run_time=runtime,
                                                    monitor_list=monitor_list))
            
            if savepath:
                animate.to_mp4(Nfps, savepath + movie_name + '.mp4')
            else:
                animate.to_mp4(Nfps, movie_name + '.mp4')

        else:
            if kwargs:
                filtered_mp_Simulation_run_kwargs = exf.filter_dict(kwargs, sim.run)
                
                sim.run(until = self.runtime, **filtered_mp_Simulation_run_kwargs)
            
            else:
                sim.run(until = self.runtime)

        print("MEEP simulation run successfully!")

    def close_simulation(self):
        """
        Close the MEEP simulation object
        """
        self.sim.reset_meep()
        print("MEEP simulation closed successfully!")

        
    def extract_data(self, 
                     data_type: str = None,
                     properties: dict = None):
        """
        Extract the data from the simulation

        Arguments
        ---------
        data_type : str 
            Type of data to be extracted
            `Current supported options: 'E-field', 'x,y,z`, 'eps_data'`
        """

        if data_type == 'E-field':
            if properties['direction'] == 'z':
                print("Extracting the E_{} data from the simulation!".format(properties['direction']))
                self.ez_data = self.sim.get_array(center=mp.Vector3(), size= self.cell, component=mp.Ez) 
                print("Extracted the E-field data!")
                return self.ez_data
            
            elif properties['direction'] == 'y':
                print("Extracting the E_{} data from the simulation!".format(properties['direction']))
                self.ey_data = self.sim.get_array(center=mp.Vector3(), size= self.cell, component=mp.Ey) 
                print("Extracted the E-field data!")
                return self.ey_data
            
            elif properties['direction'] == 'x':
                print("Extracting the E_{} data from the simulation!".format(properties['direction']))
                self.ex_data = self.sim.get_array(center=mp.Vector3(), size= self.cell, component=mp.Ex) 
                print("Extracted the E-field data!")
                return self.ex_data
            
            else:
                warnings.warn("The direction is not recognised. Assuming the default direction to be z")
                print("Extracting the E-field data from the simulation!")
                ez_data = self.sim.get_array(center=mp.Vector3(), size= self.cell, component=mp.Ez) 
                print("Extracted the E-field data!")
                return self.ez_data
            
        elif data_type == 'xyzw':
            self.x, self.y, self.z, self.w = self.sim.get_array_metadata() # x,y,z axis and the corresponding weights!!
            return self.x, self.y, self.z, self.w
        
        elif data_type == 'eps_data':
            self.eps_data = self.sim.get_array(center=mp.Vector3(), size= self.cell, component=mp.Dielectric)
            return self.eps_data
            
        else:
            raise ValueError('The data type is not recognised. Please provide a valid/existing data type mate!')
    

    # & ###########################################################################################################
    # & ###########################################################################################################
    # * FAR FIELD BEAM FUNCTIONS (TOOK FROM THE MEEPART REPOSITORY)
    
    

    
    
    # ^ ###########################################################################################################
    # ^ ###########################################################################################################
    # ^ PLOTTING FUNCTIONS
    
    def sim_master_plot(self,
                        e_field_dat: np.array = None,
                        field_comp: str = None,
                        amp_dB_ = False,
                        detector_pos_dat = None,
                        save: bool = False,
                        save_name: str = None,
                        sim_ax_args = [(4, 5), (1, 0), 2, 4],
                        x_above_sim_axes_args = [(4, 5), (0, 0), 1, 4],
                        y_right_sim_axes_args = [(4, 5), (1, 4), 2, 1],
                        fig_args = (18,12),
                        kwargs: dict = None):
        """
        Function to plot the master plot

        Arguments
        ---------
        e_field_dat : np.array
            E-field data to plot in the simulation box

        field_comp : str
            E-field component to plot in the simulation box

        amp_dB_ : bool
            If True, the field amplitudes are plotted in dB scale
            If False, the field amplitudes are plotted in linear scale

        detector_pos : float
            Position of the airy plot in the x-direction

        **kwargs : dict
            Additional keyword arguments for the MEEP plotting function plot2D()

        """
        x_pos = [k - self.cell_size[0]/2 for k in range(0, int(self.cell_size[0]), 50)]
        y_pos = [k - self.cell_size[1]/2 for k in range(0, int(self.cell_size[1]), 50)]

        if field_comp == 'Ez':
            comp= mp.Ez
        elif field_comp == 'Ey':
            comp= mp.Ey
        elif field_comp == 'Ex':
            comp= mp.Ex

        else:
            warnings.warn("The field component is not recognised. Assuming the default field component to be Ez")
            comp= mp.Ez

        # For additional keyword arguments, we need to filter the dictionary to check
        if kwargs:
            mpsat_plt.master_plot(field= comp,
                                x_pos= x_pos,
                                y_pos= y_pos,
                                sim= self.sim,
                                sx= self.cell_size[0],
                                sy= self.cell_size[1],
                                field_amplitudes= self.ez_data*self.ez_data,
                                x= self.x,
                                y = self.y,
                                dpml_params = [self.dpml, self.factor_dpml],
                                sim_ax_args = sim_ax_args,
                                x_above_sim_axes_args = x_above_sim_axes_args,
                                y_right_sim_axes_args = y_right_sim_axes_args,
                                fig_args = fig_args,
                                amp_dB = amp_dB_,
                                save= save,
                                save_name= save_name,
                                detector_pos_dat= detector_pos_dat,
                                plot2d_kwargs= kwargs)
            
        else:
            mpsat_plt.master_plot(field= comp,
                                x_pos= x_pos,
                                y_pos= y_pos,
                                sim= self.sim,
                                sx= self.cell_size[0],
                                sy= self.cell_size[1],
                                field_amplitudes= self.ez_data*self.ez_data,
                                x= self.x,
                                y = self.y,
                                dpml_params = [self.dpml, self.factor_dpml],
                                sim_ax_args = sim_ax_args,
                                x_above_sim_axes_args = x_above_sim_axes_args,
                                y_right_sim_axes_args = y_right_sim_axes_args,
                                fig_args = fig_args,
                                amp_dB = amp_dB_,
                                save= save,
                                detector_pos_dat= detector_pos_dat,
                                save_name= save_name)
        
        # ~ EDIT THE PLOT TO MOVE IT TO THE PLOTS.PY FILE SO THAT IT WILL BE
        # ~ EASIER TO debug and maintain the code
        #return


    # For Plotting the Amplitudes of the E-field in 3D
    def plot_fieldAmp(self,
                      save: bool = False,
                      save_name: str = None):
        """
        Plot the amplitude of the E-field in 3D

        Arguments
        ---------
        save : bool
            If True, the plot is saved
            If False, the plot is not saved

        save_name : str
            Name of the plot to be saved
        """
        mpsat_plt.plot_field_amplitudes(x = self.x, 
                                        y = self.y, 
                                        field_amplitudes = self.ez_data*self.ez_data)
        #return 


