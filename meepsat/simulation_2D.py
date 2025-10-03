"""
In case we want to use our own version of MEEP

import sys
import os

# Add the module's directory to the Python path
module_path = os.path.expanduser('~/Phd_work/MEEP/my_MEEP/MEEP_original_package')
if module_path not in sys.path:
    sys.path.append(module_path)
"""

"""
This module contains all the necessary functions for initialising, running and anlysing the simulation
"""

import warnings
from typing import Callable
#import meep_testings as mp
import meep as mp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib import rc

import meepsat.components_2D_meep as comp # Importing the components made using the MEEP functions
import meepsat.components_2D_eps as comp_eps # Importing the components made using the epsilon functions
# import meep_visualization_meepsat_ver as mpsat_plt # Importing the plotting functions
import meepsat.extra_functions as exf # Importing the extra functions

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
                import mp_sim_run, json_to_script
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
                mp_sim_run.set_animation_params(anim_params= {'image_every': image_every, 
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
                mp_sim_run.set_volume_monitor_registry(monitor_list, savepath, image_every)
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


