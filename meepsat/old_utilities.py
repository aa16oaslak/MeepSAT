
##~ THE FOLLOWING WAS TAKEN OUT FROM THE PREVIUS VERSION OF THE SIMULATION_2D.PY FILE TO REMOVE THE PROPERTIES DICT FROM THE FUNCTION ARGUMENTS
##~ IT IS NOW INCLUDED IN THE OLD_UTILITIES.PY FILE FOR FUTURE REFERENCE

def old_add_meep_component(self, 
                    name: str = None,
                    eps_component_list: list = None, 
                    properties: dict = None):
    
    """
    Creates the various MEEP objects/components in the simulation

    Arguments
    ---------
    name : str
        Name of the component
        `Current supported options: 'source', 'ImagePlane', 'ApertureStop', 'lens', 'detector' (case-sensitive)

    properties : dict
        dict containing the properties of the component
        {type, center, size, component etc}
        For e.g: 
        `{'source': continuous_planewaves, plane_wave etc (etc will be included in the future versions)`, 
        `'ImagePlane`: block, circle etc (etc will be included in the future versions)}`
        `Similary for other components, the properties will be defined in the future versions`

    """
    print("Adding the component: ", name)
    self.comp_name = name

    if self.comp_name == 'source':
        
        # Check if the properties are provided as a dict or not
        # The recommended way to do this is in the following way:
        # {'type': 'continuous_planewaves', 'center': (x,y,z), 'size': (sx,sy,sz), 'component': X, Y,Z}
        if properties:

            if properties['type'] == 'continuous_planewaves':
                if properties['prop_comp'] == 'Z':
                    self.prop_comp = mp.Ez
                elif properties['prop_comp'] == 'Y':
                    self.prop_comp = mp.Ey
                elif properties['prop_comp'] == 'X':
                    self.prop_comp = mp.Ex
                else:
                    warnings.warn('The component is not recognised. Assuming the default component to be Z')
                    self.prop_comp = mp.Ez
                
                source_obj = comp.source(name = 'continuous_planewaves',
                                        center = properties['center'],
                                        size = properties['size'],
                                        component = self.prop_comp,
                                        freq = self.freq)
                
                self.source = source_obj
                
            ### Add other source types here in the future versions
            # elif properties['type'] == 'gaussian_beam':

                
        else:
            warnings.warn("Assuming the default source to be a continuous plane wave source in 2D with centre (-sx/2-15, 0,0), size = (0, sy, 0) and propagating in the z-direction!")
            
            source_obj = comp.source(name ='continuous_planewaves',
                                            center = (-self.cell_size[0]/2-15, 0, 0),   
                                            size = (0, self.cell_size[1], 0),
                                            component = mp.Ez,
                                            freq = self.freq)
            
            self.source = source_obj

        print("Source object created successfully! Here is(are) the source object(s): ", self.source)
        return self.source
                
    elif self.comp_name == 'detector':

        if properties:
            if properties['type'] == 'meep_block':
                detector_obj = comp.ImagePlane_block_detectors(name = 'meep_block',
                                                                    diameter= properties['diameter'],
                                                                    pos_x= properties['pos_x'],
                                                                    thickness= properties['thickness'],
                                                                    n_refr= properties['n_refr'],
                                                                    conductivity= properties['conductivity'])
                
                # Return an instance of the detector object, so that user can use all the methods of the detector object available in components_2D_meep.py file
                self.detector = detector_obj
        else:
            warnings.warn("Assuming the default detector to be a meep block detector with diameter = sy, pos_x = sx-20, thickness = 2, n_refr = 1 and conductivity = 0.01")
            
            detector_obj = comp.ImagePlane_block_detectors(name = 'meep_block',
                                                                diameter= self.cell_size[1],
                                                                pos_x= self.cell_size[0]-20,
                                                                thickness= 2,
                                                                n_refr= 1.1,
                                                                conductivity= 0.01)
            self.detector = detector_obj

        print("Detector object created successfully! Here is the detector object: ", self.detector)
        return self.detector
            
            ### Add other detector types here in the future versions
            # elif properties['type'] == 'circle':  

    # Defining the various boundary layers here
    elif self.comp_name == 'boundary_layer':
        if properties:
            if properties['type'] == 'meep_pml':
                """
                NEED TO ADD THE FUNCTIONALITY FOR THE BOUNDARY LAYER
                """
                self.boundary_layer = [mp.PML(self.dpml)]
            ### Add other boundary layer types here in the future versions
        else:
            warnings.warn("Assuming the default boundary layer to be PML with thickness = 2")
            self.boundary_layer = [mp.PML(2)]

        print("Boundary layer object created successfully! Here is the boundary layer object: ", self.boundary_layer)
        return self.boundary_layer
    
    ### ^ ADD REST OF THE MEEP OPTICAL COMPONENTS HERE IN THE FUTURE VERSIONS
    
    else:
        raise ValueError('The component name is not recognised. Please provide a valid/existing component name mate!')

    """
    elif self.comp_name == 'convex_lens':
        
        if properties:
            if properties['type'] == 'lens1_epsilon_function':
                self.lens_obj = comp.convex_Lens_epsilon_fun(name = 'lens1_epsilon_function',
                                                            properties= properties)
                return self.lens_obj
            
            elif properties['type'] == 'lens2_epsilon_function':
                self.lens_obj = comp.convex_Lens_epsilon_fun(name = 'lens2_epsilon_function',
                                                            properties= properties)
                return self.lens_obj
            
            print("Lens object created successfully using the epsilon function! Here is the lens object: ", self.lens_obj)
            ### Add other lens types here in the future versions
    """

# ~ components_2D_meep.py/ Source class
class Source():
    """
    Class defining the various source possible in the real world for a telescope
    """

    def __init__(self,
                 type= None,
                 center = None,
                 size= None,
                 component = None,
                 freq = None,
                 wvl = None,
                 angle = 0,
                 extra_args = None):
        #**kwargs):
        
        """
        Defines the a MEEP source object

        
        Arguments
        ---------
        type : str
            Type of the source: continuopus_planewaves OR gaussian OR something else 
            (default : None)

        center : list
            Center of the source in the x, y and z directions (default : None)
            Format : [x, y, z]

        size : list
            Size of the source in the x, y and z directions (default : None)
            Format : [sx, sy, sz]

        component : mp.Ez, mp.Ex, mp.Ey, mp.Hx, mp.Hy, mp.Hz
            Propagating component of the source (default : None)

        freq : float
            Frequency of the source in MEEP units (default : None)

        wvl : float
            Wavelength of the source in MEEP units (default : None)

        angle : float
            rot_angle : float, optional
                    Angle by which the plane wave is rotated w.r.t vertical        

        **extra_args : dict
            Additional arguments for the various types of sources
            See the different source functions for more details
            For e.g., width, cutoff etc for gaussian beam modulated planewaves       

        **kwargs : dict
            Additional arguments for the meep.Source()
            https://meep.readthedocs.io/en/latest/Python_User_Interface/#source
            https://meep.readthedocs.io/en/latest/Python_User_Interface/#continuoussource
        """        
        #To store the list of all the sources
        self.source_aperture = []
        self.object_type = 'source'

        if type is None:
            warnings.warn("No name given to the source object: Taking the default source as continuous_planewaves")
            self.name = 'continuous_planewaves'
        else:
            self.name = type

        self.center = center
        self.size = size

        if component is None:
            warnings.warn("No component given to the source object: Taking the default source as Ez")
            self.component = mp.Ez
        elif component == 'Ez':
            self.component = mp.Ez
        elif component == 'Ex':
            self.component = mp.Ex
        elif component == 'Ey':
            self.component = mp.Ey
        elif component == 'Hx':
            self.component = mp.Hx
        elif component == 'Hy':
            self.component = mp.Hy
        elif component == 'Hz':
            self.component = mp.Hz
        
        if freq:
            self.freq = freq
            self.wvl = 1/freq
        elif wvl:
            self.wvl = wvl
            self.freq = 1/wvl

        # ~ Do check this extra_args parameter later on how we can improve it 
        # ~ OR cause less confusion
        self.extra_args = extra_args
        #self.additional_args = kwargs

        # Check if angle is in the additional arguments
        if 'angle' in self.extra_args:
            self.rot_angle = self.extra_args['angle']
            self.rot_angle *= np.pi/180
            print("Angle of the source:{self.rot_angle} rad = {angle} degrees")

    def amp_func(self, P):
            '''
            Returns amplitude of source with added phase to 
            emulate source rotation

            Arguments
            ---------
            P : mp.Vector3
                Meep position object at which the source is evaluated.

            Returns
            -------
            amp : complex
                Complex amplitude of source at P.
            '''

            k = mp.Vector3(2*np.pi*np.cos(self.rot_angle)/self.wvl,
                           2*np.pi*np.sin(self.rot_angle)/self.wvl,
                           0)
            return np.exp(1j* k.dot(P))
        
    def gaussian_beam_mod_planewaves(self):
        """
        Return gaussian beam modulating planewaves source
        """
        if self.additional_args:
            filtered_kwrg_for_Source = exf.filter_dict(self.additional_args, mp.Source)
            print("Additional arguments for the source: ", filtered_kwrg_for_Source)
            filtered_kwrg_for_GaussianSource = exf.filter_dict(self.additional_args, mp.GaussianSource)
            print("Additional arguments for the Gaussian source: ", filtered_kwrg_for_GaussianSource)
            # Create a Gaussian pulse modulating the plane wave source
            source = mp.Source(mp.ContinuousSource(frequency=self.freq),
                               center= mp.Vector3(self.center[0], 
                                                 self.center[1], 
                                                 self.center[2]),
                               size= mp.Vector3(self.size[0],
                                                    self.size[1],
                                                    self.size[2]),
                               component=self.component,
                               src=mp.SourceTime(mp.GaussianSource(frequency=self.freq, 
                                                                    width=self.width, 
                                                                    cutoff=self.cutoff),  # Gaussian pulse with specific width and cutoff
                                                 **filtered_kwrg_for_Source))
        
        else:
            source = mp.Source(mp.ContinuousSource(frequency=self.freq),
                            center=mp.Vector3(self.center[0], 
                                                self.center[1], 
                                                self.center[2]),
                            size= mp.Vector3(self.size[0],
                                                self.size[1],
                                                self.size[2]),
                            component=self.component,
                            src=mp.SourceTime(function=mp.GaussianSource(frequency=self.freq, 
                                                                        width=self.width, 
                                                                        cutoff=self.cutoff)))
        
        print("Gaussian beam modulated planewaves source created successfully!: {}".format(source))
            
        return source

    def assemble(self):
        """
        Return Gaussian beam source
        """
        if self.additional_args is not None:
            source_filtered_kwrg = exf.filter_dict(self.additional_args, mp.Source)
            print("Additional arguments for the source: ", source_filtered_kwrg)
            source_type_filtered_kwrg = exf.filter_dict(self.additional_args, mp.GaussianSource)
            print("Additional arguments for the source type: ", source_type_filtered_kwrg)

            source = mp.Source(mp.GaussianSource(frequency=self.freq, 
                                                 width=self.width, 
                                                 cutoff=self.cutoff,
                                                 **source_type_filtered_kwrg),
                               center= self.center,
                               size= self.size,
                               component=self.component,
                               amp_func=self.amp_func,
                               **source_filtered_kwrg)
            
        else:
            source = mp.Source(mp.GaussianSource(frequency=self.freq, 
                                                 width=self.width, 
                                                 cutoff=self.cutoff),
                               center= self.center,
                               size= self.size,
                               component=self.component,
                               amp_func=self.amp_func)
            
        return source

    def continuous_planewaves(self):
        """
        Return continuous planewaves source
        """
        if self.extra_args is not None:
            source_filtered_kwrg = exf.filter_dict(self.extra_args, mp.Source)
            print("Additional arguments for the source: ", source_filtered_kwrg)
            source_type_filtered_kwrg = exf.filter_dict(self.extra_args, mp.ContinuousSource)
            print("Additional arguments for the source type: ", source_type_filtered_kwrg)


            source = mp.Source(mp.ContinuousSource(frequency=self.freq, 
                                                   **source_type_filtered_kwrg),
                           center=mp.Vector3(self.center[0], 
                                             self.center[1], 
                                             self.center[2]),
                           size= mp.Vector3(self.size[0],
                                            self.size[1],
                                            self.size[2]),
                           component=self.component,
                           amp_func=self.amp_func,
                           **source_filtered_kwrg)
            
        else:
            source = mp.Source(mp.ContinuousSource(frequency=self.freq),
                            center=mp.Vector3(self.center[0], 
                                                self.center[1], 
                                                self.center[2]),
                            size= mp.Vector3(self.size[0],
                                                self.size[1],
                                                self.size[2]),
                            component=self.component,
                            amp_func=self.amp_func)
        return source


def epsilon_data(self,
                return_dat: bool = False):
    """
    Extracts the epsilon data (basically the various components)
    This will be stored at the beginning of the simulation.
    Returns
    -------
    epsilon_data : array
        Epsilon data of the simulation.
    """
    print('Extracting epsilon data...')
    epsilon_data = self.sim.get_array(center=mp.Vector3(), 
                                        size= self.mpsat_sim.cell, 
                                        component=mp.Dielectric)
    print('Epsilon data extracted successfully!')
    if return_dat == True:
        return epsilon_data
    

import psutil
import os
def print_memory_usage():
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    print(f"Memory usage: {mem_info.rss / 1024 ** 2:.2f} MB")

print_memory_usage()

## 
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# MEEPULATOR CODE
print("Data requested: ", required_data)
# Initialize the MEEPulator object
from meepsat.meepulator import MEEPulator

if mpsat_sim is None:
    warnings.warn("No MEEPSAT simulation object provided. Assuming the default simulation object to be self.mpsat_sim created earlier")
    mpsat_sim = self.self

meepulator = MEEPulator(sim = sim,
                        mpsat_sim= mpsat_sim,
                        required_data = required_data,
                        filename= savepath + movie_name,
                        format= 'h5')

print("MEEPulator object created successfully!", meepulator.available_data())


#^ PRIMARY DATA
# Storing the primary data
meepulator.init_primary_data_keys()
meepulator.store_primary_data()


#^ SECONDARY DATA
# Initializing the secondary data keys
meepulator.init_secondary_data_keys()
# Function to store the secondary data
def store_secondary_data_mp_at_every(sim):
    """
    Uses all the required functions to calculate the required data at every timestep.

    Returns
    -------
    A callable function to be used in mp.at_every.
    """
    for sec_data in required_data:
            meepulator.append_data_to_list(key=sec_data, 
                                    value=getattr(meepulator, sec_data)(return_dat=True))
    print('Data extracted successfully at timestep:', meepulator.current_time_step())

self.meepulator_func = store_secondary_data_mp_at_every