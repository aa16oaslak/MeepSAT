import sys
import os
import site
from pathlib import Path
#import meep_testings as mp
import meep as mp
import numpy as np
import warnings
import math

import matplotlib.pyplot as plt
from matplotlib import rc

# Extra functions
import meepsat.helpers as exf

# * ############################################################################################################
# * ############################################################################################################

# Defining some global functions that will decrease the length of the code
def set_sims_obj(self, mpsat_sim):
    """
    Set the MEEPSAT simulation object in the various classes
    
    Parameters
    ----------
    self : object
        Class object

    sim : meep.Simulation
        MEEP simulation object

    mpsat_sim : MEEPSAT
        MEEPSAT simulation object
    """
    if mpsat_sim is None:
        raise ValueError("MEEPSAT simulation object is missing!")
    else:
        self.mpsat_sim = mpsat_sim

    return self.mpsat_sim

def set_center(self, center= None, default_center = None):
    """
    Set the material center in the various classes
    
    Parameters
    ----------
    self : object
        Class object

    center : mp.Vector3
        Center of the material in the x, y and z directions
        Format : mp.Vector3(x, y, z)
    """
    if center is not None:
        self.center = center
    else:
        # There should always be a default center present in the class
        if default_center is None:
            raise ValueError(f"No center given to the material object in the {self.__class__.__name__} class!")
        else:
            warnings.warn(f"No center given to the material object: Taking the default center as {default_center}")
            self.center = default_center
    
    return self.center
        
def set_size(self, size= None, default_size = None):
    """
    Set the material size in the various classes
    
    Parameters
    ----------
    self : object
        Class object

    size : mp.Vector3
        Size of the material in the x, y and z directions
        Format : mp.Vector3(sx, sy, sz)
    """
    if size is not None:
        self.size = size
    else:
        # There should always be a default size present in the class
        if default_size is None:
            raise ValueError(f"No size given to the material object in the {self.__class__.__name__} class!")
        else:
            warnings.warn(f"No size given to the material object: Taking the default size as {default_size}")
            self.size = default_size
    
    return self.size

def set_prop_component(self, component= None, default_component = None):
    """
    Set the component in the various classes
    
    Parameters
    ----------
    self : object
        Class object

    component : str or meep component
        Propagating component of the source, can be a string ('Ez', 'Ex', 'Ey', 'Hx', 'Hy', 'Hz')
        or the actual MEEP component (mp.Ez, mp.Ex, mp.Ey, mp.Hx, mp.Hy, mp.Hz)
    """
    avail_components = ['Ez', 'Ex', 'Ey', 'Hx', 'Hy', 'Hz']
    meep_components = [mp.Ez, mp.Ex, mp.Ey, mp.Hx, mp.Hy, mp.Hz]
    
    # If component is None, use default_component
    if component is None:
        if default_component is None:
            raise ValueError(f"No component given to the object in the {self.__class__.__name__} class!")
        else:
            warnings.warn(f"No component given: Taking the default component as {default_component}")
            self.component = default_component
            return self.component
    
    # If component is already a MEEP component, use it directly
    if component in meep_components:
        self.component = component
        return self.component
    
    # If component is a string, convert to the corresponding MEEP component
    if isinstance(component, str):
        if component in avail_components:
            idx = avail_components.index(component)
            self.component = meep_components[idx]
            return self.component
    
    # If we reach here, the component is invalid
    raise ValueError(f"Invalid component given to the source object in the {self.__class__.__name__} class! "
                     f"Please choose from {avail_components} or use MEEP components directly (mp.Ez, mp.Ex, etc.)")

def set_freq_wvl(self, freq= None, wvl= None):
    """
    Set the frequency and wavelength in the various classes
    
    Parameters
    ----------
    self : object
        Class object

    freq : float
        Frequency of the source in MEEP units (default : None)
    
    wvl : float
        Wavelength of the source in MEEP units (default : None)
        If freq is given, wvl is calculated as 1/freq
    """
    if freq is not None:
        self.freq = freq
        self.wvl = 1/freq
    elif wvl is not None:
        self.wvl = wvl
        self.freq = 1/wvl
    else:
        raise ValueError("Frequency or wavelength is missing! Please provide either the frequency or the wavelength")
    
    return self.freq, self.wvl

def set_source_angle(self, angle= None):
    """
    Set the angle of the source in the various classes
    
    Parameters
    ----------
    self : object
        Class object

    angle : float
        Angle by which the plane wave is rotated w.r.t vertical (default : None)

    Returns
    -------
    self.rot_angle : float
        Angle of the source in radians
    """
    if angle is not None:
        self.rot_angle = angle
        self.rot_angle *= np.pi/180
        print(f"Angle of the source:{self.rot_angle} rad = {angle} degrees")
    else:
        self.rot_angle = 0

    return self.rot_angle

def set_material_obj(self, epsilon_real, epsilon_imag, freq):
    """
    Set the material object in the various classes
    
    Parameters
    ----------
    self : object
        Class object

    epsilon_real : float
        Real part of the permittivity of the material

    epsilon_imag : float
        Imaginary part, i.e. conductivity, of the material

    freq : float
        Frequency at which the sim will be run.
        Needed to set the material property accordingly
    """
    self.epsilon_real = epsilon_real
    self.epsilon_imag = epsilon_imag
    self.freq = freq
    self.conductivity = epsilon_imag*2*np.pi*freq/epsilon_real


# * ############################################################################################################

###& SOURCE
# ~ CONTINUOUS PLANE WAVES
class ContinuousPlaneWaves():
    """
    Class defining the continuous plane waves source
    """
    def __init__(self,
                 mpsat_sim,
                 center = None,
                 size = None,
                 component = None,
                 freq = None,
                 wvl = None,
                 angle = 0,
                 kwargs= None):
        """
        Parameters
        ----------
        mpsat_sim : MEEPSAT
            MEEPSAT simulation object

        center : mp.Vector3
            Center of the source in the x, y and z directions (default : mp.Vector3(0, 0, 0))
            Format : mp.Vector3(x, y, z)
        
        size : mp.Vector3
            Size of the source in the x, y and z directions (default : None)
            Format : mp.Vector3(sx, sy, sz)

        component : str or meep component
            Propagating component of the source, can be a string ('Ez', 'Ex', 'Ey', 'Hx', 'Hy', 'Hz')
            or the actual MEEP component (mp.Ez, mp.Ex, mp.Ey, mp.Hx, mp.Hy, mp.Hz)
        
        freq : float
            Frequency of the source in MEEP units (default : mpsat_sim.freq)
        
        wvl : float
            Wavelength of the source in MEEP units (default : mpsat_sim.wvl)
            If freq is given, wvl is calculated as 1/freq

        angle : float (optional)
            Angle by which the plane wave is rotated w.r.t vertical (default : 0)

        **kwargs : dict
            Additional arguments for the meep.Source()
            https://meep.readthedocs.io/en/latest/Python_User_Interface/#source
            https://meep.readthedocs.io/en/latest/Python_User_Interface/#continuoussource
        """
        # Sims object
        self.mpsat_sim = set_sims_obj(self, mpsat_sim)
        # Centre
        self.center = set_center(self, center, default_center = mp.Vector3(0, 0, 0))
        # Size
        self.size = set_size(self, size, default_size = mp.Vector3(0, self.mpsat_sim.cell_size[1], 0))
        # Propagating component
        self.component = set_prop_component(self, component, default_component = mp.Ez)
        # Frequency and wavelength
        self.freq, self.wvl = set_freq_wvl(self, freq, wvl)
        # Angle of the source
        self.rot_angle = set_source_angle(self, angle)        
        # Additional arguments for both mp.Source() and mp.ContinuousSource()
        self.additional_args = kwargs

        print("Source object created with the following parameters:")
        print("Center: ", self.center)
        print("Size: ", self.size)
        print("Component: ", self.component)
        print("Frequency: ", self.freq)
        print("Wavelength: ", self.wvl)
        print("Angle: ", self.rot_angle)
        print("Additional arguments: ", self.additional_args)

    def amp_func(self, P):
        '''
        Adopted from MEEPART package
        ---
        Returns amplitude of source with added phase to 
        emulate source rotation

        Parameters
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
    
    def assemble(self):
        """
        Return continuous planewaves source
        """
        if self.additional_args is not None:
            source_filtered_kwrg = exf.filter_dict(self.additional_args, mp.Source)
            print("Additional arguments for the Source: ", source_filtered_kwrg)
            source_type_filtered_kwrg = exf.filter_dict(self.additional_args, mp.ContinuousSource)
            print("Additional arguments for the ContinuousSource: ", source_type_filtered_kwrg)

            source = mp.Source(mp.ContinuousSource(frequency=self.freq, 
                                                   **source_type_filtered_kwrg),
                               center= self.center,
                               size= self.size,
                               component=self.component,
                               amp_func=self.amp_func,
                               **source_filtered_kwrg)
            
        else:
            source = mp.Source(mp.ContinuousSource(frequency=self.freq),
                               center= self.center,
                               size= self.size,
                               component=self.component,
                               amp_func=self.amp_func)
        
        print("Continuous plane waves source assembled!")
        return source

class GaussianBeam():

    def __init__(self,
                 mpsat_sim,
                 center = None,
                 size = None,
                 component = None,
                 freq = None,
                 wvl = None,
                 angle = 0,
                 width = 10,
                 cutoff = 0,
                 kwargs= None):

        """
        Parameters
        ----------

        center : mp.Vector3
            Center of the source in the x, y and z directions (default : None)
            Format : mp.Vector3(x, y, z)

        size : mp.Vector3
            Size of the source in the x, y and z directions (default : None)
            Format : mp.Vector3(sx, sy, sz)

        component : mp.Ez, mp.Ex, mp.Ey, mp.Hx, mp.Hy, mp.Hz
            Propagating component of the source (default : None)

        freq : float
            Frequency of the source in MEEP units (default : mpsat_sim.freq)
        
        wvl : float
            Wavelength of the source in MEEP units (default : mpsat_sim.wvl)
            If freq is given, wvl is calculated as 1/freq

        angle : float (optional)
            Angle by which the plane wave is rotated w.r.t vertical (default : 0)

        width : float
            Width of the Gaussian pulse (default : 10)

        cutoff : float
            Cutoff of the Gaussian pulse (default : 5)

        **kwargs : dict
            Additional arguments for the meep.Source() and meep.GaussianSource()
            https://meep.readthedocs.io/en/latest/Python_User_Interface/#source
            https://meep.readthedocs.io/en/latest/Python_User_Interface/#gaussiansource  

            Note: width and cutoff (arguments for mp.GaussianSource) are already defined
            in the function definition because they are specific to the Gaussian beam                    
        """

        # Sims object
        self.mpsat_sim = set_sims_obj(self, mpsat_sim)
        # Centre
        self.center = set_center(self, center, default_center = mp.Vector3(0, 0, 0))
        # Size
        self.size = set_size(self, size, default_size = mp.Vector3(0, self.mpsat_sim.cell_size[1], 0))
        # Propagating component
        self.component = set_prop_component(self, component, default_component = mp.Ez)
        # Frequency and wavelength
        self.freq, self.wvl = set_freq_wvl(self, freq, wvl)
        # Angle of the source (in radians)
        self.rot_angle = set_source_angle(self, angle)
        # k-vector
        self.k_vector = self.calculate_wave_vector(self.rot_angle, self.wvl)
        # Width of the Gaussian pulse
        self.width = width
        # Cutoff of the Gaussian pulse
        self.cutoff = cutoff
        # Additional arguments for both mp.Source() and mp.GaussianSource()
        self.additional_args = kwargs

    def help_gaussian_beam(self, taper_angle, wvl,
                                beam_waist = None,
                                taper = None):
        '''
        For a gaussian beam source
        Provides taper when given beam waist and provides beam waist
        when given taper, at a given taper angle and wavelength in meep units.
        Arguments
        ---------
        taper_angle : float
            Angle in degrees at which the taper is given
        wvl : float
            Wavelength of the source
        beam_waist : float, optional
            Size of the beam waist in MEEP units
        taper : float, optional
            Taper in dB
        '''                        

        a = 20*np.log10((1 + np.cos(np.radians(taper_angle)))/2)
        b = 10*(2*np.pi)**2 * (1-np.cos(np.radians(taper_angle)))*np.log10(np.exp(1)) 

        if beam_waist is None :
            w0 = np.sqrt(- wvl**2 * (taper - a)/b)
            print('The beam waist is {:.2e} MEEP units'.format(w0))

        if taper is None :
            A = a - b* beam_waist**2 / wvl**2
            print('The taper at angle {:.1f} deg is {:.2f} dB'.format(taper_angle, A))

    def calculate_wave_vector(self, angle_rad, wavelength):
        """
        Calculate the wave vector based on the angle and wavelength.

        Parameters
        ----------
        angle : float
            Angle of the source in radians.
        wavelength : float
            Wavelength of the source.

        Returns
        -------
        k_vector : mp.Vector3
            Wave vector for the given angle and wavelength.
            The length of the wave vector is ignored in Gaussian beam sources.
        """
        kx = 2 * np.pi * np.cos(angle_rad) / wavelength
        ky = 2 * np.pi * np.sin(angle_rad) / wavelength
        return mp.Vector3(kx, ky, 0)

    """def gaussianProfile(self,
                        vec):
        w0 = self.width
        return np.exp(-np.square((vec.x-(self.center.x))/w0))"""

    def assemble(self):
        """
        Return Gaussian beam source
        """

        if self.additional_args is not None:
            continuous_source_filtered_kwrg = exf.filter_dict(self.additional_args, mp.ContinuousSource)
            print("Additional arguments for the ContinuousSource: ", continuous_source_filtered_kwrg)
            source_type_filtered_kwrg = exf.filter_dict(self.additional_args, mp.GaussianBeam2DSource)
            print("Additional arguments for GaussianBeamSource: ", source_type_filtered_kwrg)

            source = mp.GaussianBeam2DSource(mp.ContinuousSource(frequency=self.freq,
                                                               **continuous_source_filtered_kwrg), 
                                            beam_w0=self.width,
                                            beam_kdir=self.k_vector,
                                            center= self.center,
                                            size= self.size,
                                            component=self.component,
                                            **source_type_filtered_kwrg)
            
            
        else:
            source = mp.GaussianBeam2DSource(mp.ContinuousSource(frequency=self.freq),
                                            beam_w0=self.width,
                                            beam_kdir=self.k_vector,
                                            center= self.center,
                                            size= self.size,
                                            component=self.component)
        print("Gaussian beam source assembled!")
        return source

#* Defining some global functions for meep block object
def meep_block(size, 
               center, 
               material,
               angle=0,
               rot_axis='x',
               e1=mp.Vector3(1, 0, 0),
               e2=mp.Vector3(0, 1, 0),
               e3=mp.Vector3(0, 0, 1),
               **kwargs):
    """
    Returns the block object for the source.

    Parameters
    ----------
    size : mp.Vector3
        Size of the block in the x, y, and z directions.
    center : mp.Vector3
        Center of the block in the x, y, and z directions.
    material : mp.Medium
        Material of the block.
    angle : float (optional) 
        Angle by which the block is rotated w.r.t the rot_axis anticlockwise (default: 0).
        Units are in degrees. 
    rot_axis : str (optional)
        Axis about which the block is rotated (default: 'x').
    e1, e2, e3 : mp.Vector3 (optional)
        Vectors defining the x, y, and z axes of the block (default: standard unit vectors).
    kwargs : dict
        Additional arguments for the meep.Block().
    """
    
    # Check for valid rotation axis
    if rot_axis not in ['x', 'y', 'z']:
        raise ValueError("Invalid rotation axis. Choose from 'x', 'y', or 'z'.")

    # Set the rotation axis
    if rot_axis == 'x':
        axis = mp.Vector3(1, 0, 0)
    elif rot_axis == 'y':
        axis = mp.Vector3(0, 1, 0)
    else:  # Default: z-axis
        axis = mp.Vector3(0, 0, 1)

    # Apply rotation if angle is non-zero
    if angle != 0:
        e1 = e1.rotate(axis, math.radians(angle))
        e2 = e2.rotate(axis, math.radians(angle))
        e3 = e3.rotate(axis, math.radians(angle))

    # Return the block with the given parameters
    return mp.Block(size=size,
                    center=center,
                    material=material,
                    e1=e1,
                    e2=e2,
                    e3=e3,
                    **kwargs)



#& APERTURE STOP


class ApertureStop(object):
    '''
    Class defining an aperture stop
    '''

    def __init__(self,
                 mpsat_sim,
                 type,
                 diameter, 
                 thickness,
                 pos_x=None,
                 pos_y=None, 
                 n_refr = 1, 
                 material= None,
                 conductivity = mp.inf,
                 rot_axis = 'x',
                 rot_angle = 0,
                 y_centre_offset = [0,0],
                 y_size_offset = [0,0]):
        '''
        Defines the attributes of the aperture stop object

        Arguments
        ---------
        mpsat_sim : object
            MEEPSAT simulation object
        type: str
            Type of the aperture stop- circular, square etc       
        diameter : float 
            Diameter of the aperture stop opening
        pos_x : float, optional
            Position of the left surface of the aperture stop along x-axis
            (mutually exclusive with pos_y)
        pos_y : float, optional
            Position of the bottom surface of the aperture stop along y-axis
            (mutually exclusive with pos_x)
        thickness : float
            Thickness of aperture stop slab
        n_refr : float, optional 
            Index of refraction of the material 
            if the stop is dielectric
            (default = 1)
        conductivity : float, optional
            Conductivity of the material (default = mp.inf)
        rot_axis : str, optional
            Axis about which the aperture stop is rotated 
            (default : 'x')
        rot_angle : float, optional
            Angle by which the aperture stop is rotated w.r.t rot_axis (default : 0 degrees)
        '''
        self.mpsat_sim = mpsat_sim
        self.type = type                
        self.thick = thickness
        
        # Check that only one position is specified
        if pos_x is not None and pos_y is not None:
            raise ValueError("Cannot specify both pos_x and pos_y. Choose one direction for the aperture stop.")
        
        if pos_x is None and pos_y is None:
            raise ValueError("Must specify either pos_x or pos_y for the aperture stop position.")
        
        # Set position based on which one is provided
        if pos_x is not None:
            # Convert the pos_x in (0,x) coordinate system to (-x/2, x/2)
            self.pos_x = pos_x - self.mpsat_sim.cell.x/2
            self.pos_y = None
            self.orientation = 'vertical'  # Blocks oriented vertically (along y-axis)
        else:
            # Convert the pos_y in (0,y) coordinate system to (-y/2, y/2)
            self.pos_y = pos_y - self.mpsat_sim.cell.y/2
            self.pos_x = None
            self.orientation = 'horizontal'  # Blocks oriented horizontally (along x-axis)
                          
        self.diameter = diameter        
        self.permittivity = n_refr**2   
        self.conductivity = conductivity
        self.material = material 
        self.object_type = 'AP_stop'
        self.rot_axis = rot_axis
        self.rot_angle = rot_angle
        self.y_centre_offset = y_centre_offset
        self.y_size_offset = y_size_offset

        print(f"Aperture stop created with orientation: {self.orientation}")
        print("type material: ", self.material)

    def square_aperture(self):
        '''
        Returns the block object for the aperture stop
        '''
        if self.material is not None:
            material = self.material
        else:
            material = mp.Medium(epsilon=self.permittivity, 
                                D_conductivity = self.conductivity)
        
        if self.orientation == 'vertical':
            # Original implementation: blocks along y-axis, positioned at pos_x
            block_size_y_up = (self.mpsat_sim.cell.y - self.diameter) / 2 
            block_size_y_down = (self.mpsat_sim.cell.y - self.diameter) / 2 
            
            size_up = mp.Vector3(self.thick, block_size_y_up + self.y_size_offset[0], 0)
            centre_up = mp.Vector3(self.pos_x + (self.thick/2),
                                self.diameter/2 + (block_size_y_up + self.y_size_offset[0])/2 + self.y_centre_offset[0],
                                0)

            size_down = mp.Vector3(self.thick, block_size_y_down + self.y_size_offset[1], 0)
            centre_down = mp.Vector3(self.pos_x + (self.thick/2),
                                    -self.diameter/2 - (block_size_y_down + self.y_size_offset[1])/2 + self.y_centre_offset[1],
                                    0)
            
        else:  # orientation == 'horizontal'
            # New implementation: blocks along x-axis, positioned at pos_y
            block_size_x_left = (self.mpsat_sim.cell.x - self.diameter) / 2 
            block_size_x_right = (self.mpsat_sim.cell.x - self.diameter) / 2 
            
            size_up = mp.Vector3(block_size_x_right + self.y_size_offset[0], self.thick, 0)
            centre_up = mp.Vector3(self.diameter/2 + (block_size_x_right + self.y_size_offset[0])/2 + self.y_centre_offset[0],
                                 self.pos_y + (self.thick/2),
                                 0)

            size_down = mp.Vector3(block_size_x_left + self.y_size_offset[1], self.thick, 0)
            centre_down = mp.Vector3(-self.diameter/2 - (block_size_x_left + self.y_size_offset[1])/2 + self.y_centre_offset[1],
                                   self.pos_y + (self.thick/2),
                                   0)
        
        aperture_stop_up = meep_block(size = size_up,
                                        center = centre_up,
                                        material = material,
                                        angle = self.rot_angle,
                                        rot_axis = self.rot_axis)
        
        aperture_stop_down = meep_block(size = size_down,
                                        center = centre_down,
                                        material = material,
                                        angle = self.rot_angle,
                                        rot_axis = self.rot_axis)
        
        print(f'Aperture stop created ({self.orientation}): Up size={size_up}, Down size={size_down}')
        print(f'Centers: Up={centre_up}, Down={centre_down}')

        return aperture_stop_up, aperture_stop_down

    def assemble(self):
        '''
        Returns the block objects for the aperture stop according to the type
        
        Returns
        -------
        tuple
            Two block objects (aperture_stop_up, aperture_stop_down)
        '''
        if self.type == 'square':
            return self.square_aperture()
        else:
            raise ValueError(f'Invalid aperture stop type: {self.type}. Currently only "square" is supported.')


# class ApertureStop(object):
#     '''
#     Class defining an aperture stop
#     '''

#     def __init__(self,
#                  mpsat_sim,
#                  type,
#                  diameter, 
#                  pos_x, 
#                  thickness, 
#                  n_refr = 1, 
#                  material= None,
#                  conductivity = mp.inf,
#                  rot_axis = 'x',
#                  rot_angle = 0,
#                  y_centre_offset = [0,0],
#                  y_size_offset = [0,0]):
#         '''
#         Defines the attributes of the aperture stop object

#         Arguments
#         ---------
#         mpsat_sim : object
#             MEEPSAT simulation object
#         type: str
#             Type of the aperture stop- circular, square etc       
#         diameter : float 
#             Diameter of the aperture stop opening
#         pos_x : float
#             Position of the left surface of the aperture stop along x-axis
#         thickness : float
#             Thickness of aperture stop slab
#         n_refr : float, optional 
#             Index of refraction of the material 
#             if the stop is dielectric
#             (default = 1)
#         conductivity : float, optional
#             Conductivity of the material (default = 1e7)
#         rot_axis : str, optional
#             Axis about which the aperture stop is rotated 
#             (default : 'x')
#         rot_angle : float, optional
#             Angle by which the aperture stop is rotated w.r.t rot_axis (default : 0 degrees)
#         '''
#         self.mpsat_sim = mpsat_sim
#         self.type = type                
#         self.thick = thickness
#         if pos_x:
#             # Convert the pos_x in (0,x) coordinate system to (-x/2, x/2)
#             self.pos_x = pos_x - self.mpsat_sim.cell.x/2
#         else:
#             self.pos_x = 0 # Center of the cell                  
#         self.diameter = diameter        
#         self.permittivity = n_refr**2   
#         self.conductivity = conductivity
#         self.material = material 
#         self.object_type = 'AP_stop'
#         self.rot_axis = rot_axis
#         self.rot_angle = rot_angle
#         self.y_centre_offset = y_centre_offset  # Offset along the y-axis, if needed
#         self.y_size_offset = y_size_offset  # Offset for the size along the y-axis, if needed

#         print("type material: ", self.material)

#     # def square_aperture(self):
#     #     '''
#     #     Returns the block object for the aperture stop
#     #     '''
#     #     #Defines the material with given properties
#     #     material = mp.Medium(epsilon=self.permittivity, 
#     #                          D_conductivity = self.conductivity)
        
#     #     # Up block of the aperture stop slab
#     #     size_up = mp.Vector3(self.thick,
#     #                          (self.mpsat_sim.cell.y - self.diameter)/2,
#     #                          0)
        
#     #     centre_up = mp.Vector3(self.pos_x - self.thick/2,
#     #                            self.diameter/2 + size_up.y/2,
#     #                              0)
        
#     #     size_down = mp.Vector3(self.thick,
#     #                              (self.mpsat_sim.cell.y - self.diameter)/2,
#     #                                 0) 
        
#     #     centre_down = mp.Vector3(self.pos_x - self.thick/2,
#     #                              -self.diameter/2 - size_down.y/2,
#     #                                 0)
        
#     #     aperture_stop_up = meep_block(size = size_up,
#     #                                     center = centre_up,
#     #                                     material = material,
#     #                                     angle = self.rot_angle,
#     #                                     rot_axis = self.rot_axis)
        
#     #     aperture_stop_down = meep_block(size = size_down,
#     #                                     center = centre_down,
#     #                                     material = material,
#     #                                     angle = self.rot_angle,
#     #                                     rot_axis = self.rot_axis)
        
#     #     print('Aperture stop created',[aperture_stop_up, aperture_stop_down])

#     #     return aperture_stop_up, aperture_stop_down

#     def square_aperture(self):
#         '''
#         Returns the block object for the aperture stop
#         '''
#         #Defines the material with given properties

#         if self.material is not None:
#             material = self.material
#         else:
#             material = mp.Medium(epsilon=self.permittivity, 
#                                 D_conductivity = self.conductivity)
        
#         # Calculate the block size (should be identical for both)
#         block_size_y_up = (self.mpsat_sim.cell.y - self.diameter) / 2 
#         block_size_y_down = (self.mpsat_sim.cell.y - self.diameter) / 2 
        
#         # Up block of the aperture stop slab
#         size_up = mp.Vector3(self.thick, block_size_y_up + self.y_size_offset[0], 0)  # Fixed: added offset directly to y-component
#         centre_up = mp.Vector3(self.pos_x + (self.thick/2),  # Fixed: changed from - to +
#                             self.diameter/2 + (block_size_y_up + self.y_size_offset[0])/2 + self.y_centre_offset[0],
#                             0)

#         # Down block of the aperture stop slab
#         size_down = mp.Vector3(self.thick, block_size_y_down + self.y_size_offset[1], 0)  # Fixed: added offset directly to y-component
#         centre_down = mp.Vector3(self.pos_x + (self.thick/2),  # Fixed: changed from - to +
#                                 -self.diameter/2 - (block_size_y_down + self.y_size_offset[1])/2 + self.y_centre_offset[1],
#                                 0)
        
#         aperture_stop_up = meep_block(size = size_up,
#                                         center = centre_up,
#                                         material = material,
#                                         angle = self.rot_angle,
#                                         rot_axis = self.rot_axis)
        
#         aperture_stop_down = meep_block(size = size_down,
#                                         center = centre_down,
#                                         material = material,
#                                         angle = self.rot_angle,
#                                         rot_axis = self.rot_axis)
        
#         print(f'Aperture stop created: Up size={size_up}, Down size={size_down}')
#         print(f'Centers: Up={centre_up}, Down={centre_down}')

#         return aperture_stop_up, aperture_stop_down
    

    # def assemble(self):
    #     '''
    #     Returns the block object for the aperture stop

    #     Arguments
    #     ---------
    #     centre_list : list, optional
    #         List of the centers of the aperture stop

    #     size_list : list, optional
    #         List of the sizes of the aperture stop
    #     '''
    #     if self.type == 'square':
    #         return self.square_aperture()
    #     else:
    #         raise ValueError('Invalid aperture stop type name')


###& LENS Mounting
# class LensMounting():
#     def __init__(self,
#                  mpsat_sim,
#                  type = None,
#                  diameter = None, 
#                  pos_x = None, 
#                  thickness = None,
#                  n_refr = None, 
#                  conductivity = None):
        

    
###& DETECTOR CLASS
class Detector():
    '''
    Class defining an image plane/ the location of the detectors
    '''

    def __init__(self,
                 type= None,
                 diameter= None, 
                 pos_x= None, 
                 thickness= None,
                 n_refr = None, 
                 conductivity = None):
        '''
        Defines the attributes of the aperture stop object

        Arguments
        ---------
        type : str, optional
            Type of Detector- block, circular etc (default : None)       
        diameter : float 
            Diameter of the image plane slab
        pos_x : float
            Position of the image plane along x-axis
        thickness : float
            Thickness of image plane slab
        n_refr : float, optional
            Index of refraction of the material 
            if the stop is dielectric
            (default = 1)
        conductivity : float, optional
            Conductivity of the material (default = np.inf)
        '''
        self.object_type = 'detector'

        self.name = type              
        self.diameter = diameter
        self.x = pos_x
        self.thickness = thickness
        self.n_refr = n_refr
        self.conductivity = conductivity

        self.center = [self.x, 0, 0]
        self.size = [self.thickness, self.diameter, 0]

    def center(self):
        return self.center
    
    def size(self):
        return self.size

    def position(self):
        if self.name is not None :
            return self.object_type + 'of type' + self.name + 'at position' + str(self.x)
        else : 
            return 'Image Plane/Detector at position ' + str(self.x)
        

    def block_detector(self):
        '''
        Returns the block object for the image plane/ detector! 
        '''
        
        meep_block_detector = mp.Block(size= mp.Vector3(self.thickness, self.diameter, 0),
                                       center= mp.Vector3(self.x, 0, 0),
                                       material= self.material)
        
        return meep_block_detector
    
    ### ^ Similarly we can add more types of detectors here
    # ^ def circular_detector(self):

    def assemble(self):
        '''
        Returns the block object for the image plane/ detector! 
        '''
        if self.conductivity != np.inf :
            #Defines the material with given properties
            self.material = mp.Medium(epsilon=self.n_refr**2, 
                                      D_conductivity = self.conductivity)
        
        else :
            #If the conductivity is infinite, Meep can define a perfect conductor
            self.material = mp.perfect_electric_conductor
        
        if self.name == 'meep_block':
            detector = self.block_detector()

        ### ^ Similarly we can add more types of detectors here
        # ^ elif self.name == 'circular':

        else:
            raise ValueError('Invalid detector type name')
        
        return detector
    
###& BOUNDARY CLASS CLASS

class Boundary():
    """
    Class defining the boundary conditions of the 2D simulation box
    """
    def __init__(self,
                 type = None,
                 thickness = None,
                 **kwargs):
        """
        Defines the attributes of the boundary object

        Arguments
        ---------
        type : str
            Type of the boundary conditions; e.g., PML (default : None)

        thickness : float
            Thickness of the boundary conditions (default : None)

        **kwargs : dict
            Additional arguments for the meep.Boundary()
            https://meep.readthedocs.io/en/latest/Python_User_Interface/#boundary
        """
        self.object_type = 'boundary_layer'

        if type is None:
            warnings.warn("No name given to the boundary object: Taking the default boundary as PML")
            self.name = 'PML'
        else:
            self.name = type

        if thickness is None:
            warnings.warn("No thickness given to the boundary object: Taking the default thickness as 2.0")
            self.thickness = 2.0
        else:
            self.thickness = thickness

        self.additional_args = kwargs

    def description(self):
        return self.object_type + ': ' + self.name + ' with thickness ' + str(self.thickness)
        
    def pml_boundary(self):
        """
        Return PML boundary conditions
        """
        if self.additional_args:
            filtered_kwrg = exf.filter_dict(self.additional_args, mp.Boundary)
            boundary = mp.PML(self.thickness, **filtered_kwrg)
        else:
            boundary = mp.PML(self.thickness)
        
        return boundary
    
    ### ^ Similarly we can add more types of boundaries here
    # ^ def periodic_boundary(self):

    def assemble(self):
        """
        Returns the boundary object according to the user input
        """
        if self.name == 'PML':
            boundary = self.pml_boundary()

        ### ^ Similarly we can add more types of boundaries here
        # ^ elif self.name == 'periodic':
        else:
            raise ValueError('Invalid boundary type name')

        return boundary
        
# * ############################################################################################################
# * ############################################################################################################
# * ############################################################################################################
# * Extracting some classes from the MEEPART package

class Filter(object):
    """
    Class defining the filter object
    """
    def __init__(self,
                 mpsat_sim,
                 name="block",
                 center=None,
                 size=None,
                 material=None,
                 angle=0,
                 rot_axis='x',
                 **kwargs):
        """
        Defines the attributes of the filter object

        Arguments
        ---------
        mpsat_sim : object
            MEEPSAT simulation object
        name : str, optional
            Type of filter (default: 'block')
        center : mp.Vector3
            Center of the filter
        size : mp.Vector3
            Size of the filter
        material : mp.Medium
            Material of the filter
        angle : float, optional
            Angle of rotation in degrees (default: 0)
        rot_axis : str, optional
            Axis of rotation ('x', 'y', or 'z') (default: 'x')
        **kwargs : dict
            Additional arguments for the meep functions
        """
        # Sims object
        self.mpsat_sim = set_sims_obj(self, mpsat_sim)
        self.name = name
        self.object_type = 'Filter'
        
        # Set center and size using your helper functions
        self.center = set_center(self, center, default_center=mp.Vector3(0, 0, 0))
        self.size = set_size(self, size, default_size=mp.Vector3(0, 0, 0))
        
        # Material
        if material is None:
            raise ValueError("Material must be specified for Filter")
        self.material = material
        
        # Rotation parameters
        self.angle = angle
        self.rot_axis = rot_axis
        self.kwargs = kwargs

    def block_filter(self):
        """
        Return the block filter object
        """
        # Use your existing meep_block utility function
        filter_block = meep_block(
            size=self.size,
            center=self.center,
            material=self.material,
            angle=self.angle,
            rot_axis=self.rot_axis,
            **self.kwargs
        )
        return filter_block
    
    def assemble(self):
        """
        Returns the filter object according to the user input
        """
        if self.name == 'block':
            filter_obj = self.block_filter()
        else:
            raise ValueError(f'Invalid filter type: {self.name}')

        return filter_obj
    

class Slab(object):
    """
    Class defining a slab object - a simple geometrical shape used for various optical elements
    """
    def __init__(self,
                 mpsat_sim,
                 name="block",
                 center=None,
                 size=None,
                 material=None,
                 angle=0,
                 rot_axis='x',
                 **kwargs):
        """
        Defines the attributes of the slab object

        Arguments
        ---------
        mpsat_sim : object
            MEEPSAT simulation object
        name : str, optional
            Type of slab (default: 'block')
        center : mp.Vector3
            Center of the slab
        size : mp.Vector3
            Size of the slab
        material : mp.Medium
            Material of the slab
        angle : float, optional
            Angle of rotation in degrees (default: 0)
        rot_axis : str, optional
            Axis of rotation ('x', 'y', or 'z') (default: 'x')
        **kwargs : dict
            Additional arguments for the meep functions
        """
        # Sims object
        self.mpsat_sim = set_sims_obj(self, mpsat_sim)
        self.name = name
        self.object_type = 'Slab'
        
        # Set center and size using helper functions
        self.center = set_center(self, center, default_center=mp.Vector3(0, 0, 0))
        self.size = set_size(self, size, default_size=mp.Vector3(0, 0, 0))
        
        # Material
        if material is None:
            raise ValueError("Material must be specified for Slab")
        self.material = material
        
        # Rotation parameters
        self.angle = angle
        self.rot_axis = rot_axis
        self.kwargs = kwargs

    def block_slab(self):
        """
        Return the block slab object
        """
        # Use the existing meep_block utility function
        slab_block = meep_block(
            size=self.size,
            center=self.center,
            material=self.material,
            angle=self.angle,
            rot_axis=self.rot_axis,
            **self.kwargs
        )
        return slab_block
    
    def assemble(self):
        """
        Returns the slab object according to the user input
        """
        if self.name == 'block':
            slab_obj = self.block_slab()
        else:
            raise ValueError(f'Invalid slab type: {self.name}')

        return slab_obj
    
#!= Modules for MEEP monitor


class VolumeMonitor():
    """
    Class defining a volume monitor for collecting data from a specific region
    """
    def __init__(self,
                 mpsat_sim,
                 name="volume_monitor",
                 center=None,
                 size=None,
                 components=None,
                 data_required=None,
                 **kwargs):
        """
        Parameters
        ----------
        mpsat_sim : MEEPSAT
            MEEPSAT simulation object
            
        name : str, optional
            Name of the monitor (default: 'volume_monitor')
            
        center : list or mp.Vector3
            Center of the monitor volume [x, y, z]
            
        size : list or mp.Vector3
            Size of the monitor volume [x, y, z]
            
        components : list, optional
            Field components to monitor (default: None, monitors all components)
            
        data_required : dict, optional
            Dictionary specifying what data to collect and when:
            {
                'at_every_timestep': int,  # Collect data every N timesteps
                'at_every': list,          # List of data types to collect at each sampling
                'at_end': list             # List of data types to collect at simulation end
            }
            
        **kwargs : dict
            Additional arguments for customization
        """
        # Sims object
        self.mpsat_sim = set_sims_obj(self, mpsat_sim)
        self.name = name
        self.object_type = 'VolumeMonitor'
        
        # Set center - handle both list and Vector3
        if isinstance(center, list):
            self.center = mp.Vector3(center[0], center[1], center[2])
        else:
            self.center = set_center(self, center, default_center=mp.Vector3(0, 0, 0))
            
        # Set size - handle both list and Vector3
        if isinstance(size, list):
            self.size = mp.Vector3(size[0], size[1], size[2])
        else:
            self.size = set_size(self, size, default_size=mp.Vector3(1, 1, 0))
            
        # Components to monitor
        self.components = components
        
        # Data collection requirements
        self.data_required = data_required if data_required else {
            'at_every_timestep': 10,
            'at_every': [],
            'at_end': []
        }
        
        # Additional args
        self.kwargs = kwargs
        
        print(f"Volume monitor '{self.name}' created at {self.center} with size {self.size}")
        print(f"Data collection: every {self.data_required.get('at_every_timestep')} timesteps")
        print(f"Collecting: {self.data_required.get('at_every')} during simulation")
        print(f"Collecting: {self.data_required.get('at_end')} at end")

    def assemble(self):
        """
        Return the assembled monitor object
        """
        # Create a volume object
        volume = mp.Volume(center=self.center, size=self.size)
        print(f"Volume monitor assembled: {volume}")
        return volume
    

# Add after the VolumeMonitor class

class FluxMonitor():
    """
    Class defining a flux monitor for calculating power transmission and reflection
    """
    def __init__(self,
                 mpsat_sim,
                 name="flux_monitor",
                 center=None,
                 size=None,
                 direction=mp.X,
                 freq_min=None,
                 freq_max=None,
                 nfreq=100,
                 monitor_type="transmission",  # "incident", "reflection", "transmission"
                 use_flux_file=None,
                 **kwargs):
        """
        Parameters
        ----------
        mpsat_sim : MEEPSAT
            MEEPSAT simulation object
            
        name : str, optional
            Name of the monitor (default: 'flux_monitor')
            
        center : list or mp.Vector3
            Center of the flux monitor plane
            
        size : list or mp.Vector3
            Size of the flux monitor plane
            
        direction : mp.direction constant
            Direction normal to the flux plane (mp.X, mp.Y, or mp.Z)
            
        freq_min : float
            Minimum frequency to monitor
            
        freq_max : float
            Maximum frequency to monitor
            
        nfreq : int
            Number of frequency points
            
        monitor_type : str
            Type of flux monitor: "incident", "reflection", or "transmission"

        use_flux_file : str, optional
            Path to saved flux data for normalization
        
        **kwargs : dict
            Additional arguments for mp.FluxRegion
        """
        self.use_flux_file = use_flux_file 
        # Sims object
        self.mpsat_sim = set_sims_obj(self, mpsat_sim)
        self.name = name
        self.object_type = 'FluxMonitor'
        self.monitor_type = monitor_type
        
        # Set center - handle both list and Vector3
        if isinstance(center, list):
            self.center = mp.Vector3(center[0], center[1], center[2])
        else:
            self.center = set_center(self, center, default_center=mp.Vector3(0, 0, 0))
            
        # Set size - handle both list and Vector3
        if isinstance(size, list):
            self.size = mp.Vector3(size[0], size[1], size[2])
        else:
            self.size = set_size(self, size, default_size=mp.Vector3(0, self.mpsat_sim.cell_size[1], 0))
            
        # Frequency parameters
        self.freq_min = freq_min if freq_min is not None else self.mpsat_sim.freq * 0.8
        self.freq_max = freq_max if freq_max is not None else self.mpsat_sim.freq * 1.2
        self.nfreq = nfreq
        
        # Direction
        self.direction = direction
        
        # Additional args
        self.kwargs = kwargs
        
        print(f"Flux monitor '{self.name}' created at {self.center} with size {self.size}")
        print(f"Frequency range: {self.freq_min} to {self.freq_max} with {self.nfreq} points")
        print(f"Monitor type: {self.monitor_type}")

    def assemble(self):
        """
        Return the assembled flux monitor object
        """
        # Create a flux region
        flux_region = mp.FluxRegion(
            center=self.center,
            size=self.size,
            direction=self.direction,
            **self.kwargs
        )
        print(f"Flux monitor {self.monitor_type} assembled: {flux_region}")
        return flux_region
    

class PyramidalAbsorbers(object):
    '''
    Class defining pyramidal absorbers along the edges of the simulation cell.
    '''
    def __init__(self,
                 mpsat_sim,
                 base_width=None,
                 top_width=0,
                 height=None,
                 layer_thickness=None,
                 num_pyramids=20,
                 n_layers=10,
                 material=None,
                 epsilon_real=2.5,
                 epsilon_imag=0,
                 freq=1/3,
                 x_coverage_start=None,
                 x_coverage_end=None,
                 y_coverage_start=None,
                 y_coverage_end=None,
                 coverage_percent=40,
                 edges=["top", "bottom"],
                 x_left_offset=0,
                 x_right_offset=0,
                 y_top_offset=0,
                 y_bottom_offset=0,
                 # Substrate parameters
                 add_substrate=True,
                 substrate_thickness=None,
                 substrate_material=None,
                 substrate_epsilon_real=None,
                 substrate_epsilon_imag=None,
                 substrate_extends_beyond_pyramids=True,
                 substrate_extension=0.5,
                 # PEC backing parameters
                 add_pec_backing=False,
                 pec_thickness=None,
                 pec_extends_beyond_substrate=True,
                 pec_extension=0,
                 name=None):
        '''
        Defines pyramidal absorbers along cell edges

        Parameters
        ----------
        mpsat_sim : MEEPSAT
            MEEPSAT simulation object
        base_width : float, optional
            Width of the pyramid base (default: calculated from coverage and num_pyramids)
        top_width : float, optional
            Width of the pyramid top layer (default: 0 for pointed pyramid)
        height : float, optional
            Total height of each pyramid (default: None, calculated from layer_thickness)
        layer_thickness : float, optional
            Thickness of each layer (default: None, calculated from height)
        num_pyramids : int, optional
            Number of pyramids along each edge (default: 20)
        n_layers : int, optional
            Number of layers per pyramid (default: 10)
        material : mp.Medium, optional
            Material for the pyramids (overrides epsilon values if provided)
        epsilon_real : float, optional 
            Real part of the permittivity (default: 2.5)
        epsilon_imag : float, optional
            Imaginary part of the permittivity (default: 0)
        freq : float, optional
            Frequency for material properties (default: 1/3)
        x_coverage_start : float, optional
            Start position for top/bottom pyramids (default: calculated from coverage_percent)
        x_coverage_end : float, optional
            End position for top/bottom pyramids (default: calculated from coverage_percent)
        y_coverage_start : float, optional
            Start position for left/right pyramids (default: calculated from coverage_percent)
        y_coverage_end : float, optional
            End position for left/right pyramids (default: calculated from coverage_percent)
        coverage_percent : float, optional
            Percentage of cell width to cover with pyramids (default: 40%)
        edges : list of str, optional
            Which edges to place pyramids on (default: ["top", "bottom"])
            Options: "top", "bottom", "left", "right"
        x_left_offset : float, optional
            Offset for left edge pyramids (default: 0)
        x_right_offset : float, optional
            Offset for right edge pyramids (default: 0)
        y_top_offset : float, optional 
            Offset for top edge pyramids (default: 0)
        y_bottom_offset : float, optional
            Offset for bottom edge pyramids (default: 0)
        add_substrate : bool, optional
            Whether to add substrate beneath pyramids (default: True)
        substrate_thickness : float, optional
            Thickness of the substrate (default: same as layer_thickness)
        substrate_material : mp.Medium, optional
            Material for the substrate (default: same as pyramid material)
        substrate_epsilon_real : float, optional
            Real part of substrate permittivity (default: same as pyramid)
        substrate_epsilon_imag : float, optional
            Imaginary part of substrate permittivity (default: same as pyramid)
        substrate_extends_beyond_pyramids : bool, optional
            Whether substrate extends beyond pyramid coverage (default: True)
        substrate_extension : float, optional
            How much substrate extends beyond pyramids on each side (default: 0.5)
        add_pec_backing : bool, optional
            Whether to add PEC backing beneath substrate (default: False)
        pec_thickness : float, optional
            Thickness of the PEC backing (default: same as substrate_thickness)
        pec_extends_beyond_substrate : bool, optional
            Whether PEC backing extends beyond substrate (default: True)
        pec_extension : float, optional
            How much PEC backing extends beyond substrate on each side (default: 0)
        name : str, optional
            Name of the object (default: None)
            
        Notes
        -----
        Priority for determining pyramid dimensions:
        1. If both height and layer_thickness are provided, height takes priority
        2. If only height is provided, layer_thickness = height / n_layers
        3. If only layer_thickness is provided, height = layer_thickness * n_layers
        4. If neither is provided, defaults are calculated based on base_width
        '''
        # Sims object
        self.mpsat_sim = set_sims_obj(self, mpsat_sim)
        
        # Basic parameters
        self.name = name if name else "Pyramidal Absorbers"
        self.object_type = 'PyramidalAbsorber'
        self.num_pyramids = num_pyramids
        self.n_layers = n_layers
        self.edges = edges
        self.coverage_percent = coverage_percent
        self.top_width = top_width
        
        # Offset parameters
        self.x_left_offset = x_left_offset
        self.x_right_offset = x_right_offset
        self.y_top_offset = y_top_offset
        self.y_bottom_offset = y_bottom_offset
        
        # Substrate parameters
        self.add_substrate = add_substrate
        self.substrate_extends_beyond_pyramids = substrate_extends_beyond_pyramids
        self.substrate_extension = substrate_extension
        
        # PEC backing parameters
        self.add_pec_backing = add_pec_backing
        self.pec_extends_beyond_substrate = pec_extends_beyond_substrate
        self.pec_extension = pec_extension
        
        self.x_coverage_start = x_coverage_start
        self.x_coverage_end = x_coverage_end
        self.y_coverage_start = y_coverage_start
        self.y_coverage_end = y_coverage_end
        
        # Validate number of pyramids
        if self.x_coverage_start is not None and self.x_coverage_end is not None:
            if self.num_pyramids != int((self.x_coverage_end - self.x_coverage_start)/base_width):
                warnings.warn(f"The number of pyramids specified ({self.num_pyramids}) cannot fit in the x coverage range ({self.x_coverage_end - self.x_coverage_start}). Reducing the number of pyramids.")
                # Round the next lower integer
                self.num_pyramids = int((self.x_coverage_end - self.x_coverage_start)/base_width) + 1 
                print(f"New number of pyramids: {self.num_pyramids}")
            elif self.num_pyramids < 1:
                raise ValueError("Number of pyramids must be at least 1.")
            else:
                self.num_pyramids = self.num_pyramids


        # Calculate x coverage area (for top/bottom)
        if self.x_coverage_start is None or self.x_coverage_end is None:
            coverage_factor = self.coverage_percent / 100.0
            # Calculate half the coverage width to center it
            half_coverage_x = (self.mpsat_sim.cell_size[0] * coverage_factor) / 2
            self.x_coverage_start = -half_coverage_x
            self.x_coverage_end = half_coverage_x            
        else:
            self.x_coverage_start = self.x_coverage_start
            self.x_coverage_end = self.x_coverage_end
            
        self.x_coverage_width = self.x_coverage_end - self.x_coverage_start

        # Calculate y coverage area (for left/right)
        if self.y_coverage_start is None or self.y_coverage_end is None:
            coverage_factor = self.coverage_percent / 100.0
            # Calculate half the coverage width to center it
            half_coverage_y = (self.mpsat_sim.cell_size[1] * coverage_factor) / 2
            self.y_coverage_start = -half_coverage_y
            self.y_coverage_end = half_coverage_y
        else:
            self.y_coverage_start = self.y_coverage_start
            self.y_coverage_end = self.y_coverage_end
            
        self.y_coverage_width = self.y_coverage_end - self.y_coverage_start

        # Set base widths (for x and y directions)
        if base_width is None:
            # Calculate base widths based on coverage and number of pyramids
            self.x_base_width = self.x_coverage_width / self.num_pyramids
            self.y_base_width = self.y_coverage_width / self.num_pyramids
        else:
            self.x_base_width = base_width
            self.y_base_width = base_width
        
        # Handle height and layer_thickness parameters with priority logic
        self._set_pyramid_dimensions(height, layer_thickness)

        # Check if imaginary part is provided and set the conductivity accordingly
        if epsilon_imag != 0:
            self.epsilon_imag = epsilon_imag
            self.D_conductivity = epsilon_imag * 2 * np.pi * freq / epsilon_real
        else:
            self.epsilon_imag = 0
            self.D_conductivity = None # No conductivity if imaginary part is zero
        
        # Set pyramid material
        if material is not None:
            self.material = material
        else:
            set_material_obj(self, epsilon_real, epsilon_imag, freq)
            self.material = mp.Medium(epsilon=self.epsilon_real, D_conductivity=self.D_conductivity)
        
        # Set substrate parameters
        self._set_substrate_parameters(substrate_thickness, substrate_material, 
                                     substrate_epsilon_real, substrate_epsilon_imag)
        
        # Set PEC backing parameters
        if pec_thickness is None:
            self.pec_thickness = self.substrate_thickness if self.add_substrate else self.layer_thickness
        else:
            self.pec_thickness = pec_thickness
            
        print(f"PyramidalAbsorbers created: {self.num_pyramids} pyramids on {self.edges}")
        print(f"Base width: {max(self.x_base_width, self.y_base_width):.3f}, "
              f"Top width: {self.top_width:.3f}")
        print(f"Height: {self.height:.3f}, Layer thickness: {self.layer_thickness:.3f}")
        if self.add_substrate:
            print(f"Substrate: thickness={self.substrate_thickness:.3f}, "
                  f"extends_beyond={self.substrate_extends_beyond_pyramids}")
        if self.add_pec_backing:
            print(f"PEC backing: thickness={self.pec_thickness:.3f}, "
                  f"extends_beyond={self.pec_extends_beyond_substrate}")
            
    def _set_substrate_parameters(self, substrate_thickness, substrate_material, 
                                substrate_epsilon_real, substrate_epsilon_imag):
        """
        Set substrate parameters with proper defaults
        
        Parameters
        ----------
        substrate_thickness : float or None
            Thickness of substrate
        substrate_material : mp.Medium or None
            Material for substrate
        substrate_epsilon_real : float or None
            Real part of substrate permittivity
        substrate_epsilon_imag : float or None
            Imaginary part of substrate permittivity
        """
        if not self.add_substrate:
            return
            
        # Set substrate thickness
        if substrate_thickness is None:
            self.substrate_thickness = self.layer_thickness  # Same as layer thickness
        else:
            self.substrate_thickness = substrate_thickness
            
        # Set substrate material
        if substrate_material is not None:
            self.substrate_material = substrate_material
        else:
            # Use substrate-specific material properties if provided, otherwise use pyramid material
            if substrate_epsilon_real is not None or substrate_epsilon_imag is not None:
                sub_eps_real = substrate_epsilon_real if substrate_epsilon_real is not None else self.epsilon_real
                sub_eps_imag = substrate_epsilon_imag if substrate_epsilon_imag is not None else self.epsilon_imag
                sub_conductivity = sub_eps_imag * 2 * np.pi * self.freq / sub_eps_real
                self.substrate_material = mp.Medium(epsilon=sub_eps_real, D_conductivity=sub_conductivity)
            else:
                self.substrate_material = self.material  # Same as pyramid material

    def _set_pyramid_dimensions(self, height, layer_thickness):
        """
        Set pyramid dimensions based on height and layer_thickness parameters
        with proper priority handling
        
        Parameters
        ----------
        height : float or None
            Total height of pyramid
        layer_thickness : float or None
            Thickness of each layer
        """
        # Priority logic for determining dimensions
        if height is not None and layer_thickness is not None:
            # Both provided - height takes priority
            self.height = height
            self.layer_thickness = self.height / self.n_layers
            warnings.warn(f"Both height ({height}) and layer_thickness ({layer_thickness}) provided. "
                         f"Using height and calculating layer_thickness = {self.layer_thickness:.3f}")
            
        elif height is not None:
            # Only height provided
            self.height = height
            self.layer_thickness = self.height / self.n_layers
            
        elif layer_thickness is not None:
            # Only layer_thickness provided
            self.layer_thickness = layer_thickness
            self.height = self.layer_thickness * self.n_layers
            
        else:
            # Neither provided - calculate defaults based on base width
            max_base_width = max(self.x_base_width, self.y_base_width)
            width_diff = max_base_width - self.top_width
            
            # Default: make the pyramid depth proportional to the width difference
            # Use a reasonable aspect ratio
            self.layer_thickness = width_diff / (2 * self.n_layers)
            self.height = self.layer_thickness * self.n_layers
            
            warnings.warn(f"Neither height nor layer_thickness provided. "
                         f"Using defaults: height={self.height:.3f}, "
                         f"layer_thickness={self.layer_thickness:.3f}")

    def calculate_layer_width(self, layer_index, base_width):
        """
        Calculate the width of a specific layer based on linear interpolation
        between base_width and top_width
        
        Parameters
        ----------
        layer_index : int
            Index of the layer (0 = bottom/base, n_layers-1 = top)
        base_width : float
            Width of the base layer
            
        Returns
        -------
        float
            Width of the specified layer
        """
        # Linear interpolation from base_width to top_width
        progress = layer_index / (self.n_layers - 1) if self.n_layers > 1 else 0
        layer_width = base_width - progress * (base_width - self.top_width)
        return max(layer_width, 0)  # Ensure non-negative width

    def __str__(self):
        return f"{self.name}: {self.num_pyramids} pyramids on {self.edges}, height={self.height}"

    def assemble(self):
        """
        Assemble all pyramidal absorbers, substrates, and PEC backing
        
        Returns
        -------
        list
            List of all geometric objects
        """
        all_objects = []
        
        # First, create PEC backing if enabled (should be placed below substrate)
        if self.add_pec_backing:
            pec_blocks = self._create_pec_backing()
            all_objects.extend(pec_blocks)
            pec_count = len(pec_blocks)
        else:
            pec_count = 0
        
        # Then create substrates if enabled
        if self.add_substrate:
            substrates = self._create_substrates()
            all_objects.extend(substrates)
            substrate_count = len(substrates)
        else:
            substrate_count = 0
        
        # Finally create pyramids
        pyramids = self._create_pyramids()
        all_objects.extend(pyramids)
        
        # Print summary
        print(f"Assembled {len(pyramids)} pyramid blocks, {substrate_count} substrate blocks, "
              f"and {pec_count} PEC backing blocks")
        print(f"Total objects: {len(all_objects)}")
        
        return all_objects

    def _create_pec_backing(self):
        """
        Create PEC backing blocks for all edges with absorbers
        
        Returns
        -------
        list
            List of PEC backing block objects
        """
        pec_blocks = []
        
        # Bottom edge PEC backing
        if "bottom" in self.edges:
            # Determine width based on extension settings
            if self.pec_extends_beyond_substrate and self.add_substrate and self.substrate_extends_beyond_pyramids:
                pec_x_start = self.x_coverage_start - self.substrate_extension - self.pec_extension
                pec_x_end = self.x_coverage_end + self.substrate_extension + self.pec_extension
            elif self.pec_extends_beyond_substrate and self.add_substrate:
                pec_x_start = self.x_coverage_start - self.pec_extension
                pec_x_end = self.x_coverage_end + self.pec_extension
            elif self.pec_extends_beyond_substrate:
                pec_x_start = self.x_coverage_start - self.pec_extension
                pec_x_end = self.x_coverage_end + self.pec_extension
            else:
                pec_x_start = self.x_coverage_start
                pec_x_end = self.x_coverage_end
                
            pec_width = pec_x_end - pec_x_start
            pec_center_x = (pec_x_start + pec_x_end) / 2
            
            # Position PEC below substrate (if present) or below pyramids
            if self.add_substrate:
                pec_center_y = (-self.mpsat_sim.cell_size[1]/2 + 
                              (self.mpsat_sim.factor_dpml*self.mpsat_sim.dpml) - 
                              self.substrate_thickness - self.pec_thickness/2 + self.y_bottom_offset)
            else:
                pec_center_y = (-self.mpsat_sim.cell_size[1]/2 + 
                              (self.mpsat_sim.factor_dpml*self.mpsat_sim.dpml) - 
                              self.pec_thickness/2 + self.y_bottom_offset)
            
            pec_blocks.append(mp.Block(
                size=mp.Vector3(pec_width, self.pec_thickness, mp.inf),
                center=mp.Vector3(pec_center_x, pec_center_y),
                material=mp.perfect_electric_conductor
            ))
        
        # Top edge PEC backing
        if "top" in self.edges:
            # Determine width based on extension settings
            if self.pec_extends_beyond_substrate and self.add_substrate and self.substrate_extends_beyond_pyramids:
                pec_x_start = self.x_coverage_start - self.substrate_extension - self.pec_extension
                pec_x_end = self.x_coverage_end + self.substrate_extension + self.pec_extension
            elif self.pec_extends_beyond_substrate and self.add_substrate:
                pec_x_start = self.x_coverage_start - self.pec_extension
                pec_x_end = self.x_coverage_end + self.pec_extension
            elif self.pec_extends_beyond_substrate:
                pec_x_start = self.x_coverage_start - self.pec_extension
                pec_x_end = self.x_coverage_end + self.pec_extension
            else:
                pec_x_start = self.x_coverage_start
                pec_x_end = self.x_coverage_end
                
            pec_width = pec_x_end - pec_x_start
            pec_center_x = (pec_x_start + pec_x_end) / 2
            
            # Position PEC above substrate (if present) or above pyramids
            if self.add_substrate:
                pec_center_y = (self.mpsat_sim.cell_size[1]/2 - 
                             (self.mpsat_sim.factor_dpml*self.mpsat_sim.dpml) + 
                             self.substrate_thickness + self.pec_thickness/2 + self.y_top_offset)
            else:
                pec_center_y = (self.mpsat_sim.cell_size[1]/2 - 
                             (self.mpsat_sim.factor_dpml*self.mpsat_sim.dpml) + 
                             self.pec_thickness/2 + self.y_top_offset)
            
            pec_blocks.append(mp.Block(
                size=mp.Vector3(pec_width, self.pec_thickness, mp.inf),
                center=mp.Vector3(pec_center_x, pec_center_y),
                material=mp.perfect_electric_conductor
            ))
        
        # Left edge PEC backing
        if "left" in self.edges:
            # Determine width based on extension settings
            if self.pec_extends_beyond_substrate and self.add_substrate and self.substrate_extends_beyond_pyramids:
                pec_y_start = self.y_coverage_start - self.substrate_extension - self.pec_extension
                pec_y_end = self.y_coverage_end + self.substrate_extension + self.pec_extension
            elif self.pec_extends_beyond_substrate and self.add_substrate:
                pec_y_start = self.y_coverage_start - self.pec_extension
                pec_y_end = self.y_coverage_end + self.pec_extension
            elif self.pec_extends_beyond_substrate:
                pec_y_start = self.y_coverage_start - self.pec_extension
                pec_y_end = self.y_coverage_end + self.pec_extension
            else:
                pec_y_start = self.y_coverage_start
                pec_y_end = self.y_coverage_end
                
            pec_width = pec_y_end - pec_y_start
            pec_center_y = (pec_y_start + pec_y_end) / 2
            
            # FIX: Position PEC between PML boundary and substrate (or pyramids if no substrate)
            # PEC should be INSIDE the simulation, not outside
            if self.add_substrate:
                # PEC goes between PML and substrate
                pec_center_x = (-self.mpsat_sim.cell_size[0]/2 + 
                            (self.mpsat_sim.factor_dpml*self.mpsat_sim.dpml) + 
                            self.pec_thickness/2 + self.x_left_offset)
            else:
                # PEC goes between PML and pyramids
                pec_center_x = (-self.mpsat_sim.cell_size[0]/2 + 
                            (self.mpsat_sim.factor_dpml*self.mpsat_sim.dpml) + 
                            self.pec_thickness/2 + self.x_left_offset)
            
            pec_blocks.append(mp.Block(
                size=mp.Vector3(self.pec_thickness, pec_width, mp.inf),
                center=mp.Vector3(pec_center_x, pec_center_y),
                material=mp.perfect_electric_conductor
            ))
        
        # Right edge PEC backing
        if "right" in self.edges:
            # Determine width based on extension settings
            if self.pec_extends_beyond_substrate and self.add_substrate and self.substrate_extends_beyond_pyramids:
                pec_y_start = self.y_coverage_start - self.substrate_extension - self.pec_extension
                pec_y_end = self.y_coverage_end + self.substrate_extension + self.pec_extension
            elif self.pec_extends_beyond_substrate and self.add_substrate:
                pec_y_start = self.y_coverage_start - self.pec_extension
                pec_y_end = self.y_coverage_end + self.pec_extension
            elif self.pec_extends_beyond_substrate:
                pec_y_start = self.y_coverage_start - self.pec_extension
                pec_y_end = self.y_coverage_end + self.pec_extension
            else:
                pec_y_start = self.y_coverage_start
                pec_y_end = self.y_coverage_end
                
            pec_width = pec_y_end - pec_y_start
            pec_center_y = (pec_y_start + pec_y_end) / 2
            
            # FIX: Position PEC INSIDE the simulation, not outside
            # PEC goes between PML boundary and absorbers (substrate or pyramids)
            if self.add_substrate:
                # PEC between PML and substrate - SUBTRACT to go inward
                pec_center_x = (self.mpsat_sim.cell_size[0]/2 - 
                            (self.mpsat_sim.factor_dpml*self.mpsat_sim.dpml) - 
                            self.pec_thickness/2 + self.x_right_offset)
            else:
                # PEC between PML and pyramids - SUBTRACT to go inward
                pec_center_x = (self.mpsat_sim.cell_size[0]/2 - 
                            (self.mpsat_sim.factor_dpml*self.mpsat_sim.dpml) - 
                            self.pec_thickness/2 + self.x_right_offset)
            
            pec_blocks.append(mp.Block(
                size=mp.Vector3(self.pec_thickness, pec_width, mp.inf),
                center=mp.Vector3(pec_center_x, pec_center_y),
                material=mp.perfect_electric_conductor
            ))
        
        return pec_blocks

    def _create_pyramids(self):
        """
        Create pyramid blocks for all edges
        
        Returns
        -------
        list
            List of pyramid block objects
        """
        pyramids = []
        
        # Bottom edge pyramids (pointing upward)
        if "bottom" in self.edges:
            for j in range(self.num_pyramids):
                pyramid_center_x = self.x_coverage_start + (j + 0.5) * self.x_base_width
                for i in range(self.n_layers):
                    layer_width = self.calculate_layer_width(i, self.x_base_width)
                    if layer_width <= 0:
                        continue
                    
                    # FIX: Start pyramids AFTER PEC and substrate
                    base_y = (-self.mpsat_sim.cell_size[1]/2 + 
                            (self.mpsat_sim.factor_dpml*self.mpsat_sim.dpml))
                    
                    if self.add_pec_backing:
                        base_y += self.pec_thickness
                    if self.add_substrate:
                        base_y += self.substrate_thickness
                        
                    layer_center_y = base_y + i * self.layer_thickness + self.layer_thickness/2 + self.y_bottom_offset
                    
                    pyramids.append(mp.Block(
                        size=mp.Vector3(layer_width, self.layer_thickness, mp.inf),
                        center=mp.Vector3(pyramid_center_x, layer_center_y),
                        material=self.material
                    ))
        
        # Top edge pyramids (pointing downward)
        if "top" in self.edges:
            for j in range(self.num_pyramids):
                pyramid_center_x = self.x_coverage_start + (j + 0.5) * self.x_base_width
                for i in range(self.n_layers):
                    layer_width = self.calculate_layer_width(i, self.x_base_width)
                    if layer_width <= 0:
                        continue
                    
                    # FIX: Start pyramids AFTER PEC and substrate
                    base_y = (self.mpsat_sim.cell_size[1]/2 - 
                            (self.mpsat_sim.factor_dpml*self.mpsat_sim.dpml))
                    
                    if self.add_pec_backing:
                        base_y -= self.pec_thickness
                    if self.add_substrate:
                        base_y -= self.substrate_thickness
                        
                    layer_center_y = base_y - i * self.layer_thickness - self.layer_thickness/2 + self.y_top_offset
                    
                    pyramids.append(mp.Block(
                        size=mp.Vector3(layer_width, self.layer_thickness, mp.inf),
                        center=mp.Vector3(pyramid_center_x, layer_center_y),
                        material=self.material
                    ))
        
        # Left edge pyramids (pointing rightward)
        if "left" in self.edges:
            for j in range(self.num_pyramids):
                pyramid_center_y = self.y_coverage_start + (j + 0.5) * self.y_base_width
                for i in range(self.n_layers):
                    layer_width = self.calculate_layer_width(i, self.y_base_width)
                    if layer_width <= 0:
                        continue
                    
                    # FIX: Start pyramids AFTER PEC and substrate
                    base_x = (-self.mpsat_sim.cell_size[0]/2 + 
                            (self.mpsat_sim.factor_dpml*self.mpsat_sim.dpml))
                    
                    if self.add_pec_backing:
                        base_x += self.pec_thickness
                    if self.add_substrate:
                        base_x += self.substrate_thickness
                        
                    layer_center_x = base_x + i * self.layer_thickness + self.layer_thickness/2 + self.x_left_offset
                    
                    pyramids.append(mp.Block(
                        size=mp.Vector3(self.layer_thickness, layer_width, mp.inf),
                        center=mp.Vector3(layer_center_x, pyramid_center_y),
                        material=self.material
                    ))
        
        # Right edge pyramids (pointing leftward)
        if "right" in self.edges:
            for j in range(self.num_pyramids):
                pyramid_center_y = self.y_coverage_start + (j + 0.5) * self.y_base_width
                for i in range(self.n_layers):
                    layer_width = self.calculate_layer_width(i, self.y_base_width)
                    if layer_width <= 0:
                        continue
                    
                    # FIX: Start pyramids AFTER PEC and substrate
                    base_x = (self.mpsat_sim.cell_size[0]/2 - 
                            (self.mpsat_sim.factor_dpml*self.mpsat_sim.dpml))
                    
                    if self.add_pec_backing:
                        base_x -= self.pec_thickness
                    if self.add_substrate:
                        base_x -= self.substrate_thickness
                        
                    layer_center_x = base_x - i * self.layer_thickness - self.layer_thickness/2 + self.x_right_offset
                    
                    pyramids.append(mp.Block(
                        size=mp.Vector3(self.layer_thickness, layer_width, mp.inf),
                        center=mp.Vector3(layer_center_x, pyramid_center_y),
                        material=self.material
                    ))
        
        return pyramids

    def _create_pyramids(self):
        """
        Create pyramid blocks for all edges
        
        Returns
        -------
        list
            List of pyramid block objects
        """
        pyramids = []
        
        # Bottom edge pyramids (pointing upward)
        if "bottom" in self.edges:
            for j in range(self.num_pyramids):
                pyramid_center_x = self.x_coverage_start + (j + 0.5) * self.x_base_width
                for i in range(self.n_layers):
                    layer_width = self.calculate_layer_width(i, self.x_base_width)
                    if layer_width <= 0:
                        continue  # Skip layers with zero or negative width
                        
                    layer_center_y = (-self.mpsat_sim.cell_size[1]/2 + 
                                    (self.mpsat_sim.factor_dpml*self.mpsat_sim.dpml) + 
                                    i * self.layer_thickness + self.layer_thickness/2 + self.y_bottom_offset)
                    
                    pyramids.append(mp.Block(
                        size=mp.Vector3(layer_width, self.layer_thickness, mp.inf),
                        center=mp.Vector3(pyramid_center_x, layer_center_y),
                        material=self.material
                    ))
        
        # Top edge pyramids (pointing downward)
        if "top" in self.edges:
            for j in range(self.num_pyramids):
                pyramid_center_x = self.x_coverage_start + (j + 0.5) * self.x_base_width
                for i in range(self.n_layers):
                    layer_width = self.calculate_layer_width(i, self.x_base_width)
                    if layer_width <= 0:
                        continue
                        
                    layer_center_y = (self.mpsat_sim.cell_size[1]/2 - 
                                    (self.mpsat_sim.factor_dpml*self.mpsat_sim.dpml) - 
                                    i * self.layer_thickness - self.layer_thickness/2 + self.y_top_offset)
                    
                    pyramids.append(mp.Block(
                        size=mp.Vector3(layer_width, self.layer_thickness, mp.inf),
                        center=mp.Vector3(pyramid_center_x, layer_center_y),
                        material=self.material
                    ))
        
        # Left edge pyramids (pointing rightward)
        if "left" in self.edges:
            for j in range(self.num_pyramids):
                pyramid_center_y = self.y_coverage_start + (j + 0.5) * self.y_base_width
                for i in range(self.n_layers):
                    layer_width = self.calculate_layer_width(i, self.y_base_width)
                    if layer_width <= 0:
                        continue
                        
                    layer_center_x = (-self.mpsat_sim.cell_size[0]/2 + 
                                    (self.mpsat_sim.factor_dpml*self.mpsat_sim.dpml) + 
                                    i * self.layer_thickness + self.layer_thickness/2 + self.x_left_offset)
                    
                    pyramids.append(mp.Block(
                        size=mp.Vector3(self.layer_thickness, layer_width, mp.inf),
                        center=mp.Vector3(layer_center_x, pyramid_center_y),
                        material=self.material
                    ))
        
        # Right edge pyramids (pointing leftward)
        if "right" in self.edges:
            for j in range(self.num_pyramids):
                pyramid_center_y = self.y_coverage_start + (j + 0.5) * self.y_base_width
                for i in range(self.n_layers):
                    layer_width = self.calculate_layer_width(i, self.y_base_width)
                    if layer_width <= 0:
                        continue
                        
                    layer_center_x = (self.mpsat_sim.cell_size[0]/2 - 
                                    (self.mpsat_sim.factor_dpml*self.mpsat_sim.dpml) - 
                                    i * self.layer_thickness - self.layer_thickness/2 + self.x_right_offset)
                    
                    pyramids.append(mp.Block(
                        size=mp.Vector3(self.layer_thickness, layer_width, mp.inf),
                        center=mp.Vector3(layer_center_x, pyramid_center_y),
                        material=self.material
                    ))
        
        return pyramids
    

# # ~ FOREBAFFLE CLASS
class Forebaffle(object):
    '''
    Class defining a triangular forebaffle structure, adjusts the size of the cells and source based on the angle of the forebaffle to ensure proper geometry.
    '''
    def __init__(self,
                 window_diameter,
                 cell_size,
                 source_center,
                 source_size,
                 window_lens1_distance,
                 cellx_sourcex_distance,
                 sourcex_FB_vertex_distance,
                 optics_tube_length,
                 boundary_thickness,
                 fb_angle_degrees,
                 fb_hypotenuse,
                 fb_location,
                 fb_material=mp.perfect_electric_conductor,
                 fb_epsilon_real=5.4,
                 fb_epsilon_imag=0.0,
                 sim_type= 'TFWD'
                ):
        '''
        Defines a triangular forebaffle structure with a specific angle
        Parameters
        ----------
        window_diameter : float
            Diameter of the window after the forebaffle
        cell_size : float
            Size of the simulation cell
        source_center : float
            Center position of the source
        source_size : float
            Size of the source
        window_lens1_distance : float
            Distance from the left surface of the window to the left surface of lens 1
        cellx_sourcex_distance : float
            Distance from the left edge of the cell to the left edge of the source
        sourcex_FB_vertex_distance : float
            Distance from the left edge of the source to the vertex of the forebaffle
        optics_tube_length : float
            Length of the optics tube
        boundary_thickness : list [Xlow, Xhigh, Ylow, Yhigh]
            Thickness of the simulation boundaries
        fb_angle_degrees : float
            Angle of the forebaffle in degrees
        fb_hypotenuse : float
            Length of the hypotenuse of the forebaffle triangle
        fb_location: str
            Type of forebaffle location, options are 'up' and 'down'
        fb_material : mp.Medium
            Material for the forebaffle
        fb_epsilon_real : float
            Real part of permittivity for the forebaffle material
        fb_epsilon_imag : float
            Imaginary part of permittivity for the forebaffle material¨
        sim_type: str
            Type of simulation (TFWD or TR) - useful for source center
        '''
        self.window_diameter = window_diameter
        self.cell_size = cell_size
        self.source_center = source_center
        self.source_size = source_size
        self.window_lens1_distance = window_lens1_distance
        self.cellx_sourcex_distance = cellx_sourcex_distance
        self.sourcex_FB_vertex_distance = sourcex_FB_vertex_distance
        self.optics_tube_length = optics_tube_length
        self.fb_angle_degrees = fb_angle_degrees
        self.fb_angle_radians = np.radians(fb_angle_degrees)
        self.fb_hypotenuse = fb_hypotenuse
        
        # Calculate the expected height of the forebaffle based on the angle and hypotenuse
        self.fb_height = np.abs(self.fb_hypotenuse * np.sin(self.fb_angle_radians))

        # Calculate the base length of the forebaffle
        self.fb_base_length = np.sqrt(self.fb_hypotenuse**2 - self.fb_height**2)

        # Adjust the cell size based on the forebaffle dimensions
        self.adjusted_cell_size = (boundary_thickness[0] + self.cellx_sourcex_distance + self.sourcex_FB_vertex_distance + self.fb_base_length + self.optics_tube_length + boundary_thickness[1],
                                   2*self.fb_height + window_diameter + boundary_thickness[2] + boundary_thickness[3],              
                                   0)
    
        # Calculate the adjustments on the centre of Source
        self.source_center_adjusted = (- self.adjusted_cell_size[0]/2 + boundary_thickness[0] + self.cellx_sourcex_distance, 
                                       0,
                                       0)
        
        # Calculate the size of the source based on the new cell size and forebaffle parameters
        self.source_size_adjusted = (0, #self.adjusted_cell_size[0] - boundary_thickness[0] - boundary_thickness[1],
                                     self.adjusted_cell_size[1] - boundary_thickness[2] - boundary_thickness[3],
                                     0)
        

        if fb_location == 'down':
            # Calculate the vertex A of the forebaffle
            self.fb_vertex_A = (self.source_center_adjusted[0] + self.sourcex_FB_vertex_distance,
                                -self.adjusted_cell_size[1]/2 + boundary_thickness[2],
                                0)


            # Calculate the Vertex B of the forebaffle
            self.fb_vertex_B = (self.fb_vertex_A[0] + self.fb_base_length,
                                self.fb_vertex_A[1] + self.fb_height,
                                0)

            # Calculate the Vertex C of the forebaffle
            self.fb_vertex_C = (self.fb_vertex_A[0] + self.fb_base_length,
                                self.fb_vertex_A[1],
                                0)
            
            #! NEED TO MAKE THE HEIGHT CONSTANT REGARDLESS OF ANGLE by increasing OR decreasing the size of the cell and shifting the forebaffle accordingly so theyintersect at -window_diameter/2
            # Check if the Vertex B exceeds or short of the window_diameter/2
            if self.fb_vertex_B[1] > -(self.window_diameter/2):
                diff = self.fb_vertex_B[1] - (self.window_diameter/2)
                # Adjust the cell size, source size, and vertex positions accordingly
                self.adjusted_cell_size = (self.adjusted_cell_size[0],
                                           self.adjusted_cell_size[1] - diff,
                                             0)
                self.source_size_adjusted = (self.source_size_adjusted[0],
                                            self.source_size_adjusted[1] - diff,
                                                0)
                
                self.fb_vertex_A = (self.fb_vertex_A[0],
                                    self.fb_vertex_A[1] - diff,
                                    0)
                self.fb_vertex_B = (self.fb_vertex_B[0],
                                    self.fb_vertex_B[1] - diff,
                                    0)
                self.fb_vertex_C = (self.fb_vertex_C[0],
                                    self.fb_vertex_C[1] - diff,
                                    0)
                
            elif self.fb_vertex_B[1] < -(self.window_diameter/2):
                diff = -(self.window_diameter/2) - self.fb_vertex_B[1]
                # Adjust the cell size, source size, and vertex positions accordingly
                self.adjusted_cell_size = (self.adjusted_cell_size[0],
                                           self.adjusted_cell_size[1] + diff,
                                             0)
                self.source_size_adjusted = (self.source_size_adjusted[0],
                                            self.source_size_adjusted[1] + diff,
                                                0)
                self.fb_vertex_A = (self.fb_vertex_A[0],
                                    self.fb_vertex_A[1] + diff,
                                    0)
                self.fb_vertex_B = (self.fb_vertex_B[0],
                                    self.fb_vertex_B[1] + diff,
                                    0)
                self.fb_vertex_C = (self.fb_vertex_C[0],
                                    self.fb_vertex_C[1] + diff,
                                    0)

        elif fb_location == 'up':
            # Calculate the vertex A of the forebaffle
            self.fb_vertex_A = (self.source_center_adjusted[0] + self.sourcex_FB_vertex_distance,
                                self.adjusted_cell_size[1]/2 - boundary_thickness[3],
                                0)


            # Calculate the Vertex B of the forebaffle
            self.fb_vertex_B = (self.fb_vertex_A[0] + self.fb_base_length,
                                self.fb_vertex_A[1] - self.fb_height,
                                0)
            # Calculate the Vertex C of the forebaffle
            self.fb_vertex_C = (self.fb_vertex_A[0] + self.fb_base_length,
                                self.fb_vertex_A[1],
                                0)
            
            #! NEED TO MAKE THE HEIGHT CONSTANT REGARDLESS OF ANGLE by increasing OR decreasing the size of the cell and shifting the forebaffle accordingly so theyintersect at +window_diameter/2
            # Check if the Vertex B exceeds or short of the window_diameter/2
            if self.fb_vertex_B[1] < (self.window_diameter/2):
                diff = (self.window_diameter/2) - self.fb_vertex_B[1]
                # Adjust the cell size, source size, and vertex positions accordingly
                self.adjusted_cell_size = (self.adjusted_cell_size[0],
                                           self.adjusted_cell_size[1] + diff,
                                                0)
                
                self.source_size_adjusted = (self.source_size_adjusted[0],
                                            self.source_size_adjusted[1] + diff,
                                                0)
                self.fb_vertex_A = (self.fb_vertex_A[0],
                                    self.fb_vertex_A[1] + diff,
                                    0)
                
                self.fb_vertex_B = (self.fb_vertex_B[0],
                                    self.fb_vertex_B[1] + diff,
                                    0)
                self.fb_vertex_C = (self.fb_vertex_C[0],
                                    self.fb_vertex_C[1] + diff,
                                    0)
                
            elif self.fb_vertex_B[1] > (self.window_diameter/2):
                diff = self.fb_vertex_B[1] - (self.window_diameter/2)
                # Adjust the cell size, source size, and vertex positions accordingly
                self.adjusted_cell_size = (self.adjusted_cell_size[0],
                                           self.adjusted_cell_size[1] - diff,
                                                0)
                
                self.source_size_adjusted = (self.source_size_adjusted[0],
                                            self.source_size_adjusted[1] - diff,
                                                0)
                self.fb_vertex_A = (self.fb_vertex_A[0],
                                    self.fb_vertex_A[1] - diff,
                                    0)
                self.fb_vertex_B = (self.fb_vertex_B[0],
                                    self.fb_vertex_B[1] - diff,
                                    0)
                self.fb_vertex_C = (self.fb_vertex_C[0],
                                    self.fb_vertex_C[1] - diff,
                                    0)
        else:
            raise ValueError("fb_location must be either 'up' or 'down'")

        #Set the material for the forebaffle
        if fb_material is not None:
            self.material = fb_material
        else:
            if fb_epsilon_imag != 0:
                self.epsilon_real = fb_epsilon_real
                self.epsilon_imag = fb_epsilon_imag
                self.conductivity = fb_epsilon_imag * 2 * np.pi * (1/3) / fb_epsilon_real
                self.material = mp.Medium(epsilon=fb_epsilon_real, D_conductivity=self.conductivity)
            else:
                self.material = mp.Medium(epsilon=fb_epsilon_real)

        # Print forebaffle creation details
        print(f"Forebaffle created at angle={self.fb_angle_degrees}°, "
              f"hypotenuse={self.fb_hypotenuse}, "
              f"base_length={self.fb_base_length}, height={self.fb_height}",
              f"At vertices: A{self.fb_vertex_A}, B{self.fb_vertex_B}, C{self.fb_vertex_C}")

    def assemble(self):
        """
        Assemble the forebaffle triangle

        Returns
        -------
        mp.Prism
            Meep prism object representing the forebaffle
        """
        # Create the prism
        forebaffle = mp.Prism(
            vertices=[mp.Vector3(*self.fb_vertex_A),
                      mp.Vector3(*self.fb_vertex_B),
                      mp.Vector3(*self.fb_vertex_C)],
            height=mp.inf,
            axis=mp.Vector3(0, 0, 1),
            material = self.material
        )

        print(f"Forebaffle assembled with vertices at {self.fb_vertex_A}, {self.fb_vertex_B}, and {self.fb_vertex_C}")
        print(f"Adjusted cell size: {self.adjusted_cell_size}")
        print(f"Adjusted source center: {self.source_center_adjusted}")
        print(f"Adjusted source size: {self.source_size_adjusted}")
        
        # Calculate AB length for verification
        AB_length = np.sqrt((self.fb_vertex_B[0] - self.fb_vertex_A[0])**2 + (self.fb_vertex_B[1] - self.fb_vertex_A[1])**2)
        print(f"Forebaffle AB length: {AB_length}, Expected hypotenuse: {self.fb_hypotenuse}")

        return forebaffle, self.adjusted_cell_size, self.source_center_adjusted, self.source_size_adjusted, self.fb_base_length, self.fb_height, self.fb_vertex_A, self.fb_vertex_B, self.fb_vertex_C


# class Forebaffle(object):
#     '''
#     Class defining a triangular forebaffle structure.
#     '''
#     def __init__(self,
#                  mpsat_sim,
#                  angle_degrees=30,
#                  x_vertex=None,
#                  y_vertex=None,
#                  leg1_length=40,
#                  leg2_length=70,
#                  height=30,
#                  material=None,
#                  epsilon=5.4,
#                  epsilon_imag=0,
#                  freq=1/3,
#                  name=None):
#         '''
#         Defines a triangular forebaffle structure with a specific angle

#         Parameters
#         ----------
#         mpsat_sim : MEEPSAT
#             MEEPSAT simulation object
#         angle_degrees : float, optional
#             Angle of the forebaffle in degrees (default: 30)
#         x_vertex : float, optional
#             X-coordinate of the vertex (default: -300)
#         y_vertex : float, optional
#             Y-coordinate of the vertex (default: bottom of simulation cell)
#         leg1_length : float, optional
#             Length of the first (horizontal) leg (default: 40)
#         leg2_length : float, optional
#             Length of the second (angled) leg (default: 70)
#         height : float, optional
#             Height of the forebaffle in z-direction (default: 30)
#         material : mp.Medium, optional
#             Material for the forebaffle (overrides epsilon if provided)
#         epsilon : float, optional
#             Permittivity of the material (default: 5.4)
#         epsilon_imag : float, optional
#             Imaginary part of permittivity (default: 0)
#         freq : float, optional
#             Frequency for material properties (default: 1/3)
#         name : str, optional
#             Name of the object (default: None)
#         '''
#         # Sims object
#         self.mpsat_sim = set_sims_obj(self, mpsat_sim)
        
#         # Basic parameters
#         self.name = name if name else "Forebaffle"
#         self.object_type = 'Forebaffle'
        
#         # Geometry parameters
#         self.angle_degrees = angle_degrees
#         self.angle_radians = np.radians(angle_degrees)
#         self.x_vertex = x_vertex if x_vertex is not None else -300
#         self.y_vertex = y_vertex if y_vertex is not None else -self.mpsat_sim.cell_size[1]/2
#         self.leg1_length = leg1_length
#         self.leg2_length = leg2_length
#         self.height = height
        
#         # Material properties
#         if material is not None:
#             self.material = material
#         else:
#             # Check if imaginary part is provided
#             if epsilon_imag != 0:
#                 self.epsilon_real = epsilon
#                 self.epsilon_imag = epsilon_imag
#                 self.conductivity = epsilon_imag * 2 * np.pi * freq / epsilon
#                 self.material = mp.Medium(epsilon=epsilon, D_conductivity=self.conductivity)
#             else:
#                 self.material = mp.Medium(epsilon=epsilon)
            
#         print(f"Forebaffle created: angle={self.angle_degrees}°, "
#               f"position=({self.x_vertex}, {self.y_vertex}), "
#               f"legs=({self.leg1_length}, {self.leg2_length})")

#     def __str__(self):
#         return f"{self.name}: angle={self.angle_degrees}°, height={self.height}"

#     def assemble(self):
#         """
#         Assemble the forebaffle prism
        
#         Returns
#         -------
#         mp.Prism
#             Meep prism object representing the forebaffle
#         """
#         # Calculate the coordinates of the three vertices
#         v1 = mp.Vector3(self.x_vertex, self.y_vertex)  # Vertex where angle is measured
#         v2 = mp.Vector3(self.x_vertex + self.leg1_length, self.y_vertex)  # End of first leg (horizontal)
#         v3 = mp.Vector3(self.x_vertex + self.leg2_length * np.cos(self.angle_radians),
#                       self.y_vertex + self.leg2_length * np.sin(self.angle_radians))  # End of second leg
        
#         # Create the prism
#         forebaffle = mp.Prism(
#             vertices=[v1, v2, v3],
#             height=self.height,
#             axis=mp.Vector3(0, 0, 1),
#             material=self.material
#         )
        
#         print(f"Forebaffle assembled with vertices at {v1}, {v2}, and {v3}")
        
#         return forebaffle


