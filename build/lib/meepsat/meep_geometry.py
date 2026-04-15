from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional, Literal, Dict, Any
import logging
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

# MeepSAT functions
import meepsat.helpers as exf
import meepsat.meshing as mesh

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

### -- RELEVANT FUNCTIONS FOR BROADBAND SOURCE -- ###
def set_broadband_freq_wvl(self, central_wvl=None, wvl_min=None, wvl_max=None):
    """
    Set the frequency and wavelength for the broadband source

    Parameters
    ----------
    self : object
        Class object

    central_wvl : float
        Central wavelength of the source in MEEP units (default : None)

    wvl_min : float
        Minimum wavelength of the source in MEEP units (default : None)

    wvl_max : float
        Maximum wavelength of the source in MEEP units (default : None)

    Returns
    -------
    self.freq : float
        Frequency of the source in MEEP units

    self.wvl : float
        Wavelength of the source in MEEP units
    """
    self.wvl_min = wvl_min
    self.wvl_max = wvl_max
    
    if central_wvl is None:
        self.central_wvl = (self.wvl_min + self.wvl_max) / 2

    self.center_freq = 1/self.central_wvl
    self.freq_width = 1/self.wvl_min - 1/self.wvl_max

    return self.freq_width, self.center_freq

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
                 rot_axis= 'x',
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

        rot_axis : str
            Axis around which the source is rotated (default : 'x')

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
        # Rotation axis of the source
        self.rot_axis = rot_axis
        # Additional arguments for both mp.Source() and mp.ContinuousSource()
        self.additional_args = kwargs

        print("Source object created with the following parameters:")
        print("Center: ", self.center)
        print("Size: ", self.size)
        print("Component: ", self.component)
        print("Frequency: ", self.freq)
        print("Wavelength: ", self.wvl)
        print("Angle: ", self.rot_angle)
        print("Rotation axis: ", self.rot_axis)
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
        
        if self.rot_axis=='x':
            k = mp.Vector3(2*np.pi*np.cos(self.rot_angle)/self.wvl,
                        2*np.pi*np.sin(self.rot_angle)/self.wvl,
                        0)
        elif self.rot_axis=='y':
            k = mp.Vector3(2*np.pi*np.sin(self.rot_angle)/self.wvl,
                        2*np.pi*np.cos(self.rot_angle)/self.wvl,
                        0)
        else:
            raise ValueError("Invalid Rotation axis. Choose either 'x' OR 'y'")
        
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



# ~ BROADBAND PLANE WAVES
class BroadbandPlaneWaveSource():
    """
    Class defining the broadband plane waves source
    """
    def __init__(self,
                 mpsat_sim,
                 center = None,
                 size = None,
                 component = None,
                 central_wvl = None,
                 wvl_min= None,
                 wvl_max= None,
                 angle = 0,
                 rot_axis= 'x',
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

        rot_axis : str
            Axis around which the source is rotated (default : 'x')

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
        self.wvl_min, self.wvl_max, self.central_wvl = wvl_min, wvl_max, central_wvl
        self.freq_width, self.center_freq = set_broadband_freq_wvl(self, central_wvl, wvl_min, wvl_max)
        # Angle of the source
        self.rot_angle = set_source_angle(self, angle)       
        # Rotation axis of the source
        self.rot_axis = rot_axis
        # Additional arguments for both mp.Source() and mp.ContinuousSource()
        self.additional_args = kwargs

        print("Source object created with the following parameters:")
        print("Center: ", self.center)
        print("Size: ", self.size)
        print("Component: ", self.component)
        print("Central Frequency and Freq Width: ", self.center_freq, self.freq_width)
        print("Wavelength Range and Central Wavelength:", self.wvl_min, self.wvl_max, self.central_wvl)
        print("Angle: ", self.rot_angle)
        print("Rotation axis: ", self.rot_axis)
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
        
        if self.rot_axis=='x':
            k = mp.Vector3(2*np.pi*np.cos(self.rot_angle)/(1/self.wvl_max),
                        2*np.pi*np.sin(self.rot_angle)/(1/self.wvl_max),
                        0)
        elif self.rot_axis=='y':
            k = mp.Vector3(2*np.pi*np.sin(self.rot_angle)/(1/self.wvl_max),
                        2*np.pi*np.cos(self.rot_angle)/(1/self.wvl_max),
                        0)
        else:
            raise ValueError("Invalid Rotation axis. Choose either 'x' OR 'y'")
        
        return np.exp(1j* k.dot(P))

    
    def assemble(self):
        """
        Return Broadband planewave Pulse
        """
        if self.additional_args is not None:
            source_filtered_kwrg = exf.filter_dict(self.additional_args, mp.Source)
            print("Additional arguments for the Source: ", source_filtered_kwrg)
            source_type_filtered_kwrg = exf.filter_dict(self.additional_args, mp.GaussianSource)
            print("Additional arguments for the GaussianSource: ", source_type_filtered_kwrg)

            source = mp.Source(mp.GaussianSource(frequency= self.center_freq,
                                                fwidth= 2*self.freq_width,
                                                **source_type_filtered_kwrg),
                            center= self.center,
                            size= self.size,
                            component=self.component,
                            # amp_func=self.amp_func,
                            **source_filtered_kwrg)
            
        else:
            source = mp.Source(mp.GaussianSource(frequency= self.center_freq,
                                                fwidth= 2*self.freq_width),
                            center= self.center,
                            size= self.size,
                            component=self.component)#,
                            # amp_func=self.amp_func)
        
        print("Broadband plane waves source assembled!")
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
               rot_axis='z',
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
        print(f"Rotating block by {angle}° around {rot_axis}-axis")


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
            Type of the aperture stop: 
                circular, square, arrow, etc.
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
            self.pos_x = pos_x #- self.mpsat_sim.cell.x/2
            self.pos_y = None
            self.orientation = 'vertical'  # Blocks oriented vertically (along y-axis)
        else:
            # Convert the pos_y in (0,y) coordinate system to (-y/2, y/2)
            self.pos_y = pos_y #- self.mpsat_sim.cell.y/2
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
    

#~ OLD PYRAMIDAL ASBORBE CLASS
class PyramidalAbsorbers(object):
    '''
    Class defining pyramidal absorbers along the edges of the simulation cell.
    '''
    def __init__(self,
                 mpsat_sim,
                 num_pyramids,
                 add_pyramids=True,
                 base_width=None,
                 top_width=0,
                 height=None,
                 layer_thickness=None,
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
        
        # self.mpsat_sim = mpsat_sim
        # Basic parameters
        self.name = name if name else "Pyramidal Absorbers"
        self.object_type = 'PyramidalAbsorber'
        self.num_pyramids = num_pyramids
        self.add_pyramids = add_pyramids
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
            print(f"Assembled {len(pec_blocks)} PEC backing blocks")
            all_objects.extend(pec_blocks)
            pec_count = len(pec_blocks)
        else:
            pec_count = 0
        
        # Then create substrates if enabled
        if self.add_substrate:
            substrates = self._create_substrates()
            print(f"Assembled {len(substrates)} substrate blocks")
            all_objects.extend(substrates)
            substrate_count = len(substrates)
        else:
            substrate_count = 0
        
        # Finally create pyramids
        if self.add_pyramids:
            pyramids = self._create_pyramids()
            print(f"Assembled {len(pyramids)} pyramid blocks")
            all_objects.extend(pyramids)
        
        # Print summary
        # print(f"Assembled {len(pyramids)} pyramid blocks, {substrate_count} substrate blocks, "
        #       f"and {pec_count} PEC backing blocks")
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
            # if self.add_substrate:
            #     pec_center_y = (-self.mpsat_sim.cell_size[1]/2 + 
            #                   (self.mpsat_sim.factor_dpml*self.mpsat_sim.dpml) + 
            #                   self.substrate_thickness + self.pec_thickness/2 + self.y_bottom_offset)
            # else:
            #     pec_center_y = (-self.mpsat_sim.cell_size[1]/2 + 
            #                   (self.mpsat_sim.factor_dpml*self.mpsat_sim.dpml) + 
            #                   self.pec_thickness/2 + self.y_bottom_offset)
            
            pec_center_y = (-self.mpsat_sim.cell_size[1]/2 + 
                      (self.mpsat_sim.factor_dpml*self.mpsat_sim.dpml) + 
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
            
            # # Position PEC above substrate (if present) or above pyramids
            # if self.add_substrate:
            #     pec_center_y = (self.mpsat_sim.cell_size[1]/2 - 
            #                  (self.mpsat_sim.factor_dpml*self.mpsat_sim.dpml) + 
            #                  self.substrate_thickness + self.pec_thickness/2 + self.y_top_offset)
            # else:
            #     pec_center_y = (self.mpsat_sim.cell_size[1]/2 - 
            #                  (self.mpsat_sim.factor_dpml*self.mpsat_sim.dpml) + 
            #                  self.pec_thickness/2 + self.y_top_offset)
            
            pec_center_y = (self.mpsat_sim.cell_size[1]/2 - 
                      (self.mpsat_sim.factor_dpml*self.mpsat_sim.dpml) - 
                      self.pec_thickness/2 - self.y_bottom_offset)
            
            
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
            
            # # FIX: Position PEC between PML boundary and substrate (or pyramids if no substrate)
            # # PEC should be INSIDE the simulation, not outside
            # if self.add_substrate:
            #     # PEC goes between PML and substrate (behind substrate)
            #     pec_center_x = (-self.mpsat_sim.cell_size[0]/2 + 
            #                 (self.mpsat_sim.factor_dpml*self.mpsat_sim.dpml) - 
            #                 self.substrate_thickness - self.pec_thickness/2 + self.x_left_offset)
            # else:
            #     # PEC goes between PML and pyramids (no substrate)
            #     pec_center_x = (-self.mpsat_sim.cell_size[0]/2 + 
            #                 (self.mpsat_sim.factor_dpml*self.mpsat_sim.dpml) - 
            #                 self.pec_thickness/2 + self.x_left_offset)
            
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
            
            # # FIX: Position PEC INSIDE the simulation, not outside
            # # PEC goes between PML boundary and absorbers (substrate or pyramids)
            # if self.add_substrate:
            #     # PEC between PML and substrate - behind the substrate
            #     pec_center_x = (self.mpsat_sim.cell_size[0]/2 - 
            #                 (self.mpsat_sim.factor_dpml*self.mpsat_sim.dpml) + 
            #                 self.substrate_thickness + self.pec_thickness/2 + self.x_right_offset)
            # else:
            #     # PEC between PML and pyramids - no substrate
            #     pec_center_x = (self.mpsat_sim.cell_size[0]/2 - 
            #                 (self.mpsat_sim.factor_dpml*self.mpsat_sim.dpml) + 
            #                 self.pec_thickness/2 + self.x_right_offset)
            
            pec_center_x = (self.mpsat_sim.cell_size[0]/2 - 
                        (self.mpsat_sim.factor_dpml*self.mpsat_sim.dpml) - 
                        self.pec_thickness/2 - self.x_left_offset)
            
            pec_blocks.append(mp.Block(
                size=mp.Vector3(self.pec_thickness, pec_width, mp.inf),
                center=mp.Vector3(pec_center_x, pec_center_y),
                material=mp.perfect_electric_conductor
            ))
        
        return pec_blocks

    def _create_substrates(self):
        """
        Create substrate blocks for all edges
        
        Returns
        -------
        list
            List of substrate block objects
        """
        substrates = []
        
        # Bottom edge substrates
        if "bottom" in self.edges:
            if self.substrate_extends_beyond_pyramids:
                substrate_x_start = self.x_coverage_start - self.substrate_extension
                substrate_x_end = self.x_coverage_end + self.substrate_extension
            else:
                substrate_x_start = self.x_coverage_start
                substrate_x_end = self.x_coverage_end
                
            substrate_width = substrate_x_end - substrate_x_start
            substrate_center_x = (substrate_x_start + substrate_x_end) / 2
            
            # Position substrate after PEC (if present) or after PML
            if self.add_pec_backing:
                substrate_center_y = (-self.mpsat_sim.cell_size[1]/2 + 
                                    (self.mpsat_sim.factor_dpml*self.mpsat_sim.dpml) + 
                                    self.pec_thickness + self.substrate_thickness/2 + self.y_bottom_offset)
            else:
                substrate_center_y = (-self.mpsat_sim.cell_size[1]/2 + 
                                    (self.mpsat_sim.factor_dpml*self.mpsat_sim.dpml) + 
                                    self.substrate_thickness/2 + self.y_bottom_offset)
            
            substrates.append(mp.Block(
                size=mp.Vector3(substrate_width, self.substrate_thickness, mp.inf),
                center=mp.Vector3(substrate_center_x, substrate_center_y),
                material=self.substrate_material
            ))
        
        # Top edge substrates
        if "top" in self.edges:
            if self.substrate_extends_beyond_pyramids:
                substrate_x_start = self.x_coverage_start - self.substrate_extension
                substrate_x_end = self.x_coverage_end + self.substrate_extension
            else:
                substrate_x_start = self.x_coverage_start
                substrate_x_end = self.x_coverage_end
                
            substrate_width = substrate_x_end - substrate_x_start
            substrate_center_x = (substrate_x_start + substrate_x_end) / 2

            # Position substrate after PEC (if present) or after PML
            if self.add_pec_backing:
                substrate_center_y = (self.mpsat_sim.cell_size[1]/2 - 
                                    (self.mpsat_sim.factor_dpml*self.mpsat_sim.dpml) - 
                                    self.pec_thickness - self.substrate_thickness/2 - self.y_bottom_offset)
            else:
                substrate_center_y = (self.mpsat_sim.cell_size[1]/2 + 
                                    (self.mpsat_sim.factor_dpml*self.mpsat_sim.dpml) - 
                                    self.substrate_thickness/2 - self.y_bottom_offset)
            
            substrates.append(mp.Block(
                size=mp.Vector3(substrate_width, self.substrate_thickness, mp.inf),
                center=mp.Vector3(substrate_center_x, substrate_center_y),
                material=self.substrate_material
            ))
        
        # Left edge substrates
        if "left" in self.edges:
            if self.substrate_extends_beyond_pyramids:
                substrate_y_start = self.y_coverage_start - self.substrate_extension
                substrate_y_end = self.y_coverage_end + self.substrate_extension
            else:
                substrate_y_start = self.y_coverage_start
                substrate_y_end = self.y_coverage_end
                
            substrate_width = substrate_y_end - substrate_y_start
            substrate_center_y = (substrate_y_start + substrate_y_end) / 2
            
            if self.add_pec_backing:
                substrate_center_x = (-self.mpsat_sim.cell_size[0]/2 + 
                                    (self.mpsat_sim.factor_dpml*self.mpsat_sim.dpml) + 
                                    self.pec_thickness + self.substrate_thickness/2 + self.x_left_offset)
            else:
                substrate_center_x = (-self.mpsat_sim.cell_size[0]/2 + 
                                    (self.mpsat_sim.factor_dpml*self.mpsat_sim.dpml) + 
                                    self.substrate_thickness/2 + self.x_left_offset)
            
            substrates.append(mp.Block(
                size=mp.Vector3(self.substrate_thickness, substrate_width, mp.inf),
                center=mp.Vector3(substrate_center_x, substrate_center_y),
                material=self.substrate_material
            ))
        
        # Right edge substrates
        if "right" in self.edges:
            if self.substrate_extends_beyond_pyramids:
                substrate_y_start = self.y_coverage_start - self.substrate_extension
                substrate_y_end = self.y_coverage_end + self.substrate_extension
            else:
                substrate_y_start = self.y_coverage_start
                substrate_y_end = self.y_coverage_end
                
            substrate_width = substrate_y_end - substrate_y_start
            substrate_center_y = (substrate_y_start + substrate_y_end) / 2
            
            if self.add_pec_backing:
                substrate_center_x = (self.mpsat_sim.cell_size[0]/2 - 
                                    (self.mpsat_sim.factor_dpml*self.mpsat_sim.dpml) - 
                                    self.pec_thickness - self.substrate_thickness/2 - self.x_left_offset)
            else:
                substrate_center_x = (self.mpsat_sim.cell_size[0]/2 - 
                                    (self.mpsat_sim.factor_dpml*self.mpsat_sim.dpml) - 
                                    self.substrate_thickness/2 - self.x_left_offset)
            
            substrates.append(mp.Block(
                size=mp.Vector3(self.substrate_thickness, substrate_width, mp.inf),
                center=mp.Vector3(substrate_center_x, substrate_center_y),
                material=self.substrate_material
            ))
        
        return substrates

    def _create_pyramids(self):
        """
        Create pyramid blocks for all edges
        
        Returns
        -------
        list
            List of pyramid block objects
        """
        pyramids = []
        
        #~ Bottom
        # Calculate base offset for pyramids based on what exists before them
        def get_base_offset():
            offset = self.mpsat_sim.factor_dpml * self.mpsat_sim.dpml
            if self.add_pec_backing:
                offset += self.pec_thickness
            if self.add_substrate:
                offset += self.substrate_thickness
            return offset
    
        # Bottom edge pyramids (pointing upward)
        if "bottom" in self.edges:
            base_offset = get_base_offset()

            for j in range(self.num_pyramids):
                pyramid_center_x = self.x_coverage_start + (j + 0.5) * self.x_base_width
                for i in range(self.n_layers):
                    layer_width = self.calculate_layer_width(i, self.x_base_width)
                    if layer_width <= 0:
                        continue  # Skip layers with zero or negative width
                        
                    layer_center_y = (-self.mpsat_sim.cell_size[1]/2 + base_offset +
                                    i * self.layer_thickness + self.layer_thickness/2 + self.y_bottom_offset)
                    
                    pyramids.append(mp.Block(
                        size=mp.Vector3(layer_width, self.layer_thickness, mp.inf),
                        center=mp.Vector3(pyramid_center_x, layer_center_y),
                        material=self.material
                    ))
        
        #~ Top
        # Top edge pyramids (pointing downward)
        if "top" in self.edges:
            base_offset = get_base_offset()
            for j in range(self.num_pyramids):
                pyramid_center_x = self.x_coverage_start + (j + 0.5) * self.x_base_width
                for i in range(self.n_layers):
                    layer_width = self.calculate_layer_width(i, self.x_base_width)
                    if layer_width <= 0:
                        continue
                        
                    layer_center_y = (self.mpsat_sim.cell_size[1]/2 - base_offset - 
                                    i * self.layer_thickness - self.layer_thickness/2 + self.y_top_offset)
                    
                    pyramids.append(mp.Block(
                        size=mp.Vector3(layer_width, self.layer_thickness, mp.inf),
                        center=mp.Vector3(pyramid_center_x, layer_center_y),
                        material=self.material
                    ))
        
        #~ Left
        # Left edge pyramids (pointing rightward)
        if "left" in self.edges:
            base_offset = get_base_offset()
            for j in range(self.num_pyramids):
                pyramid_center_y = self.y_coverage_start + (j + 0.5) * self.y_base_width
                for i in range(self.n_layers):
                    layer_width = self.calculate_layer_width(i, self.y_base_width)
                    if layer_width <= 0:
                        continue
                        
                    layer_center_x = (-self.mpsat_sim.cell_size[0]/2 + base_offset + 
                                    i * self.layer_thickness + self.layer_thickness/2 + self.x_left_offset)
                    
                    pyramids.append(mp.Block(
                        size=mp.Vector3(self.layer_thickness, layer_width, mp.inf),
                        center=mp.Vector3(layer_center_x, pyramid_center_y),
                        material=self.material
                    ))
        
        #~ Right
        # Right edge pyramids (pointing leftward)
        if "right" in self.edges:
            base_offset = get_base_offset()
            for j in range(self.num_pyramids):
                pyramid_center_y = self.y_coverage_start + (j + 0.5) * self.y_base_width
                for i in range(self.n_layers):
                    layer_width = self.calculate_layer_width(i, self.y_base_width)
                    if layer_width <= 0:
                        continue
                        
                    layer_center_x = (self.mpsat_sim.cell_size[0]/2 - base_offset -
                                    i * self.layer_thickness - self.layer_thickness/2 + self.x_right_offset)
                    
                    pyramids.append(mp.Block(
                        size=mp.Vector3(self.layer_thickness, layer_width, mp.inf),
                        center=mp.Vector3(layer_center_x, pyramid_center_y),
                        material=self.material
                    ))
        
        return pyramids
    

#~ NEW ABSORBER CLASS SUPPORTING DIFFERENT TYPES OF ABSORBERS

class Absorbers:
    def __init__(self,
                 p,
                 taper_type,
                 grid_size_sx,
                 grid_size_sy,
                 resolution,
                 center_x_mm,
                 center_y_mm,
                 eps_array,
                 geometry_objects,
                 z0,
                 z1,
                 orientation,
                 angle_axis,
                 h= None,
                 p_h_ratio= None,
                 # Substrate parameters
                 substrate_thickness=None,
                 substrate_material=None,
                 add_substrate = False,
                 default_substrate_material = 1,
                 mesh_filter_option='min',
                 epsilon_r=None,
                 epsilon_i=None,
                 material=None,
                 plot_alpha=False,
                 plot_profile=False,
                 plot_mesh= False,
                 savepath=None
                 ):
        
    
        # Dimensions of the absorbers
        self.p = p # base
        
        if p_h_ratio:
            self.p_h_ratio = p_h_ratio # p/h ratio
            h = p*p_h_ratio # height
            self.h = h
        else:
            self.h = h
        
        self.p_h_ratio = p_h_ratio # p/h ratio
        self.l_array = np.linspace(0,h, 1000) # l goes from 0 to h
        self.theta = np.arctan(self.p/(2*self.h)) # angle w.r.t base
        self.orientation = orientation
        self.angle_axis = angle_axis

        # Absorber material and impedance properties
        self.epsilon_r = epsilon_r
        self.epsilon_i = epsilon_i
        if material:
            self.material = material
            # Update epsilon_r for create_absorber_from_profile function later
            self.epsilon_r = material.epsilon
        elif epsilon_r:
            self.material = mp.Medium(epsilon=epsilon_r)
        elif epsilon_i:
            D_conductivity = epsilon_i * 2 * np.pi * freq / epsilon_r
            self.material = mp.Medium(epsilon=epsilon_r, D_conductivity=D_conductivity)
        else:
            raise ValueError("Invalid material properties")
        
        self.z0 = z0
        self.z1 = z1

        # Type of the taper
        self.taper_type = taper_type # ['Pyramidal', 'linear', 'exponential']
        
        # Simulation box parameters
        self.grid_size_sx = grid_size_sx
        self.grid_size_sy = grid_size_sy
        self.resolution = resolution
        self.center_x_mm = center_x_mm
        self.center_y_mm = center_y_mm
        
        # Epsilon Map and Geometry object array
        self.eps_array = eps_array
        self.geometry_objects = geometry_objects
        
        # Substrate parameters
        self.add_substrate = add_substrate
        self.substrate_thickness = substrate_thickness
        self.substrate_material = substrate_material
        # Default substrate material
        self.default_substrate_material = self.material

        # Triangular mesh option
        self.mesh_filter_option = mesh_filter_option

        # Plotting options
        self.plot_alpha = plot_alpha
        self.plot_profile = plot_profile
        self.plot_mesh = plot_mesh
        
        
        # Save path
        if savepath:
            self.savepath = savepath
            os.makedirs(self.savepath, exist_ok=True)
        else:
            # Save in the current director
            self.savepath = './'

    def assemble(self):
        # Calculate the filling factor (alpha_array) 
        if self.taper_type == "Pyramidal":
            self.alpha_array = self.alpha_pyramidal(self.p, self.theta, self.l_array)
        elif self.taper_type == "Exponential":
            self.alpha_array = self.alpha_exponential(self.z0, self.z1, self.h, self.epsilon_r, self.l_array)
        elif self.taper_type == "Linear":
            self.alpha_array = self.alpha_linear(self.z0, self.z1, self.h, self.epsilon_r, self.l_array)
        else:
            raise ValueError("Invalid taper type")
        
        self.w_array = self.p * np.sqrt(self.alpha_array)

        if self.plot_alpha:
            self._plot_alpha()
        if self.plot_profile:
            self._plot_profile()

        # Create the absorber filled with triangular mesh
        self.absorber, self.tri = self.create_absorber_from_profile(
            grid_size_sx=self.grid_size_sx,
            grid_size_sy=self.grid_size_sy,
            eps_array=self.eps_array,
            center_x_mm=self.center_x_mm,
            center_y_mm=self.center_y_mm,
            pyramid_height=self.h,
            base_width=self.p,
            alpha_profile=self.alpha_array,
            orientation=self.orientation,
            angle_axis=self.angle_axis,
            add_substrate=self.add_substrate,
            substrate_thickness=self.substrate_thickness,
            substrate_material=self.substrate_material,
            material_value=self.epsilon_r,
            resolution=self.resolution
        )

        self.absorber_prisms = mesh.convert_triangles_to_prisms(gridx_size_mm=self.grid_size_sx,
                                                                gridy_size_mm=self.grid_size_sy,
                                                                tri=self.tri,
                                                                material=self.material,
                                                                # resolution=self.resolution,
                                                                thickness=1.0 # It won't affect in 2D
        )

        # Check if absorber_prisms is a list and extend, otherwise append
        if isinstance(self.absorber_prisms, list):
            self.geometry_objects.extend(self.absorber_prisms)
        else:
            self.geometry_objects.append(self.absorber_prisms)

        
        return self.geometry_objects

    def _plot_alpha(self):
        plt.figure(figsize=(10, 6))
        plt.plot(self.l_array/self.h, self.alpha_array, label = self.taper_type, color = 'blue')
        plt.title('Alpha Values for Different Absorber profiles')
        plt.xlabel('Normalized Height (l/h)')
        plt.ylabel('Alpha (Filling factor)')
        plt.grid()
        plt.legend()
        plt.savefig(self.savepath + 'alpha_values_linear_absorber.png')
            

    def _plot_profile(self):
        plt.figure(figsize=(10, 6))
        plt.plot(self.w_array, self.l_array)
        plt.xlabel("Width (w)")
        plt.ylabel("Height (l)")
        plt.title("Linear profile")
        plt.savefig(self.savepath + 'linear_profile.png')
        plt.close()
        

    # Pyramid Taper
    def alpha_pyramidal(self, p, theta, l):
        return ((p - 2*np.tan(theta)*l)**2)/p**2

    # Exponential Taper
    def alpha_exponential(self, z0, z1, h, e_r, l):
        """Exponential impedance profile from z0 to z1 over height h"""
        b = z0 / z1
        a = (1/h) * np.log(b)
        epsilon_l = b * np.exp(-2 * a * l)
        alpha = ((epsilon_l - 1) / (e_r - 1))
        
        # Normalize alpha to [0, 1] range
        # Benefits:
            # The bottom (highest alpha) reaches 1.0
            # The top (lowest alpha) reaches ~0
            # Values are spread across the full [0, 1] range
        alpha_min = np.min(alpha)
        alpha_max = np.max(alpha)
        if alpha_max > alpha_min:
            alpha = (alpha - alpha_min) / (alpha_max - alpha_min)
        
        alpha = np.clip(alpha, 1e-6, 1.0)
        
        return alpha

    # Linear Taper
    def alpha_linear(self, z0, z1, h, e_r, l):
        m = (z1-z0)/h
        epsilon_l = (z0/(m*l + z0))**2
        alpha = (epsilon_l - 1) / (e_r - 1)
        
        # Reverse the profile so it decreases with height
        # This helps in inverting the pyramids
        alpha = alpha[::-1]

        return alpha



    def create_absorber_from_profile(self,
                                     grid_size_sx, 
                                    grid_size_sy, 
                                    eps_array,
                                    center_x_mm,
                                    center_y_mm,
                                    pyramid_height, 
                                    base_width,
                                    alpha_profile,
                                    orientation="+y",
                                    angle_axis = "x",
                                    add_substrate=True,
                                    substrate_thickness=1.0,
                                    substrate_material = None,
                                    material_value=1.0,
                                    resolution=1.0):
        """
        Create a pyramidal absorber using an impedance profile.
        
        Parameters:
        -----------
        alpha_profile : ndarray
            Filling factor profile (0 to 1) as function of height
        """
        absorber_array = eps_array.copy()
        scaled_pyramid_height = int(pyramid_height * resolution)
        scaled_base_width = int(base_width * resolution)
        scaled_grid_size_sx = int(grid_size_sx * resolution)
        scaled_grid_size_sy = int(grid_size_sy * resolution)
        
        # Convert to pixel coordinates
        center_x = int((center_x_mm+grid_size_sx/2) * resolution)
        center_y = int((center_y_mm+grid_size_sy/2) * resolution)

        if add_substrate:
            # Convert single values to lists for uniform handling
            if substrate_thickness is not None and not isinstance(substrate_thickness, (list, tuple)):
                substrate_thickness = [substrate_thickness]
            if substrate_material is not None and not isinstance(substrate_material, (list, tuple)):
                substrate_material = [substrate_material]
            
            # Validate that lists have same length
            if substrate_thickness and substrate_material:
                if len(substrate_thickness) != len(substrate_material):
                    raise ValueError("substrate_thickness and substrate_material lists must have same length")
            
            # Create empty lists to store the values of the centre, size, angle, material of the different substrates
            centre_x_substrate = []
            centre_y_substrate = []
            size_x_substrate = []
            size_y_substrate = []
            angle_substrate = []

        for layer in range(scaled_pyramid_height):
            # Get alpha value from profile for this layer
            # Calculating the index in the `alpha_profile` array that
            # corresponds to the current layer of the pyramidal absorber being created.
            profile_idx = min(int(layer / scaled_pyramid_height * len(alpha_profile)), len(alpha_profile) - 1)
            alpha = alpha_profile[profile_idx]
            
            # Width varies based on filling factor
            current_width = int(scaled_base_width * np.sqrt(alpha))
            
            # Layer count variable
            layer_count = 0

            
            if orientation == "+y":
                y_pos = center_y + layer
                if 0 <= y_pos < scaled_grid_size_sy:
                    for x in range(max(0, center_x - current_width//2), 
                                min(scaled_grid_size_sx, center_x + current_width//2 + 1)):
                        # Material property varies with alpha (impedance matching)
                        absorber_array[y_pos, x] = material_value 
                # Add substrate
                if layer_count == 0:   
                    if add_substrate:
                        centre_x_substrate, centre_y_substrate, size_x_substrate, size_y_substrate, angle_substrate = self._calculate_substrate_positions(orientation=orientation,                                                                                                                                                 center_x=center_x,
                                                                                                                                                    center_y=center_y,
                                                                                                                                                    substrate_thickness=substrate_thickness,
                                                                                                                                                    substrate_material=substrate_material,
                                                                                                                                                    scaled_base_width=scaled_base_width,
                                                                                                                                                    resolution=resolution,
                                                                                                                                                    angle_axis=angle_axis)
                
                layer_count += 1
                
            elif orientation == "-y":
                y_pos = center_y - layer
                if 0 <= y_pos < scaled_grid_size_sy:
                    for x in range(max(0, center_x - current_width//2), 
                                min(scaled_grid_size_sx, center_x + current_width//2 + 1)):
                        # Material property varies with alpha (impedance matching)
                        absorber_array[y_pos, x] = material_value
                # Add substrate
                if layer_count == 0:   
                    if add_substrate:
                        centre_x_substrate, centre_y_substrate, size_x_substrate, size_y_substrate, angle_substrate = self._calculate_substrate_positions(orientation=orientation,
                                                                                                                                                    center_x=center_x,
                                                                                                                                                    center_y=center_y,
                                                                                                                                                    substrate_thickness=substrate_thickness,
                                                                                                                                                    substrate_material=substrate_material,
                                                                                                                                                    scaled_base_width=scaled_base_width,
                                                                                                                                                    resolution=resolution,
                                                                                                                                                    angle_axis=angle_axis)
                
                layer_count += 1
                

            elif orientation == "+x":
                x_pos = center_x + layer
                if 0 <= x_pos < scaled_grid_size_sx:
                    for y in range(max(0, center_y - current_width//2), 
                                min(scaled_grid_size_sy, center_y + current_width//2 + 1)):
                        # Material property varies with alpha (impedance matching)
                        absorber_array[y, x_pos] = material_value 
                # Add substrate                           
                if layer_count == 0:   
                    if add_substrate:
                        centre_x_substrate, centre_y_substrate, size_x_substrate, size_y_substrate, angle_substrate = self._calculate_substrate_positions(orientation=orientation,
                                                                                                                                                    center_x=center_x,
                                                                                                                                                    center_y=center_y,
                                                                                                                                                    substrate_thickness=substrate_thickness,
                                                                                                                                                    substrate_material=substrate_material,
                                                                                                                                                    scaled_base_width=scaled_base_width,
                                                                                                                                                    resolution=resolution,
                                                                                                                                                    angle_axis=angle_axis)                    

                layer_count += 1
                
            elif orientation == "-x":
                x_pos = center_x - layer
                if 0 <= x_pos < scaled_grid_size_sx:
                    for y in range(max(0, center_y - current_width//2), 
                                min(scaled_grid_size_sy, center_y + current_width//2 + 1)):
                        # Material property varies with alpha (impedance matching)
                        absorber_array[y, x_pos] = material_value
                # Add substrate       
                if layer_count == 0:   
                    if add_substrate:

                        centre_x_substrate, centre_y_substrate, size_x_substrate, size_y_substrate, angle_substrate = self._calculate_substrate_positions(orientation=orientation,
                                                                                                                                                    center_x=center_x,
                                                                                                                                                    center_y=center_y,
                                                                                                                                                    substrate_thickness=substrate_thickness,
                                                                                                                                                    substrate_material=substrate_material,
                                                                                                                                                    scaled_base_width=scaled_base_width,
                                                                                                                                                    resolution=resolution,
                                                                                                                                                    angle_axis=angle_axis)

                layer_count += 1
                
            elif isinstance(orientation, float):
                orientation_rad = np.radians(orientation)
                if angle_axis == "x":
                    # Center position along the angled axis
                    x_pos = int(center_x + layer * np.cos(orientation_rad))
                    y_pos = int(center_y + layer * np.sin(orientation_rad))
                    
                    # Direction perpendicular to the angle (for width)
                    perp_x = -np.sin(orientation_rad)
                    perp_y = np.cos(orientation_rad)
                    
                    # Draw the width perpendicular to the angle direction
                    for w in range(-current_width//2, current_width//2 + 1):
                        x_line = int(x_pos + w * perp_x)
                        y_line = int(y_pos + w * perp_y)
                        if 0 <= x_line < scaled_grid_size_sx and 0 <= y_line < scaled_grid_size_sy:
                            absorber_array[y_line, x_line] = material_value

                    # Add substrate
                    if layer_count == 0:   
                        if add_substrate:
                            centre_x_substrate, centre_y_substrate, size_x_substrate, size_y_substrate, angle_substrate = self._calculate_substrate_positions(orientation=orientation,
                                                                                                                                                        center_x=center_x,
                                                                                                                                                        center_y=center_y,
                                                                                                                                                        substrate_thickness=substrate_thickness,
                                                                                                                                                        substrate_material=substrate_material,
                                                                                                                                                        scaled_base_width=scaled_base_width,
                                                                                                                                                        resolution=resolution,
                                                                                                                                                        angle_axis=angle_axis)
                                
                elif angle_axis == "y":
                    # TODO: FIX THE BUG HERE!!
                    # # Center position along the angled axis
                    # x_pos = int(center_x + layer * np.sin(orientation_rad))
                    # y_pos = int(center_y + layer * np.cos(orientation_rad))
                    
                    # # Direction perpendicular to the angle (for width)
                    # perp_x = np.cos(orientation_rad)
                    # perp_y = -np.sin(orientation_rad)
                    
                    # # Draw the width perpendicular to the angle direction
                    # for w in range(-current_width//2, current_width//2 + 1):
                    #     x_line = int(x_pos + w * perp_x)
                    #     y_line = int(y_pos + w * perp_y)
                    #     if 0 <= x_line < scaled_grid_size_sx and 0 <= y_line < scaled_grid_size_sy:
                    #         absorber_array[y_line, x_line] = material_value 

                    # if layer_count == 0:   
                    #     if add_substrate:
                    #         centre_x_substrate = center_x - (substrate_thickness/2)*resolution*np.cos(orientation_rad)
                    #         centre_y_substrate = center_y - (substrate_thickness/2)*resolution*np.sin(orientation_rad)
                    #         size_x_substrate = substrate_thickness * resolution
                    #         size_y_substrate = scaled_base_width
                    #         angle_substrate = orientation  
                    raise Warning("Y-axis orientation is not yet implemented yet!!")

                layer_count += 1
                
            else:
                raise ValueError("Invalid orientation type")
            
        if add_substrate:
            import meepsat.meep_geometry as comp_meep
            for i in range(len(substrate_thickness)):
                # Convert substrate dimensions from pixels to mm
                size_x_substrate_mm = size_x_substrate[i] / resolution
                size_y_substrate_mm = size_y_substrate[i] / resolution
                substrate_thickness_mm = substrate_thickness[i]
                
                # Convert center from pixels to mm
                centre_x_substrate_mm = centre_x_substrate[i] / resolution - grid_size_sx/2
                centre_y_substrate_mm = centre_y_substrate[i] / resolution - grid_size_sy/2
                
                print(f"Substrate size: ({size_x_substrate_mm:.2f}, {size_y_substrate_mm:.2f}) mm")
                print(f"Substrate center: ({centre_x_substrate_mm:.2f}, {centre_y_substrate_mm:.2f}) mm")
                print(f"Substrate angle: {angle_substrate[i]}")
                
                
                
                substrate = comp_meep.meep_block(
                    size=mp.Vector3(size_x_substrate_mm, 
                                size_y_substrate_mm, 
                                0),
                    center=mp.Vector3(centre_x_substrate_mm,
                                    centre_y_substrate_mm,
                                    0),
                    angle=angle_substrate[i],
                    material=substrate_material[i] if substrate_material[i] is not None else self.default_substrate_material
                )
                self.geometry_objects.append(substrate)
        
        # TRIANGULAR MESHGRID inside the absorber
        tri = mesh._create_triangular_mesh(epsilon_array= absorber_array,
                                    epsilon_val= material_value,
                                    grid_size_sx= grid_size_sx,
                                    grid_size_sy= grid_size_sy,
                                    resolution= resolution,
                                    filter_option="min",
                                    plot= True,
                                    figname='absorber_triangular_mesh.png')

        return absorber_array, tri


    def _calculate_substrate_positions(self, orientation, substrate_thickness, 
                                        substrate_material, scaled_base_width,
                                        center_x, center_y, resolution, angle_axis='x'):

        """
        Calculate the positions and dimensions of substrate layers.
        This function computes the center coordinates, sizes, and rotation angles
        for substrate layers based on the specified orientation and layer configuration.
        Parameters
        ----------
        orientation : str or float
            The orientation of the substrate layers. Can be one of "+x", "-x", "+y", "-y"
            for axis-aligned orientations, or a float representing the angle in degrees
            for rotated orientations.
        substrate_thickness : list of float
            Thickness values for each substrate layer in simulation units.
        substrate_material : str
            The material type of the substrate (used for material properties).
        scaled_base_width : float
            The width of the substrate base in simulation units.
        center_x : float
            The x-coordinate of the substrate center in simulation units.
        center_y : float
            The y-coordinate of the substrate center in simulation units.
        resolution : float
            The resolution factor for converting thickness values to spatial dimensions.
        angle_axis : str, optional
            The axis around which rotation is applied (default: 'x').
        Returns
        -------
        centre_x_substrate : list of float
            X-coordinates of the center of each substrate layer.
        centre_y_substrate : list of float
            Y-coordinates of the center of each substrate layer.
        size_x_substrate : list of float
            X-dimension sizes of each substrate layer.
        size_y_substrate : list of float
            Y-dimension sizes of each substrate layer.
        angle_substrate : list of float
            Rotation angles (in degrees) of each substrate layer.
        Notes
        -----
        The function iteratively stacks substrate layers, calculating positions based
        on cumulative thicknesses. For the first layer (i=0), positioning is relative
        to the initial center point. For subsequent layers (i>0), positioning is
        relative to the previous layer's center position.
        """

        centre_x_substrate = []
        centre_y_substrate = []
        size_x_substrate = []
        size_y_substrate = []
        angle_substrate = []
        
        for i in range(len(substrate_thickness)):
            if i == 0:
                if orientation == "+y":
                    cx, cy = center_x, center_y - (substrate_thickness[i]/2)*resolution
                    sx, sy = scaled_base_width, substrate_thickness[i] * resolution
                    angle0 = 0
                elif orientation == "-y":
                    cx, cy = center_x, center_y + (substrate_thickness[i]/2)*resolution
                    sx, sy = scaled_base_width, substrate_thickness[i] * resolution
                    angle0 = 0
                elif orientation == "-x":
                    cx, cy = center_x + (substrate_thickness[i]/2)*resolution, center_y
                    sx, sy = substrate_thickness[i] * resolution, scaled_base_width
                    angle0 = 0
                elif orientation == "+x":
                    cx, cy = center_x - (substrate_thickness[i]/2)*resolution, center_y
                    sx, sy = substrate_thickness[i] * resolution, scaled_base_width
                    angle0 = 0
                    
                elif isinstance(orientation, float):
                    orientation_rad = np.radians(orientation)
                    cx = center_x - (substrate_thickness[0]/2)*resolution*np.cos(orientation_rad)
                    cy = center_y - (substrate_thickness[0]/2)*resolution*np.sin(orientation_rad)
                    sx = substrate_thickness[i] * resolution
                    sy = scaled_base_width
                    angle0 = orientation
            else:
                if orientation == "-y":
                    cx, cy = centre_x_substrate[i-1], centre_y_substrate[i-1] + ((substrate_thickness[i-1] + substrate_thickness[i])/2)*resolution
                    sx, sy = scaled_base_width, substrate_thickness[i] * resolution
                    angle0 = 0
                elif orientation == "+y":
                    cx, cy = centre_x_substrate[i-1], centre_y_substrate[i-1] - ((substrate_thickness[i-1] + substrate_thickness[i])/2)*resolution
                    sx, sy = scaled_base_width, substrate_thickness[i] * resolution
                    angle0 = 0
                elif orientation == "-x":
                    cx, cy = centre_x_substrate[i-1] + ((substrate_thickness[i-1] + substrate_thickness[i])/2)*resolution, centre_y_substrate[i-1]
                    sx, sy = substrate_thickness[i] * resolution, scaled_base_width
                    angle0 = 0
                elif orientation == "+x":
                    cx, cy = centre_x_substrate[i-1] - ((substrate_thickness[i-1] + substrate_thickness[i])/2)*resolution, centre_y_substrate[i-1]
                    sx, sy = substrate_thickness[i] * resolution, scaled_base_width
                    angle0 = 0
                elif isinstance(orientation, float):
                    orientation_rad = np.radians(orientation)
                    cx = centre_x_substrate[i-1] - ((substrate_thickness[i-1] + substrate_thickness[i])/2)*resolution*np.cos(orientation_rad)
                    cy = centre_y_substrate[i-1] - ((substrate_thickness[i-1] + substrate_thickness[i])/2)*resolution*np.sin(orientation_rad)
                    sx = substrate_thickness[i] * resolution
                    sy = scaled_base_width
                    angle0 = orientation

            centre_x_substrate.append(cx)
            centre_y_substrate.append(cy)
            size_x_substrate.append(sx)
            size_y_substrate.append(sy)
            angle_substrate.append(angle0)

        return centre_x_substrate, centre_y_substrate, size_x_substrate, size_y_substrate, angle_substrate

# # ~ FOREBAFFLE CLASS

logger = logging.getLogger(__name__)


# * ############################################################################################################
# * BASE CLASSES - Define abstract base class first
# * ############################################################################################################

class ForebaffleComponent(ABC):
    """Abstract base class for forebaffle components."""
    
    @abstractmethod
    def get_geometry(self, parent_forebaffle) -> List[mp.GeometricObject]:
        """Return MEEP geometric objects for this component."""
        pass
    
    # @abstractmethod
    # def get_eps_map(self, parent_forebaffle) -> np.ndarray:
    #     """Return the epsilon map for this component."""
    #     pass


# * ############################################################################################################
# * DATACLASSES - Configuration classes
# * ############################################################################################################

# Absorbers over Forebaffle sides - base, height, hypotenuse 
@dataclass
class AbsorberLayer:
    """Configuration for an absorber layer on a forebaffle side."""
    side: Literal['base', 'height', 'hypotenuse']
    thickness: float
    material: mp.Medium = None
    epsilon_real: float = 1.0
    epsilon_imag: float = 0.0
    
    def get_material(self, freq: float = 1/3) -> mp.Medium:
        """Get the material with absorption properties."""
        if self.material is not None:
            return self.material
        return mp.Medium(epsilon=self.epsilon_real, 
                         D_conductivity=self.epsilon_imag * 2 * np.pi * freq / self.epsilon_real)


class AbsorberComponent(ForebaffleComponent):
    """Component that adds absorber layers to forebaffle sides."""
    
    def __init__(self, absorber_layers: List[AbsorberLayer]):
        self.absorber_layers = absorber_layers
    
    def get_geometry(self, parent_forebaffle) -> List[mp.GeometricObject]:
        """Generate absorber layer geometries based on parent forebaffle."""
        geometries = []
        v1, v2, v3 = parent_forebaffle.calculate_vertices()
        
        for layer in self.absorber_layers:
            if layer.side == 'base':
                geom = self._create_base_absorber(v1, v2, layer, parent_forebaffle)
            elif layer.side == 'height':
                geom = self._create_height_absorber(v2, v3, layer, parent_forebaffle)
            elif layer.side == 'hypotenuse':
                geom = self._create_hypotenuse_absorber(v1, v3, v2, layer, parent_forebaffle)
            
            if geom:
                geometries.append(geom)
        
        return geometries
    
    def _ensure_ccw_order(self, vertices):
        """
        Ensure vertices are in counter-clockwise order using the shoelace formula.
        
        Parameters
        ----------
        vertices : list of mp.Vector3
            List of vertices to check
            
        Returns
        -------
        list of mp.Vector3
            Vertices in counter-clockwise order
        """
        # Calculate signed area using shoelace formula
        n = len(vertices)
        area = 0
        for i in range(n):
            j = (i + 1) % n
            area += vertices[i].x * vertices[j].y
            area -= vertices[j].x * vertices[i].y
        
        # If area is negative, vertices are clockwise - reverse them
        if area < 0:
            return list(reversed(vertices))
        return vertices
    
    def _create_base_absorber(self, v1, v2, layer, parent):
        """Create absorber along the base edge (v1-v2)."""
        # Base is horizontal in most cases
        # Offset perpendicular to the base, outward from the triangle
        
        # Determine outward direction
        # For quadrant 1 (0-90°) and 4 (270-360°), offset downward
        # For quadrant 2 (90-180°) and 3 (180-270°), offset downward still works
        angle = parent.angle_degrees
        
        if 0 <= angle < 180:
            # Triangle is above base, offset downward
            offset_y = -layer.thickness
            offset_x = 0
        else:
            # Triangle is below base, offset upward
            offset_y = layer.thickness
            offset_x = 0
        
        # Create vertices - order matters for MEEP!
        vertices = [
            v1,  # Original edge start
            v2,  # Original edge end
            mp.Vector3(v2.x + offset_x, v2.y + offset_y),  # Offset edge end
            mp.Vector3(v1.x + offset_x, v1.y + offset_y),  # Offset edge start
        ]
        
        # Ensure counter-clockwise ordering
        vertices = self._ensure_ccw_order(vertices)
        
        return mp.Prism(
            vertices=vertices,
            height=parent.height,
            axis=mp.Vector3(0, 0, 1),
            material=layer.get_material()
        )
    
    def _create_height_absorber(self, v2, v3, layer, parent):
        """Create absorber along the height edge (v2-v3)."""
        # Height is vertical in most cases
        angle = parent.angle_degrees
        
        if 0 <= angle < 90 or 270 <= angle < 360:
            # Triangle is to the left of height, offset to the right
            offset_x = layer.thickness
            offset_y = 0
        else:
            # Triangle is to the right of height, offset to the left
            offset_x = -layer.thickness
            offset_y = 0
        
        vertices = [
            v2,  # Original edge start
            v3,  # Original edge end
            mp.Vector3(v3.x + offset_x, v3.y + offset_y),  # Offset edge end
            mp.Vector3(v2.x + offset_x, v2.y + offset_y),  # Offset edge start
        ]
        
        vertices = self._ensure_ccw_order(vertices)
        
        return mp.Prism(
            vertices=vertices,
            height=parent.height,
            axis=mp.Vector3(0, 0, 1),
            material=layer.get_material()
        )
    
    def _create_hypotenuse_absorber(self, v1, v3, v2, layer, parent):
        """
        Create absorber along the hypotenuse edge (v1-v3).
        
        Parameters
        ----------
        v1, v3 : mp.Vector3
            Endpoints of the hypotenuse
        v2 : mp.Vector3
            The right-angle vertex (used to determine outward direction)
        """
        # Calculate perpendicular direction
        dx = v3.x - v1.x
        dy = v3.y - v1.y
        length = np.sqrt(dx**2 + dy**2)
        
        if length == 0:
            logger.warning("Zero-length hypotenuse detected")
            return None
        
        # Two possible perpendicular directions (90° rotation)
        perp_x1 = -dy / length
        perp_y1 = dx / length
        
        # Check which direction points away from v2
        # Vector from midpoint of hypotenuse to v2
        mid_x = (v1.x + v3.x) / 2
        mid_y = (v1.y + v3.y) / 2
        to_v2_x = v2.x - mid_x
        to_v2_y = v2.y - mid_y
        
        # Dot product tells us if perpendicular points toward or away from v2
        dot = perp_x1 * to_v2_x + perp_y1 * to_v2_y
        
        # If dot product is positive, flip the direction
        if dot > 0:
            perp_x1 = -perp_x1
            perp_y1 = -perp_y1
        
        # Apply thickness
        offset_x = perp_x1 * layer.thickness
        offset_y = perp_y1 * layer.thickness
        
        vertices = [
            v1,  # Original edge start
            v3,  # Original edge end
            mp.Vector3(v3.x + offset_x, v3.y + offset_y),  # Offset edge end
            mp.Vector3(v1.x + offset_x, v1.y + offset_y),  # Offset edge start
        ]
        
        vertices = self._ensure_ccw_order(vertices)
        
        return mp.Prism(
            vertices=vertices,
            height=parent.height,
            axis=mp.Vector3(0, 0, 1),
            material=layer.get_material()
        )

#--- FLAIRING ON THE FOREBAFFLE TIP ---#
@dataclass
class FlareConfig:
    """Configuration for a flare extending from a vertex."""
    # Type of flaring: 
    # 1. 'linear' - straight line by using mp.Block 
    # 2. 'spline' - spline function pointing outwards
    flaring_type: str # Which type of flaring material
    
    # The below describes the parameters for each flaring type
    # 1. linear
    linear: Dict[str, Any] = field(default_factory=lambda: {
        "length": 1.0,
        "thickness": 1.0,
        "theta2_axis": 'x'
    })
    # 2. spline
    # spline: Dict[str, Any] = field(default_factory=lambda: {
    #     ""
    # })

    material: mp.Medium = None # Meep Material
    epsilon_real: float = 1.0 # Real permittivity
    epsilon_imag: float = 0.0 # Imaginary permittivity
    which_vertex: str = 'v3' # From which vertex of the baffle, the user wants to extend the flaring structure (default from the v3)
    
    

    def get_material(self, freq: float = 1/3) -> mp.Medium:
        """Get the material with absorption properties."""
        if self.material is not None:
            return self.material
        return mp.Medium(epsilon=self.epsilon_real, 
                         D_conductivity=self.epsilon_imag * 2 * np.pi * freq / self.epsilon_real)

class FlairComponent(ForebaffleComponent):
    """Component representing the flaring structure on the forebaffle tip."""
    def __init__(self, flairs: List[FlareConfig]):
        self.flairs = flairs
        
    def get_geometry(self, parent_forebaffle) -> List[mp.GeometricObject]:
        """Generate flair geometries based on parent forebaffle."""
        geometries = self.create_flairs(parent_forebaffle)
        return geometries
    
    def get_eps_map(self, parent_forebaffle) -> np.ndarray:
        """Return the epsilon map (flairs don't modify it directly)."""
        return parent_forebaffle.epsilon_map

    def find_vertex_in_epsilon(self, vertex, parent_forebaffle):
        """
        Find the epsilon value at a vertex location in the epsilon map.
        
        Parameters
        ----------
        vertex : mp.Vector3 or tuple/list
            Vertex coordinates in real units
        parent_forebaffle : Forebaffle
            Parent forebaffle object containing simulation parameters
            
        Returns
        -------
        float
            Epsilon value at the vertex location
            
        Raises
        ------
        ValueError
            If vertex is outside the epsilon map bounds
        """
        # Extract coordinates - handle both mp.Vector3 and tuple/list
        if isinstance(vertex, mp.Vector3):
            x, y = vertex.x, vertex.y
        else:
            x, y = vertex[0], vertex[1]
        
        # Get simulation parameters
        resolution = parent_forebaffle.mpsat_sim.resolution
        epsilon_map = parent_forebaffle.epsilon_map
        
        # Calculate effective cell size (excluding PML on both sides)
        pml_thickness = parent_forebaffle.mpsat_sim.factor_dpml * parent_forebaffle.mpsat_sim.dpml
        xsize = parent_forebaffle.mpsat_sim.cell_size[0] - 4 * pml_thickness
        ysize = parent_forebaffle.mpsat_sim.cell_size[1] - 4 * pml_thickness
        
        # Transform from real coordinates to pixel indices
        # Real coords: origin at cell center, range [-size/2, +size/2]
        # Pixel coords: origin at array corner, range [0, size*resolution]
        x_idx = int((x + xsize / 2) * resolution)
        y_idx = int((y + ysize / 2) * resolution)
        
        # Validate bounds - note: epsilon_map.shape = (rows, cols) = (y_size, x_size)
        if not (0 <= y_idx < epsilon_map.shape[0] and 
                0 <= x_idx < epsilon_map.shape[1]):
            raise ValueError(
                f"Vertex at ({x:.3f}, {y:.3f}) maps to pixel indices ({x_idx}, {y_idx}), "
                f"which is outside epsilon map bounds {epsilon_map.shape} (y, x). "
                f"Valid ranges: y=[0, {epsilon_map.shape[0]-1}], x=[0, {epsilon_map.shape[1]-1}]"
            )
        
        # Access array with [row, col] = [y, x] indexing
        return epsilon_map[y_idx, x_idx]

    def _get_vertex(self, parent_forebaffle, which_vertex: str):
        """Get the specified vertex from the parent forebaffle."""
        v1, v2, v3 = parent_forebaffle.calculate_vertices()
        vertex_map = {'v1': v1, 'v2': v2, 'v3': v3}
        
        if which_vertex not in vertex_map:
            raise ValueError(f"Invalid vertex '{which_vertex}'. Must be 'v1', 'v2', or 'v3'")
        
        return vertex_map[which_vertex]

    def create_flairs(self, parent_forebaffle):
        """Create all flaring components."""
        meep_geometry = []
        
        # Iterate through all flair configurations
        for flair_config in self.flairs:
            vertex = self._get_vertex(parent_forebaffle, flair_config.which_vertex)
            eps_pixel_at_vertex = self.find_vertex_in_epsilon(vertex, parent_forebaffle)
            
            # Check flaring type from the config
            if flair_config.flaring_type == 'linear':
                linear_flair = self._create_linear_flair(vertex, eps_pixel_at_vertex, flair_config, parent_forebaffle)
                meep_geometry.append(linear_flair)

            elif flair_config.flaring_type == 'spline':
                # self._create_spline_flair(vertex, parent_forebaffle)
                pass
            else: 
                raise ValueError("Please give a valid flairing type")
        
        return meep_geometry  # Return only the geometry list, not a tuple
    
    def _create_linear_flair(self, vertex, eps_pixel_at_vertex, flair_config, parent_fb):
        """Create a linear flair extending from the specified vertex."""
        res = parent_fb.mpsat_sim.resolution
        characteristic_length = 1 #mm
        unit_pixel_length = characteristic_length / res
        linear_params = flair_config.linear
        length = linear_params["length"]
        thickness = linear_params["thickness"]
        theta2 = linear_params["theta2"]
        theta2_axis = linear_params["theta2_axis"]
        flair_material = flair_config.get_material()

        # # Calculate the center of the MEEP block using sin-cos approach depending on the rotation axis
        # if theta2_axis == 'x':
        #     x_center = vertex.x + thickness * math.cos(theta2)
        #     y_center = vertex.y + thickness * math.sin(theta2) 
        # elif theta2_axis == 'y':
        #     x_center = vertex.x + thickness * math.sin(theta2)
        #     y_center = vertex.y + thickness * math.cos(theta2)
        # else:
        #     raise ValueError("Invalid rotation axis. Must be 'x' or 'y'.")

        angle_rad = math.radians(theta2) if isinstance(theta2, (int, float)) else theta2
        
        if theta2_axis == 'x':
            offset_x = length/2 * math.cos(angle_rad)
            offset_y = length/2 * math.sin(angle_rad)
        elif theta2_axis == 'y':
            offset_x = length/2 * math.sin(angle_rad)
            offset_y = length/2 * math.cos(angle_rad)

        if theta2<=90:
            offset_vertex_x = thickness/2 * math.cos(90)
            offset_vertex_y = thickness/2 * math.sin(90)
            
            new_vertex = vertex
            new_vertex.x = vertex.x - offset_vertex_x
            new_vertex.y = vertex.y - offset_vertex_y
            
            x_center = new_vertex.x + offset_x -unit_pixel_length 
            y_center = new_vertex.y + offset_y -unit_pixel_length

        elif 90< theta2 < 180:
            offset_vertex_x = thickness/2 * -math.cos(90)
            offset_vertex_y = thickness/2 * math.sin(90)
            
            new_vertex = vertex
            new_vertex.x = vertex.x - offset_vertex_x
            new_vertex.y = vertex.y - offset_vertex_y

            x_center = new_vertex.x + offset_x +unit_pixel_length
            y_center = new_vertex.y + offset_y +unit_pixel_length

        # x_center = vertex.x + offset_x +unit_pixel_length 
        # y_center = vertex.y + offset_y +unit_pixel_length
        # Use meep_block from
        import meepsat.meep_geometry as mpsat_geom

        # x_center, y_center = x_center - thickness, y_center #- thickness
        flair_block_meep = mpsat_geom.meep_block(size = mp.Vector3(length, thickness, 0),
                                                 center = mp.Vector3(x_center, y_center, 0),
                                                 material = flair_material,
                                                 angle= theta2,
                                                 rot_axis= 'z') # This will be always Z

        return flair_block_meep
    
# * ############################################################################################################
# * MAIN FOREBAFFLE CLASS
# * ############################################################################################################


logger = logging.getLogger(__name__)


# * ############################################################################################################
# * BASE CLASSES - Define abstract base class first
# * ############################################################################################################

class ForebaffleComponent(ABC):
    """Abstract base class for forebaffle components."""
    
    @abstractmethod
    def get_geometry(self, parent_forebaffle) -> List[mp.GeometricObject]:
        """Return MEEP geometric objects for this component."""
        pass
    
    # @abstractmethod
    # def get_eps_map(self, parent_forebaffle) -> np.ndarray:
    #     """Return the epsilon map for this component."""
    #     pass


# * ############################################################################################################
# * DATACLASSES - Configuration classes
# * ############################################################################################################

# Absorbers over Forebaffle sides - base, height, hypotenuse 
@dataclass
class AbsorberLayer:
    """Configuration for an absorber layer on a forebaffle side."""
    side: Literal['base', 'height', 'hypotenuse']
    thickness: float
    material: mp.Medium = None
    epsilon_real: float = 1.0
    epsilon_imag: float = 0.0
    
    def get_material(self, freq: float = 1/3) -> mp.Medium:
        """Get the material with absorption properties."""
        if self.material is not None:
            return self.material
        return mp.Medium(epsilon=self.epsilon_real, 
                         D_conductivity=self.epsilon_imag * 2 * np.pi * freq / self.epsilon_real)


class AbsorberComponent(ForebaffleComponent):
    """Component that adds absorber layers to forebaffle sides."""
    
    def __init__(self, absorber_layers: List[AbsorberLayer]):
        self.absorber_layers = absorber_layers
    
    def get_geometry(self, parent_forebaffle) -> List[mp.GeometricObject]:
        """Generate absorber layer geometries based on parent forebaffle."""
        geometries = []
        v1, v2, v3 = parent_forebaffle.calculate_vertices()
        
        for layer in self.absorber_layers:
            if layer.side == 'base':
                geom = self._create_base_absorber(v1, v2, layer, parent_forebaffle)
            elif layer.side == 'height':
                geom = self._create_height_absorber(v2, v3, layer, parent_forebaffle)
            elif layer.side == 'hypotenuse':
                geom = self._create_hypotenuse_absorber(v1, v3, v2, layer, parent_forebaffle)
            
            if geom:
                geometries.append(geom)
        
        return geometries
    
    def _ensure_ccw_order(self, vertices):
        """
        Ensure vertices are in counter-clockwise order using the shoelace formula.
        
        Parameters
        ----------
        vertices : list of mp.Vector3
            List of vertices to check
            
        Returns
        -------
        list of mp.Vector3
            Vertices in counter-clockwise order
        """
        # Calculate signed area using shoelace formula
        n = len(vertices)
        area = 0
        for i in range(n):
            j = (i + 1) % n
            area += vertices[i].x * vertices[j].y
            area -= vertices[j].x * vertices[i].y
        
        # If area is negative, vertices are clockwise - reverse them
        if area < 0:
            return list(reversed(vertices))
        return vertices
    
    def _create_base_absorber(self, v1, v2, layer, parent):
        """Create absorber along the base edge (v1-v2)."""
        # Base is horizontal in most cases
        # Offset perpendicular to the base, outward from the triangle
        
        # Determine outward direction
        # For quadrant 1 (0-90°) and 4 (270-360°), offset downward
        # For quadrant 2 (90-180°) and 3 (180-270°), offset downward still works
        angle = parent.angle_degrees
        
        if 0 <= angle < 180:
            # Triangle is above base, offset downward
            offset_y = -layer.thickness
            offset_x = 0
        else:
            # Triangle is below base, offset upward
            offset_y = layer.thickness
            offset_x = 0
        
        # Create vertices - order matters for MEEP!
        vertices = [
            v1,  # Original edge start
            v2,  # Original edge end
            mp.Vector3(v2.x + offset_x, v2.y + offset_y),  # Offset edge end
            mp.Vector3(v1.x + offset_x, v1.y + offset_y),  # Offset edge start
        ]
        
        # Ensure counter-clockwise ordering
        vertices = self._ensure_ccw_order(vertices)
        
        return mp.Prism(
            vertices=vertices,
            height=parent.height,
            axis=mp.Vector3(0, 0, 1),
            material=layer.get_material()
        )
    
    def _create_height_absorber(self, v2, v3, layer, parent):
        """Create absorber along the height edge (v2-v3)."""
        # Height is vertical in most cases
        angle = parent.angle_degrees
        
        if 0 <= angle < 90 or 270 <= angle < 360:
            # Triangle is to the left of height, offset to the right
            offset_x = layer.thickness
            offset_y = 0
        else:
            # Triangle is to the right of height, offset to the left
            offset_x = -layer.thickness
            offset_y = 0
        
        vertices = [
            v2,  # Original edge start
            v3,  # Original edge end
            mp.Vector3(v3.x + offset_x, v3.y + offset_y),  # Offset edge end
            mp.Vector3(v2.x + offset_x, v2.y + offset_y),  # Offset edge start
        ]
        
        vertices = self._ensure_ccw_order(vertices)
        
        return mp.Prism(
            vertices=vertices,
            height=parent.height,
            axis=mp.Vector3(0, 0, 1),
            material=layer.get_material()
        )
    
    def _create_hypotenuse_absorber(self, v1, v3, v2, layer, parent):
        """
        Create absorber along the hypotenuse edge (v1-v3).
        
        Parameters
        ----------
        v1, v3 : mp.Vector3
            Endpoints of the hypotenuse
        v2 : mp.Vector3
            The right-angle vertex (used to determine outward direction)
        """
        # Calculate perpendicular direction
        dx = v3.x - v1.x
        dy = v3.y - v1.y
        length = np.sqrt(dx**2 + dy**2)
        
        if length == 0:
            logger.warning("Zero-length hypotenuse detected")
            return None
        
        # Two possible perpendicular directions (90° rotation)
        perp_x1 = -dy / length
        perp_y1 = dx / length
        
        # Check which direction points away from v2
        # Vector from midpoint of hypotenuse to v2
        mid_x = (v1.x + v3.x) / 2
        mid_y = (v1.y + v3.y) / 2
        to_v2_x = v2.x - mid_x
        to_v2_y = v2.y - mid_y
        
        # Dot product tells us if perpendicular points toward or away from v2
        dot = perp_x1 * to_v2_x + perp_y1 * to_v2_y
        
        # If dot product is positive, flip the direction
        if dot > 0:
            perp_x1 = -perp_x1
            perp_y1 = -perp_y1
        
        # Apply thickness
        offset_x = perp_x1 * layer.thickness
        offset_y = perp_y1 * layer.thickness
        
        vertices = [
            v1,  # Original edge start
            v3,  # Original edge end
            mp.Vector3(v3.x + offset_x, v3.y + offset_y),  # Offset edge end
            mp.Vector3(v1.x + offset_x, v1.y + offset_y),  # Offset edge start
        ]
        
        vertices = self._ensure_ccw_order(vertices)
        
        return mp.Prism(
            vertices=vertices,
            height=parent.height,
            axis=mp.Vector3(0, 0, 1),
            material=layer.get_material()
        )

#--- FLAIRING ON THE FOREBAFFLE TIP ---#
@dataclass
class FlareConfig:
    """Configuration for a flare extending from a vertex."""
    # Type of flaring: 
    # 1. 'linear' - straight line by using mp.Block 
    # 2. 'spline' - spline function pointing outwards
    flaring_type: str # Which type of flaring material
    
    # The below describes the parameters for each flaring type
    # 1. linear
    linear: Dict[str, Any] = field(default_factory=lambda: {
        "length": 1.0,
        "thickness": 1.0,
        "theta2_axis": 'x'
    })
    # 2. spline
    # spline: Dict[str, Any] = field(default_factory=lambda: {
    #     ""
    # })

    material: mp.Medium = None # Meep Material
    epsilon_real: float = 1.0 # Real permittivity
    epsilon_imag: float = 0.0 # Imaginary permittivity
    which_vertex: str = 'v3' # From which vertex of the baffle, the user wants to extend the flaring structure (default from the v3)
        

    def get_material(self, freq: float = 1/3) -> mp.Medium:
        """Get the material with absorption properties."""
        if self.material is not None:
            return self.material
        return mp.Medium(epsilon=self.epsilon_real, 
                         D_conductivity=self.epsilon_imag * 2 * np.pi * freq / self.epsilon_real)

class FlairComponent(ForebaffleComponent):
    """Component representing the flaring structure on the forebaffle tip."""
    def __init__(self, flairs: List[FlareConfig]):
        self.flairs = flairs
        
    def get_geometry(self, parent_forebaffle) -> List[mp.GeometricObject]:
        """Generate flair geometries based on parent forebaffle."""
        geometries = self.create_flairs(parent_forebaffle)
        return geometries
    
    def get_eps_map(self, parent_forebaffle) -> np.ndarray:
        """Return the epsilon map (flairs don't modify it directly)."""
        return parent_forebaffle.epsilon_map

    def find_vertex_in_epsilon(self, vertex, parent_forebaffle):
        """
        Find the epsilon value at a vertex location in the epsilon map.
        
        Parameters
        ----------
        vertex : mp.Vector3 or tuple/list
            Vertex coordinates in real units
        parent_forebaffle : Forebaffle
            Parent forebaffle object containing simulation parameters
            
        Returns
        -------
        float
            Epsilon value at the vertex location
            
        Raises
        ------
        ValueError
            If vertex is outside the epsilon map bounds
        """
        # Extract coordinates - handle both mp.Vector3 and tuple/list
        if isinstance(vertex, mp.Vector3):
            x, y = vertex.x, vertex.y
        else:
            x, y = vertex[0], vertex[1]
        
        # Get simulation parameters
        resolution = parent_forebaffle.mpsat_sim.resolution
        epsilon_map = parent_forebaffle.epsilon_map
        
        # Calculate effective cell size (excluding PML on both sides)
        pml_thickness = parent_forebaffle.mpsat_sim.factor_dpml * parent_forebaffle.mpsat_sim.dpml
        xsize = parent_forebaffle.mpsat_sim.cell_size[0] - 4 * pml_thickness
        ysize = parent_forebaffle.mpsat_sim.cell_size[1] - 4 * pml_thickness
        
        # Transform from real coordinates to pixel indices
        # Real coords: origin at cell center, range [-size/2, +size/2]
        # Pixel coords: origin at array corner, range [0, size*resolution]
        x_idx = int((x + xsize / 2) * resolution)
        y_idx = int((y + ysize / 2) * resolution)
        
        # Validate bounds - note: epsilon_map.shape = (rows, cols) = (y_size, x_size)
        if not (0 <= y_idx < epsilon_map.shape[0] and 
                0 <= x_idx < epsilon_map.shape[1]):
            raise ValueError(
                f"Vertex at ({x:.3f}, {y:.3f}) maps to pixel indices ({x_idx}, {y_idx}), "
                f"which is outside epsilon map bounds {epsilon_map.shape} (y, x). "
                f"Valid ranges: y=[0, {epsilon_map.shape[0]-1}], x=[0, {epsilon_map.shape[1]-1}]"
            )
        
        # Access array with [row, col] = [y, x] indexing
        return epsilon_map[y_idx, x_idx]

    def _get_vertex(self, parent_forebaffle, which_vertex: str):
        """Get the specified vertex from the parent forebaffle."""
        v1, v2, v3 = parent_forebaffle.calculate_vertices()
        vertex_map = {'v1': v1, 'v2': v2, 'v3': v3}
        
        if which_vertex not in vertex_map:
            raise ValueError(f"Invalid vertex '{which_vertex}'. Must be 'v1', 'v2', or 'v3'")
        
        return vertex_map[which_vertex]

    def create_flairs(self, parent_forebaffle):
        """Create all flaring components."""
        meep_geometry = []
        
        # Iterate through all flair configurations
        for flair_config in self.flairs:
            vertex = self._get_vertex(parent_forebaffle, flair_config.which_vertex)
            eps_pixel_at_vertex = self.find_vertex_in_epsilon(vertex, parent_forebaffle)
            
            # Check flaring type from the config
            if flair_config.flaring_type == 'linear':
                linear_flair = self._create_linear_flair(vertex, eps_pixel_at_vertex, flair_config, parent_forebaffle)
                meep_geometry.append(linear_flair)

            elif flair_config.flaring_type == 'spline':
                # self._create_spline_flair(vertex, parent_forebaffle)
                pass
            else: 
                raise ValueError("Please give a valid flairing type")
        
        return meep_geometry  # Return only the geometry list, not a tuple
    
    def _create_linear_flair(self, vertex, eps_pixel_at_vertex, flair_config, parent_fb):
        """Create a linear flair extending from the specified vertex."""
        res = parent_fb.mpsat_sim.resolution
        characteristic_length = 1 #mm
        unit_pixel_length = characteristic_length / res
        linear_params = flair_config.linear
        length = linear_params["length"]
        thickness = linear_params["thickness"]
        theta2 = linear_params["theta2"]
        theta2_axis = linear_params["theta2_axis"]
        flair_material = flair_config.get_material()

        # # Calculate the center of the MEEP block using sin-cos approach depending on the rotation axis
        # if theta2_axis == 'x':
        #     x_center = vertex.x + thickness * math.cos(theta2)
        #     y_center = vertex.y + thickness * math.sin(theta2) 
        # elif theta2_axis == 'y':
        #     x_center = vertex.x + thickness * math.sin(theta2)
        #     y_center = vertex.y + thickness * math.cos(theta2)
        # else:
        #     raise ValueError("Invalid rotation axis. Must be 'x' or 'y'.")

        angle_rad = math.radians(theta2) if isinstance(theta2, (int, float)) else theta2
        
        if theta2_axis == 'x':
            offset_x = length/2 * math.cos(angle_rad)
            offset_y = length/2 * math.sin(angle_rad)
        elif theta2_axis == 'y':
            offset_x = length/2 * math.sin(angle_rad)
            offset_y = length/2 * math.cos(angle_rad)

        if theta2<=90:
            offset_vertex_x = thickness/2 * math.cos(90)
            offset_vertex_y = thickness/2 * math.sin(90)
            
            new_vertex = vertex
            new_vertex.x = vertex.x - offset_vertex_x
            new_vertex.y = vertex.y - offset_vertex_y
            
            x_center = new_vertex.x + offset_x -unit_pixel_length 
            y_center = new_vertex.y + offset_y -unit_pixel_length

        elif 90< theta2 < 180:
            offset_vertex_x = thickness/2 * -math.cos(90)
            offset_vertex_y = thickness/2 * math.sin(90)
            
            new_vertex = vertex
            new_vertex.x = vertex.x - offset_vertex_x
            new_vertex.y = vertex.y - offset_vertex_y

            x_center = new_vertex.x + offset_x +unit_pixel_length
            y_center = new_vertex.y + offset_y +unit_pixel_length

        # x_center = vertex.x + offset_x +unit_pixel_length 
        # y_center = vertex.y + offset_y +unit_pixel_length
        # Use meep_block from
        import meepsat.meep_geometry as mpsat_geom

        # x_center, y_center = x_center - thickness, y_center #- thickness
        flair_block_meep = mpsat_geom.meep_block(size = mp.Vector3(length, thickness, 0),
                                                 center = mp.Vector3(x_center, y_center, 0),
                                                 material = flair_material,
                                                 angle= theta2,
                                                 rot_axis= 'z') # This will be always Z

        return flair_block_meep
    
# * ############################################################################################################
# * MAIN FOREBAFFLE CLASS
# * ############################################################################################################


class Forebaffle(object):
    '''
    Class defining a triangular forebaffle structure.
    '''
    def __init__(self,
                 mpsat_sim,
                 epsilon_map,
                 shape= 'linear',
                 angle_degrees=30,
                 x_vertex=None,
                 y_vertex=None,
                 material=None,
                 epsilon_real=5.4,
                 epsilon_imag=0,
                 freq=1/3,
                 name=None,
                 components: Optional[List[ForebaffleComponent]] = None,
                 # Linear Forebaffle parameters :
                 hypotenuse=70,
                 base=None,
                 height=None,
                 # Spline Forebaffle parameters:
                 num_periods=1,
                 amplitude=5,
                 no_of_points=300,
                 scaling_factor=3,
                 spline_degree=3,
                 spline_smoothing=1,
                 fb_thickness=10,
                 add_absorber=True,
                 absorber_side= 'above',
                 absorber_epsilon_real=5.4,
                 absorber_epsilon_imag=0,
                 absorber_thickness=5
):
        '''
        Defines a right angled triangular forebaffle structure with a specific opening angle

        Parameters
        ----------
        mpsat_sim : MEEPSAT
            MEEPSAT simulation object
        epsilon_map: np.ndarray
            Epsilon map for the whole system
            This is useful for adding random geometries which are not trivial using MEEPs objects
        shape: str
            Shape of the forebaffle (default: 'linear')
            Available Shapes: ['linear', 'spline']
        angle_degrees : float, optional
            Angle of the forebaffle in degrees (default: 30)
        x_vertex : float, optional
            X-coordinate of the vertex (default: -300)
            v1 in the code
        y_vertex : float, optional
            Y-coordinate of the vertex (default: bottom of simulation cell)
            v1 in the code
        base : float, optional
            Length of the base of the right angled trianglular forebaffle (default: 40)
        hypotenuse : float, optional
            Length of the hypotenuse of the right angled trianglular forebaffle (default: 70)
        height : float, optional
            Length of the height of the right angled trianglular forebaffle (default: 30)
        material : mp.Medium, optional
            Material for the forebaffle (overrides epsilon if provided)
        epsilon_real : float, optional
            Permittivity of the material (default: 5.4)
        epsilon_imag : float, optional
            Imaginary part of permittivity (default: 0)
        freq : float, optional
            Frequency for material properties (default: 1/3)
        name : str, optional
            Name of the object (default: None)
            
        'THE BELOW PARAMETERS ARE FOR SPLINE FOREBAFFLE DESIGN'
        num_periods: int, optional (only needed if shape = 'spline')
            Number of oscillations between start and end (default: 1)
        amplitude: float, optional (only needed if shape = 'spline')
            Amplitude of oscillation in mm (default: 5)
        no_of_points: int, optional (only needed if shape = 'spline')
            Number of points between start and end (default: 300)
        scaling_factor: float, optional (only needed if shape = 'spline')
            Frequency factor for the oscillation (default: 3)
        spline_degree: int, optional (only needed if shape = 'spline')
            Degree of the spline (default: 3)
        spline_smoothing: float, optional (only needed if shape = 'spline')
            Smoothing factor for the spline (default: 1)
        fb_thickness: float, optional
            Thickness of the forebaffle (default: 10)
        add_absorber: bool, optional
            Whether to add an absorber layer (default: True)
        absorber_side: str, optional
            Which side of the spline to add the absorber (default: 'above')
            available options: ['above', 'below']
        absorber_epsilon_real: float, optional
            Real part of the permittivity for the absorber (default: 5.4)
        absorber_epsilon_imag: float, optional
            Imaginary part of the permittivity for the absorber (default: 0)
        absorber_thickness: float, optional
            Thickness of the absorber layer (default: 2.0)
        '''
        self.mpsat_sim = mpsat_sim
        self.epsilon_map = epsilon_map
        
        # Basic parameters
        self.name = name if name else "Forebaffle"
        self.object_type = 'Forebaffle'
        self.fb_shape = shape
        
        # Geometry parameters
        self.angle_degrees = angle_degrees
        self.angle_radians = np.radians(angle_degrees)
        self.x_vertex = x_vertex if x_vertex is not None else -300
        self.y_vertex = y_vertex if y_vertex is not None else -self.mpsat_sim.cell_size[1]/2
        self.hypotenuse = hypotenuse

        
        if base is None and height is None:
            self.base, self.height = self._calculate_base_height_from_angle_hypotenuse(
                angle_degrees, hypotenuse
            )
            print("base",self.base,"\t","height",self.height)
        elif base is None or height is None:
            raise ValueError("Either provide hypotenuse + angle OR provide all the sides of the forebaffle.")
        else:
            self.base = base
            self.height = height

        # Material properties to be used in simulation using meep geometries
        if material is not None:
            self.material = material
        else:
            # Check if epsilon_real is -inf (perfect conductor)
            if np.isinf(epsilon_real) and epsilon_real < 0:
                self.material = mp.perfect_electric_conductor
            # Check if imaginary part is provided
            elif epsilon_imag != 0:
                self.epsilon_real = epsilon_real
                self.epsilon_imag = epsilon_imag
                self.conductivity = epsilon_imag * 2 * np.pi * freq / epsilon_real
                self.material = mp.Medium(epsilon=self.epsilon_real, D_conductivity=self.conductivity)
            else:
                self.material = mp.Medium(epsilon=epsilon_real)

        # Component system for additional features
        self.components = components if components else []
        
        # Spline parameters
        if shape == 'spline':
            self.spline_num_periods = num_periods
            self.spline_amplitude = amplitude
            self.spline_no_of_points = no_of_points
            self.spline_scaling_factor = scaling_factor
            self.spline_fb_thickness = fb_thickness
            self.spline_degree = spline_degree
            self.spline_smoothing = spline_smoothing
            self.spline_add_absorbers = add_absorber
            self.spline_absorber_side = absorber_side
            self.spline_abs_thickness =absorber_thickness
            self.spline_abs_epsilon_real = absorber_epsilon_real
            self.spline_abs_epsilon_imag = absorber_epsilon_imag

    def __str__(self):
        return f"{self.name}: angle={self.angle_degrees}°, height={self.height}"
    
    def _calculate_base_height_from_angle_hypotenuse(self, angle_degrees, hypotenuse):
        angle_radians = np.radians(angle_degrees)
        # Using ASTC rule from Trigonometry
        if 0 <= angle_degrees <= 90:
            base = float(hypotenuse * np.cos(angle_radians))
            height = float(hypotenuse * np.sin(angle_radians))
        elif 90 < angle_degrees <= 180:
            base = float(hypotenuse * -1* np.cos(angle_radians))
            height = float(hypotenuse * np.sin(angle_radians))  
        elif 180 < angle_degrees <= 270:
            base = float(hypotenuse * -1* np.cos(angle_radians))
            height = float(hypotenuse * -1* np.sin(angle_radians))
        else:
            base = float(hypotenuse * np.cos(angle_radians))
            height = float(hypotenuse * -1* np.sin(angle_radians))
        return base, height

    def calculate_vertices(self):
        """
        Calculate the vertices of the triangular forebaffle based on the provided parameters
        
        Returns
        """
        # Calculate the coordinates of the three vertices
        v1 = mp.Vector3(self.x_vertex, self.y_vertex)  # Vertex where angle is measured
        
        # The right angle is at v2, so we calculate v2 based on the base and angle

        perp_height = self.height
        if 0 <= self.angle_degrees < 90:
            # Quadrant 1: base right, perpendicular up
            v2 = mp.Vector3(self.x_vertex + self.base, self.y_vertex)
            v3 = mp.Vector3(self.x_vertex + self.base, self.y_vertex + perp_height)
            
        elif 90 <= self.angle_degrees < 180:
            # Quadrant 2: base left, perpendicular up
            v2 = mp.Vector3(self.x_vertex - self.base, self.y_vertex)
            v3 = mp.Vector3(self.x_vertex - self.base, self.y_vertex + perp_height)
            
        elif 180 <= self.angle_degrees < 270:
            # Quadrant 3: base left, perpendicular down
            v2 = mp.Vector3(self.x_vertex - self.base, self.y_vertex)
            v3 = mp.Vector3(self.x_vertex - self.base, self.y_vertex - perp_height)
            
        else:  # 270 <= self.angle_degrees < 360
            # Quadrant 4: base right, perpendicular down
            v2 = mp.Vector3(self.x_vertex + self.base, self.y_vertex)
            v3 = mp.Vector3(self.x_vertex + self.base, self.y_vertex - perp_height)
                
        print(f"Calculated vertices: v1={v1}, v2={v2}, v3={v3}")
        print(f"Quadrant: {int(self.angle_degrees // 90) + 1}")
        
        # Consider adding the boundary layer thickness to the vertex positions
        if self.name == 'Right Forebaffle':
            boundary_layer_size = 0#self.mpsat_sim.dpml * self.mpsat_sim.factor_dpml
            if boundary_layer_size > 0:
                # For right forebaffle, we need to consider the boundary layer on the right side
                v1 = mp.Vector3(v1.x - boundary_layer_size, v1.y)
                v2 = mp.Vector3(v2.x - boundary_layer_size, v2.y)
                v3 = mp.Vector3(v3.x - boundary_layer_size, v3.y)
                print(f"Adjusted vertices for boundary layer: v1={v1}, v2={v2}, v3={v3}")

        elif self.name == 'Left Forebaffle':
            boundary_layer_size = 0#self.mpsat_sim.dpml * self.mpsat_sim.factor_dpml
            if boundary_layer_size > 0:
                # For left forebaffle, we need to consider the boundary layer on the left side
                v1 = mp.Vector3(v1.x + boundary_layer_size, v1.y)
                v2 = mp.Vector3(v2.x + boundary_layer_size, v2.y)
                v3 = mp.Vector3(v3.x + boundary_layer_size, v3.y)
                print(f"Adjusted vertices for boundary layer: v1={v1}, v2={v2}, v3={v3}")
                
        elif self.name == 'Top Forebaffle':
            boundary_layer_size = 0#self.mpsat_sim.dpml * self.mpsat_sim.factor_dpml
            if boundary_layer_size > 0:
                # For top forebaffle, we need to consider the boundary layer on the top side
                v1 = mp.Vector3(v1.x, v1.y - boundary_layer_size)
                v2 = mp.Vector3(v2.x, v2.y - boundary_layer_size)
                v3 = mp.Vector3(v3.x, v3.y - boundary_layer_size)
                print(f"Adjusted vertices for boundary layer: v1={v1}, v2={v2}, v3={v3}")
                
        elif self.name == 'Bottom Forebaffle':
            boundary_layer_size = 0#self.mpsat_sim.dpml * self.mpsat_sim.factor_dpml
            if boundary_layer_size > 0:
                # For bottom forebaffle, we need to consider the boundary layer on the bottom side
                v1 = mp.Vector3(v1.x, v1.y + boundary_layer_size)
                v2 = mp.Vector3(v2.x, v2.y + boundary_layer_size)
                v3 = mp.Vector3(v3.x, v3.y + boundary_layer_size)
                print(f"Adjusted vertices for boundary layer: v1={v1}, v2={v2}, v3={v3}")
                
        else:
            logger.warning(f"Unknown forebaffle name '{self.name}' - no boundary layer adjustment applied")
            
        return v1, v2, v3
    

        
    def _create_spline_forebaffle_with_prisms(self, start_vertex, end_vertex):
        """
        Create a spline forebaffle using multiple MEEP prism objects.
        
        This creates a stepped approximation of the spline curve using rectangular
        prism elements, similar to step file export in CAD software.
        
        Parameters
        ----------
        start_vertex, end_vertex : mp.Vector3
            Start and end points of the spline
            
        Returns
        -------
        List[mp.GeometricObject]
            List of prism objects forming the spline structure
        """
        from scipy.interpolate import UnivariateSpline
        
        # Get simulation parameters
        x_start, y_start = start_vertex.x, start_vertex.y
        x_end, y_end = end_vertex.x, end_vertex.y
        
        # Spline parameters
        num_periods = self.spline_num_periods
        amplitude = self.spline_amplitude
        factor = self.spline_scaling_factor
        no_of_points = self.spline_no_of_points
        
        # Generate spline curve
        x_points = np.linspace(x_start, x_end, num=no_of_points)
        y_base = np.linspace(y_start, y_end, len(x_points))
        y_periodic = y_base + amplitude * np.sin(factor * np.pi * num_periods * 
                                                (x_points - x_start) / (x_end - x_start))
        # Calculate offset correction before spline
        y_start_uncorrected = y_periodic[0]
        y_end_uncorrected = y_periodic[-1]
        offset_start = y_start - y_start_uncorrected
        offset_end = y_end - y_end_uncorrected
        
        # Apply linear interpolation of offset across the entire curve
        # This ensures endpoints match while preserving spline shape
        offset_correction = np.linspace(offset_start, offset_end, len(y_periodic))
        y_periodic = y_periodic + offset_correction

        # Create smooth spline
        spline = UnivariateSpline(x_points, y_periodic, k=self.spline_degree, 
                                s=self.spline_smoothing)
        
        # Create prism objects
        geometries = []
        thickness = self.spline_fb_thickness
        
        # # Create material with proper absorption handling
        # if self.epsilon_imag != 0:
        #     # Use conductivity for absorption
        #     freq = 1/3  # Default frequency
        #     conductivity = self.epsilon_imag * 2 * np.pi * freq / self.epsilon_real
        #     material = mp.Medium(epsilon=self.epsilon_real, D_conductivity=conductivity)
        # else:
        #     material = mp.Medium(epsilon=self.epsilon_real)
        material = self.material
        
        # Use fewer segments for prism creation (e.g., 1/10 of original points)
        prism_segments = max(10, no_of_points // 10)  # At least 10 segments
        x_prism_points = np.linspace(x_start, x_end, prism_segments)
        
        for i in range(len(x_prism_points) - 1):
            x1 = x_prism_points[i]
            x2 = x_prism_points[i + 1]
            
            # Evaluate spline at segment endpoints
            y1 = float(spline(x1))
            y2 = float(spline(x2))
            
            # Create a quadrilateral prism for this segment
            # Vertices at top and bottom of the segment
            v1_bottom = mp.Vector3(x1, y1 - thickness/2)
            v2_bottom = mp.Vector3(x2, y2 - thickness/2)
            v2_top = mp.Vector3(x2, y2 + thickness/2)
            v1_top = mp.Vector3(x1, y1 + thickness/2)
            
            # Create prism (quadrilateral)
            prism = mp.Prism(
                vertices=[v1_bottom, v2_bottom, v2_top, v1_top],
                height=self.height,
                axis=mp.Vector3(0, 0, 1),
                material=material
            )
            geometries.append(prism)
        
        # Add absorber layers if needed
        if self.spline_add_absorbers:
            # Create absorber material properly
            if self.spline_abs_epsilon_imag != 0:
                freq = 1/3
                abs_conductivity = self.spline_abs_epsilon_imag * 2 * np.pi * freq / self.spline_abs_epsilon_real
                absorber_material = mp.Medium(epsilon=self.spline_abs_epsilon_real, 
                                             D_conductivity=abs_conductivity)
            else:
                absorber_material = mp.Medium(epsilon=self.spline_abs_epsilon_real)
            
            absorber_thickness = self.spline_abs_thickness
            
            x_prism_points = np.linspace(x_start, x_end, prism_segments)
            
            for i in range(len(x_prism_points) - 1):
                x1 = x_prism_points[i]
                x2 = x_prism_points[i + 1]
                
                y1 = float(spline(x1))
                y2 = float(spline(x2))
                
                if self.spline_absorber_side in ['above', 'both']:
                    # Absorber above
                    v1_inner = mp.Vector3(x1, y1 + thickness/2)
                    v2_inner = mp.Vector3(x2, y2 + thickness/2)
                    v2_outer = mp.Vector3(x2, y2 + thickness/2 + absorber_thickness)
                    v1_outer = mp.Vector3(x1, y1 + thickness/2 + absorber_thickness)
                    
                    absorber_prism = mp.Prism(
                        vertices=[v1_inner, v2_inner, v2_outer, v1_outer],
                        height=self.height,
                        axis=mp.Vector3(0, 0, 1),
                        material=absorber_material
                    )
                    geometries.append(absorber_prism)
                
                if self.spline_absorber_side in ['below', 'both']:
                    # Absorber below
                    v1_outer = mp.Vector3(x1, y1 - thickness/2 - absorber_thickness)
                    v2_outer = mp.Vector3(x2, y2 - thickness/2 - absorber_thickness)
                    v2_inner = mp.Vector3(x2, y2 - thickness/2)
                    v1_inner = mp.Vector3(x1, y1 - thickness/2)
                    
                    absorber_prism = mp.Prism(
                        vertices=[v1_outer, v2_outer, v2_inner, v1_inner],
                        height=self.height,
                        axis=mp.Vector3(0, 0, 1),
                        material=absorber_material
                    )
                    geometries.append(absorber_prism)
        
        return geometries

    def assemble(self):
        """
        Assemble the forebaffle with all components.
        
        Returns
        -------
        List[mp.GeometricObject]
            List of MEEP geometric objects (main structure + components)
        """
        self.v1, self.v2, self.v3 = self.calculate_vertices()
        geometries = []
                
        if self.fb_shape == 'linear':

            # Main forebaffle structure
            main_forebaffle = mp.Prism(
                vertices=[self.v1, self.v2, self.v3],
                height=self.height,
                axis=mp.Vector3(0, 0, 1),
                material=self.material
            )
            
            geometries.append(main_forebaffle)
            
            # Add component geometries
            for component in self.components:
                geometries.extend(component.get_geometry(self))
                    
            logger.info(f"Forebaffle assembled with {len(geometries)} geometric objects")
            
            return geometries
        
        # elif self.fb_shape == 'spline':
        #     self.epsilon_map = self._create_spline_forebaffle(epsilon_map= self.epsilon_map,
        #                                                       start_vertex=v3,
        #                                                       end_vertex=v1)

        #     logger.info(f"Forebaffle assembled with spline shape. The following parameters were used:\n"
        #                 f"    start vertex: {v1}\n"
        #                 f"    end vertex: {v3}\n"
        #                 f"    forebaffle_epsilon: {self.epsilon_real + self.epsilon_imag * 1j}\n"
        #                 f"    forebaffle_thickness: {self.spline_fb_thickness}\n"
        #                 f"    absorber_epsilon: {self.spline_abs_epsilon_real + self.spline_abs_epsilon_imag * 1j}\n"
        #                 f"    absorber_thickness: {self.spline_abs_thickness}\n"
        #                 f"    spline_add_absorbers: {self.spline_add_absorbers}\n"
        #                 f"    spline_absorber_side: {self.spline_absorber_side}")
            
        #     return self.epsilon_map
        
        # elif self.fb_shape == 'spline':
            
        #     geometries = self._create_spline_forebaffle_with_prisms(
        #         start_vertex=v3,
        #         end_vertex=v1
        #     )
        #     logger.info(f"Forebaffle assembled with {len(geometries)} spline prism segments")
        #     return geometries
        elif self.fb_shape == 'spline':
            
            # Ensure start_vertex has smaller x-coordinate for monotonic spline
            start_v = self.v3
            end_v = self.v1
            if start_v.x > end_v.x:
                start_v, end_v = end_v, start_v
            
            geometries = self._create_spline_forebaffle_with_prisms(
                start_vertex=start_v,
                end_vertex=end_v
            )
            logger.info(f"Forebaffle assembled with {len(geometries)} spline prism segments")
            return geometries

        else:
            raise ValueError(f"Unknown forebaffle shape '{self.fb_shape}'")
        

    


