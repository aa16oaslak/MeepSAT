import sys
import os
import site
from pathlib import Path
#import meep_testings as mp
import meep as mp
import numpy as np
import warnings
import scipy.optimize as sc
import h5py
import math

import matplotlib.pyplot as plt
from matplotlib import rc

import meepsat.helpers as exf

# ! THE FOLLOWING SEGMENT OF THE CODE WAS TAKEN FROM THE MEEPART CODE
# ! AND ADAPTED TO THE NEW STRUCTURE OF THE CODE IN THE MEEPSAT PACKAGE
  
# ~ LENS CLASS
class AsphericLens(object):
    '''
    Class defining an aspheric lens of arbitrary shape and 
    position, and creating the function of sag (curvature) 
    used to create the permitttivity map
    '''
    
    def __init__(self,
                 diameter, 
                 r1, 
                 r2, 
                 thick,
                 c1 = 0, c2 = 0, 
                 # lens type
                 lens_type = 'aspheric',
                 # Add higher-order aspheric coefficients
                 a1_coeffs = None, a2_coeffs = None,
                 name = None, 
                 x = 0., y = 0., 
                 n_refr = 1.52, 
                 #! Single ARC parameters (useless- but not touching for now to avoid bugs)       
                 AR_left = None, 
                 AR_right = None,
                 AR_material = 1.52/4,
                 #! Parameters for multi-layer ARCs (single + multi-layer can be used simultaneously)
                 AR_left_layers = None,
                 AR_left_materials = None,
                 AR_right_layers = None,
                 AR_right_materials = None,
                 #! Adding stepped pyramid ARC here
                 ARC_type = None,
                 step_ARC_nlayers = None,
                 step_ARC_pitch = None,
                 step_ARC_kerf = None,
                 step_ARC_depth = None,
                 step_ARC_width = None,
                 step_ARC_material = None,
                 step_ARC_angle = 'perpendicular_to_surface',
                 step_ARC_rot_axis = 'z',
                 step_ARC_offset = [0,0],
                 delam_thick = 0,
                 delam_width = 10,
                 radial_slope = 0,
                 axial_slope = 0,
                 surf_err_width = 1,
                 surf_err_scale = 0,
                 custom_def = False,
                 eps = None,
                 mpsat_sim = None):
        '''
        Defines the attributes of the Lens object

        Arguments
        ---------       
        diameter : float 
            Diameter of the lens
        r1 : float
            Left surface curvature radius
        r2 : float  
            Right surface cruvature radius
        thick : float
            Thickness of lens on the optical axis
        name : str, optional
            Name of object (default : None)
        c1 : float, optional
            Left surface aspheric parameter (default : 0)
        c2 : float, optional
            Right surface aspheric parameter (default : 0)
        lens_type : str, optional
            Type of lens, either 'aspheric' or 'extended_aspheric'
            (default : 'extended_aspheric')
        a1_coeffs : list, optional
            Higher-order aspheric coefficients for left surface (Currently supported for the first 3 coefficients)
            (default : None)
        a2_coeffs : list, optional
            Higher-order aspheric coefficients for right surface (Currently supported for the first 3 coefficients)
            (default : None)
        x : float, optional
            Position of center of left surface along x axis (default : 0)
        y : float, optional
            Position of center of left surface along y axis (default : 0)
        n_refr : float, optional
            Index of refraction of the lens. 
            Set to HDPE by default.
            (default : 1.52) 

    
        #! Single ARC parameters (useless- but not touching for now to avoid bugs)       
        AR_left : float, optional
            Anti Reflection coating thickness of left surface of the lens
            (default : None) 
        AR_right : float, optional
            Anti Reflection coating thickness of right surface of the lens
            (default : None) 
        AR_material : float, optional
            Refractive index of the AR coating material

        #! Parameters for multi-layer ARCs (single + multi-layer can be used simultaneously)
        AR_left_layers : list, optional
            List of thicknesses of each layer in the multi-layer ARC on the left surface
            (default : None)
        AR_left_materials : list, optional
            List of refractive indices of each layer in the multi-layer ARC on the left surface
            (default : None)
        AR_right_layers : list, optional
            List of thicknesses of each layer in the multi-layer ARC on the right surface
            (default : None)
        AR_right_materials : list, optional
            List of refractive indices of each layer in the multi-layer ARC on the right surface
            (default : None)


        #! stepped pyramid ARC parameters
        ARC_type : str, optional
            Type of ARC, either 'stepped_pyramid' or 'default' in the current version
        
        step_ARC_nlayers : int, optional
            Number of layers in the stepped pyramid ARC (default : None)
        
        step_ARC_pitch : float, optional
            Pitch of the stepped pyramid ARC (default : None)

        step_ARC_kerf : float, optional
            Kerf of the stepped pyramid ARC (default : None)

        step_ARC_depth : float, optional
            Depth of the stepped pyramid ARC (default : None)

        step_ARC_width : float, optional
            Width of the stepped pyramid ARC (default : None)

        step_ARC_material : float, optional
            Refractive index of the stepped pyramid ARC material (default : None)

        step_ARC_angle : str, optional
            Angle of the stepped pyramid ARC, either 'perpendicular_to_surface' or None
            If 'perpendicular_to_surface', the ARC is perpendicular to the surface of the lens.
            (default : 'perpendicular_to_surface')

        step_ARC_rot_axis : str, optional
            Axis of rotation for the stepped pyramid ARC, either 'x', 'y', or 'z'
            (default : 'z' i.e. rotation in the xy-plane)

        step_ARC_offset : float, optional
            Offset of the stepped pyramid ARCs base layer's edge center from the lens surface (default : [0, 0])
            [0, 0] means no offset, i.e. the base layer's edge center is at the lens surface.
            
        delam_thick : float, optional
            Thickness of delaminated lumps at their center
            (default : 0)
        delam_width : float, optional
            Width of delaminated lumps along y-axis
            Used in a division, hence default is not 0.
            (default : 10)
        radial_slope : float, optional
            Derivative of the index of refraction w.r.t y-axis (default : 0)
        axial_slope : float, optional
            Derivative of the index of refraction w.r.t x-axis (default : 0)
        surf_err_scale : float, optional
            Width of the gaussian of the distribution of surface errors
            (default : 0)
        surf_err_width : float, optional
            Size of the bins of same surface error (default : 1)
        custom_def : bool, optional 
            Enables custom deformation function (default : False)

        # ^ ######### ^ #
        eps: np.array, optional
           Dielectric map of the other components in the system 
        
        mpsat_sim: object
            MEEPSAT object produced from sim_init() in simulation_2D.py
        '''
        self.name = name                  
        self.diameter = diameter        
        self.r1 = r1                    
        self.r2 = r2                    
        self.c1 = c1                    
        self.c2 = c2

        # Define the lens type
        self.lens_type = lens_type
        if lens_type not in ['aspheric', 'extended_aspheric']:
            raise ValueError("lens_type must be either 'aspheric' or 'extended aspheric'")
        
        # Add higher-order coefficients
        self.a1_coeffs = a1_coeffs if a1_coeffs is not None else None
        self.a2_coeffs = a2_coeffs if a2_coeffs is not None else None
        
        
        self.thick = thick              
        self.x = x                      
        self.y = y                      
        self.eps = n_refr**2            
        self.object_type = 'Lens'

        #! Single ARC parameters (useless- but not touching for now to avoid bugs)       
        self.AR_left = AR_left          
        self.AR_right = AR_right        
        self.AR_material = AR_material #!material of the AR coating  

        #! Parameters for multi-layer ARCs (single + multi-layer can be used simultaneously)
        self.left_layers=AR_left_layers 
        self.left_materials=AR_left_materials
        self.right_layers=AR_right_layers 
        self.right_materials=AR_right_materials
        
        #! stepped pyramid ARC parameters
        if ARC_type == 'stepped_pyramid':
            self.ARC_type = ARC_type
            self.step_ARC_nlayers = step_ARC_nlayers
            self.step_ARC_pitch = step_ARC_pitch
            self.step_ARC_kerf = step_ARC_kerf
            self.step_ARC_depth = step_ARC_depth
            self.step_ARC_width = step_ARC_width
            self.step_ARC_material = step_ARC_material
            self.step_ARC_angle = step_ARC_angle
            self.step_ARC_rot_axis = step_ARC_rot_axis
            self.step_ARC_offset = step_ARC_offset

        self.delam_thick = delam_thick  
        self.delam_width = delam_width  
        self.radial_slope = radial_slope
        self.axial_slope = axial_slope  
        self.surf_err_width = surf_err_width    
        self.surf_err_scale = surf_err_scale    
        self.custom_def = custom_def
        self.permittivity_map = eps # ~ THIS IS THE EPSILON MAP

        # Extracting the required parameters from the MEEPSAT object
        self.res = mpsat_sim.resolution 
        self.dpml = mpsat_sim.factor_dpml*mpsat_sim.dpml # mpsat_sim.dpml 
        # 2 times because there's pml on both sides
        self.size_x, self.size_y, self.size_z = mpsat_sim.cell_size[0] - 2*self.dpml, mpsat_sim.cell_size[1] - 2*self.dpml, mpsat_sim.cell_size[2]
        self.mpsat_sim = mpsat_sim # ~ MEEPSAT object

        #TESTING IMPORTED DEFORMED PROFILE AS CSV
        #deform = []
        #with open('deformedsurface.csv') as csvfile:
        #    reader = csv.reader(csvfile, delimiter=',')
        #    k = 0
        #    for row in reader:
        #        k+= 1 
        #        if k>=11 :
        #            deform.append(np.float(row[2]))
        #deform0 = 2*deform[0]-deform[1]
        #deform.insert(0, deform0)
        #self.deform = deform

    def position(self):
        if self.name is not None : 
            return self.name + ' at position ' + str(self.x)
        else :
            return 'Lens at position ' + str(self.x)

    def even_asphere_lens_eqn(self, y, r, k, higher_order_coeffs):#=[0, 0, 0]):
        '''
        Aspheric lens equation for even aspheric coefficients
        Arguments
        ---------
        y : float
            Distance from optical axis at which the sag is computed
        r : float
            Curvature radius of the lens
        k : float
            Aspheric coefficient
        higher_order_coeffs [A2, A3, A4] : list, optional
            Higher-order aspheric coefficients (default : [0, 0, 0])
        '''
        A2, A3, A4 = higher_order_coeffs
        sag_value = (y**2/r) / (1 + np.sqrt(1 - (1 + k)*y**2/r**2)) + A2 * y**2 + A3 * y**4 + A4 * y**6
        return sag_value

    def extended_asphere_lens_eqn(self, y, r, k, higher_order_coeffs):#=[0, 0, 0]):
        '''
        Aspheric lens equation for extended aspheric coefficients
        Arguments
        ---------
        y : float
            Distance from optical axis at which the sag is computed
        r : float
            Curvature radius of the lens
        k : float
            Aspheric coefficient
        higher_order_coeffs [A1, A2, A3] : list, optional
            Higher-order aspheric coefficients (default : [0, 0, 0])
        '''
        A1, A2, A3 = higher_order_coeffs
        sag_value = (y**2/r) / (1 + np.sqrt(1 - (1 + k)*y**2/r**2)) + A1 * y + A2 * y**2 + A3 * y**3
        return sag_value
    
    def left_surface(self, y):
        '''
        Aspheric lens equation for left surface

        Arguments
        ---------
        y : float
            Distance from optical axis at which the sag is computed

        Returns
        -------
        sag : float
            Sag at at distance y from optical axis.
        '''
        higher_order_coeffs_left = self.a1_coeffs if self.a1_coeffs is not None else [0, 0, 0]

        if self.lens_type == 'aspheric':
            left_surface = self.even_asphere_lens_eqn(y= y, 
                                                       r= self.r1, 
                                                       k = self.c1, 
                                                       higher_order_coeffs= higher_order_coeffs_left)
        elif self.lens_type == 'extended_aspheric':
            left_surface = self.extended_asphere_lens_eqn(y= y, 
                                                       r= self.r1, 
                                                       k = self.c1, 
                                                       higher_order_coeffs= higher_order_coeffs_left)
        else:
            raise ValueError("Invalid lens type. Use 'aspheric' or 'extended_aspheric'.")
        
        if self.r1 != np.inf:
            return left_surface
        else:
            # If the radius is infinite, returns a flat surface, i.e. 0 sag
            return 0
        

    def right_surface(self, y):
        '''
        Aspheric lens equation for right surface

        Arguments
        ---------
        y : float
            Distance from optical axis at which the sag is computed

        Returns
        -------
        sag : float
            Sag at at distance y from optical axis.
        '''
        higher_order_coeffs_right = self.a2_coeffs if self.a1_coeffs is not None else [0, 0, 0]
        
        if self.lens_type == 'aspheric':
            right_surface = self.even_asphere_lens_eqn(y= y, 
                                                       r= self.r2, 
                                                       k = self.c2, 
                                                       higher_order_coeffs= higher_order_coeffs_right)
        elif self.lens_type == 'extended_aspheric':
            right_surface = self.extended_asphere_lens_eqn(y= y, 
                                                            r= self.r2, 
                                                            k = self.c2, 
                                                            higher_order_coeffs= higher_order_coeffs_right)
        else:
            raise ValueError("Invalid lens type. Use 'aspheric' or 'extended_aspheric'.")
        
        if self.r2 != np.inf:
            return right_surface
        else:
            # If the radius is infinite, returns a flat surface, i.e. 0 sag
            return 0

    def delamination(self, y, y0):       
        '''
        Returns the air layer thickness that makes delamination, it is 
        zero everywhere excpet where there's the lump, centered on y0, defined by
        its width and thickness

        Arguments
        ---------
        y : float
            Distance from optical axis at which 
            the delamination is evaluated
        y0 : float
            Center of the delaminated lump
        Returns
        -------
        delam : float
            Delamination layer thickness along x-axis at y
        '''

        thick = self.delam_thick
        width = self.delam_width
        return np.abs(min((((y-y0)/width)**2-1)*thick, 0))

    def cust_def(self, y):
        '''
        Returns custom deformation function

        Arguments
        ---------
        y : float
            Distance from optical axis at which 
            the deformation is evaluated
        Returns
        -------
        deform : float
            Deformation of surface along x-axis at y
        '''
        if self.custom_def :
            # ~ Insert here the custom function
            return 0

        else :
            return 0
        
    def make_lens_bubbles(self, radius, nb_clusters, nb_per_cluster):
            '''
            Introduces clusters of air bubbles inside the lenses of the system, 
            each cluster has a central bubble and a number of smaller bubble gathered
            around this central bubble
            
            Arguments
            -----------------
            radius : float
                Radius of the central bubble
            nb_clusters : float
                Number of clusters per lens
            nb_per_cluster : 
                Number of bubbles surrounding the main one in each
                cluster
            Notes
            -----
            This function alters the permittivity map. 
            '''

            res = self.res

            #Function which, given a radius, that 
            #returns the indices of the points within 
            #the circle centered on (0,0)
            def bubble(rad):
                '''
                Introduces clusters of air bubbles inside the lenses of the system, 
                each cluster has a central bubble and a number of smaller bubble gathered
                around this central bubble
            
                Arguments
                -----------------
                rad : float
                    Radius of the bubble

                Returns
                -------
                bubble : array
                    Array of indexes within radius
                '''
                bubble = []
                for k in range(-rad, rad+1):
                    for j in range(-rad, rad+1):
                        if k**2 + j**2 <= rad**2 :
                            bubble.append([k,j])
                return np.array(bubble)

            #List of centers of bubbles
            list_centers = []

            #List of radii of bubbles
            list_radii = []

            #Iterate for all lenses
            for component in self.components:

                if component.object_type == 'Lens':

                    #Lens thickness
                    thick = component.thick*res

                    #So that the bubbles aren't generated 
                    #on the very edge of the lenses
                    low = np.int64(np.around(self.size_y*res*0.1))
                    high = np.int64(np.around(self.size_y*res*0.9))

                    #Iterate over cluster numbers
                    for i in range(nb_clusters):

                        #The center of the lens can be anywhere on the y axis
                        y0 = np.random.randint(low = low, high = high)
                    
                        #Left surface sag
                        x_left = np.int64(np.around((
                            component.left_surface(y0/res - self.size_y/2) + 
                            component.x)*res))
                        #Right surface sag       
                        x_right = np.int64(np.around((
                            component.right_surface(y0/res - self.size_y/2) + 
                            component.x)*res + 
                            thick))

                        #The center of the cluster has to be inside the lens
                        x0 = np.random.randint(low = x_left, high = x_right+1)

                        #Radius of the main can vary by 10 percent
                        radius_0 = radius*(0.9 + np.random.random()*0.2)
                    
                        #Update lists
                        list_centers.append([x0,y0])
                        list_radii.append(radius_0)

                        #Iterate over the number of surrounding bubbles
                        for k in range(nb_per_cluster):

                            #The center of each surrounding bubble is random, within
                            #a certain distance of the central bubble
                            phi = np.random.random()*2*np.pi
                            r = radius_0*(1 + np.random.random()*3)

                            #change of variables
                            x_k = np.int64(np.around(r*np.cos(phi)*res))
                            y_k = np.int64(np.around(r*np.sin(phi)*res))

                            #The radius is a function of distance, the farther the 
                            #smaller
                            radius_k = radius_0*np.exp(-r/(3*radius_0))*np.random.random()

                            #Update lists
                            list_centers.append([x0+x_k, y0+y_k])
                            list_radii.append(radius_k)

            list_centers = np.array(list_centers)
            list_radii = np.array(list_radii)
            list_all = []

            #Making bubbles for all centers and radii
            for k in range(len(list_centers)):
                radius_k = np.int64(np.around(list_radii[k]*res))
                bubble_k = bubble(radius_k)
                for u in bubble_k : 
                    list_all.append(list_centers[k] + u)

            #Update the map
            for index in list_all : 
                self.permittivity_map[index[0], index[1]] = 1
        

    def write_lens(self, comp, eps_map, res):
        '''
        The lens equation returns a sag (distance from plane orth. to
        optical axis) as a function of distance from optical axis y,
        so the code cycles through the different y to change the 
        dielectric map between left surface and right surface
        ---------
        comp : component
            Lens component object
        eps_map : 2D or 3D array
            Dielectric map on which the lens will be written
        res : float
            Resolution of map
        '''

        # The y axis has its zero in the middle of the cell, the offset
        # is mid_y
        mid_y = np.int64(self.size_y*res/2)

        #Thickness of the lens on optical axis
        thick = comp.thick*res

        #Generate the center of the lumps made by delamination, 
        #different for the left and right surface
        high = np.int64(np.around(self.size_y/2))
        y0_left = np.random.randint(low = -high, high = high)
        y0_right = np.random.randint(low = -high, high = high)

        radius = np.int64(np.float64(comp.diameter*res/2))

        #Generates the bins of random surface errors.
        if comp.surf_err_scale!=0 :
            nb_bins = int(comp.diameter/comp.surf_err_width)
            err_left = np.around(np.random.normal(scale = comp.surf_err_scale*res,
                                                  size = nb_bins))
            err_right = np.around(np.random.normal(scale = comp.surf_err_scale*res, 
                                                   size = nb_bins))

        if comp.surf_err_scale == 0:
            nb_bins = int(comp.diameter/comp.surf_err_width)
            err_left = np.zeros(nb_bins)
            err_right = np.zeros(nb_bins)
        
        #Iterates y over the radius, as the lenses are symmetric
        #above and below the optical axis
        for y_res in range(radius) :           

            #Left surface sag
            x_left = np.int64(np.around((
                        comp.left_surface(y_res/res) + self.dpml + 
                        comp.x - comp.cust_def((y_res+mid_y)/res))*res))
            #Right surface sag       
            x_right = np.int64(np.around((
                        comp.right_surface(y_res/res) + 
                        comp.x + self.dpml -
                        comp.cust_def((y_res+mid_y)/res))*res + 
                        thick))
            
            #Above and below the optical axis :
            y_positive = int(self.dpml*res + mid_y + y_res)
            y_negative = int(self.dpml*res + mid_y - y_res)

            #Get the delamination as a function of y on left surface
            delam_pos_L = np.int64(np.around(res*
                comp.delamination(y_res/res, y0_left)))
            delam_neg_L = np.int64(np.around(res*
                comp.delamination(-y_res/res, y0_left)))

            #Get the delamination as a function of y on right surface
            delam_pos_R = np.int64(np.around(res*
                comp.delamination(y_res/res, y0_right)))
            delam_neg_R = np.int64(np.around(res*
                comp.delamination(-y_res/res, y0_right)))
            
            #Gradient in the index
            #ONLY WORKS WHEN NO SURFACE DEFECT
            radial_slope = comp.radial_slope/res
            axial_slope = comp.axial_slope/res
            if radial_slope != 0 or axial_slope != 0 : 
            
                eps0 = comp.eps
                x0 = np.int64(np.around(comp.x*res))
                x_range = range(x_left, x_right+1) 
                #The value is squared as the permittivity is index squared
                eps_line = [eps0 + 
                            (y_res*radial_slope)**2 + 
                            ((k-x0)*axial_slope)**2 for k in x_range]
            if radial_slope ==0 and axial_slope == 0 :
                eps_line = comp.eps


            #Surface error
            err_bin_idx = int(np.around(y_res/res/comp.surf_err_width))
            err_left_pos = int(err_left[err_bin_idx])
            err_left_neg = int(err_left[- err_bin_idx])

            err_right_pos = int(err_left[err_bin_idx]) 
            err_right_neg = int(err_left[- err_bin_idx])

            x_left_neg = int(x_left + err_left_neg)
            x_left_pos = int(x_left + err_left_pos)

            x_right_neg = int(x_right + err_right_neg)
            x_right_pos = int(x_right + err_right_pos)


            #Write lens between left and right surface below optical axis
            eps_map[x_left_neg: x_right_neg+1, y_negative] *= eps_line
            
            #So that the center line is not affected twice :
            if y_res != 0 :
                #Write lens between left and right surface above optical axis
                eps_map[x_left_pos: x_right_pos+1, y_positive] *= eps_line
            
            #Write AR coating on left surface
            if comp.AR_left is not None :

                AR_thick = np.int64(np.around(comp.AR_left*res))

                eps_map[x_left_neg - AR_thick - delam_neg_L: x_left_neg - 
                        delam_neg_L, y_negative] *= comp.AR_material

                if y_res != 0 :
                    eps_map[x_left_pos - AR_thick - delam_pos_L: x_left_pos - 
                            delam_pos_L, y_positive] *= comp.AR_material
            
            #Write AR coating on right surface                    
            if comp.AR_right is not None :
                
                AR_thick = np.int64(np.around(comp.AR_right*res))

                eps_map[x_right_neg + 1 + delam_neg_R: AR_thick + x_right_neg + 
                        1 + delam_neg_R, y_negative] *= comp.AR_material

                if y_res != 0 :
                    eps_map[x_right_pos + 1 + delam_pos_R: AR_thick + 
                            x_right_pos + 1 + delam_pos_R, 
                            y_positive] *= comp.AR_material
                    
                    
    
    def assemble(self):
        """
        Assembling the lens object by callling the write_lens method
        """
        self.write_lens(self, self.permittivity_map, self.res)
        return self.permittivity_map
        

    ### ^ FOR PLOTTING AND SAVING THE EPSILON MAP ^ ###

    def plot_lenses(self, save = False):
        '''
        Plots the permittivity map, where we can see only the lenses,
        allows to check their dispostion and shape
        '''
        extent = (0, 
                  len(self.permittivity_map[:])/self.res,
                  0,
                  len(self.permittivity_map[:][0])/self.res)
        plt.figure(dpi = 150)
        plt.title('Permittivity map')
        plt.imshow(self.permittivity_map.transpose(), extent = extent)
        if save:
            plt.savefig('Lenses.png')
        plt.show()
        plt.close()
    
    # def write_h5file(self, parallel=False, filename='epsilon_map'):
    #     '''
    #     Writes the file that will then be 
    #     read within the MEEP simulation

    #     Arguments
    #     ---------
    #     parallel : bool, optional
    #         If the computation is run in parallel (default : False)
    #     filename : str, optional
    #         Name of the permittivity map file written. 
    #         Needs to be the same name given to the MEEP simulation
    #         (default : 'epsilon_map')
    #     '''
    #     self.mapname = filename
    #     if parallel:
    #         from mpi4py import MPI
    #         comm = MPI.COMM_WORLD
    #         if not h5py.get_config().mpi:
    #             raise ValueError("h5py was built without MPI support, can't use mpio driver")
            
    #         with h5py.File(filename + '.h5', 'w', driver='mpio', comm=comm) as h:
    #             size_x = len(self.permittivity_map[:, 0])
    #             size_y = len(self.permittivity_map[0, :])
    #             dset = h.create_dataset('eps', (size_x, size_y), dtype='float32')
    #             dset[:, :] = self.permittivity_map
    #     else:
    #         with h5py.File(filename + '.h5', 'w') as h:
    #             size_x = len(self.permittivity_map[:, 0])
    #             size_y = len(self.permittivity_map[0, :])
    #             dset = h.create_dataset('eps', (size_x, size_y), 
    #                                     dtype='float32', 
    #                                     compression='gzip')
    #             dset[:, :] = self.permittivity_map

    # def write_h5file(self, parallel=False, filename='epsilon_map'):
    #     '''
    #     Writes the file that will then be 
    #     read within the MEEP simulation

    #     Arguments
    #     ---------
    #     parallel : bool, optional
    #         If the computation is run in parallel (default : False)
    #     filename : str, optional
    #         Name of the permittivity map file written. 
    #         Needs to be the same name given to the MEEP simulation
    #         (default : 'epsilon_map')
    #     '''
    #     self.mapname = filename
        
    #     if parallel:
    #         from mpi4py import MPI
    #         comm = MPI.COMM_WORLD
    #         rank = comm.Get_rank()
            
    #         if not h5py.get_config().mpi:
    #             raise ValueError("h5py was built without MPI support, can't use mpio driver")
            
    #         with h5py.File(filename + '.h5', 'w', driver='mpio', comm=comm) as h:
    #             size_x = len(self.permittivity_map[:, 0])
    #             size_y = len(self.permittivity_map[0, :])
    #             dset = h.create_dataset('eps', (size_x, size_y), dtype='float32')
                
    #             # Use collective I/O - all processes must participate
    #             with dset.collective:
    #                 # Only rank 0 writes the data
    #                 if rank == 0:
    #                     dset[:, :] = self.permittivity_map.astype('float32')
            
    #         comm.barrier()
    #     else:
    #         with h5py.File(filename + '.h5', 'w') as h:
    #             size_x = len(self.permittivity_map[:, 0])
    #             size_y = len(self.permittivity_map[0, :])
    #             dset = h.create_dataset('eps', (size_x, size_y), 
    #                                     dtype='float32', 
    #                                     compression='gzip')
    #             dset[:, :] = self.permittivity_map

    # def write_h5file(self, parallel=False, filename='epsilon_map'):
    #     '''
    #     Writes the file that will then be 
    #     read within the MEEP simulation

    #     Arguments
    #     ---------
    #     parallel : bool, optional
    #         If the computation is run in parallel (default : False)
    #     filename : str, optional
    #         Name of the permittivity map file written. 
    #         Needs to be the same name given to the MEEP simulation
    #         (default : 'epsilon_map')
    #     '''
    #     self.mapname = filename
        
    #     # Check if permittivity_map is properly initialized
    #     if self.permittivity_map is None:
    #         raise ValueError("permittivity_map is None. Call assemble() before write_h5file()")
        
    #     # Make sure permittivity_map is a proper 2D array
    #     try:
    #         size_x = len(self.permittivity_map[:, 0])
    #         size_y = len(self.permittivity_map[0, :])
    #     except (IndexError, AttributeError, TypeError):
    #         raise ValueError("permittivity_map has incorrect shape or type. Expected a 2D array.")
        
    #     if parallel:
    #         try:
    #             from mpi4py import MPI
    #             comm = MPI.COMM_WORLD
    #             rank = comm.Get_rank()
    #             size = comm.Get_size()
                
    #             if not h5py.get_config().mpi:
    #                 raise ValueError("h5py was built without MPI support, can't use mpio driver")
                
    #             print(f"Process {rank}/{size}: Creating HDF5 file with parallel I/O")
                
    #             # Create file with parallel access
    #             with h5py.File(filename + '.h5', 'w', driver='mpio', comm=comm) as h:
    #                 # Create dataset
    #                 dset = h.create_dataset('eps', (size_x, size_y), dtype='float32')
                    
    #                 # Use collective I/O for better compatibility with ROMIO drivers
    #                 dset.write_direct(self.permittivity_map.astype('float32'))
                    
    #                 # Force synchronization to ensure all processes have written their data
    #                 h.flush()
                    
    #             # Additional MPI barrier to ensure all processes complete writing
    #             comm.barrier()
    #             print(f"Process {rank}/{size}: HDF5 file written successfully")
                
    #         except Exception as e:
    #             from mpi4py import MPI
    #             rank = MPI.COMM_WORLD.Get_rank()
    #             print(f"Process {rank}: Error in parallel write: {str(e)}")
                
    #             # Fallback to serial write on rank 0 only
    #             if rank == 0:
    #                 print("Falling back to serial write...")
    #                 try:
    #                     with h5py.File(filename + '.h5', 'w') as h:
    #                         dset = h.create_dataset('eps', (size_x, size_y), 
    #                                                 dtype='float32', 
    #                                                 compression='gzip')
    #                         dset[:, :] = self.permittivity_map
    #                     print("Serial fallback write successful")
    #                 except Exception as fallback_error:
    #                     print(f"Serial fallback also failed: {str(fallback_error)}")
    #                     raise
                
    #             # Ensure all processes wait for rank 0 to complete
    #             MPI.COMM_WORLD.barrier()
                
    #     else:
    #         try:
    #             with h5py.File(filename + '.h5', 'w') as h:
    #                 dset = h.create_dataset('eps', (size_x, size_y), 
    #                                         dtype='float32', 
    #                                         compression='gzip')
    #                 dset[:, :] = self.permittivity_map
    #             print(f"HDF5 file written successfully in serial mode")
    #         except Exception as e:
    #             print(f"Error in serial write: {str(e)}")
    #             raise

    #! def write_h5file(self, parallel=False, filename='epsilon_map'):
    #     '''
    #     Writes the file that will then be 
    #     read within the MEEP simulation

    #     Arguments
    #     ---------
    #     parallel : bool, optional
    #         If the computation is run in parallel (default : False)
    #     filename : str, optional
    #         Name of the permittivity map file written. 
    #         Needs to be the same name given to the MEEP simulation
    #         (default : 'epsilon_map')
    #     '''
    #     self.mapname = filename
        
    #     if parallel:
    #         from mpi4py import MPI
    #         comm = MPI.COMM_WORLD
    #         rank = comm.Get_rank()
            
    #         # Only rank 0 writes the file
    #         if rank == 0:
    #             with h5py.File(filename + '.h5', 'w') as h:
    #                 size_x = len(self.permittivity_map[:, 0])
    #                 size_y = len(self.permittivity_map[0, :])
    #                 dset = h.create_dataset('eps', (size_x, size_y), 
    #                                         dtype='float32')
    #                 dset[:, :] = self.permittivity_map.astype('float32')
            
    #         # Wait for rank 0 to finish writing
    #         comm.barrier()
    #     else:
    #         with h5py.File(filename + '.h5', 'w') as h:
    #             size_x = len(self.permittivity_map[:, 0])
    #             size_y = len(self.permittivity_map[0, :])
    #             dset = h.create_dataset('eps', (size_x, size_y), 
    #                                     dtype='float32', 
    #                                     compression='gzip')
    #             dset[:, :] = self.permittivity_map

    def write_h5file(self, parallel=False, filename='epsilon_map'):
        '''
        Writes the file that will then be 
        read within the MEEP simulation

        Arguments
        ---------
        parallel : bool, optional
            If the computation is run in parallel (default : False)
        filename : str, optional
            Name of the permittivity map file written. 
            Needs to be the same name given to the MEEP simulation
            (default : 'epsilon_map')
        '''
        import os
        
        # Store the full path
        self.mapname = filename
        full_path = filename + '.h5'
        
        if parallel:
            from mpi4py import MPI
            comm = MPI.COMM_WORLD
            rank = comm.Get_rank()
            
            # Only rank 0 writes the file
            if rank == 0:
                # Ensure directory exists
                file_dir = os.path.dirname(full_path) if os.path.dirname(full_path) else '.'
                os.makedirs(file_dir, exist_ok=True)
                
                # Remove old file if it exists
                if os.path.exists(full_path):
                    os.remove(full_path)
                
                with h5py.File(full_path, 'w') as h:
                    size_x = len(self.permittivity_map[:, 0])
                    size_y = len(self.permittivity_map[0, :])
                    dset = h.create_dataset('eps', (size_x, size_y), 
                                            dtype='float32')
                    dset[:, :] = self.permittivity_map.astype('float32')
                    h.flush()  # Ensure data is written to disk
                
                print(f"Rank 0: HDF5 file written to {os.path.abspath(full_path)}")
            
            # Wait for rank 0 to finish writing
            comm.barrier()
            
            # All ranks verify the file exists
            if not os.path.exists(full_path):
                raise FileNotFoundError(f"HDF5 file not found at {os.path.abspath(full_path)}")
            
            # Additional barrier to ensure file system sync
            comm.barrier()            

        else:
            # Ensure directory exists
            file_dir = os.path.dirname(full_path) if os.path.dirname(full_path) else '.'
            os.makedirs(file_dir, exist_ok=True)
            
            # Remove old file if it exists
            if os.path.exists(full_path):
                os.remove(full_path)
            
            with h5py.File(full_path, 'w') as h:
                size_x = len(self.permittivity_map[:, 0])
                size_y = len(self.permittivity_map[0, :])
                dset = h.create_dataset('eps', (size_x, size_y), 
                                        dtype='float32', 
                                        compression='gzip')
                dset[:, :] = self.permittivity_map
                h.flush()  # Ensure data is written to disk
            
            print(f"HDF5 file written to {os.path.abspath(full_path)}")


    def write_lens_nARC(self, comp, eps_map, res, 
                    AR_left_layers=None, AR_left_materials=None,
                    AR_right_layers=None, AR_right_materials=None):
        '''
        Enhanced version of write_lens that supports multiple AR coating layers.
        The lens equation returns a sag (distance from plane orth. to
        optical axis) as a function of distance from optical axis y,
        so the code cycles through the different y to change the 
        dielectric map between left surface and right surface
        ---------
        Parameters:
        comp : component
            Lens component object
        eps_map : 2D or 3D array
            Dielectric map on which the lens will be written
        res : float
            Resolution of map
        AR_left_layers : list of float, optional
            List of thicknesses for each AR coating layer on the left surface
        AR_left_materials : list of float, optional
            List of permittivity values for each AR coating layer on the left surface
        AR_right_layers : list of float, optional
            List of thicknesses for each AR coating layer on the right surface
        AR_right_materials : list of float, optional
            List of permittivity values for each AR coating layer on the right surface
        '''
        # Validate AR coating parameters
        if AR_left_layers is not None and AR_left_materials is None:
            raise ValueError("AR_left_materials must be provided with AR_left_layers")
        if AR_right_layers is not None and AR_right_materials is None:
            raise ValueError("AR_right_materials must be provided with AR_right_layers")
        
        if AR_left_layers is not None and len(AR_left_layers) != len(AR_left_materials):
            raise ValueError("AR_left_layers and AR_left_materials must have the same length")
        if AR_right_layers is not None and len(AR_right_layers) != len(AR_right_materials):
            raise ValueError("AR_right_layers and AR_right_materials must have the same length")

        # The y axis has its zero in the middle of the cell, the offset is mid_y
        mid_y = np.int64(self.size_y*res/2)

        # Thickness of the lens on optical axis
        thick = comp.thick*res

        # Generate the center of the lumps made by delamination, 
        # different for the left and right surface
        high = np.int64(np.around(self.size_y*0.9/2))
        y0_left = np.random.randint(low=-high, high=high)
        y0_right = np.random.randint(low=-high, high=high)

        radius = np.int64(comp.diameter*res/2)

        # Generates the bins of random surface errors.
        if comp.surf_err_scale != 0:
            nb_bins = int(comp.diameter/comp.surf_err_width)
            err_left = np.around(np.random.normal(scale=comp.surf_err_scale*res,
                                                size=nb_bins))
            err_right = np.around(np.random.normal(scale=comp.surf_err_scale*res, 
                                                size=nb_bins))
        else:
            nb_bins = int(comp.diameter/comp.surf_err_width)
            err_left = np.zeros(nb_bins)
            err_right = np.zeros(nb_bins)

        # Iterate y over the radius, as the lenses are symmetric
        # above and below the optical axis
        for y_res in range(radius):           
            
            # Left surface sag
            x_left = np.int64(np.around((
                        comp.left_surface(y_res/res) + self.dpml + 
                        comp.x - comp.cust_def((y_res+mid_y)/res))*res))
            # Right surface sag       
            x_right = np.int64(np.around((
                        comp.right_surface(y_res/res) + 
                        comp.x + self.dpml -
                        comp.cust_def((y_res+mid_y)/res))*res + 
                        thick))
            
            # Above and below the optical axis:
            y_positive = int(self.dpml*res + mid_y + y_res)
            y_negative = int(self.dpml*res + mid_y - y_res)

            # Get the delamination as a function of y on left surface
            delam_pos_L = np.int64(np.around(res*
                comp.delamination(y_res/res, y0_left)))
            delam_neg_L = np.int64(np.around(res*
                comp.delamination(-y_res/res, y0_left)))

            # Get the delamination as a function of y on right surface
            delam_pos_R = np.int64(np.around(res*
                comp.delamination(y_res/res, y0_right)))
            delam_neg_R = np.int64(np.around(res*
                comp.delamination(-y_res/res, y0_right)))
            
            # Gradient in the index
            # ONLY WORKS WHEN NO SURFACE DEFECT
            radial_slope = comp.radial_slope/res
            axial_slope = comp.axial_slope/res
            if radial_slope != 0 or axial_slope != 0: 
                eps0 = comp.eps
                x0 = np.int64(np.around(comp.x*res))
                x_range = range(x_left, x_right+1) 
                # The value is squared as the permittivity is index squared
                eps_line = [eps0 + 
                            (y_res*radial_slope)**2 + 
                            ((k-x0)*axial_slope)**2 for k in x_range]
            else:
                eps_line = comp.eps

            # Surface error
            err_bin_idx = int(np.around(y_res/res/comp.surf_err_width))
            err_left_pos = int(err_left[err_bin_idx])
            err_left_neg = int(err_left[- err_bin_idx])

            err_right_pos = int(err_left[err_bin_idx]) 
            err_right_neg = int(err_left[- err_bin_idx])

            x_left_neg = int(x_left + err_left_neg)
            x_left_pos = int(x_left + err_left_pos)

            x_right_neg = int(x_right + err_right_neg)
            x_right_pos = int(x_right + err_right_pos)

            # Write lens between left and right surface below optical axis
            eps_map[x_left_neg:x_right_neg+1, y_negative] *= eps_line
            
            # So that the center line is not affected twice:
            if y_res != 0:
                # Write lens between left and right surface above optical axis
                eps_map[x_left_pos:x_right_pos+1, y_positive] *= eps_line
            
            # Write multi-layer AR coating on left surface
            if AR_left_layers is not None:
                # Start position is directly at the lens surface
                start_pos_neg = x_left_neg - delam_neg_L
                start_pos_pos = x_left_pos - delam_pos_L
                
                # Apply each layer, moving outward from the lens surface
                for i, (layer_thick, material) in enumerate(zip(AR_left_layers, AR_left_materials)):
                    AR_thick = np.int64(np.around(layer_thick*res))
                    
                    # Below optical axis
                    eps_map[start_pos_neg - AR_thick:start_pos_neg, y_negative] *= material
                    
                    # Above optical axis (if not on axis)
                    if y_res != 0:
                        eps_map[start_pos_pos - AR_thick:start_pos_pos, y_positive] *= material
                    
                    # Move starting position outward for next layer
                    start_pos_neg -= AR_thick
                    start_pos_pos -= AR_thick
            
            # Write multi-layer AR coating on right surface                    
            if AR_right_layers is not None:
                # Start position is directly at the lens surface
                start_pos_neg = x_right_neg + 1 + delam_neg_R
                start_pos_pos = x_right_pos + 1 + delam_pos_R
                
                # Apply each layer, moving outward from the lens surface
                for i, (layer_thick, material) in enumerate(zip(AR_right_layers, AR_right_materials)):
                    AR_thick = np.int64(np.around(layer_thick*res))
                    
                    # Below optical axis
                    eps_map[start_pos_neg:start_pos_neg + AR_thick, y_negative] *= material
                    
                    # Above optical axis (if not on axis)
                    if y_res != 0:
                        eps_map[start_pos_pos:start_pos_pos + AR_thick, y_positive] *= material
                    
                    # Move starting position outward for next layer
                    start_pos_neg += AR_thick
                    start_pos_pos += AR_thick

    # Example usage:
    def assemble_with_multi_arc(self):
        """
        Assembling the lens object with multi-layer AR coatings
        
        Parameters:
        left_layers: list of float
            Thicknesses of each AR coating layer on the left surface
        left_materials: list of float
            Permittivity values for each AR coating layer on the left surface
        right_layers: list of float
            Thicknesses of each AR coating layer on the right surface
        right_materials: list of float
            Permittivity values for each AR coating layer on the right surface
        """
        self.write_lens_nARC(self, self.permittivity_map, self.res,
                        AR_left_layers=self.left_layers,
                        AR_left_materials=self.left_materials,
                        AR_right_layers=self.right_layers,
                        AR_right_materials=self.right_materials)
        return self.permittivity_map
    
    # # Example usage:
    # # Create a lens with multi-layer AR coatings
    # lens = AsphericLens(diameter=10, r1=20, r2=-20, thick=5, mpsat_sim=sim_obj)

    # # Define the AR coating layers
    # left_layers = [0.25, 0.3, 0.15]  # Three layers with different thicknesses
    # left_materials = [1.4, 1.2, 1.3]  # Corresponding refractive indices squared

    # right_layers = [0.2, 0.25]  # Two layers on the right surface
    # right_materials = [1.3, 1.2]  # Corresponding refractive indices squared

    # # Apply the multi-layer AR coatings
    # lens.assemble_with_multi_arc(
    #     left_layers=left_layers, 
    #     left_materials=left_materials,
    #     right_layers=right_layers, 
    #     right_materials=right_materials
    # )

    #*The below functions are helpful for adding stepped pyramid ARC in the 
    #*lens permittivity map

    def meep_block(self,
                size, 
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

    def generate_discretized_points(self, center, angle, distance, num_points=1):
        """
        Generates discretized points along a line at a specified angle from the center.

        Parameters
        ----------
        center : tuple
            Center coordinates (x, y) from which the points are generated.
        angle : float
            Angle in degrees (w.r.t X axis) at which the points are generated.
        distance : float
            Distance from the center to the points.
        num_points : int (optional)
            Number of points to generate along the line (default: 1).

        Returns
        -------
        list of tuples
            List of generated points as (x, y) coordinates.
        """
        angle_rad = np.radians(angle)
        if num_points == 1:
            #! I did some debugging here, that I don't know :( --> But it works LoL
            x = center[0] + distance * np.sin(angle_rad)
            y = center[1] + distance * np.cos(angle_rad)
            points = [(x, y)]
            return points
        else:
            # Generate points from -distance/2 to distance/2
            distance_arr = np.linspace(-distance/2, distance/2, num_points)
            points = [(center[0] + d * np.cos(angle_rad), 
                    center[1] + d * np.sin(angle_rad)) for d in distance_arr]

            return points


    def stepped_pyramid_geometery(self,
                                nlayers,
                                base_bottom_edge_center,
                                pitch,
                                depth_arr,
                                kerf_arr,
                                width_arr=None,
                                material=mp.Medium(index=1.45),
                                rot_axis = 'z',
                                angle=0):
        
        """
        Function to generate the geometry of a stepped pyramid with specified parameters.

        Parameters:
        nlayers (int): Number of layers in the stepped pyramid.
        center (tuple): Base center coordinates (x,y) on which the pyramids will be attached.
        pitch (float): Pitch of the pyramid.
        depth_arr (list): List of depths for each layer, starting from the bottom layer.
        kerf_arr (list): List of kerfs for each layer, starting from the bottom layer.
        width_arr (list, optional): List of widths for each layer. If not provided, it will be calculated as pitch - depth for each layer.
        material (mp.Medium): Material of the blocks (default: mp.Medium(index=1.45)).
        rot_axis (str): Axis about which the blocks are rotated ('x', 'y', or 'z') (default: 'x').
        angle (float): Angle by which the blocks are rotated w.r.t the x-axis (default: 0).
                    Unit: degrees.

        Returns:
        A list of MEEP block objects representing the geometry of a single stepped pyramid geometry.
        """
        # Initialize the list to hold the blocks
        blocks = []

        if width_arr is None:
            print("Width array is not provided. It will be calculated as pitch - kerf for each layer.")
            width_arr = [pitch - kerf for kerf in kerf_arr]

        # Loop through each layer to create the blocks
        # Epty array to hold the centers of the base layer
        base_center = []

        # Initialise the depth to calculate d1 + d2 + ... + dn
        # This will be used to calculate the center of each layer after the base layer
        # The base layer is at depth = depth_arr[0]/2
        depth_from_previous_layers_centre = 0
        for i in range(nlayers):
            # Calculate the center of the base layer of the pyramid
            if i == 0:
                base_layer_center = self.generate_discretized_points(center= base_bottom_edge_center,
                                                                angle= angle,
                                                                distance= depth_arr[i]/2,
                                                                num_points=1)
                base_center.append(base_layer_center[0])
                # Create the block for the base layer
                base_layer_block = self.meep_block(size=mp.Vector3(width_arr[i], depth_arr[i], 0),
                                            center= mp.Vector3(base_layer_center[0][0], base_layer_center[0][1], 0),
                                            material=material,
                                            angle=angle,
                                            rot_axis=rot_axis)

                blocks.append(base_layer_block)


            # For rest of the layers, calculate the center based on the base layer's center
            # and add the depth of the current layer to the previous depth (basically cumulative depth from the base layer)
            else:
                
                depth_from_previous_layers_centre += depth_arr[i-1]/2 + depth_arr[i]/2
                other_layers_center = self.generate_discretized_points(center= base_center[0],
                                                                angle= angle,
                                                                distance= depth_from_previous_layers_centre,
                                                                num_points=1)
                
                # Create the block for the other layers
                other_layers_block = self.meep_block(size=mp.Vector3(width_arr[i], depth_arr[i], 0),
                                                center= mp.Vector3(other_layers_center[0][0], other_layers_center[0][1], 0),
                                                material=material,
                                                angle=angle,
                                                rot_axis=rot_axis)

                blocks.append(other_layers_block)

        return blocks

    
    
    """
    Instead of adding the stepped pyramid ARC to the  permittivity map of the lens,
    we will instead return the MEEP bllock objects representing the stepped pyramid ARC coating.
    These blocks can then be added into the geometry list of the MEEP simulation.
    This allows for more flexibility in how the ARC coating is applied and visualized. 
    ##! MOST IMPORTANT: CURRENT STEPPED PYRAMID ARC CANNOT BE APPLIED TO THE LENS SAGS 
    ##! WITH DELAMINATION AND SURFACE ERRORS ###!
    """

    def extract_lens_surface_coordinates(self, comp, res):
        """
        Extracts the x,y coordinates of the left and right lens surfaces,
        returns them in Meep units centered at (0,0).
        """
        mid_y = np.int64(self.size_y * res / 2)
        thick = comp.thick * res
        radius = np.int64(comp.diameter * res / 2)

        left_surface_coords = []
        right_surface_coords = []

        if comp.surf_err_scale != 0:
            nb_bins = int(comp.diameter / comp.surf_err_width)
            err_left = np.around(np.random.normal(scale=comp.surf_err_scale * res, size=nb_bins))
            err_right = np.around(np.random.normal(scale=comp.surf_err_scale * res, size=nb_bins))
        else:
            nb_bins = int(comp.diameter / comp.surf_err_width)
            err_left = np.zeros(nb_bins)
            err_right = np.zeros(nb_bins)

        for y_res in range(radius):
            # print(f"Processing y_res: {y_res}")
            x_left = np.int64(np.around((
                comp.left_surface(y_res / res) + self.dpml +
                comp.x - comp.cust_def((y_res + mid_y) / res)) * res))
            x_right = np.int64(np.around((
                comp.right_surface(y_res / res) +
                comp.x + self.dpml -
                comp.cust_def((y_res + mid_y) / res)) * res + thick))

            y_positive = int(self.dpml * res + mid_y + y_res)
            y_negative = int(self.dpml * res + mid_y - y_res)

            err_bin_idx = int(np.around(y_res / res / comp.surf_err_width))
            err_left_pos = int(err_left[err_bin_idx])
            err_left_neg = int(err_left[-err_bin_idx])
            err_right_pos = int(err_right[err_bin_idx])
            err_right_neg = int(err_right[-err_bin_idx])

            x_left_neg = int(x_left + err_left_neg)
            x_left_pos = int(x_left + err_left_pos)
            x_right_neg = int(x_right + err_right_neg)
            x_right_pos = int(x_right + err_right_pos)

            # First convert from array indices to physical coordinates
            x_left_neg_phys = x_left_neg / res
            x_left_pos_phys = x_left_pos / res  
            x_right_neg_phys = x_right_neg / res
            x_right_pos_phys = x_right_pos / res
            y_negative_phys = y_negative / res
            y_positive_phys = y_positive / res

            # Then center at (0,0) by subtracting half the total size (including PML)
            total_size_x = self.size_x + 2 * self.dpml
            total_size_y = self.size_y + 2 * self.dpml

            x_left_neg_meep = x_left_neg_phys - (total_size_x / 2)
            x_left_pos_meep = x_left_pos_phys - (total_size_x / 2)
            x_right_neg_meep = x_right_neg_phys - (total_size_x / 2)
            x_right_pos_meep = x_right_pos_phys - (total_size_x / 2)
            y_negative_meep = y_negative_phys - (total_size_y / 2)
            y_positive_meep = y_positive_phys - (total_size_y / 2)

            # # Convert to Meep units and center at (0,0)
            # x_left_neg_meep = (x_left_neg / res) - ((self.size_x) / 2) - self.dpml/4 #! managed somehow
            # x_left_pos_meep = (x_left_pos / res) - ((self.size_x) / 2)  - self.dpml/4 #! managed somehow
            # x_right_neg_meep = (x_right_neg / res) - ((self.size_x) / 2)  - self.dpml/4 #! managed somehow
            # x_right_pos_meep = (x_right_pos / res) - ((self.size_x) / 2)  - self.dpml/4#! managed somehow
            # y_negative_meep = (y_negative / res) - ((self.size_y) / 2)  - self.dpml/2 #!fucking working 
            # y_positive_meep = (y_positive / res) - ((self.size_y) / 2)  - self.dpml/2 #!fucking working

            #if not np.isclose(np.around(y_negative_meep, 2), 0.0):
            left_surface_coords.append((x_left_neg_meep, y_negative_meep))
            right_surface_coords.append((x_right_neg_meep, y_negative_meep))
            #if not np.isclose(np.around(y_positive_meep, 2), 0.0):
            left_surface_coords.append((x_left_pos_meep, y_positive_meep))
            right_surface_coords.append((x_right_pos_meep, y_positive_meep))


        return {
            'left_surface': left_surface_coords,
            'right_surface': right_surface_coords
        }


    def create_arc_blocks_vectorized(self,
                                     x_left_steps, x_right_steps, angle_left, angle_right, y_steps_left, y_steps_right, 
                                    arc_layer_pitch, arc_layer_depth, arc_layer_kerf, arc_layer_width, arc_material):
        """
        Vectorized creation of ARC stepped pyramid blocks for better performance
        """
        all_blocks = []
        
        # Create arrays for all left and right centers
        left_centers = np.column_stack((x_left_steps, y_steps_left))
        right_centers = np.column_stack((x_right_steps, y_steps_right))
        
        # Batch create left stepped pyramids
        for i, (center, angle) in enumerate(zip(left_centers, angle_left)):
            left_pyramid = self.stepped_pyramid_geometery(
                nlayers=self.step_ARC_nlayers,
                base_bottom_edge_center=tuple(center),
                pitch=arc_layer_pitch,
                depth_arr=arc_layer_depth,
                kerf_arr=arc_layer_kerf,
                width_arr=arc_layer_width,
                material=arc_material,
                rot_axis=self.step_ARC_rot_axis,
                angle=angle
            )

            all_blocks.extend(left_pyramid)
        
        # Batch create right stepped pyramids
        for i, (center, angle) in enumerate(zip(right_centers, angle_right)):
            right_pyramid = self.stepped_pyramid_geometery(
                nlayers=self.step_ARC_nlayers,
                base_bottom_edge_center=tuple(center),
                pitch=arc_layer_pitch,
                depth_arr=arc_layer_depth,
                kerf_arr=arc_layer_kerf,
                width_arr=arc_layer_width,
                material=arc_material,
                rot_axis=self.step_ARC_rot_axis,
                angle=angle
            )


            all_blocks.extend(right_pyramid)
        
        return all_blocks

    # Sort the coordinates by y-values and remove duplicates
    def prepare_spline_data(self, x_coords, y_coords):
        """
        Prepare data for spline interpolation by sorting and removing duplicates
        """
        # Combine and sort by y-coordinates
        combined = list(zip(y_coords, x_coords))
        combined.sort(key=lambda item: item[0])  # Sort by y
        
        # Remove duplicates (keep first occurrence)
        unique_data = []
        prev_y = None
        for y, x in combined:
            if prev_y is None or not np.isclose(y, prev_y):
                unique_data.append((y, x))
                prev_y = y
        
        if len(unique_data) < 2:
            raise ValueError("Not enough unique points for interpolation")
        
        y_unique, x_unique = zip(*unique_data)
        return np.array(x_unique), np.array(y_unique)

    def write_lens_with_stepped_pyramid_ARC_v2(self, comp):
        """
        In this version, we will assume that the lens surfaces are:
        - Centered at (0,0), instead of at self.x and self.y
        - We will first generate the lens sags for left and right surfaces centered at (0,0)
        - Then self.x, self.y is given in the 0 to x,y coordinate system; convert this to -x/2 - x/2 and -y/2 - y/2 coordinates
        - Do a coordinate shift for the lens sags in the (-x/2, x/2) and (-y/2, y/2) coordinate system 
          (basically add the self.x and self.y coordinates in the (-x/2, x/2) and (-y/2, y/2) coordinate system)
        - Expectation: The lens surfaces will be centered at the required physcial coordinates of the system, 
          and the ARC coating will be applied on the lens surfaces.
        """
        # Check if all required parameters for stepped pyramid ARC are provided
        if self.step_ARC_nlayers is None or self.step_ARC_pitch is None or \
            self.step_ARC_kerf is None or self.step_ARC_depth is None or \
            self.step_ARC_material is None:
            raise ValueError("All stepped pyramid ARC parameters must be provided.")
        
        # Calculate the width if not provided
        if self.step_ARC_width is None:
            print("Width array is not provided. It will be calculated as pitch - kerf for each stepped pyramid ARC layer.")
            self.step_ARC_width = [self.step_ARC_pitch - kerf for kerf in self.step_ARC_kerf]

        def even_asphere_lens_eqn(y, r, k, A2=0, A3=0, A4=0):
            # y =y/10
            # r = r/10
            return (y**2/r) / (1 + np.sqrt(1 - (1 + k)*y**2/r**2)) + A2 * y**2 + A3 * y**4 + A4 * y**6

        # Defining the y array
        y_arc_steps = np.arange(-self.diameter/2 + self.dpml - self.step_ARC_pitch/2, self.diameter/2 + self.step_ARC_pitch/2, self.step_ARC_pitch)
        # start = -self.diameter / 2
        # stop = self.diameter / 2
        # step = self.step_ARC_pitch

        # num_points = int(np.floor((stop - start) / step)) + 1
        # y_arc_steps = np.linspace(start, stop, num_points)

        # Extracting the left and right surface coordinates using the lens sag equations
        x_left_arc_steps = even_asphere_lens_eqn(y_arc_steps, self.r1, self.c1, self.a1_coeffs[0], self.a1_coeffs[1], self.a1_coeffs[2]) + self.x - self.size_x/2 - comp.cust_def(y_arc_steps) + self.dpml + self.step_ARC_offset[0] #! Note: we need to check with cust_def
        x_right_arc_steps = even_asphere_lens_eqn(y_arc_steps, self.r2, self.c2, self.a2_coeffs[0], self.a2_coeffs[1], self.a2_coeffs[2])  + self.x + self.thick - self.size_x/2 - comp.cust_def(y_arc_steps) + self.dpml + self.step_ARC_offset[1] #! Note: we need to check with cust_def

        # Calculate the slope using scipy of each point by considering the adjacent points
        # from scipy.ndimage import gaussian_filter1d
        slope_left = np.gradient(x_left_arc_steps, y_arc_steps)
        slope_right = np.gradient(x_right_arc_steps, y_arc_steps)

        # # Calculate the perpendicular angle of the slope in radians
        angle_left = np.rad2deg(np.arctan(slope_left))
        angle_left = -angle_left - 90  # Adjusting the angle to be perpendicular on the left lens surface
        angle_right = np.rad2deg(np.arctan(slope_right))
        angle_right = -angle_right + 90  # Adjusting the angle to be perpendicular on the right lens surface

        # Create the ARC blocks for both left and right surfaces
        all_blocks = self.create_arc_blocks_vectorized(
            x_left_steps=x_left_arc_steps,
            x_right_steps=x_right_arc_steps,
            angle_left=angle_left,
            angle_right=angle_right,
            y_steps_left=y_arc_steps,
            y_steps_right=y_arc_steps,
            arc_layer_pitch=self.step_ARC_pitch,
            arc_layer_depth=self.step_ARC_depth,
            arc_layer_kerf=self.step_ARC_kerf,
            arc_layer_width=self.step_ARC_width,
            arc_material=self.step_ARC_material
        )

        return all_blocks
        



    def write_lens_with_stepped_pyramid_ARC(self):
        """
        Writes the lens surfaces with stepped pyramid ARC coating.
        This method generates the stepped pyramid ARC coating on the lens surfaces
        

        Returns:
        -------
        Adds the stepped pyramid ARC coating to the permittivity map of the lens.
        """
        from scipy.interpolate import UnivariateSpline
        if self.step_ARC_nlayers is None or self.step_ARC_pitch is None or \
            self.step_ARC_kerf is None or self.step_ARC_depth is None or \
            self.step_ARC_material is None:
            raise ValueError("All stepped pyramid ARC parameters must be provided.")
        
        # Calculating the separation between the base layers of the ARC by considering the kerf and width
        if self.step_ARC_width is None:
            # If width is not provided, calculate it as pitch - kerf
            print("Width array is not provided. It will be calculated as pitch - kerf for each stepped pyramid ARC layer.")
            self.step_ARC_width = [self.step_ARC_pitch - kerf for kerf in self.step_ARC_kerf]
        #!=====
        left_surface_coords = self.extract_lens_surface_coordinates(self, self.res)['left_surface']
        right_surface_coords = self.extract_lens_surface_coordinates(self, self.res)['right_surface']

        x_left = np.array([coord[0] for coord in left_surface_coords])
        y_left = np.array([coord[1] for coord in left_surface_coords])
        x_right = np.array([coord[0] for coord in right_surface_coords])
        y_right = np.array([coord[1] for coord in right_surface_coords])

        # Interpolate the left and right surface coordinates to get evenly spaced points
        # from scipy.interpolate import interp1d
        # interp_left = interp1d(y_left, x_left, bounds_error=False, fill_value="extrapolate")
        # interp_right = interp1d(y_right, x_right, bounds_error=False, fill_value="extrapolate")
        # Prepare data for spline interpolation
        try:
            x_left_clean, y_left_clean = self.prepare_spline_data(x_left, y_left)
            x_right_clean, y_right_clean = self.prepare_spline_data(x_right, y_right)
            
            # Create splines with cleaned data
            interp_left = UnivariateSpline(y_left_clean, x_left_clean, s=0)
            interp_right = UnivariateSpline(y_right_clean, x_right_clean, s=0)
            
        except ValueError as e:
            print(f"Spline interpolation failed: {e}")
            print("Falling back to linear interpolation...")
            
            # Fallback to linear interpolation
            from scipy.interpolate import interp1d
            interp_left = interp1d(y_left, x_left, bounds_error=False, fill_value="extrapolate")
            interp_right = interp1d(y_right, x_right, bounds_error=False, fill_value="extrapolate")

        # Extracting the N-1 top and bottom y coordinates of the left and right surfaces
        y_min_left = np.min(y_left)
        y_max_left = np.max(y_left)
        y_min_right = np.min(y_right)
        y_max_right = np.max(y_right)

        # Generate evenly spaced points along the y-axis for the left and right surfaces according to the pitch
        y_arc_steps_left = np.arange(y_min_left, y_max_left, self.step_ARC_pitch)
        y_arc_steps_right = np.arange(y_min_right, y_max_right, self.step_ARC_pitch)

        # Calculate the x-coordinates for the left edge of the ARC layers
        x_left_arc_steps = interp_left(y_arc_steps_left)
        # Calculate the x-coordinates for the right edge of the ARC layers
        x_right_arc_steps = interp_right(y_arc_steps_right)

        # # Calculate the slope using scipy of each point by considering the adjacent points
        # # from scipy.ndimage import gaussian_filter1d
        slope_left = np.gradient(x_left_arc_steps, y_arc_steps_left)
        slope_right = np.gradient(x_right_arc_steps, y_arc_steps_right)

        # # Calculate the perpendicular angle of the slope in radians
        angle_left = np.rad2deg(np.arctan(slope_left))
        angle_left = -angle_left - 90  # Adjusting the angle to be perpendicular on the left lens surface
        angle_right = np.rad2deg(np.arctan(slope_right))
        angle_right = -angle_right + 90  # Adjusting the angle to be perpendicular on the right lens surface

        # Create the ARC blocks for both left and right surfaces
        all_blocks = self.create_arc_blocks_vectorized(
            x_left_steps=x_left_arc_steps,
            x_right_steps=x_right_arc_steps,
            angle_left=angle_left,
            angle_right=angle_right,
            y_steps_left=y_arc_steps_left,
            y_steps_right=y_arc_steps_right,
            arc_layer_pitch=self.step_ARC_pitch,
            arc_layer_depth=self.step_ARC_depth,
            arc_layer_kerf=self.step_ARC_kerf,
            arc_layer_width=self.step_ARC_width,
            arc_material=self.step_ARC_material
        )

        return all_blocks

    
    def assemble_with_stepped_pyramid_ARC(self):
        """
        Assembling the lens object with stepped pyramid ARC coating.
        This method generates the stepped pyramid ARC coating on the lens surfaces
        """
        # First assemble the lens itself
        self.permitivitty_map = self.assemble()
        
        # Then generate the stepped pyramid blocks for ARC coating
        #! self.stepped_pyramid_blocks = self.write_lens_with_stepped_pyramid_ARC()
        self.stepped_pyramid_blocks = self.write_lens_with_stepped_pyramid_ARC_v2(self)

        # Return both the permittivity map and the stepped pyramid blocks
        return self.permittivity_map, self.stepped_pyramid_blocks
            

#~ Feedhorn class
class FeedHorn(object):
    """
    Class defining an FeedHorn from a txt file containing the geometry information about
    the Horn in r vs z.
    """
    def __init__(self,
                 mpsat_sim,
                 eps,
                 focal_plane_x,
                 focal_plane_y_range,
                 feedhorn_y_range,
                 # Feedhorn params
                 txt_file,
                 t_m,
                 t_f,
                 w2,
                 thick_x,
                 savepath,
                 central_metal_thickness = 0,
                 plot = False,
                 eps_pec = -1e-10,
                 eps_air = 1
                 ):
    
        """
        Arguments
        ---------
        mpsat_sim: object
            MEEPSAT object produced from sim_init() in simulation_2D.py
        
        eps: np.array
            Dielectric map of the other components in the system 
        
        focal_plane_x: float
            X-coordinate of the focal plane
        
        focal_plane_y_range: tuple
            Y-coordinate range of the focal plane

        feedhorn_y_range: tuple
            Y-coordinate range of the feedhorn distribution on the focal plane

        txt_file: str
            path to the text file containing the geometry information about
            the Horn in r vs z.

        t_m: float
            Thickness of the metal gap of the two consecutive apertures 

        t_f: float
            Gap between the centers of the two feedhorns

        w2: float
            Feedhorn's aperture width

        thick_x: float
            Total extent of the feedhorn in the X-axis

        savepath: str
            savepath for the generated plots and the data files 

        central_metal_thickness: float
            The thickness of the central metal layer separating feedhorn arrays of different wafers

        plot: bool
            Whether to generate plots 

        eps_pec: float
            Permittivity of the perfect electric conductor (PEC)

        eps_air: float
            Permittivity of air

        """

        self.mpsat_sim = mpsat_sim
        self.eps = eps
        self.focal_plane_y_range = focal_plane_y_range
        self.feedhorn_y_range = feedhorn_y_range

        # if self.focal_plane_y_range < self.feedhorn_y_range:
        #     raise(ValueError("focal_plane_y_range must be greater than or equal to feedhorn_y_range"))

        self.txt_file = txt_file
        self.t_m = t_m
        self.t_f = t_f
        self.w2 = w2
        self.thick_x = thick_x
        self.central_metal_thickness = central_metal_thickness
        self.savepath = savepath
        self.plot = plot
        self.eps_pec = eps_pec
        self.eps_air = eps_air

        self.focal_plane_x = focal_plane_x + thick_x #! Because we want the forebaffles opening aperture at the position of the focal plane

        # Extract some parameters from mpsat_sim
        self.sx = mpsat_sim.cell_size[0]
        self.sy = mpsat_sim.cell_size[1]
        self.res = mpsat_sim.resolution
        

    def load_txt_dat(self):
        import pandas as pd
        self.data = pd.read_csv(self.txt_file, sep=r'\s+')  # Fixed regex warning
        self.data['r_pos'] = self.data['r']*10
        self.data['r_neg'] = -self.data['r_pos']
        self.cumulative_z = np.cumsum(self.data['z']*10)

        if self.plot == True:
            plt.figure(figsize=(10, 6))
            plt.plot(self.cumulative_z, self.data['r_pos'])
            plt.plot(self.cumulative_z, self.data['r_neg'])
            plt.xlabel('z (mm)')
            plt.ylabel('r (mm)')
            plt.title('z vs r')
            plt.grid(True)
            plt.savefig(self.savepath + 'step1_z_column_plot.png')
            plt.close()

        return self.data
    
    

    def fit_spline_to_dat(self, s_factor=0, no_points=1000):
        from scipy.interpolate import UnivariateSpline
        r_pos_spline = UnivariateSpline(self.cumulative_z, self.data['r_pos'], s=s_factor) 
        r_neg_spline = UnivariateSpline(self.cumulative_z, self.data['r_neg'], s=s_factor)
        
        # Fit the spline
        z_new = np.linspace(self.cumulative_z.min(), self.cumulative_z.max(), no_points)
        r_pos_fitted = r_pos_spline(z_new)
        r_neg_fitted = r_neg_spline(z_new)

        if self.plot == True:
            plt.figure(figsize=(10, 6))
            plt.plot(z_new, r_pos_fitted, label='Fitted r_pos')
            plt.plot(z_new, r_neg_fitted, label='Fitted r_neg')
            plt.xlabel('z (mm)')
            plt.ylabel('r (mm)')
            plt.title('Fitted splines over original data')
            plt.legend()
            plt.grid(True)
            plt.savefig(self.savepath + 'step2_fitted_splines_plot.png')
            plt.close()

        return r_pos_spline, r_neg_spline



    def create_coordinate_grids(self):
        """Create x, y coordinate arrays for the grid"""
        # Match the epsilon map dimensions which have +1
        self.x = np.linspace(-self.sx/2, self.sx/2, int(self.sx * self.res) + 1)
        self.y = np.linspace(-self.sy/2, self.sy/2, int(self.sy * self.res) + 1)
        return self.x, self.y

    def define_focal_plane_axis(self):
        """Create the focal plane axis array"""
        self.focal_plane_axis = np.linspace(
            self.focal_plane_y_range[0], 
            self.focal_plane_y_range[1], 
            int((self.focal_plane_y_range[1] - self.focal_plane_y_range[0]) * self.res)
        )
        return self.focal_plane_axis

    def fill_pec_region(self):
        """Fill the focal plane region with PEC - VECTORIZED"""
        x, y = self.create_coordinate_grids()
        
        # Create meshgrid - use indexing='ij' to match epsilon array dimensions
        # epsilon array is (len(x), len(y)), so X varies along axis 0, Y along axis 1
        X, Y = np.meshgrid(x, y, indexing='ij')
        
        # Create boolean mask for PEC region
        mask_pec = ((X <= self.focal_plane_x) & 
                    (X >= (self.focal_plane_x - self.thick_x)) & 
                    (Y >= self.focal_plane_y_range[0]) & 
                    (Y <= self.focal_plane_y_range[1]))
        
        # Apply the mask
        self.eps[mask_pec] = self.eps_pec


    def calculate_feedhorn_centers(self):
        """Calculate feedhorn center positions"""
        n_feedhorns_positive = int(np.floor(self.feedhorn_y_range[1] / self.t_f)) + 1
        n_feedhorns_negative = int(np.floor(abs(self.feedhorn_y_range[0]) / self.t_f))
        
        if self.central_metal_thickness == 0:
            feedhorn_centers_positive = np.arange(0, n_feedhorns_positive) * self.t_f
            feedhorn_centers_negative = -np.arange(1, n_feedhorns_negative + 1) * self.t_f
        else:
            feedhorn_centers_positive = np.arange(self.central_metal_thickness/2, n_feedhorns_positive) * self.t_f
            feedhorn_centers_negative = -np.arange(self.central_metal_thickness/2, n_feedhorns_negative + 1) * self.t_f

        self.feedhorn_centers = np.sort(np.concatenate([feedhorn_centers_negative, feedhorn_centers_positive]))
        
        return self.feedhorn_centers

    def fill_feedhorn_profiles(self, r_pos_spline, r_neg_spline):
        """Fill air inside feedhorns using spline functions - VECTORIZED"""
        x, y = self.create_coordinate_grids()
        
        # Create meshgrid - use indexing='ij' to match epsilon array dimensions
        X, Y = np.meshgrid(x, y, indexing='ij')
        
        # Mask for x within feedhorn extent
        mask_x = (X <= self.focal_plane_x) & (X >= (self.focal_plane_x - self.cumulative_z.max()))
        
        # Calculate z position for all grid points
        z_pos = self.focal_plane_x - X
        
        # Mask for valid z positions
        mask_z = (z_pos >= 0) & (z_pos <= self.cumulative_z.max())
        
        # Combine masks
        mask_region = mask_x & mask_z
        
        # Get the radial bounds at each z position (vectorized spline evaluation)
        # Only evaluate where mask_region is True
        z_pos_valid = z_pos[mask_region]
        r_upper_valid = r_pos_spline(z_pos_valid)
        r_lower_valid = r_neg_spline(z_pos_valid)
        
        # For each feedhorn center, check if points are inside
        for centre in self.feedhorn_centers:
            # Calculate y distance from feedhorn center for the entire grid
            y_dist = Y - centre
            
            # Create temporary arrays for r_upper and r_lower for the full grid
            r_upper_grid = np.full_like(X, np.nan)
            r_lower_grid = np.full_like(X, np.nan)
            
            # Fill in the valid regions
            r_upper_grid[mask_region] = r_upper_valid
            r_lower_grid[mask_region] = r_lower_valid
            
            # Check if points are inside this feedhorn
            mask_feedhorn = mask_region & (y_dist >= r_lower_grid) & (y_dist <= r_upper_grid)
            
            # Fill with air
            self.eps[mask_feedhorn] = self.eps_air

    
    def plot_focal_plane(self):
        """Plot the simulation grid with focal plane axis"""
        if not self.plot:
            return
            
        x, y = self.create_coordinate_grids()
        focal_plane_axis = self.define_focal_plane_axis()
        
        # Create meshgrid for plotting
        X, Y = np.meshgrid(x, y, indexing='ij')
        
        plt.figure(figsize=(10, 8))
        plt.pcolormesh(X, Y, np.ones_like(X), cmap='gray', alpha=0.3, shading='auto')
        plt.plot(np.full_like(focal_plane_axis, self.focal_plane_x), focal_plane_axis, 
                'r-', linewidth=2, label='Focal plane axis')
        plt.xlabel('x (mm)')
        plt.ylabel('y (mm)')
        plt.title('Simulation Grid with Focal Plane')
        plt.legend()
        plt.grid(True)
        plt.axis('equal')
        plt.savefig(self.savepath + 'step3_focal_plane_plot.png')
        plt.close()


    def plot_pec_region(self):
        """Plot the simulation grid with PEC region filled"""
        if not self.plot:
            return
            
        x, y = self.create_coordinate_grids()
        focal_plane_axis = self.define_focal_plane_axis()
        
        plt.figure(figsize=(10, 8))
        plt.pcolormesh(x, y, self.eps.T, cmap='RdBu', shading='auto', vmin=-0.2, vmax=1)
        plt.plot(np.full_like(focal_plane_axis, self.focal_plane_x), focal_plane_axis, 
                'r-', linewidth=2, label='Focal plane axis')
        plt.xlabel('x (mm)')
        plt.ylabel('y (mm)')
        plt.title('Simulation Grid with PEC Region')
        plt.colorbar(label='Epsilon')
        plt.legend()
        plt.grid(True)
        plt.axis('equal')
        plt.savefig(self.savepath + 'step3b_focal_plane_with_PEC.png')
        plt.close()


    def plot_feedhorn_centers(self):
        """Plot feedhorn centers on the focal plane"""
        if not self.plot:
            return
            
        x, y = self.create_coordinate_grids()
        focal_plane_axis = self.define_focal_plane_axis()
        
        plt.figure(figsize=(10, 8))
        plt.pcolormesh(x, y, self.eps.T, cmap='RdBu', alpha=0.3, shading='auto', vmin=-0.2, vmax=1)
        plt.plot(np.full_like(focal_plane_axis, self.focal_plane_x), focal_plane_axis, 
                'r-', linewidth=2, label='Focal plane axis')
        
        # Draw each feedhorn as a circle
        for center in self.feedhorn_centers:
            circle = plt.Circle((self.focal_plane_x, center), self.w2/2, 
                            color='blue', fill=False, linewidth=2)
            plt.gca().add_patch(circle)
            plt.plot(self.focal_plane_x, center, 'bo', markersize=5)
        
        plt.xlabel('x (mm)')
        plt.ylabel('y (mm)')
        plt.title('Simulation Grid with Focal Plane and Feedhorns')
        plt.legend()
        plt.grid(True)
        plt.axis('equal')
        plt.savefig(self.savepath + 'step4_focal_plane_with_feedhorns_plot.png')
        plt.close()


    def plot_final_geometry(self):
        """Plot the final feedhorn geometry with all profiles filled"""
        if not self.plot:
            return
            
        x, y = self.create_coordinate_grids()
        
        plt.figure(figsize=(12, 10))
        plt.pcolormesh(x, y, self.eps.T, cmap='RdBu', shading='auto', vmin=-0.2, vmax=1)
        plt.xlabel('x (mm)')
        plt.ylabel('y (mm)')
        plt.title('Simulation Grid with Feedhorns (Air-filled)')
        plt.colorbar(label='Epsilon')
        plt.grid(True)
        plt.axis('equal')
        plt.savefig(self.savepath + 'step5_feedhorns_with_profiles.png')
        plt.close()

    def add_absorbers_to_extra_PEC(self):
        # # Calculate the remaining length remaining on the PEC layer in the focal plane
        # extra_layer_negative_side = self.focal_plane_y_range[0] - self.feedhorn_y_range[0]
        # extra_layer_positive_side = self.focal_plane_y_range[1] - self.feedhorn_y_range[1]

        # absorber_range_y_neg = [self.feedhorn_y_range[0], self.feedhorn_y_range[0]-extra_layer_negative_side]
        # absorber_range_y_pos = [self.feedhorn_y_range[1], self.feedhorn_y_range[1]+extra_layer_positive_side]

        # import meepsat.meep_geometry as comp_meep
        # absorbers_y_neg = comp_meep.PyramidalAbsorbers(self.mpsat_sim,
        #                                  base_width = 6,
        #                                  height = 9,
        #                                  n_layers = 70,
        #                                  top_width = 0.5,
        #                                  epsilon_real = 5.4,
        #                                  epsilon_imag = 0.8,
        #                                  freq = data["sources"]["source1"]["frequecy"],
        #                                  add_substrate=True,
        #                                  substrate_thickness=7,#p,
        #                                  substrate_material=None, # If None, then it will be same as the absorber material
        #                                  substrate_extends_beyond_pyramids=False,
        #                                  substrate_extension=1,
        #                                  y_top_offset=-forebaffle_height +mpsat_sim.dpml*mpsat_sim.factor_dpml,# + 0.35,
        #                                  y_bottom_offset= +forebaffle_height-mpsat_sim.dpml*mpsat_sim.factor_dpml,# -0.35,
        #                                 #  num_pyramids = 150,
        #                                  x_coverage_start = -size_x/2 + cellx_sourcex_distance + sourcex_FB_vertex_distance + forebaffle_base,
        #                                  x_coverage_end = size_x/2 + 10,# - mpsat_sim.dpml*mpsat_sim.factor_dpml + 1,
        #                                  add_pec_backing = True,
        #                                  pec_thickness = forebaffle_height-7, # PEC thickness same as the forebaffle perpendicular height)
        #                                  pec_extends_beyond_substrate = False,
        #                                  pec_extension = 1, # pec extends beyond the substrate by 1 mm
        #                                  name = "absorbers"
        #                                 )
        pass


    def assemble(self):
        """Assemble the complete feedhorn geometry"""
        # Load and fit data
        self.load_txt_dat()
        
        # Plot step 3a: focal plane
        self.plot_focal_plane()
        
        # Get splines from fit_spline_to_dat
        r_pos_spline, r_neg_spline = self.fit_spline_to_dat()
        
        # Fill regions
        self.fill_pec_region()
        
        # Plot step 3b: PEC region
        self.plot_pec_region()
        
        self.calculate_feedhorn_centers()
        
        # Plot step 4: feedhorn centers
        self.plot_feedhorn_centers()
        
        self.fill_feedhorn_profiles(r_pos_spline, r_neg_spline)
        
        # if self.feedhorn_y_range != self.focal_plane_y_range:
        #     self.add_absorbers_to_extra_PEC()
        
        # Plot step 5: final geometry
        self.plot_final_geometry()
        
        return self.eps