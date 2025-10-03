import meep as mp
import numpy as np
import math

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
                    angle=angle,
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
                 pos_x, 
                 thickness, 
                 n_refr = 1, 
                 conductivity = 1e7,
                 rot_axis = 'x',
                 rot_angle = 0):
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
        pos_x : float
            Position of the left surface of the aperture stop along x-axis
        thickness : float
            Thickness of aperture stop slab
        n_refr : float, optional 
            Index of refraction of the material 
            if the stop is dielectric
            (default = 1)
        conductivity : float, optional
            Conductivity of the material (default = 1e7)
        rot_axis : str, optional
            Axis about which the aperture stop is rotated 
            (default : 'x')
        rot_angle : float, optional
            Angle by which the aperture stop is rotated w.r.t rot_axis (default : 0 degrees)
        '''
        self.mpsat_sim = mpsat_sim
        self.type = type                
        self.thick = thickness
        if pos_x:
            self.pos_x = pos_x
        else:
            self.pos_x = mpsat_sim.cell_size[0]/2 # Center of the cell                  
        self.diameter = diameter        
        self.permittivity = n_refr**2   
        self.conductivity = conductivity
        self.object_type = 'AP_stop'
        self.rot_axis = rot_axis
        self.rot_angle = rot_angle
    
    def square_aperture(self):
        '''
        Returns the block object for the aperture stop
        '''
        #Defines the material with given properties
        material = mp.Medium(epsilon=self.permittivity, 
                             D_conductivity = self.conductivity)
        
        # Up block of the aperture stop slab
        size_up = mp.Vector3(self.thick,
                             (self.mpsat_sim.cell.y - self.diameter)/2,
                             0)
        
        centre_up = mp.Vector3(self.pos_x - self.thick/2,
                               self.diameter/2 + size_up.y/2,
                                 0)
        
        size_down = mp.Vector3(self.thick,
                                 (self.mpsat_sim.cell.y - self.diameter)/2,
                                    0) 
        
        centre_down = mp.Vector3(self.pos_x - self.thick/2,
                                 -self.diameter/2 - size_down.y/2,
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
        
        return [aperture_stop_up, aperture_stop_down]
    

    def assemble(self):
        '''
        Returns the block object for the aperture stop

        Arguments
        ---------
        centre_list : list, optional
            List of the centers of the aperture stop

        size_list : list, optional
            List of the sizes of the aperture stop
        '''
        if self.type == 'square':
            return self.square_aperture()
        else:
            raise ValueError('Invalid aperture stop type name')

# Main simulation setup
def run_simulation(Nfps = 10, 
                   image_every = 5,
                   movie_name = 'aperture_stop',
                   runtime = 30):
    
    import sys
    import os
    

    # Set up simulation parameters
    cell_size = mp.Vector3(16, 16, 0)  # 2D simulation in the x-y plane
    resolution = 10  # Simulation resolution

    mpsat_sim = mpsat.sim_init(sim_name='aperture_stop',
                              cell_size=cell_size,
                              freq=1,
                              resolution=resolution,
                              boundary_layer_type='PML',
                              boundary_layer_size=1,
                              factor_dpml=1)
                              
    # Create a simulation object
    sim = mp.Simulation(cell_size=cell_size,
                        boundary_layers=[mp.PML(1.0)],  # PML boundary
                        resolution=resolution,
                        sources=[mp.Source(mp.ContinuousSource(frequency=0.15),
                                           component=mp.Ez,
                                           center=mp.Vector3(0, 0),
                                           size=mp.Vector3(0, 1))])    
    # Define aperture stop parameters
    aperture_stop = ApertureStop(mpsat_sim=mpsat_sim,
                                 type='square',
                                 diameter=6.0,  # Diameter of the aperture stop
                                 pos_x=4,  # Position of the aperture stop along the x-axis
                                 thickness=1.0)  # Thickness of the aperture stop

    # Assemble the aperture stop
    aperture_stop_blocks = aperture_stop.assemble()
    sim.geometry = aperture_stop_blocks
    
    import matplotlib.pyplot as plt
    f = plt.figure(dpi = 150)
                
    #Animate object
    """
    field_func = lambda x: 20*np.log10(np.abs(x))

    def colorbar(ax):
        matplotlib.colorbar.ColorbarBase(ax=ax)
        return ax
    """

    animate = mp.Animate2D(sim,
                        f = f,
                        fields=mp.Ez,
                        realtime=True,
                        field_parameters={'alpha':0.8,
                                            'cmap':'RdBu',
                                            'interpolation':'none'},
                        boundary_parameters={'hatch':'o', 
                                                'linewidth':1.5, 
                                                'facecolor':'y', 
                                                'edgecolor':'b', 
                                                'alpha':0.3})
    # Run the simulation
    sim.run(mp.at_every(image_every, animate), until=runtime)

    animate.to_mp4(Nfps, movie_name + '.mp4')

# Run the simulation
if __name__ == '__main__':
    run_simulation(Nfps = 10, 
                   image_every = 5,
                   movie_name = 'aperture_stop',
                   runtime = 30)    
