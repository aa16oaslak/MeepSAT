
import sys
import os

# Dynamically determine the path to the MEEPSAT library
notebook_dir = os.path.dirname(os.path.abspath('../../../../'))
main_dir = os.path.join(notebook_dir)#, 'meepsat')
meepsat_dir = os.path.join(main_dir, 'meepsat')
sys.path.append(main_dir)
sys.path.append(meepsat_dir)

print('The path to the notebook directory is:', notebook_dir)
print('The path to the main directory is:', main_dir)
print('The path to the MEEPSAT library is:', meepsat_dir)

# For saving the output generated from this Tutorial
savepath = os.path.join(os.path.dirname(os.path.abspath('./')), 'output_files/')



import meep as mp
import numpy as np
import h5py
import matplotlib.pyplot as plt
    


# Importing the MEEPSAT librarires
import meepsat.simulation_2D as sim
import meepsat.components_2D_meep as comp_meep
import meepsat.components_2D_eps as comp_eps
import meepsat.visualization as mpsat_plot



#~ Initialising MEEPSAT Simulation
cell_X, cell_Y, cell_Z = 900, 448, 0 # Cell Size without considering the PML thickness and its factor

# Initialize the simulation with the different parameters
mpsat_sim = sim.sim_init(sim_name='example',
                        cell_size= [cell_X, cell_Y, cell_Z], # [sx, sy, sz] in mm
                        freq= 0.3, 
                        resolution= 3,
                        boundary_layer_type= 'PML',
                        boundary_layer_size= 2,
                        factor_dpml= 2)

# Print the simulation parameters
mpsat_sim.print_simulation_parameters()


source_list = []

#~ Adding Source: GaussianBeam
gaussian_source_init = comp_meep.GaussianBeam(mpsat_sim=mpsat_sim,
                                              center= mp.Vector3(445,
                                                                0,
                                                                0),
                                              size= mp.Vector3(0,
                                                              448,
                                                              0),
                                              component= 'Ez',
                                              freq= 0.3,
                                              angle= 180,
                                              width= 2.22,
                                              kwargs= {'beam_x0': mp.Vector3(0,
                                                                             0,
                                                                             0),
                                                       'beam_E0': mp.Vector3(0,
                                                                             0,
                                                                             1),
                                                       'start_time': 0,
                                                       'end_time': mp.inf})
source_list.append(gaussian_source_init.assemble())



#~ Adding Boundary: PML
boundary_init = comp_meep.PML(mpsat_sim=mpsat_sim,
                                  size= 2,
                                  kwargs= {})

# Creating an MEEP object for the boundary using the assemble function
pml = boundary_init.assemble()



#~ Adding Epsilon Map
size_x, size_y, size_z = mpsat_sim.cell_size[0], mpsat_sim.cell_size[1], mpsat_sim.cell_size[2]
res = mpsat_sim.resolution
# Create the epsilon map: total size of the simulation cell in all the axis multiplied by the resolution + 1
epsilon_map = np.ones(((size_x + mpsat_sim.factor_dpml*mpsat_sim.dpml)*res+1, 
                            (size_y + mpsat_sim.factor_dpml*mpsat_sim.dpml)*res+1), dtype = 'float32')



#~ Adding Aperture: Square
aperture1_init = comp_meep.ApertureStop(mpsat_sim=mpsat_sim,
                                       type='square',
                                       diameter=408.678,
                                       pos_x=130,
                                       thickness=2,
                                       n_refr=1.45,
                                       conductivity=mp.inf)

aperture1_up, aperture1_down = aperture1_init.assemble()
mpsat_sim.add_meep_geometry(aperture1_up)
mpsat_sim.add_meep_geometry(aperture1_down)



#~ Adding Lens: lens1
lens1_init = comp_eps.AsphericLens(name = 'plano_hyperbolic_lens',
                                r1 = 100000,
                                r2 = -456,
                                c1 = 100000,
                                c2 = -2.4649,
                                thick = 80,
                                diameter = 448,
                                x = 80,
                                n_refr= 1.57,
                                AR_left= 0.45,
                                AR_right= 0.45,
                                AR_material= None,
                                eps= epsilon_map,
                                mpsat_sim= mpsat_sim)

lens1 = lens1_init.assemble()
mpsat_sim.add_eps_geometry(lens1)
# Plot the lens
# lens1_init.plot_lenses()

#~ Saving the Epsilon Map
# Saving the epsilon_map initiated after the final to a .h5 file, so that meep can read it
lens1_init.write_h5file(filename= savepath + 'example__epsilon_map.h5',
                        parallel= False) # Make it True if you runnning the code in parallel



# Initialize the meep simulation object with the different parameters
sim = mpsat_sim.meep_sim_obj(cell_size= mpsat_sim.cell,
                            boundary_layers= [pml],
                            geometry= mpsat_sim.meep_geometry,
                            sources= source_list,
                            epsilon_h5_file= savepath + 'simple_one_lens_old')



#~ Runtime of the simulation
run_time = {'command': 'until= 1000'}
#~ Requested data by the user
data_required= {'at_every_timestep': 5, 'at_every': ['animate_Ez2_dB', 'Ez2_dB'], 'at_end': ['output_efield_z']}



mpsat_sim.run_simulation(sim = sim,
                         mpsat_sim = mpsat_sim,
                         runtime = run_time,
                         get_mp4 = True,
                         image_every = 5,
                         Nfps = 15,
                         savepath = savepath,
                         movie_name = 'example_animation',
                         requested_data = data_required,
                         save_data = False)
