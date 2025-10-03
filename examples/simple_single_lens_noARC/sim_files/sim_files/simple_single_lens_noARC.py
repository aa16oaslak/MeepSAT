import sys
import os
import site
from pathlib import Path
import meep as mp
import numpy as np
import h5py
import matplotlib.pyplot as plt
import time
import json


# Importing the MEEPSAT librarires
import meepsat.simulation_2D as sim
import meepsat.components_2D_meep as comp_meep
import meepsat.components_2D_eps as comp_eps
import meepsat.mp_sim_run as mp_sim_run
import meepsat.json_to_script as json_to_script
import meepsat.analysis as mpsat_analysis

def read_json(json_file):
    # Read the JSON file
    with open(json_file, "r") as f:
        data = json.load(f)
    return data

#!===========


# Check if JSON path was provided as command line argument
json_file_path = sys.argv[1]
print(f"Using JSON file from command line argument: {json_file_path}")
data = read_json(json_file_path)

#! Add more parameteters to twick if needed
source_freq, res, savepath_dir, runtime, beam_waist = sys.argv[2], sys.argv[3], sys.argv[4], float(sys.argv[5]), float(sys.argv[6])
data["sources"]['source1']["frequecy"] = float(source_freq)
data["simulation"]['primary_params']['resolution'] = int(res)
data["output"]["savepath"]["path"] = str(Path(savepath_dir)) + "/" 
# Update the beam waist for the source in the JSON data
data["sources"]['source1']["extra_args"]["width"] = float(beam_waist)  # in mm

#Checking if the resolution criteria is met or not for the source's frequency
# resolution/frequency ratio should be at least 10
#! GO DOWN FOR THE SMALLEST FREQUENCY
if data["simulation"]['primary_params']['resolution'] / data["sources"]['source1']["frequecy"] < 10:
    print("Resolution criteria doesn't meet the criteria. Increasing the resolution to meet the criteria.")
    data["simulation"]['primary_params']['resolution'] = int(data["sources"]['source1']["frequecy"] * 10)
#~ ---------------------------------------------

#! Checking the smallest frequency's resolution criteria
# The highest index of refraction in my current systrenm is absorbers's
highest_n = np.sqrt(5.4)  
wavelength_meep_ = 1 / data["sources"]['source1']["frequecy"]  # Wavelength in MEEP unit
wavelength_meep_inside_medium = wavelength_meep_ / highest_n  # Wavelength in MEEP unit inside the medium
freq_inside_medium = 1 / wavelength_meep_inside_medium  # Frequency inside the medium
if data["simulation"]['primary_params']['resolution'] / freq_inside_medium < 10:
    print("Resolution criteria doesn't meet the criteria for the smallest frequency. Increasing the resolution to meet the criteria.")
    data["simulation"]['primary_params']['resolution'] = int(freq_inside_medium * 10)


# Now assuming the smallest frequency present in the system is the ARC thickness
arc_thickness = 1/(4*data["sources"]['source1']["frequecy"])
arc_refractive_index = np.sqrt(data["lenses"]["lens1"]["n_refr"])
wavelength_meep_arc = arc_refractive_index/ data["sources"]['source1']["frequecy"]  # Wavelength in MEEP unit for ARC
# set the smallest frequency
data["simulation"]['primary_params']['smallest_freq'] = float(min(1/wavelength_meep_arc, 1/arc_thickness))
if data["simulation"]['primary_params']['resolution'] / data["simulation"]['primary_params']['smallest_freq']< 10:
    print("Resolution criteria doesn't meet the criteria for the ARC thickness. Increasing the resolution to meet the criteria.")
    data["simulation"]['primary_params']['resolution'] = int(10 * data["simulation"]['primary_params']['smallest_freq'])

# # Set the arc thickness in the simulation parameters
# data["lenses"]["lens1"]["AR_left_layers"] = [arc_thickness]
# data["lenses"]["lens1"]["AR_right_layers"] = [arc_thickness]

# # Set the arc materials in the simulation parameters
# data["lenses"]["lens1"]["AR_left_materials"] = [arc_refractive_index]
# data["lenses"]["lens1"]["AR_right_materials"] = [arc_refractive_index]

#~ ---------------------------------------------
# Checking the PML thickness and factor
# It should at least 1/2 wavelength_meep_ of the source
min_pml_thickness = 0.5*wavelength_meep_
given_pml_thickness = data['boundary_layers']['boundary']['size']*data['boundary_layers']['boundary']['factor_dpml']
if given_pml_thickness < min_pml_thickness:
    print(f"PML thickness {given_pml_thickness} is less than the minimum required {min_pml_thickness}. Setting PML thickness to {min_pml_thickness}.")
    data['boundary_layers']['boundary']['size'] = min_pml_thickness

#~ ---------------------------------------------

savepath = data["output"]["savepath"]["path"]
os.makedirs(savepath, exist_ok=True)
print('Output directory path:', savepath)

#~ ---------------------------------------------

#~ Initialising MEEPSAT Simulation
cell_X, cell_Y, cell_Z = data["simulation"]['primary_params']['cell_size']['x'], data["simulation"]['primary_params']['cell_size']['y'], data["simulation"]['primary_params']['cell_size']['z'] # Cell Size without considering the PML thickness and its factor


# Initialize the simulation with the different parameters
mpsat_sim = sim.sim_init(sim_name= str(data["simulation"]["name"]),
                        cell_size= [cell_X, cell_Y, cell_Z], # [sx, sy, sz] in mm
                        smallest_freq= data["simulation"]['primary_params']['smallest_freq'], 
                        resolution= data["simulation"]['primary_params']['resolution'],
                        boundary_layer_type= data['boundary_layers']['boundary']['type'],
                        boundary_layer_size= data['boundary_layers']['boundary']['size'],
                        factor_dpml= data['boundary_layers']['boundary']['factor_dpml'])
#~ ---------------------------------------------
# Print the simulation parameters
mpsat_sim.print_simulation_parameters()

#~ Adding Sources
source_list = []
exec(json_to_script.source_script(data))
#~ ---------------------------------------------

#~ Adding Boundary
x_left_boundary = mp.PML(thickness=mpsat_sim.dpml*mpsat_sim.factor_dpml, direction=mp.X, side=mp.Low)
x_right_boundary = mp.PML(thickness=mpsat_sim.dpml*mpsat_sim.factor_dpml, direction=mp.X, side=mp.High)
y_down_boundary = mp.PML(thickness=mpsat_sim.dpml*mpsat_sim.factor_dpml, direction=mp.Y, side=mp.Low)
y_up_boundary = mp.PML(thickness=mpsat_sim.dpml*mpsat_sim.factor_dpml, direction=mp.Y, side=mp.High)

custom_boundary_layers = [x_left_boundary, x_right_boundary, y_down_boundary, y_up_boundary]
#~ ---------------------------------------------

#~ Adding Epsilon Map
size_x, size_y, size_z = mpsat_sim.cell_size[0], mpsat_sim.cell_size[1], mpsat_sim.cell_size[2]
res = int(mpsat_sim.resolution)  # Ensure resolution is an integer
# Create the epsilon map: total size of the simulation cell in all the axis multiplied by the resolution + 1
epsilon_map = np.ones((int((size_x)*res+1), 
                       int((size_y)*res+1)), dtype = 'float32')



#~ Adding lens (if given)
exec(json_to_script.add_lens(data))
#~ ---------------------------------------------

#~ Adding box slabs (if given)
exec(json_to_script.add_slab(data))
#~ ---------------------------------------------

#~ Adding aperture (if given)
exec(json_to_script.add_aperture(data))
#~ ---------------------------------------------

#~ Adding absorbers
#~ Add pyramidal absorbers with substrate 
# absorbers = comp_meep.PyramidalAbsorbers(mpsat_sim,
#                                          base_width = 6,
#                                          height = 9,
#                                          num_pyramids = 150,
#                                          n_layers = 45,
#                                          top_width = 0.5,
#                                          epsilon_real = 5.4,
#                                          epsilon_imag = 0.8,
#                                          freq = data["sources"]["source1"]["frequecy"],
#                                          add_substrate=True,
#                                          substrate_thickness=4 + 7,
#                                          y_top_offset=-7,
#                                          y_bottom_offset=7,
#                                          coverage_percent=100)

# mpsat_sim.meep_geometry.extend(absorbers.assemble())


#! DEFINING THE SIMULATION OBJECT
# Mirror symmetry along y direction (x=0 plane)
symmetries = [mp.Mirror(mp.Y, phase=+1)] 

simulation = mp.Simulation(
    cell_size=mpsat_sim.cell,
    sources=source_list,
    resolution=mpsat_sim.resolution,
    boundary_layers=custom_boundary_layers,
    geometry=mpsat_sim.meep_geometry,
    epsilon_input_file = data["output"]["savepath"]["path"] + data["output"]["epsilon_h5_file"]["filename"] +"_epsilon_map" + ".h5",
    symmetries = symmetries,
    force_complex_fields= True)

# #~ Set PEC boundary conditions
# simulation.set_boundary(direction=mp.Y, side=mp.Low, condition=mp.Metallic)
# simulation.set_boundary(direction=mp.Y, side=mp.High, condition=mp.Metallic)

#~ ---------------------------------------------
# Define frequency parameters (for single frequency source)
fcen = float(data["sources"]["source1"]["frequecy"])
df = 0     # Zero bandwidth since we have a single frequency
nfreq = 1  # Just one frequency point
wavelength_meep = 1 / float(data["sources"]["source1"]["frequecy"])  # Wavelength in MEEP unit
wavelength_factor = 1 #meaning 1 mm = 1 MEEP unit
wavelength_natural_real= wavelength_meep * wavelength_factor * 1e-3  # Wavelength in natural units in m

#* 1. Power monitor at the aperture plane
x_aperture = 20 #+2  
x_aperture_in_sim = x_aperture - (size_x/2) + 1 # + (mpsat_sim.factor_dpml*mpsat_sim.dpml))  # Adjust focal point to simulation coordinates
#x_aperture_in_sim = x_aperture - (size_x/2 + (mpsat_sim.factor_dpml*mpsat_sim.dpml))  # Adjust focal point to simulation coordinates


# Volume for field monitoring at aperture
aperture_line_monitor = mp.Volume(
    center=mp.Vector3(x_aperture_in_sim, 0, 0),
    size=mp.Vector3(0, size_y, 0) 
)

# Adding a DFT fields monitor to capture near fields at aperture
aperture_monitor = simulation.add_dft_fields([mp.Ex, mp.Ey, mp.Ez, mp.Hx, mp.Hy, mp.Hz],
                                            fcen, df, nfreq,
                                            where=aperture_line_monitor)


# Aperture plane flux monitor for power calculation
aperture_flux_region = mp.FluxRegion(
    center=mp.Vector3(x_aperture_in_sim, 0, 0),
    size=mp.Vector3(0, size_y, 0)
)
# Add FLUX monitor for power calculation using FluxRegion
aperture_flux_monitor = simulation.add_flux(fcen, df, nfreq, aperture_flux_region)

#* Defining an function to extract the power from the aperture monitor at a given timestep during the simulation
def extract_aperture_power(simulation):
    # Get the Ez array for the aperture monitor
    aperture_fields = simulation.get_array(vol=aperture_line_monitor, 
                                           component= mp.Ez,
                                           cmplx= True)
    # Calculate the power (magnitude squared)
    aperture_power = np.abs(aperture_fields)**2

    # Save the power data to an npz compressed file
    np.savez_compressed(os.path.join(savepath, "aperture_power_{0}.npz".format(simulation.meep_time())),
                        field=aperture_fields,
                        power=aperture_power,
                        y_coords=np.linspace(-size_y/2, size_y/2, len(aperture_fields)))
    
    print(f"Aperture power data saved to {os.path.join(savepath, 'aperture_power.npz')} at timestep {simulation.meep_time()}.")


#* Define an function to extract the phasors information from the DFT monitor at the aperture plane
def extract_aperture_phasors(simulation):
    # # Get the DFT array for the aperture monitor
    # aperture_dft_fields = [simulation.get_dft_array(aperture_monitor, c, 0) for c in [mp.Ex, mp.Ey, mp.Ez, mp.Hx, mp.Hy, mp.Hz]]
    # # Save the data to an npz compressed file
    # np.savez_compressed(os.path.join(savepath, "aperture_phasors_{0}.npz".format(simulation.meep_time())),
    #                     ex=aperture_dft_fields[0],
    #                     ey=aperture_dft_fields[1],
    #                     ez=aperture_dft_fields[2],
    #                     hx=aperture_dft_fields[3],
    #                     hy=aperture_dft_fields[4],
    #                     hz=aperture_dft_fields[5])
    #                     #y_coords=np.linspace(-size_y/2, size_y/2, len(aperture_dft_fields[0])))
    
    # print(f"Aperture phasors data saved to {os.path.join(savepath, 'aperture_phasors.npz')} at timestep {simulation.meep_time()}.")

    # Saving the DFT fields using output_dft instead
    simulation.output_dft(aperture_monitor, 
                          os.path.join(savepath, "aperture_phasors_{0}".format(simulation.meep_time())))
    print(f"Aperture phasors data saved to {os.path.join(savepath, 'aperture_phasors')} at timestep {simulation.meep_time()}.")


#*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

#* 2. Near-to-Far Field (N2F) monitor
ffb_line_monitor= mp.Near2FarRegion(center=mp.Vector3(x_aperture_in_sim-9, 0, 0), 
                                    size=mp.Vector3(0, size_y, 0),
                                    direction=mp.X, 
                                    weight = -1)

n2f_monitor = simulation.add_near2far(fcen, 
                            df, 
                            nfreq, 
                            ffb_line_monitor) 


ffb_monitor_lens1_1mm = simulation.add_dft_fields([mp.Ez],
                                                fcen, df, nfreq,
                                                where=mp.Volume(center=mp.Vector3(x_aperture_in_sim-9, 0, 0),
                                                                size=mp.Vector3(0, size_y, 0)))

#* Defining an function to extract the power from the ffb_monitor_lens1_1mm monitor at a given timestep during the simulation
def extract_lens1_1mm_power(simulation):
    # Get the DFT array for the ffb_monitor_lens1_1mm monitor
    lens1_1mm_fields = simulation.get_array(vol=mp.Volume(center=mp.Vector3(x_aperture_in_sim-9, 0, 0),
                                                          size=mp.Vector3(0, size_y, 0)),
                                          component=mp.Ez,
                                          cmplx=True)
    # Calculate the power (magnitude squared)
    lens1_1mm_power = np.abs(lens1_1mm_fields)**2

    # Save the power data to an npz compressed file
    np.savez_compressed(os.path.join(savepath, "lens1_1mm_power_{0}.npz".format(simulation.meep_time())),
                        field=lens1_1mm_fields,
                        power=lens1_1mm_power,
                        y_coords=np.linspace(-size_y/2, size_y/2, len(lens1_1mm_fields)))
    
    print(f"Lens1 1mm power data saved to {os.path.join(savepath, 'lens1_1mm_power.npz')} at timestep {simulation.meep_time()}.")


#* 2b) Adding another N2F monitor, 9 mm after the aperture
ffb_line_monitor_2= mp.Near2FarRegion(center=mp.Vector3(x_aperture_in_sim, 0, 0),
                                    size=mp.Vector3(0, size_y, 0),
                                    direction=mp.X, 
                                    weight = -1)

n2f_monitor_2 = simulation.add_near2far(fcen,
                            df, 
                            nfreq, 
                            ffb_line_monitor_2)

ffb_monitor_lens1_11mm = simulation.add_dft_fields([mp.Ez],
                                                fcen, df, nfreq,
                                                where=mp.Volume(center=mp.Vector3(x_aperture_in_sim, 0, 0),
                                                                size=mp.Vector3(0, size_y, 0)))

#* Defining an function to extract the power from the ffb_monitor_lens1_11mm monitor at a given timestep during the simulation
def extract_aperture_11mm__power(simulation):
    # Get the DFT array for the ffb_monitor_lens1_11mm monitor
    lens1_11mm_fields = simulation.get_array(vol=mp.Volume(center=mp.Vector3(x_aperture_in_sim, 0, 0),
                                                          size=mp.Vector3(0, size_y, 0)),
                                          component=mp.Ez,
                                          cmplx=True)
    # Calculate the power (magnitude squared)
    lens1_11mm_power = np.abs(lens1_11mm_fields)**2

    # Save the power data to an npz compressed file
    np.savez_compressed(os.path.join(savepath, "lens1_11mm_power_{0}.npz".format(simulation.meep_time())),
                        field=lens1_11mm_fields,
                        power=lens1_11mm_power,
                        y_coords=np.linspace(-size_y/2, size_y/2, len(lens1_11mm_fields)))
    
    print(f"Lens1 11mm power data saved to {os.path.join(savepath, 'lens1_11mm_power.npz')} at timestep {simulation.meep_time()}.")



#*%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

#* 3. DFT Monitors near the edges of the simulation cell and just above the absorber
# TOP Absorber
top_absorber_monitor = mp.Volume(
    center=mp.Vector3(0, size_y/2 - 1,0),
    size=mp.Vector3(size_x, 0, 0)  
)

top_absorber_dft = simulation.add_dft_fields([mp.Ez],
                                             fcen, df, nfreq,
                                             where=top_absorber_monitor)

# BOTTOM Absorber
bottom_absorber_monitor = mp.Volume(
    center=mp.Vector3(0, -size_y/2 + 1,0),
    size=mp.Vector3(size_x, 0, 0)
)   

bottom_absorber_dft = simulation.add_dft_fields([mp.Ez],
                                                fcen, df, nfreq,
                                                where=bottom_absorber_monitor)

# Top Edge
top_edge_monitor = mp.Volume(
    center=mp.Vector3(0, size_y/2-mpsat_sim.dpml*mpsat_sim.factor_dpml , 0),
    size=mp.Vector3(size_x, 0, 0)  
)

top_edge_dft = simulation.add_dft_fields([mp.Ez],
                                          fcen, df, nfreq,
                                          where=top_edge_monitor)

# Bottom Edge
bottom_edge_monitor = mp.Volume(
    center=mp.Vector3(0, -size_y/2+ mpsat_sim.dpml*mpsat_sim.factor_dpml , 0),
    size=mp.Vector3(size_x, 0, 0)
)

bottom_edge_dft = simulation.add_dft_fields([mp.Ez],
                                            fcen, df, nfreq,
                                            where=bottom_edge_monitor)

#*4 . Power monitor near the source
source_monitor_region = mp.Volume(
    #center=mp.Vector3(data["sources"]["source1"]["center_x"] -30, data["sources"]["source1"]["center_y"], data["sources"]["source1"]["center_z"]),
    center= mp.Vector3(73, 0, 0),
    size=mp.Vector3(0, size_y, 0)
)
# Adding a DFT fields monitor to capture near fields at the source
source_monitor = simulation.add_dft_fields([mp.Ez],
                                           fcen, df, nfreq,
                                           where=source_monitor_region)

# Defining an function to extract the power from the source monitor at a given timestep during the simulation
def extract_source_power(simulation):
    # # Get the DFT array for the source monitor
    # source_fields = simulation.get_array(vol=source_monitor_region, 
    #                                      component= mp.Ez,
    #                                      cmplx= True)

    # Extract the fields: Ex, Ey, Ez, Hx, Hy, Hz
    source_fields = [simulation.get_source(component= c,
                                         vol=source_monitor_region) for c in [mp.Ex, mp.Ey, mp.Ez, mp.Hx, mp.Hy, mp.Hz]]

    # Calculate the power (magnitude squared)
    source_power = np.abs(source_fields)**2

    # Save the power data to an npz compressed file
    np.savez_compressed(os.path.join(savepath, "source_power_{0}.npz".format(simulation.meep_time())),
                        field=source_fields,
                        power=source_power,
                        y_coords=np.linspace(-size_y/2, size_y/2, len(source_fields)))
    
    print(f"Source power data saved to {os.path.join(savepath, 'source_power.npz')} at timestep {simulation.meep_time()}.")


# Dfining an function to extract the phasors information from the DFT monitor at the source plane
def extract_source_phasors(simulation):
    # Get the DFT array for the source monitor
    source_dft_fields = [simulation.get_dft_array(source_monitor, c, 0) for c in [mp.Ex, mp.Ey, mp.Ez, mp.Hx, mp.Hy, mp.Hz]]
    # Save the data to an npz compressed file
    np.savez_compressed(os.path.join(savepath, "source_phasors_{0}.npz".format(simulation.meep_time())),
                        ex=source_dft_fields[0],
                        ey=source_dft_fields[1],
                        ez=source_dft_fields[2],
                        hx=source_dft_fields[3],
                        hy=source_dft_fields[4],
                        hz=source_dft_fields[5])
                        #y_coords=np.linspace(-size_y/2, size_y/2, len(source_dft_fields[0])))
    
    print(f"Source phasors data saved to {os.path.join(savepath, 'source_phasors.npz')} at timestep {simulation.meep_time()}.")


#*5. Defining the step functions for power extraction near the absorbers and edges

def extract_power_near_absorbers_and_edges(simulation):
    # Get the DFT array for the top absorber monitor
    top_absorber_fields = simulation.get_array(vol=top_absorber_monitor, 
                                              component= mp.Ez,
                                              cmplx= True)
    bottom_absorber_fields = simulation.get_array(vol=bottom_absorber_monitor,
                                                 component= mp.Ez,
                                                 cmplx= True)
    
    # Calculate the power (magnitude squared)
    top_absorber_power = np.abs(top_absorber_fields)**2
    bottom_absorber_power = np.abs(bottom_absorber_fields)**2

    # Get the DFT array for the top edge monitor
    top_edge_fields = simulation.get_array(vol=top_edge_monitor,
                                          component= mp.Ez,
                                          cmplx= True)
    bottom_edge_fields = simulation.get_array(vol=bottom_edge_monitor,
                                             component= mp.Ez,
                                             cmplx= True)

    # Calculate the power (magnitude squared)
    top_edge_power = np.abs(top_edge_fields)**2
    bottom_edge_power = np.abs(bottom_edge_fields)**2
    
    # Save the power data to an npz compressed file
    np.savez_compressed(os.path.join(savepath, "absorber_power_{0}.npz".format(simulation.meep_time())),
                        top_field=top_absorber_fields,
                        bottom_field=bottom_absorber_fields,
                        top_power=top_absorber_power,
                        bottom_power=bottom_absorber_power,
                        top_edge_field=top_edge_fields,
                        bottom_edge_field=bottom_edge_fields,
                        top_edge_power=top_edge_power,
                        bottom_edge_power=bottom_edge_power,
                        x_coords=np.linspace(-size_x/2, size_x/2, len(top_absorber_fields)))
    
    print(f"Absorber power data saved to {os.path.join(savepath, 'absorber_power.npz')} at timestep {simulation.meep_time()}.")

# #* Whole simulation box DFT monitor for flux calculation
# # Define a flux region that covers the entire simulation cell
# total_flux_region = mp.FluxRegion(
#     center=mp.Vector3(0, 0, 0),
#     size=mp.Vector3(size_x + mpsat_sim.factor_dpml*mpsat_sim.dpml, size_y + mpsat_sim.factor_dpml*mpsat_sim.dpml, 0)
# )

# # Defining a monitor to capture the fields in the entire simulation cell
# dft_cell = simulation.add_dft_fields([mp.Ez],
#                                     fcen, df, nfreq,
#                                     where=total_flux_region)

#~ ---------------------------------------------

# Extract the epsilon map from the simulation object
simulation.run(until=0)
epsilon_map = simulation.get_epsilon() 

# Visualize with custom labels
plt.figure(figsize=(8, 4))
mp.plot2D(simulation,
          eps_parameters={'vmin': 1, 'vmax': 3, 'cmap': 'binary'})
#plt.imshow(epsilon_map.T, interpolation='spline36', cmap='binary', origin='lower')
# plt.show()
plt.savefig(savepath + "/epsilon_map" + ".png", dpi=300)
# Show plot for 5 seconds
# plt.pause(5)
# plt.close()

#~ ---------------------------------------------

# Set animation parameters
mp_sim_run.set_animation_params(anim_params= {'image_every': data["output"]["animation_options"]["image_every"], 
                                              'Nfps': data["output"]["animation_options"]["Nfps"], 
                                              'anim_file_name': savepath + "/"+ data["output"]["animation_options"]["movie_name"] + ".mp4"})

simulation.use_output_directory(savepath)

timestepDuration = data["output"]["animation_options"]["image_every"]
total_time= runtime

f = plt.figure(dpi = 150)
animate = mp.Animate2D(simulation,
                        f = f,
                        fields=mp.Ez,
                        realtime=True,
                        eps_parameters={'vmin': 1, 'vmax': 3, 'cmap': 'binary'})

# Calculate the timeperiod of the source frequency 
period = 1 / fcen  # Time period in MEEP time units
# We need atleast 10 points per period to properly sample the wave
points_per_period = 10
dt = period / points_per_period  # Time step in MEEP time units

t0 = 390 # Time after which we start extracting power near the absorbers

# step_func_power_near_edge = mp.after_time(t0, mp.at_every(dt, extract_power_near_absorbers))

# #! ======================================================
#! REFERENCE SIMS FOR NORMALIZATION
reference_sim = mp.Simulation(
    cell_size=mpsat_sim.cell,
    sources=source_list,
    resolution=mpsat_sim.resolution,
    boundary_layers=custom_boundary_layers,
    geometry=[],
    symmetries = symmetries,
    force_complex_fields= True)
    
# Add a flux region at the source position to measure the input power
source_flux_region_ref = mp.FluxRegion(
    center=mp.Vector3(70, 0, 0), 
    size=mp.Vector3(0, size_y, 0)
)

source_flux_monitor_ref = reference_sim.add_flux(fcen, df, nfreq, source_flux_region_ref)

# Fnction to calculate the flux array at a given timestep
def extract_flux_data_from_monitor(simulation):
    # Extract the total flux from the monitor (for normalization)
    flux_data = mp.get_fluxes(source_flux_monitor_ref)
    # Save the flux data to an npz compressed file
    np.savez_compressed(os.path.join(savepath, "source_total_flux_reference_{0}.npz".format(simulation.meep_time())),
                        flux=flux_data)    


total_time_ref = 30
t0_ref = 20 # Time after which we start extracting flux data

reference_sim.run(
                  mp.after_time(t0_ref, mp.at_every(dt, mp_sim_run.extract_flux_data_from_monitor)),
                  mp.after_time(t0_ref, mp.at_every(dt, mp_sim_run.extract_source_power)),
                  #mp.after_time(t0_ref, mp.at_every(dt, extract_source_phasors)),
                  until = total_time_ref
                  )

# #! ======================================================

# #~ Function to exract the poynting flux for the entire simulation box at a given timestep
# # def extract_total_flux_map(simulation):
# #     # Define a box to capture the fields in the entire simulation cell
# #     box = mp.Volume(center=mp.Vector3(0, 0, 0),
# #                     size=mp.Vector3(size_x + mpsat_sim.factor_dpml*mpsat_sim.dpml, size_y + mpsat_sim.factor_dpml*mpsat_sim.dpml, 0)
# #     )

# #     # # Run the current simulation to some short time to capture the fields
# #     # simulation.run(until=1)

# #     (Ex,Ey,Ez)     = [simulation.get_array(vol=box, component=c, cmplx=True) for c in [mp.Ex, mp.Ey, mp.Ez]]
# #     eps            = simulation.get_array(vol=box, component=mp.Dielectric)
# #     (x,y,z,w)      = simulation.get_array_metadata(vol=box)
# #     energy_density = np.real(eps*(np.conj(Ex)*Ex + np.conj(Ey)*Ey + np.conj(Ez)*Ez)) # array
# #     energy         = np.sum(w*energy_density)                                        # scalar

# #     # Get the DFT array for the total flux monitor
# #     Ez = simulation.get_dft_array(dft_cell, mp.Ez, 0)
# #     Hy = simulation.get_dft_array(dft_cell, mp.Hy, 0)
# #     (x,y,z,w) = simulation.get_array_metadata(dft=dft_cell)
# #     flux_density = 0.5 * np.real(np.conj(Ez) * Hy)
# #     flux = np.sum(w * flux_density)


# #     # Save the flux data to an npz compressed file
# #     np.savez_compressed(os.path.join(savepath, "poynting_flux_{0}.npz".format(simulation.meep_time())),
# #                         eps=eps,
# #                         Ez=Ez,
# #                         Hy=Hy,
# #                         energy_density=energy_density,
# #                         energy=energy,
# #                         flux_density=flux_density,
# #                         flux=flux,
# #                         x_coords=x,
# #                         y_coords=y,
# #                         weights=w)

def extract_xyzw(simulation):
    # Define a box to capture the fields in the entire simulation cell
    box = mp.Volume(center=mp.Vector3(0, 0, 0),
                    size=mp.Vector3(size_x, size_y, 0)
    )
    (x,y,z,w) = simulation.get_array_metadata(vol=box)

    # Save the xyzw data to an npz compressed file
    np.savez_compressed(os.path.join(savepath, "xyzw_{0}.npz".format(simulation.meep_time())),
                        x_coords=x,
                        y_coords=y,
                        weights=w)


simulation.run(mp.at_beginning(mp.output_epsilon(frequency= fcen)),
               mp.at_every(timestepDuration, animate), 
               mp.at_every(timestepDuration, mp_sim_run.Ez2_dB),
               mp.after_time(t0, mp.at_every(dt, extract_power_near_absorbers_and_edges)),
               mp.after_time(t0, mp.at_every(dt, extract_aperture_power)),
               mp.after_time(t0, mp.at_every(dt, extract_lens1_1mm_power)),
               mp.after_time(t0, mp.at_every(dt, extract_aperture_11mm__power)),
               #mp.after_time(t0, mp.at_every(dt, extract_total_flux_map)),
               mp.after_time(t0, mp.at_every(dt, mp.output_hfield)),
               mp.after_time(t0, mp.at_every(dt, mp.output_efield)),
               mp.after_time(t0, mp.at_every(dt, mp.output_dfield)),
               mp.after_time(t0, mp.at_every(dt, mp.output_sfield)),
               mp.after_time(t0, mp.at_every(dt, mp.output_bfield)),
               #mp.after_time(t0, mp.at_every(dt, extract_xyzw)),
               mp.after_time(t0, mp.at_every(dt, extract_aperture_phasors)),
               mp.at_end(mp_sim_run.save_animation),
               mp.at_end(mp.output_hfield),
               mp.at_end(mp.output_efield),
               mp.at_end(mp.output_bfield),
               mp.at_end(mp.output_dfield),
               mp.at_end(mp.output_sfield),
               mp.at_end(extract_xyzw),
               until = total_time)

#Save the animation using to_gif
animate.to_gif(data["output"]["animation_options"]["Nfps"],
               savepath + "/"+ data["output"]["animation_options"]["movie_name"] + "efield.gif")

#! POST-PROCESSING
#~ ---------------------------------------------
#~ ---------------------------------------------
# Power calculation at the top and bottom edges of the simulation cell + absorbers monitor
top_absorber_fields = simulation.get_array(component=mp.Ez, vol=top_absorber_monitor, cmplx=True)
bottom_absorber_fields = simulation.get_array(component=mp.Ez, vol=bottom_absorber_monitor, cmplx=True)
top_absorber_power = np.abs(top_absorber_fields)**2
bottom_absorber_power = np.abs(bottom_absorber_fields)**2
# Convert to dB
top_absorber_power_dB = 10 * np.log10(top_absorber_power + 1e-10)  # Add small value to avoid log(0)
bottom_absorber_power_dB = 10 * np.log10(bottom_absorber_power + 1e-10)  # Add small value to avoid log(0)
x_coords_top_absorber_power = np.linspace(-size_x/2, size_x/2, len(top_absorber_fields))
x_coords_bottom_absorber_power = np.linspace(-size_x/2, size_x/2, len(bottom_absorber_fields))

# Now edge power
top_edge_fields = simulation.get_array(component=mp.Ez, vol=top_edge_monitor, cmplx=True)
bottom_edge_fields = simulation.get_array(component=mp.Ez, vol=bottom_edge_monitor, cmplx=True)
top_edge_power = np.abs(top_edge_fields)**2
bottom_edge_power = np.abs(bottom_edge_fields)**2
# Convert to dB
top_edge_power_dB = 10 * np.log10(top_edge_power + 1e-10)  # Add small value to avoid log(0)
bottom_edge_power_dB = 10 * np.log10(bottom_edge_power + 1e-10)  # Add small value to avoid log(0)
x_coords_top_edge_power = np.linspace(-size_x/2, size_x/2, len(top_edge_fields)) 
x_coords_bottom_edge_power = np.linspace(-size_x/2, size_x/2, len(bottom_edge_fields))

# Plot, save everything as a hdf5 file
plt.figure(figsize=(12, 6))
plt.plot(x_coords_top_absorber_power, -top_absorber_power_dB, label='-Top Absorber Power (dB)', color='blue', linewidth=2)
plt.plot(x_coords_bottom_absorber_power, bottom_absorber_power_dB, label='Bottom Absorber Power (dB)', color='orange', linewidth=2)
plt.plot(x_coords_top_edge_power, -top_edge_power_dB, label='-Top Edge Power (dB)', color='green', linewidth=2)
plt.plot(x_coords_bottom_edge_power, bottom_edge_power_dB, label='Bottom Edge Power (dB)', color='red', linewidth=2)
plt.xlabel('X Position (mm)')
plt.ylabel('Power (dB)')
plt.title('Power Distribution at Absorbers and Edges')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(data["output"]["savepath"]["path"] + data["simulation"]["name"] + "_absorber_edge_power.png", dpi=300, bbox_inches='tight')
#plt.show()

# Save power data to HDF5 file
# with h5py.File(data["output"]["savepath"]["path"] + data["simulation"]["name"] + "_absorber_edge_power.h5", "w") as f:
#     f.create_dataset("top_absorber_power_dB", data=-top_absorber_power_dB)
#     f.create_dataset("bottom_absorber_power_dB", data=bottom_absorber_power_dB)
#     f.create_dataset("top_edge_power_dB", data=-top_edge_power_dB)
#     f.create_dataset("bottom_edge_power_dB", data=bottom_edge_power_dB)
#     f.create_dataset("x_coords_top_absorber_power", data=x_coords_top_absorber_power)
#     f.create_dataset("x_coords_bottom_absorber_power", data=x_coords_bottom_absorber_power)
#     f.create_dataset("x_coords_top_edge_power", data=x_coords_top_edge_power)
#     f.create_dataset("x_coords_bottom_edge_power", data=x_coords_bottom_edge_power)
# print(f"Power data saved to: {data['output']['savepath']['path']}{data['simulation']['name']}_absorber_edge_power.h5")

# Replace HDF5 save with JSON save
absorber_edge_data = {
    "top_absorber_power_dB": (-top_absorber_power_dB).tolist(),
    "bottom_absorber_power_dB": bottom_absorber_power_dB.tolist(),
    "top_edge_power_dB": (-top_edge_power_dB).tolist(),
    "bottom_edge_power_dB": bottom_edge_power_dB.tolist(),
    "x_coords_top_absorber_power": x_coords_top_absorber_power.tolist(),
    "x_coords_bottom_absorber_power": x_coords_bottom_absorber_power.tolist(),
    "x_coords_top_edge_power": x_coords_top_edge_power.tolist(),
    "x_coords_bottom_edge_power": x_coords_bottom_edge_power.tolist()
}

with open(data["output"]["savepath"]["path"] + data["simulation"]["name"] + "_absorber_edge_power.json", "w") as f:
    json.dump(absorber_edge_data, f, indent=2)
print(f"Power data saved to: {data['output']['savepath']['path']}{data['simulation']['name']}_absorber_edge_power.json")


# #~ ---------------------------------------------
# #* Near-to-Far Field (N2F) transformation near 11 mm before the lens 1
# 1000 points
mpsat_analysis.get_MEEP_ff(
            simulation=simulation,
            ff_distance= 1e8, #!ffb_distance_mm,
            ff_angle=45,
            ff_npts=1000,
            n2f_obj=n2f_monitor_2,
            saveplot=True,
            #parallel=exec(data["simulation"]["parallel"]),
            parallel=True,
            saveh5=True,
            filename=data["output"]["savepath"]["path"] + data["simulation"]["name"] + "n2f_monitor_11mm_before_lens1_ff_1000points",
            ylim=-40,
            plot_title=data["simulation"]["name"] + " - FFB Amplitude for " + "{}".format(round(wavelength_natural_real, 4)) + "m"
)
 

# 2000 points
mpsat_analysis.get_MEEP_ff(
            simulation=simulation,
            ff_distance= 1e8, #!ffb_distance_mm,
            ff_angle=45,
            ff_npts=2000,
            n2f_obj=n2f_monitor_2,
            saveplot=True,
            #parallel=exec(data["simulation"]["parallel"]),
            parallel=True,
            saveh5=True,
            filename=data["output"]["savepath"]["path"] + data["simulation"]["name"] + "n2f_monitor_11mm_before_lens1_ff_2000points",
            ylim=-40,
            plot_title=data["simulation"]["name"] + " - FFB Amplitude for " + "{}".format(round(wavelength_natural_real, 4)) + "m"
)

# 4000 points
mpsat_analysis.get_MEEP_ff(
            simulation=simulation,
            ff_distance= 1e8, #!ffb_distance_mm,
            ff_angle=45,
            ff_npts=4000,
            n2f_obj=n2f_monitor_2,
            saveplot=True,
            #parallel=exec(data["simulation"]["parallel"]),
            parallel=True,
            saveh5=True,
            filename=data["output"]["savepath"]["path"] + data["simulation"]["name"] + "n2f_monitor_11mm_before_lens1_ff_4000points",
            ylim=-40,
            plot_title=data["simulation"]["name"] + " - FFB Amplitude for " + "{}".format(round(wavelength_natural_real, 4)) + "m"
)




complex_field_at_11mm_before_lens1 = mpsat_analysis.get_complex_field(sim = simulation,
                                                            simres = mpsat_sim.resolution,
                                                            aper_size = data["lenses"]["lens1"]["diameter"],
                                                            aper_pos_x = x_aperture_in_sim,
                                                            wvl = wavelength_meep, 
                                                            plot_amp = True, 
                                                            saveh5 = True, 
                                                            filename = data["output"]["savepath"]["path"] + 'complex_field_at_11mm_before_lens1',
                                                            parallel=True)

freq_11mm, FFTs_11mm = mpsat_analysis.custom_beam_FT(sim_res= mpsat_sim.resolution,
                    list_efields = complex_field_at_11mm_before_lens1,
                    aper_size = data["lenses"]["lens1"]["diameter"],
                    zero_pad = 15,
                    savebeam = True,
                    parallel = True,
                    filename = data["output"]["savepath"]["path"] + 'beam_FT_at_11mm_before_lens1')

# Save the frequency and FFTs data to JSON
fft_data_11mm = {
    "frequency": np.array(freq_11mm).tolist(),
    "FFTs": np.array(FFTs_11mm).tolist(),
    "wavelength_m": wavelength_natural_real,
    "x_position_mm": x_aperture_in_sim - 48.38,  
    "aperture_diameter_mm": data["lenses"]["lens1"]["diameter"]
}

# Save to JSON file
with open(data["output"]["savepath"]["path"] + data["simulation"]["name"] + "_beam_FT_11mm_before_lens1.json", "w") as f:
    json.dump(fft_data_11mm, f, indent=2)
print(f"Beam Fourier Transform data saved to: {data['output']['savepath']['path']}{data['simulation']['name']}_beam_FT_11mm_before_lens1.json")

# #~ ---------------------------------------------

# #*Near-to-Far Field (N2F) transformation near 1 mm before the lens 1
# 1000 points
mpsat_analysis.get_MEEP_ff(
            simulation=simulation,
            ff_distance= 1e8, #!ffb_distance_mm,
            ff_angle=45,
            ff_npts=1000,
            n2f_obj=n2f_monitor,
            saveplot=True,
            #parallel=exec(data["simulation"]["parallel"]),
            parallel=True,
            saveh5=True,
            filename=data["output"]["savepath"]["path"] + data["simulation"]["name"] + "n2f_monitor_ff",
            ylim=-40,
            plot_title=data["simulation"]["name"] + " - FFB Amplitude for " + "{}".format(round(wavelength_natural_real, 4)) + "m"
)

# 2000 points
mpsat_analysis.get_MEEP_ff(
            simulation=simulation,
            ff_distance= 1e8, #!ffb_distance_mm,
            ff_angle=45,
            ff_npts=2000,
            n2f_obj=n2f_monitor,
            saveplot=True,
            #parallel=exec(data["simulation"]["parallel"]),
            parallel=True,
            saveh5=True,
            filename=data["output"]["savepath"]["path"] + data["simulation"]["name"] + "n2f_monitor_ff_2000points",
            ylim=-40,
            plot_title=data["simulation"]["name"] + " - FFB Amplitude for " + "{}".format(round(wavelength_natural_real, 4)) + "m"
)

# 4000 points
mpsat_analysis.get_MEEP_ff(
            simulation=simulation,
            ff_distance= 1e8, #!ffb_distance_mm,
            ff_angle=45,
            ff_npts=4000,
            n2f_obj=n2f_monitor,
            saveplot=True,
            #parallel=exec(data["simulation"]["parallel"]),
            parallel=True,
            saveh5=True,
            filename=data["output"]["savepath"]["path"] + data["simulation"]["name"] + "n2f_monitor_ff_4000points",
            ylim=-40,
            plot_title=data["simulation"]["name"] + " - FFB Amplitude for " + "{}".format(round(wavelength_natural_real, 4)) + "m"
)

complex_field_at_aperture = mpsat_analysis.get_complex_field(sim = simulation,
                                                            simres = mpsat_sim.resolution,
                                                            aper_size = data["lenses"]["lens1"]["diameter"],
                                                            aper_pos_x = x_aperture_in_sim,
                                                            wvl = wavelength_meep, 
                                                            plot_amp = True, 
                                                            saveh5 = True, 
                                                            filename = data["output"]["savepath"]["path"] + 'complex_field_at_aperture',
                                                            parallel=True)

freq, FFTs = mpsat_analysis.custom_beam_FT(sim_res= mpsat_sim.resolution,
                    list_efields = complex_field_at_aperture,
                    aper_size = data["lenses"]["lens1"]["diameter"],
                    zero_pad = 15,
                    savebeam = True,
                    parallel = True,
                    filename = data["output"]["savepath"]["path"] + 'beam_FT_at_aperture')

# Save the frequency and FFTs data to JSON
fft_data = {
    "frequency": np.array(freq).tolist(),
    "FFTs": np.array(FFTs).tolist(),
    "wavelength_m": wavelength_natural_real,
    "x_position_mm": x_aperture,
    "aperture_diameter_mm": data["lenses"]["lens1"]["diameter"]
}   

# Save to JSON file
with open(data["output"]["savepath"]["path"] + data["simulation"]["name"] + "_beam_FT.json", "w") as f:
    json.dump(fft_data, f, indent=2)
print(f"Beam Fourier Transform data saved to: {data['output']['savepath']['path']}{data['simulation']['name']}_aperture_beam_FT.json")

# Extract field data at aperture
aperture_fields = simulation.get_array(component=mp.Ez, vol=aperture_line_monitor, cmplx=True)
y_coords = np.linspace(-size_y/2, size_y/2, len(aperture_fields))

# Calculate power density (|E|^2)
power_density = np.abs(aperture_fields)**2

# Normalize to see relative distribution
power_density_normalized = power_density / np.max(power_density)
power_density_dB = 10 * np.log10(power_density_normalized + 1e-10)  # Add small value to avoid log(0)

# Extract total power through aperture
aperture_power = mp.get_fluxes(aperture_flux_monitor)[0]
print(f"Total power at aperture: {aperture_power}")

# Get aperture diameter from JSON file
aperture_diameter = data["lenses"]["lens1"]["diameter"]
aperture_radius = aperture_diameter / 2

# Plot the beam profile showing sidelobes
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(y_coords, power_density_normalized, 'b-', linewidth=2, label='Beam Profile')
# Add aperture boundary lines
plt.axvline(x=aperture_radius, color='cyan', linestyle='--', linewidth=2, alpha=0.7, label=f'Aperture Edge (+{aperture_radius:.1f} mm)')
plt.axvline(x=-aperture_radius, color='cyan', linestyle='--', linewidth=2, alpha=0.7, label=f'Aperture Edge (-{aperture_radius:.1f} mm)')
plt.xlabel('Y position (mm)')
plt.ylabel('Normalized Power')
plt.title(f'Aperture Beam Profile (Linear Scale)\nAperture Diameter: {aperture_diameter} mm')
plt.grid(True, alpha=0.3)
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(y_coords, power_density_dB, 'r-', linewidth=2, label='Beam Profile (dB)')
# Add aperture boundary lines
plt.axvline(x=aperture_radius, color='cyan', linestyle='--', linewidth=2, alpha=0.7, label=f'Aperture Edge (+{aperture_radius:.1f} mm)')
plt.axvline(x=-aperture_radius, color='cyan', linestyle='--', linewidth=2, alpha=0.7, label=f'Aperture Edge (-{aperture_radius:.1f} mm)')
plt.xlabel('Y position (mm)')
plt.ylabel('Power (dB)')
plt.title(f'Aperture Beam Profile (dB Scale)\nAperture Diameter: {aperture_diameter} mm')
plt.ylim([-60, 0])  # Show down to -60 dB to see sidelobes
plt.grid(True, alpha=0.3)
plt.legend()

plt.tight_layout()
plt.savefig(data["output"]["savepath"]["path"] + data["simulation"]["name"] + "_aperture_beam_profile.png", dpi=300, bbox_inches='tight')
#plt.show()

# Calculate power within aperture vs total power
# Find indices within aperture bounds
aperture_mask = (y_coords >= -aperture_radius) & (y_coords <= aperture_radius)
power_within_aperture = np.sum(power_density[aperture_mask])
total_power_profile = np.sum(power_density)
aperture_efficiency = power_within_aperture / total_power_profile

print(f"Aperture diameter: {aperture_diameter} mm")
print(f"Power within aperture: {power_within_aperture:.6e}")
print(f"Total power in profile: {total_power_profile:.6e}")
print(f"Aperture efficiency: {aperture_efficiency:.4f} ({aperture_efficiency*100:.2f}%)")

# Save aperture data
aperture_data = {
    "y_coordinates_mm": y_coords.tolist(),
    "power_density_normalized": power_density_normalized.tolist(),
    "power_density_dB": power_density_dB.tolist(),
    "total_aperture_power": aperture_power,
    "frequency": fcen,
    "wavelength_m": wavelength_natural_real,
    "x_position_mm": x_aperture,
    "aperture_diameter_mm": aperture_diameter,
    "aperture_radius_mm": aperture_radius,
    "power_within_aperture": float(power_within_aperture),
    "total_power_profile": float(total_power_profile),
    "aperture_efficiency": float(aperture_efficiency)
}

# Save to JSON file
with open(data["output"]["savepath"]["path"] + data["simulation"]["name"] + "_aperture_analysis.json", "w") as f:
    json.dump(aperture_data, f, indent=2)

# Save to HDF5 for more detailed analysis
# Add before the HDF5 file writing code
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()

# Determine if running in parallel
parallel = comm.Get_size() > 1  # This will be True if more than 1 process

if parallel:
    # Check if h5py supports MPI
    if not h5py.get_config().mpi:
        raise ValueError("h5py was built without MPI support, can't use mpio driver")
    
    # Use MPI-IO driver for parallel writing
    with h5py.File(data["output"]["savepath"]["path"] + data["simulation"]["name"] + "_aperture_fields.h5", 
                  "w", driver='mpio', comm=comm) as f:
        f.create_dataset("y_coordinates", data=y_coords)
        f.create_dataset("Ez_real", data=np.real(aperture_fields))
        f.create_dataset("Ez_imag", data=np.imag(aperture_fields))
        f.create_dataset("power_density", data=power_density)
        f.attrs["total_power"] = aperture_power
        f.attrs["frequency"] = fcen
        f.attrs["wavelength_m"] = wavelength_natural_real
        f.attrs["aperture_diameter_mm"] = aperture_diameter
        f.attrs["aperture_efficiency"] = aperture_efficiency
else:
    # Only have rank 0 write the file in non-parallel mode
    if rank == 0:
        with h5py.File(data["output"]["savepath"]["path"] + data["simulation"]["name"] + "_aperture_fields.h5", 
                      "w", libver='latest') as f:
            f.create_dataset("y_coordinates", data=y_coords, compression='gzip')
            f.create_dataset("Ez_real", data=np.real(aperture_fields), compression='gzip')
            f.create_dataset("Ez_imag", data=np.imag(aperture_fields), compression='gzip')
            f.create_dataset("power_density", data=power_density, compression='gzip')
            f.attrs["total_power"] = aperture_power
            f.attrs["frequency"] = fcen
            f.attrs["wavelength_m"] = wavelength_natural_real
            f.attrs["aperture_diameter_mm"] = aperture_diameter
            f.attrs["aperture_efficiency"] = aperture_efficiency
        print(f"Aperture field data saved to {data['output']['savepath']['path'] + data['simulation']['name']}_aperture_fields.h5")
    comm.barrier()  # Ensure all processes wait for file creation

print(f"Aperture analysis saved to: {data['output']['savepath']['path']}")
print(f"- Beam profile plot: {data['simulation']['name']}_aperture_beam_profile.png")
print(f"- Analysis data: {data['simulation']['name']}_aperture_analysis.json")
print(f"- Field data: {data['simulation']['name']}_aperture_fields.h5")

# Save the data to a JSON file
with open(data["output"]["savepath"]["path"] + data["simulation"]["name"] + "_simulation_data.json", "w") as f:
    json.dump(data, f, indent=2)
print(f"Simulation parameters saved to: {data['output']['savepath']['path']}{data['simulation']['name']}_simulation_data.json")