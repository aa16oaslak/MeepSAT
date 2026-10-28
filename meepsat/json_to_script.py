import json
import os

#! THE FOLLOWING FUNCTION MIGHT CAUSE ERRORS IF THE PATHS ARE NOT SET CORRECTLY


def temporary_meepasat_path_set(json_data=None):
    # Check if savepath is provided in JSON data
    savepath_from_json = ""
    if json_data and "output" in json_data and "savepath" in json_data["output"]:
        savepath_from_json = json_data["output"]["savepath"]["path"]
    
    script = f"""
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


# Handle savepath - use JSON path if provided, otherwise create default
"""
    
    # Rest of your function remains the same...
    if savepath_from_json:
        script += f"""
# Using savepath from JSON configuration
savepath = '{savepath_from_json}'
os.makedirs(savepath, exist_ok=True)
"""
    else:
        script += f"""
# Creating default output directory in parent directory
output_dir = os.path.join(os.path.dirname(os.getcwd()), 'output_files')
os.makedirs(output_dir, exist_ok=True)
savepath = output_dir + '/'
"""

    script += f"""
# Fix imports by making sure Python can find the meepsat submodules directly
sys.path.insert(0, meepsat_dir)  # Prioritize the meepsat directory

print('Main directory path:', main_dir)
print('MEEPSAT library path:', meepsat_dir)
print('Output directory path:', savepath)
print('Python path now includes:', sys.path[:5])  # Show the first 5 paths
"""
    return script

#! <-- THE FUNCTION ENDS HERE --> #

def import_modules():
    script = f"""
import meep as mp
import numpy as np
import h5py
import matplotlib.pyplot as plt
    """
    return script


def import_meepsat_modules():
    script = f"""
# Importing the MEEPSAT librarires
import meepsat.simulation_2D as sim
import meepsat.components_2D_meep as comp_meep
import meepsat.components_2D_eps as comp_eps
import meepsat.visualization as mpsat_plot
import meepsat.mp_sim_run as mp_sim_run
"""
    return script

def init_meepsat_sim(data):
    script = f"""
#~ Initialising MEEPSAT Simulation
cell_X, cell_Y, cell_Z = {data["simulation"]['primary_params']['cell_size']['x']}, {data["simulation"]['primary_params']['cell_size']['y']}, {data["simulation"]['primary_params']['cell_size']['z']} # Cell Size without considering the PML thickness and its factor

# Initialize the simulation with the different parameters
mpsat_sim = sim.sim_init(sim_name='{data["simulation"]["name"]}',
                        cell_size= [cell_X, cell_Y, cell_Z], # [sx, sy, sz] in mm
                        smallest_freq= {data["simulation"]['primary_params']['smallest_freq']}, 
                        resolution= {data["simulation"]['primary_params']['resolution']},
                        boundary_layer_type= '{data['boundary_layers']['boundary']['type']}',
                        boundary_layer_size= {data['boundary_layers']['boundary']['size']},
                        factor_dpml= {data['boundary_layers']['boundary']['factor_dpml']},
                        precision= '{data["simulation"]['primary_params'].get('precision', 'auto')}')

# Print the simulation parameters
mpsat_sim.print_simulation_parameters()
"""
    return script

#! <-- VARIOUS SOURCE SCRIPT SEGMENTS --> #
def GaussianBeam(source_data):
    # Handle the special mp.inf case
    if source_data["extra_args"]["end_time"] == "mp.inf":
        end_time = "mp.inf"
    else:
        end_time = source_data["extra_args"]["end_time"]
        
    script = f"""
#~ Adding Source: GaussianBeam
gaussian_source_init = comp_meep.GaussianBeam(mpsat_sim=mpsat_sim,
                                              center= mp.Vector3({source_data["center_x"]},
                                                                {source_data["center_y"]},
                                                                {source_data["center_z"]}),
                                              size= mp.Vector3({source_data["size_x"]},
                                                              {source_data["size_y"]},
                                                              {source_data["size_z"]}),
                                              component= '{source_data["component"]}',
                                              freq= {source_data["frequency"]},
                                              angle= {source_data["extra_args"]["angle"]},
                                              width= {source_data["extra_args"]["width"]},
                                              kwargs= {{'beam_x0': mp.Vector3({source_data["extra_args"]["beam_x0"]["x"]},
                                                                             {source_data["extra_args"]["beam_x0"]["y"]},
                                                                             {source_data["extra_args"]["beam_x0"]["z"]}),
                                                       'beam_E0': mp.Vector3({source_data["extra_args"]["beam_E0"]["x"]},
                                                                             {source_data["extra_args"]["beam_E0"]["y"]},
                                                                             {source_data["extra_args"]["beam_E0"]["z"]}),
                                                       'start_time': {source_data["extra_args"]["start_time"]},
                                                       'end_time': {end_time}}})
source_list.append(gaussian_source_init.assemble())
"""
    return script

def continuous_planewaves(data):
    script = f"""
#~ Adding Source: ContinuousPlaneWaves  
cpw_source_init = comp_meep.ContinuousPlaneWaves(mpsat_sim=mpsat_sim,
                                                center= mp.Vector3({data["center_x"]},
                                                                  {data["center_y"]},
                                                                  {data["center_z"]}),
                                                size= mp.Vector3({data["size_x"]},
                                                                {data["size_y"]},
                                                                {data["size_z"]}),
                                                component= '{data["component"]}',
                                                freq= {data["frequency"]},
                                                angle= {data["extra_args"]["angle"]},
                                                rot_axis= '{data["extra_args"].get("rot_axis", "z")}',
                                                kwargs= {{"is_integrated": {data["extra_args"]["is_integrated"]},
                                                         "start_time": {data["extra_args"]["start_time"]},
                                                         "end_time": {data["extra_args"]["end_time"]}}})
source_list.append(cpw_source_init.assemble())
"""
    return script

def broadband_planewaves(source_data, source_name):
    """
    BroadbandPlaneWaveSource: a Gaussian pulse covering [wvl_min, wvl_max]

    The band is given either as "wvl_min"/"wvl_max" or as "freq_min"/"freq_max"
    (MEEP units), with an optional "central_wvl". "angle" and "rot_axis" are read
    from "extra_args", whose other entries go to mp.Source/mp.GaussianSource.
    """
    if "wvl_min" in source_data and "wvl_max" in source_data:
        wvl_min, wvl_max = source_data["wvl_min"], source_data["wvl_max"]
    elif "freq_min" in source_data and "freq_max" in source_data:
        wvl_min, wvl_max = 1 / source_data["freq_max"], 1 / source_data["freq_min"]
    else:
        raise ValueError(f"Source {source_name} must define either wvl_min and wvl_max "
                         "or freq_min and freq_max")

    extra_args = dict(source_data.get("extra_args", {}))
    angle = extra_args.pop("angle", 0)
    rot_axis = extra_args.pop("rot_axis", "x")
    kwargs = to_script_value(extra_args) if extra_args else None

    script = f"""
#~ Adding Source: BroadbandPlaneWave
{source_name}_init = comp_meep.BroadbandPlaneWaveSource(mpsat_sim=mpsat_sim,
                                                center= mp.Vector3({source_data["center_x"]},
                                                                  {source_data["center_y"]},
                                                                  {source_data["center_z"]}),
                                                size= mp.Vector3({source_data["size_x"]},
                                                                {source_data["size_y"]},
                                                                {source_data["size_z"]}),
                                                component= '{source_data["component"]}',
                                                central_wvl= {source_data.get("central_wvl", None)},
                                                wvl_min= {wvl_min},
                                                wvl_max= {wvl_max},
                                                angle= {angle},
                                                rot_axis= '{rot_axis}',
                                                kwargs= {kwargs})
source_list.append({source_name}_init.assemble())
"""
    return script

def feedhorn_pattern(source_data, source_name):
    """
    FeedhornPatternSource: the line source that radiates a feedhorn far-field cut
    read from an RTCC .pat file

    "pattern_file" is the .pat file (a relative path is taken from the directory the
    script runs in) and "pattern_phi" the cut [deg] (default 0). The cut nearest to
    "pattern_freq_ghz" is used, which a previous run records in its output JSON;
    otherwise the source "frequency" (MEEP units, 1/mm) is converted to GHz.
    "extra_args" may hold the FeedhornPatternSource options (angle, theta_max,
    points_per_wvl, conjugate_phase, amplitude, huygens_sign, label), start_time and
    end_time; other mp.ContinuousSource arguments go in "source_kwargs". Any other
    extra_args entry (e.g. the GaussianBeam beam_x0, beam_E0 and width) is ignored,
    so a GaussianBeam source can be switched to this type by changing its "type".
    """
    if "pattern_file" not in source_data:
        raise ValueError(f"Source {source_name} of type {source_data['type']} must define 'pattern_file'")

    if "pattern_freq_ghz" in source_data:
        freq_ghz = source_data["pattern_freq_ghz"]
    elif "frequency" in source_data:
        freq_ghz = f"{source_data['frequency']} * comp_meep.C_MM_GHZ"
    else:
        raise ValueError(f"Source {source_name} must define either 'frequency' or 'pattern_freq_ghz'")

    extra_args = source_data.get("extra_args", {})
    class_args = ["angle", "theta_max", "points_per_wvl", "conjugate_phase",
                  "amplitude", "huygens_sign", "label"]
    params = {key: extra_args[key] for key in class_args if key in extra_args}

    # Arguments of mp.ContinuousSource
    source_kwargs = {key: extra_args[key] for key in ("start_time", "end_time") if key in extra_args}
    source_kwargs.update(source_data.get("source_kwargs", {}))
    if source_kwargs:
        params["kwargs"] = source_kwargs

    extra_params = ""
    if params:
        extra_params = ",\n" + " " * 61 + kwargs_script(params, ",\n" + " " * 61)

    script = f"""
#~ Adding Source: FeedhornPattern
{source_name}_init = comp_meep.FeedhornPatternSource.from_pat('{source_data["pattern_file"]}',
                                                             freq_ghz= {freq_ghz},
                                                             phi= {source_data.get("pattern_phi", 0.0)},
                                                             center= mp.Vector3({source_data["center_x"]},
                                                                               {source_data["center_y"]},
                                                                               {source_data["center_z"]}),
                                                             size= mp.Vector3({source_data["size_x"]},
                                                                             {source_data["size_y"]},
                                                                             {source_data["size_z"]}),
                                                             component= '{source_data["component"]}'{extra_params})
if mp.am_master():
    print({source_name}_init.summary())
# Two sources: the electric and magnetic currents of the one-way line source
source_list.extend({source_name}_init.assemble())
"""
    return script

def source_script(data):
    script = f"""source_list = []"""

    for source in data["sources"]:
        source_data = data["sources"][source]
        # Check the type of source
        if source_data['type'] == 'GaussianBeam':
            script = script + '\n' + GaussianBeam(source_data)
        elif source_data['type'] == 'ContinuousPlaneWaves':
            script = script + '\n' + continuous_planewaves(source_data)
        elif source_data['type'] in ('BroadbandPlaneWave', 'BroadbandPlaneWaveSource'):
            script = script + '\n' + broadband_planewaves(source_data, source)
        elif source_data['type'] in ('FeedhornPattern', 'FeedhornPatternSource'):
            script = script + '\n' + feedhorn_pattern(source_data, source)
        else:
            raise ValueError(f"Source type {source_data['type']} not recognized.")

    return script

#! <-- THE END: VARIOUS SOURCE SCRIPT SEGMENTS --> #
#^====================================================================================================================^#

#! <-- BOUNDARY SCRIPT SEGMENTS --> #
def PML_boundary(boundary_data):
    """
    PML layers, appended to boundary_layers

    "thickness" is mpsat_sim.dpml (the "size") unless given, either as a number or
    as an expression such as "mpsat_sim.dpml*mpsat_sim.factor_dpml". sim_init pads
    the cell by factor_dpml*dpml in each direction, i.e. factor_dpml*dpml/2 per side.

    Without "sides" the PML covers every side. "sides" lists the sides to cover, as
    "x_low", "x_high", "y_low", "y_high" (one side) or "x", "y" (both sides).
    """
    thickness = to_script_value(boundary_data.get("thickness", "mpsat_sim.dpml"))
    sides = boundary_data.get("sides", None)

    if sides is None:
        script = f"""
#~ Adding Boundary: PML
# PML Layers
boundary = comp_meep.Boundary(type='PML',
                            thickness= {thickness})

# Creating an MEEP object for the boundary using the assemble function
pml = boundary.assemble()
boundary_layers.append(pml)
"""
        return script

    script = "\n#~ Adding Boundary: PML on each side\n"
    for side in sides:
        direction, _, end = side.lower().partition("_")
        if direction not in ("x", "y", "z") or end not in ("", "low", "high"):
            raise ValueError(f"PML side {side} not recognized: use 'x_low', 'x_high', "
                             "'y_low', 'y_high', 'x' or 'y'")
        side_param = f", side=mp.{end.capitalize()}" if end else ""
        script += (f"boundary_layers.append(mp.PML(thickness={thickness}, "
                   f"direction=mp.{direction.upper()}{side_param}))\n")
    return script

def boundary_script(data):
    script = "\nboundary_layers = []\n"
    for boundary in data["boundary_layers"]:
        boundary_data = data["boundary_layers"][boundary]
        if boundary_data["type"] == "PML":
            script += PML_boundary(boundary_data)
        else:
            raise ValueError(f"Boundary type {boundary_data['type']} not recognized.")
    return script
#! <-- THE END: BOUNDARY SCRIPT SEGMENTS --> #

#^====================================================================================================================^#
#! <-- SYMMETRY SCRIPT SEGMENTS --> #
def symmetry_script(data):
    """
    The list of MEEP symmetries, e.g.
        "symmetries": {"symmetry": {"type": "Mirror", "direction": "Y", "phase": 1}}
    "type" is "Mirror", "Rotate2", "Rotate4" or "none" (no symmetry)
    """
    script = "\n#~ Adding Symmetries\nsymmetries = []\n"

    for symmetry_name, symmetry_data in data.get("symmetries", {}).items():
        symmetry_type = symmetry_data.get("type", "none")
        if symmetry_type.lower() == "none":
            continue
        if symmetry_type not in ("Mirror", "Rotate2", "Rotate4"):
            raise ValueError(f"Symmetry type {symmetry_type} not recognized.")
        if "direction" not in symmetry_data:
            raise ValueError(f"Symmetry {symmetry_name} must define its 'direction' ('X', 'Y' or 'Z')")

        direction = symmetry_data["direction"].upper()
        # A mirror plane through the origin only holds if the sources sit on it
        if symmetry_type == "Mirror":
            for source_name, source_data in data.get("sources", {}).items():
                if source_data.get(f"center_{direction.lower()}", 0) != 0:
                    print(f"WARNING: source {source_name} is off the mirror plane {direction} = 0, "
                          f"so the {symmetry_name} symmetry does not hold")

        script += (f"symmetries.append(mp.{symmetry_type}(mp.{direction}, "
                   f"phase={symmetry_data.get('phase', 1)}))\n")

    return script
#! <-- THE END: SYMMETRY SCRIPT SEGMENTS --> #

#^====================================================================================================================^#
def init_epsilon_map(data):
    script= f"""
#~ Adding Epsilon Map
size_x, size_y, size_z = mpsat_sim.cell_size[0], mpsat_sim.cell_size[1], mpsat_sim.cell_size[2]
res = int(mpsat_sim.resolution)  # Ensure resolution is an integer
# Create the epsilon map: total size of the simulation cell in all the axis multiplied by the resolution + 1
epsilon_map = np.ones((int((size_x)*res+1), 
                       int((size_y)*res+1)), dtype = mpsat_sim.real_dtype)
"""
    return script

#^====================================================================================================================^#
def convert_string_to_object_reference(value, param_name=None):
    """
    Converts string references to Python object references for the script.
    
    If value is a string that starts with specific prefixes like 'mp.',
    it will be returned without quotes so it can be evaluated as a Python expression.
    
    Args:
        value: The value to potentially convert
        param_name: Optional parameter name to exclude from conversion
        
    Returns:
        The value appropriately formatted for Python code
    """
    # Parameters that should always remain as strings
    string_params = ["name", "lens_type", "ARC_type", "step_ARC_angle", "step_ARC_rot_axis"]
    
    # If this is a parameter that should always be a string, return it quoted
    if param_name in string_params:
        return f"'{value}'"
    
    # Special handling for specific string values that should remain as strings
    string_values = ["default", "stepped_pyramid", "perpendicular_to_surface", "extended_aspheric", "aspheric"]
    
    if isinstance(value, str):
        # Check if the value is in our list of string values that should stay quoted
        if value in string_values:
            return f"'{value}'"
            
        # Prefixes that indicate the string should be treated as a Python object reference
        reference_prefixes = ["mp.", "np.", "None", "mpsat_sim."]
        
        # Check if the string starts with any of our reference prefixes
        for prefix in reference_prefixes:
            if value.startswith(prefix):
                return value  # Return without quotes so it's interpreted as a reference
                
        # If it doesn't match any prefix pattern, return with quotes
        return f"'{value}'"
    
    # For non-string values, return as-is
    return value


def add_lens(data):
    if "lenses" not in data:
        return ""  # Empty string since the lens is not a mandatory component
    else:
        script = ""
        for lens in data["lenses"]:
            lens_data = data["lenses"][lens]
            
            # Build a dictionary of parameters with appropriate conversions
            params = {}
            for key, value in lens_data.items():
                params[key] = convert_string_to_object_reference(value, key)
            
            # Check if this lens has multi-layer AR coating parameters
            has_multi_arc = ('AR_left_layers' in lens_data or 'AR_right_layers' in lens_data)
            
            # Check if this lens has stepped pyramid ARC coating parameters
            has_stepped_pyramid_arc = ('ARC_type' in lens_data and lens_data['ARC_type'] == 'stepped_pyramid')

            script += f"""
#~ Adding Lens: {lens}
{lens}_init = comp_eps.AsphericLens(name = {params.get('name', "'lens'")},
                                r1 = {params.get('r1', 'np.inf')},
                                r2 = {params.get('r2', 'np.inf')},
                                c1 = {params.get('c1', 0)},
                                c2 = {params.get('c2', 0)},
                                lens_type = {params.get('lens_type', "'extended_aspheric'")},
                                a1_coeffs = {params.get('a1_coeffs', 'None')},
                                a2_coeffs = {params.get('a2_coeffs', 'None')},
                                thick = {params['thick']},
                                diameter = {params['diameter']},
                                x = {params['x']},
                                n_refr= {params['n_refr']},
                                AR_left= {params.get('AR_left', 'None')},
                                AR_right= {params.get('AR_right', 'None')},
                                AR_material= {params.get('AR_material', 'None')},
                                AR_left_layers={params.get('AR_left_layers', 'None')},
                                AR_left_materials={params.get('AR_left_materials', 'None')},
                                AR_right_layers={params.get('AR_right_layers', 'None')},
                                AR_right_materials={params.get('AR_right_materials', 'None')},
                                ARC_type= {params.get('ARC_type', "'default'")},
                                step_ARC_nlayers= {params.get('step_ARC_nlayers', 'None')},
                                step_ARC_pitch= {params.get('step_ARC_pitch', 'None')},
                                step_ARC_kerf= {params.get('step_ARC_kerf', 'None')},
                                step_ARC_depth= {params.get('step_ARC_depth', 'None')},
                                step_ARC_width= {params.get('step_ARC_width', 'None')},
                                step_ARC_material= {params.get('step_ARC_material', 'None')},
                                step_ARC_angle= {params.get('step_ARC_angle', "'perpendicular_to_surface'")},
                                step_ARC_rot_axis= {params.get('step_ARC_rot_axis', "'z'")},
                                step_ARC_offset= {params.get('step_ARC_offset', [0,0])},
                                eps= epsilon_map,
                                mpsat_sim= mpsat_sim)
"""
            # Rest of the function remains the same...
            if has_multi_arc:
                script += f"""
# Using multi-layer AR coating assembly
{lens} = {lens}_init.assemble_with_multi_arc()
"""
            elif has_stepped_pyramid_arc:
                script += f"""
# Using stepped pyramid ARC coating assembly
{lens},  {lens}_stepped_pyramid_ARC_blocks = {lens}_init.assemble_with_stepped_pyramid_ARC()
# Adding the list of stepped pyramid ARC MEEP blocks to meepsat geometry
for stepped_pyramid_ARC_blocks in {lens}_stepped_pyramid_ARC_blocks:
    mpsat_sim.add_meep_geometry(stepped_pyramid_ARC_blocks)
"""
            else:
                script += f"""
# Using standard assembly with single-layer AR coating
{lens} = {lens}_init.assemble()
"""

            script += f"""
mpsat_sim.add_eps_geometry({lens})
# Plot the lens
# {lens}_init.plot_lenses()
"""
        
        # Check if we need to save the epsilon map
        save_epsilon = "epsilon_h5_file" in data.get("output", {})

        # Add code to save epsilon map only if needed
        if save_epsilon:
            print("Saving epsilon map to HDF5 file...")
            script += f"""
#~ Saving the Epsilon Map
# Saving the epsilon_map initiated after the final to a .h5 file, so that meep can read it
{lens}_init.write_h5file(filename= '{data["output"]["savepath"]["path"]}' + '{data["output"]["epsilon_h5_file"]["filename"]}_epsilon_map',
                        parallel= {data["simulation"]["parallel"]})
"""
        return script
    


def to_script_value(value, param_name=None):
    """
    Writes a JSON value as Python code, going into lists and dicts so that
    nested strings such as "mp.metal" also become object references
    """
    if isinstance(value, dict):
        items = ", ".join(f"'{key}': {to_script_value(val, key)}" for key, val in value.items())
        return "{" + items + "}"
    if isinstance(value, list):
        return "[" + ", ".join(str(to_script_value(val, param_name)) for val in value) + "]"
    return convert_string_to_object_reference(value, param_name)


def kwargs_script(params, separator=", "):
    """
    Writes a dict of JSON parameters as keyword arguments for the script
    """
    return separator.join(f"{key}={to_script_value(value, key)}" for key, value in params.items())


def vector3_script(value):
    """
    Writes a JSON vector as an mp.Vector3: [x, y, z], {"x": .., "y": .., "z": ..},
    or a string such as "mp.Vector3(0, 1, 0)", which is written as it is
    """
    if isinstance(value, dict):
        value = [value.get("x", 0), value.get("y", 0), value.get("z", 0)]
    if isinstance(value, (list, tuple)):
        return "mp.Vector3(" + ", ".join(str(val) for val in value) + ")"
    return value


def material_script(value):
    """
    Writes a JSON material for the script: a string such as "mp.metal" is a reference,
    a number is the refractive index and a dict holds the mp.Medium arguments,
    e.g. {"epsilon": 2.25, "D_conductivity": 0.01}
    """
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return f"mp.Medium(index={value})"
    if isinstance(value, dict):
        return f"mp.Medium({kwargs_script(value)})"
    return convert_string_to_object_reference(value, "material")


def source_frequency(data):
    """
    Frequency of the first source that gives one, at which the lossy materials
    that don't give their own are converted to conductivities
    """
    return next((source["frequency"] for source in data.get("sources", {}).values()
                 if isinstance(source, dict) and "frequency" in source), None)


def output_savepath(data):
    """
    The output directory of the JSON, for the components that save plots
    """
    return data.get("output", {}).get("savepath", {}).get("path", "./")


def aperture_absorber_layers(aperture_data, aperture_name, default_freq=None):
    """
    Writes the absorber layers of an aperture stop as a list of dicts for the script,
    which ApertureStop turns into ApertureAbsorberLayer objects

    Each JSON layer holds ApertureAbsorberLayer fields, e.g.
        {"side": "front", "thickness": 3, "epsilon_real": 4.0, "epsilon_imag": 0.4,
         "mu_real": 1.0, "mu_imag": 0.3}
    or {"side": "inner", "thickness": 1, "material": "mp.metal"}.
    Fields shared by all the layers (e.g. the ECCOSORB eps and mu) can be given
    once in "absorber_defaults"; a layer's own fields override them. A layer
    without "freq" gets the source frequency, so that epsilon_imag and mu_imag
    are converted to conductivities at the simulated frequency.
    """
    layers = aperture_data.get("absorber_layers", None)
    if not layers:
        return None

    defaults = aperture_data.get("absorber_defaults", {})
    layer_strings = []
    for i, layer in enumerate(layers):
        layer = {**defaults, **layer}
        if "side" not in layer or "thickness" not in layer:
            raise ValueError(f"Absorber layer {i} of aperture {aperture_name} must define "
                             "'side' ('front', 'back' or 'inner') and 'thickness'")

        if "freq" not in layer and default_freq is not None:
            if "material" not in layer:
                print(f"No frequency specified for absorber layer {i} of {aperture_name}, "
                      f"using the source frequency: {default_freq}")
            layer["freq"] = default_freq

        params = ", ".join(f"{key}={to_script_value(value, key)}" for key, value in layer.items())
        layer_strings.append(f"dict({params})")

    return "[" + ",\n                                                        ".join(layer_strings) + "]"


def aperture_stop(aperture_data, aperture_name, default_freq=None):
    # Handle the special mp.inf case for conductivity
    if aperture_data["conductivity"] == "mp.inf":
        conductivity = "mp.inf"
    else:
        conductivity = aperture_data["conductivity"]

    # Determine which position parameter(s) to use
    pos_x = aperture_data.get("pos_x", None)
    pos_y = aperture_data.get("pos_y", None)
    orientation = aperture_data.get("orientation", None)

    if pos_x is None and pos_y is None:
        raise ValueError(f"Aperture {aperture_name} must have either pos_x or pos_y defined")

    if pos_x is not None and pos_y is not None and orientation is None:
        raise ValueError(f"Aperture {aperture_name} defines both pos_x and pos_y, "
                         "so it must also define the orientation "
                         "('vertical' or 'horizontal')")

    # Build position parameter string (both positions can be given together)
    pos_params = []
    if pos_x is not None:
        pos_params.append(f"pos_x={pos_x}")
    if pos_y is not None:
        pos_params.append(f"pos_y={pos_y}")
    if orientation is not None:
        pos_params.append(f"orientation='{orientation}'")

    # Arrow tip shape and the bare-tip switch: only passed when given in the JSON,
    # so that the ApertureStop defaults apply otherwise
    for key in ("arrow_height", "arrow_base", "arrow_angle", "arrow_tip_offset", "absorber_on_tip"):
        if key in aperture_data:
            pos_params.append(f"{key}={aperture_data[key]}")
    pos_param = ",\n                                       ".join(pos_params)

    # Absorber layers on the faces of the stop
    absorber_layers = aperture_absorber_layers(aperture_data, aperture_name, default_freq)

    script = f"""
#~ Adding Aperture: {aperture_data["type"].capitalize()}
{aperture_name}_init = comp_meep.ApertureStop(mpsat_sim=mpsat_sim,
                                       type='{aperture_data["type"]}',
                                       diameter={aperture_data["diameter"]},
                                       thickness={aperture_data["thickness"]},
                                       {pos_param},
                                       n_refr={aperture_data.get("n_refr", 1.0)},
                                       conductivity={conductivity},
                                       material={aperture_data.get("material", "None")},
                                       y_centre_offset={aperture_data.get("y_centre_offset", [0, 0])},
                                       y_size_offset={aperture_data.get("y_size_offset", [0, 0])},
                                       print_vertices={aperture_data.get("print_vertices", False)},
                                       absorber_layers={absorber_layers},
                                       preserve_aperture={aperture_data.get("preserve_aperture", False)}
)
"""

    if aperture_data["type"] == "arrow":
        # The arrow stop also returns the two triangular tips
        script += f"""
{aperture_name}_up, {aperture_name}_down, {aperture_name}_tip_up, {aperture_name}_tip_down = {aperture_name}_init.assemble()
mpsat_sim.add_meep_geometry({aperture_name}_up)
mpsat_sim.add_meep_geometry({aperture_name}_down)
mpsat_sim.add_meep_geometry({aperture_name}_tip_up)
mpsat_sim.add_meep_geometry({aperture_name}_tip_down)
"""
    else:
        script += f"""
{aperture_name}_up, {aperture_name}_down = {aperture_name}_init.assemble()
mpsat_sim.add_meep_geometry({aperture_name}_up)
mpsat_sim.add_meep_geometry({aperture_name}_down)
"""

    # Added after the stop, so the absorbers take precedence where they overlap it
    if absorber_layers:
        script += f"""
# Absorber layers on the faces of {aperture_name}
{aperture_name}_absorbers = {aperture_name}_init.get_absorbers()
mpsat_sim.meep_geometry.extend({aperture_name}_absorbers)
"""

    return script

def add_aperture(data):
    script = ""
    # First, check if the apertures are present in the data
    if "apertures" not in data:
        return script # Empty string since the aperture is not an mandatory component
    else:
        # Frequency of the first source, for the lossy absorber layers that don't give their own
        default_freq = source_frequency(data)
        for aperture in data["apertures"]:
            aperture_data = data["apertures"][aperture]
            if aperture_data["type"] in ("square", "arrow"):
                script += aperture_stop(aperture_data, aperture_name=aperture, default_freq=default_freq)
            else:
                raise ValueError(f"Aperture type {aperture_data['type']} not recognized.")
        return script

#^====================================================================================================================^#
def add_slab(data):
    """
    Create code for adding slabs defined in the JSON file
    
    Parameters
    ----------
    data : dict
        JSON data containing slab definitions
        
    Returns
    -------
    script : str
        Python script code for creating slabs
    """
    script = ""
    
    # Check if slabs are present in the data
    if "slabs" not in data:
        return script  # Empty string since slabs are not mandatory
    else:
        print("Slab found in the JSON data")
        script += "\n#~ Adding Slabs\n"
        
        for slab_name, slab_data in data["slabs"].items():
            # Handle material specially - it might be a complex expression
            material_str = slab_data.get("material", "None")
            
            # Process center and size parameters - they might be strings like "mp.Vector3(...)"
            center_str = slab_data.get("center", "None")
            size_str = slab_data.get("size", "None")
            
            script += f"""
# Slab: {slab_name}
{slab_name}_init = comp_meep.Slab(
    mpsat_sim=mpsat_sim,
    name='{slab_data.get("name", "block")}',
    center={center_str},
    size={size_str},
    material={material_str},
    angle={slab_data.get("angle", 0)},
    rot_axis='{slab_data.get("rot_axis", "x")}'
)
{slab_name}_obj = {slab_name}_init.assemble()
mpsat_sim.add_meep_geometry({slab_name}_obj)
"""
        return script

#^====================================================================================================================^#
def add_absorbers(data):
    """
    Create code for adding absorbers defined in the JSON file
    
    Parameters
    ----------
    data : dict
        JSON data containing absorber definitions
        
    Returns
    -------
    script : str
        Python script code for creating absorbers
    """
    script = ""
    
    # Check if absorbers are present in the data
    if "absorbers" not in data:
        return script  # Empty string since absorbers are not mandatory
    else:
        print("Absorber found in the JSON data")
        script += "\n#~ Adding Absorbers\n"
        
        for absorber_name, absorber_data in data["absorbers"].items():
            # Build a dictionary of parameters with appropriate conversions
            params = {}
            for key, value in absorber_data.items():
                params[key] = convert_string_to_object_reference(value, key)

            # Deal with special absorber parameters if needed
            #!) freq
            if "freq" in absorber_data:
                print(f"Absorber frequency found: {absorber_data['freq']}")
                params["freq"] = convert_string_to_object_reference(absorber_data["freq"], "freq")
            else:
                print(f"No frequency specified for absorber, using the current source frequency: {data['sources']['source1']['frequency']}")
                params["freq"] = data["sources"]["source1"]["frequency"]

            
            script += f"""
# Absorber: {absorber_name}
{absorber_name}_init = comp_meep.PyramidalAbsorbers(
    mpsat_sim=mpsat_sim,
    base_width={params.get('base_width', 'None')},
    top_width={params.get('top_width', 0)},
    height={params.get('height', 'None')},
    layer_thickness={params.get('layer_thickness', 'None')},
    num_pyramids={params.get('num_pyramids', 20)},
    n_layers={params.get('n_layers', 10)},
    material={params.get('material', 'None')},
    epsilon_real={params.get('epsilon_real', 2.5)},
    epsilon_imag={params.get('epsilon_imag', 0)},
    freq={params.get('freq', '1/3')},
    x_coverage_start={params.get('x_coverage_start', 'None')},
    x_coverage_end={params.get('x_coverage_end', 'None')},
    y_coverage_start={params.get('y_coverage_start', 'None')},
    y_coverage_end={params.get('y_coverage_end', 'None')},
    coverage_percent={params.get('coverage_percent', 40)},
    edges={params.get('edges', '["top", "bottom"]')},
    x_left_offset={params.get('x_left_offset', 0)},
    x_right_offset={params.get('x_right_offset', 0)},
    y_top_offset={params.get('y_top_offset', 0)},
    y_bottom_offset={params.get('y_bottom_offset', 0)},
    add_substrate={params.get('add_substrate', 'True')},
    substrate_thickness={params.get('substrate_thickness', 'None')},
    substrate_material={params.get('substrate_material', 'None')},
    substrate_epsilon_real={params.get('substrate_epsilon_real', 'None')},
    substrate_epsilon_imag={params.get('substrate_epsilon_imag', 'None')},
    substrate_extends_beyond_pyramids={params.get('substrate_extends_beyond_pyramids', 'True')},
    substrate_extension={params.get('substrate_extension', 0.5)},
    add_pec_backing={params.get('add_pec_backing', 'False')},
    pec_thickness={params.get('pec_thickness', 'None')},
    pec_extends_beyond_substrate={params.get('pec_extends_beyond_substrate', 'True')},
    pec_extension={params.get('pec_extension', 0)},
    name={params.get('name', 'None')}
)

{absorber_name}_blocks = {absorber_name}_init.assemble()
# Add all absorber blocks to the MEEP geometry
mpsat_sim.meep_geometry.extend({absorber_name}_blocks)
"""
        return script

#^====================================================================================================================^#
def add_wall_absorbers(data):
    """
    Create code for the tapered absorbers (meep_geometry.Absorbers) defined in the
    JSON file, e.g. the absorbers lining the optics tube walls

    Each entry of "wall_absorbers" holds the Absorbers arguments, e.g.
        {"p": 6, "p_h_ratio": 1.5, "taper_type": "Linear", "orientation": "+y",
         "epsilon_r": 5.4, "epsilon_i": 0.8, "start_point": [-38, -200], "end_point": [813, -200],
         "add_substrate": true, "substrate_thickness": [2, 257],
         "substrate_material": [null, "mp.perfect_electric_conductor"]}
    The cell size, resolution and epsilon map come from the simulation. "freq"
    defaults to the source frequency and "savepath" to the output directory.
    """
    script = ""

    # Check if wall absorbers are present in the data
    if "wall_absorbers" not in data:
        return script  # Empty string since absorbers are not mandatory

    script += "\n#~ Adding Wall Absorbers\n"
    for absorber_name, absorber_data in data["wall_absorbers"].items():
        for key in ("p", "taper_type", "orientation"):
            if key not in absorber_data:
                raise ValueError(f"Wall absorber {absorber_name} must define '{key}'")
        if "h" not in absorber_data and "p_h_ratio" not in absorber_data:
            raise ValueError(f"Wall absorber {absorber_name} must define either 'h' or 'p_h_ratio'")

        params = {"center_x_mm": 0, "center_y_mm": 0, "z0": None, "z1": None, "angle_axis": "x",
                  "freq": source_frequency(data), "savepath": output_savepath(data)}
        params.update(absorber_data)
        params_str = kwargs_script({key: value for key, value in params.items() if key != "material"},
                                   ",\n" + " " * 4)
        if "material" in params:
            params_str += f",\n    material={material_script(params['material'])}"

        script += f"""
# Wall absorber: {absorber_name}
{absorber_name}_init = comp_meep.Absorbers(
    grid_size_sx=mpsat_sim.cell_size[0],
    grid_size_sy=mpsat_sim.cell_size[1],
    resolution=int(mpsat_sim.resolution),
    eps_array=epsilon_map,
    geometry_objects=[],
    {params_str}
)
mpsat_sim.meep_geometry.extend({absorber_name}_init.assemble())
"""
    return script

#^====================================================================================================================^#
def forebaffle_components(forebaffle_data):
    """
    Writes the components of a linear forebaffle: "absorber_layers" (AbsorberLayer
    fields, side 'base', 'height' or 'hypotenuse') and "flares" (FlareConfig fields)
    """
    components = []

    def config_script(class_name, config):
        params = {key: value for key, value in config.items() if key != "material"}
        params_str = kwargs_script(params)
        if "material" in config:
            params_str += f", material={material_script(config['material'])}"
        return f"comp_meep.{class_name}({params_str})"

    if forebaffle_data.get("absorber_layers"):
        layers = ", ".join(config_script("AbsorberLayer", layer) for layer in forebaffle_data["absorber_layers"])
        components.append(f"comp_meep.AbsorberComponent([{layers}])")
    if forebaffle_data.get("flares"):
        flares = ", ".join(config_script("FlareConfig", flare) for flare in forebaffle_data["flares"])
        components.append(f"comp_meep.FlairComponent([{flares}])")

    return components


def add_forebaffles(data):
    """
    Create code for the forebaffles defined in the JSON file

    Each entry of "forebaffles" holds the Forebaffle arguments, e.g.
        {"shape": "spline", "angle_degrees": 162, "x_vertex": -36, "y_vertex": 200,
         "hypotenuse": 800, "material": "mp.perfect_electric_conductor", "num_periods": 0,
         "amplitude": 1, "fb_thickness": 10, "add_absorber": true, "absorber_side": "below",
         "absorber_epsilon_real": 5.4, "absorber_epsilon_imag": 0.8, "absorber_thickness": 6}
    "freq" defaults to the source frequency and "name" to the entry name. A linear
    forebaffle can also hold "absorber_layers" and "flares" (see forebaffle_components).
    """
    script = ""

    # Check if forebaffles are present in the data
    if "forebaffles" not in data:
        return script  # Empty string since forebaffles are not mandatory

    script += "\n#~ Adding Forebaffles\n"
    for forebaffle_name, forebaffle_data in data["forebaffles"].items():
        params = {"name": forebaffle_name, "freq": source_frequency(data)}
        params.update({key: value for key, value in forebaffle_data.items()
                       if key not in ("material", "absorber_layers", "flares")})
        if params["freq"] is None:
            raise ValueError(f"Forebaffle {forebaffle_name} must define 'freq' when no source gives a frequency")

        params_str = kwargs_script(params, ",\n" + " " * 4)
        if "material" in forebaffle_data:
            params_str += f",\n    material={material_script(forebaffle_data['material'])}"

        components = forebaffle_components(forebaffle_data)
        if components:
            if forebaffle_data.get("shape", "linear") != "linear":
                raise ValueError(f"Forebaffle {forebaffle_name}: absorber_layers and flares are only "
                                 "built for a linear forebaffle (a spline one has its own absorber_* options)")
            params_str += f",\n    components=[{', '.join(components)}]"

        script += f"""
# Forebaffle: {forebaffle_name}
{forebaffle_name}_init = comp_meep.Forebaffle(
    mpsat_sim=mpsat_sim,
    epsilon_map=epsilon_map,
    {params_str}
)
mpsat_sim.meep_geometry.extend({forebaffle_name}_init.assemble())
"""
    return script

#^====================================================================================================================^#
def add_detectors(data):
    """
    Create code for the detector/image-plane slabs defined in the JSON file, e.g.
        "detectors": {"image_plane": {"type": "meep_block", "diameter": 300, "pos_x": 760,
                                      "thickness": 2, "n_refr": 1.0, "conductivity": "np.inf"}}
    An infinite conductivity (the default) makes the detector a perfect conductor.
    """
    script = ""

    # Check if detectors are present in the data
    if "detectors" not in data:
        return script  # Empty string since detectors are not mandatory

    script += "\n#~ Adding Detectors\n"
    for detector_name, detector_data in data["detectors"].items():
        for key in ("diameter", "pos_x", "thickness"):
            if key not in detector_data:
                raise ValueError(f"Detector {detector_name} must define '{key}'")

        # Detector tests the conductivity against np.inf, which mp.inf (1e20) is not
        conductivity = detector_data.get("conductivity", "np.inf")
        if conductivity == "mp.inf":
            conductivity = "np.inf"

        script += f"""
# Detector: {detector_name}
{detector_name}_init = comp_meep.Detector(type='{detector_data.get("type", "meep_block")}',
                                       diameter={detector_data["diameter"]},
                                       pos_x={detector_data["pos_x"]},
                                       thickness={detector_data["thickness"]},
                                       n_refr={detector_data.get("n_refr", 1.0)},
                                       conductivity={conductivity})
mpsat_sim.add_meep_geometry({detector_name}_init.assemble())
"""
    return script

#^====================================================================================================================^#
def add_filters(data):
    """
    Create code for the filters defined in the JSON file, e.g.
        "filters": {"IR_blocker": {"type": "block", "center": [150, 0, 0], "size": [3, 300, 0],
                                   "material": 1.5, "angle": 0, "rot_axis": "z"}}
    "center" and "size" are [x, y, z] lists or "mp.Vector3(...)" strings; "material"
    is a reference ("mp.metal"), a refractive index or a dict of mp.Medium arguments.
    Other entries are passed on to meep_block.
    """
    script = ""

    # Check if filters are present in the data
    if "filters" not in data:
        return script  # Empty string since filters are not mandatory

    script += "\n#~ Adding Filters\n"
    for filter_name, filter_data in data["filters"].items():
        if "material" not in filter_data:
            raise ValueError(f"Filter {filter_name} must define its 'material'")

        handled = ("type", "name", "center", "size", "material", "angle", "rot_axis")
        extra = {key: value for key, value in filter_data.items() if key not in handled}
        extra_str = (",\n    " + kwargs_script(extra, ",\n    ")) if extra else ""

        script += f"""
# Filter: {filter_name}
{filter_name}_init = comp_meep.Filter(
    mpsat_sim=mpsat_sim,
    name='{filter_data.get("type", filter_data.get("name", "block"))}',
    center={vector3_script(filter_data.get("center", "None"))},
    size={vector3_script(filter_data.get("size", "None"))},
    material={material_script(filter_data["material"])},
    angle={filter_data.get("angle", 0)},
    rot_axis='{filter_data.get("rot_axis", "x")}'{extra_str}
)
{filter_name}_obj = {filter_name}_init.assemble()
mpsat_sim.add_meep_geometry({filter_name}_obj)
"""
    return script

#^====================================================================================================================^#
def add_feedhorns(data):
    """
    Create code for the feedhorn arrays (permittivity_components.FeedHorn) defined in
    the JSON file, which are written into epsilon_map, e.g.
        "feedhorns": {"array1": {"focal_plane_x": 773, "focal_plane_y_range": [-200, 200],
                                 "feedhorn_y_range": [0, 0], "txt_file": "horn_profile.txt",
                                 "t_m": 0.2, "t_f": 6.2, "w2": 6, "thick_x": 30,
                                 "central_metal_thickness": 2, "eps_pec": -1e20}}
    "savepath" defaults to the output directory.

    The epsilon map is saved to the HDF5 file by add_lens (or write_epsilon_map
    without lenses), so this code has to run before that.
    """
    script = ""

    # Check if feedhorns are present in the data
    if "feedhorns" not in data:
        return script  # Empty string since feedhorns are not mandatory

    script += "\n#~ Adding Feedhorn Arrays (written into the epsilon map)\n"
    for feedhorn_name, feedhorn_data in data["feedhorns"].items():
        for key in ("focal_plane_x", "focal_plane_y_range", "feedhorn_y_range",
                    "txt_file", "t_m", "t_f", "w2", "thick_x"):
            if key not in feedhorn_data:
                raise ValueError(f"Feedhorn array {feedhorn_name} must define '{key}'")

        params = {"savepath": output_savepath(data)}
        params.update(feedhorn_data)
        params_str = kwargs_script(params, ",\n" + " " * 4)

        script += f"""
# Feedhorn array: {feedhorn_name}
{feedhorn_name}_init = comp_eps.FeedHorn(
    mpsat_sim=mpsat_sim,
    eps=epsilon_map,
    {params_str}
)
# Writes the feedhorns into epsilon_map in place
{feedhorn_name}_init.assemble()
"""
    return script

#^====================================================================================================================^#
def add_mirrors(data):
    """
    Create code for the extended-polynomial mirrors (permittivity_components.Mirror)
    defined in the JSON file

        "mirrors": {
            "surfaces": [...],             # ZEMAX surface table, for "surface_index"
            "assemble_as": "geometry",     # MEEP prisms ("geometry") or "epsilon_map"
            "material": "mp.perfect_electric_conductor",   # "geometry" only
            "n_segments": 200,             # "geometry" only
            "placement": {"extra_zemax_points": [[y, z], ...], "margin_mm": 0},
            "defaults": {"R_N": 400, "N": 7, "thickness_mm": 15, "eps_value": 12},
            "elements": {
                "M1": {"surface_index": 4, "half_width_mm": 419, "excel_values": [...],
                       "ap_y": -9, "optical_side_zemax_mm": "M2"},
                ...
            }
        }

    Each element is built with Mirror.from_prescription when it gives "surface_index",
    Mirror.from_excel when it gives "excel_values" (with "centre_zemax_mm"), or
    Mirror with "A_ij"; "defaults" holds the arguments shared by all of them.
    "optical_side_zemax_mm" is the (y, z) the mirror reflects towards, the name of
    another element (its centre) or a surface index of "surfaces". An element can
    have its own "material". With "placement", the shared ZEMAX -> MEEP offset is
    computed by Mirror.zemax_placement and applied to every mirror; the cell size
    is not changed, so a warning is printed if the mirrors don't fit.

    With "assemble_as": "epsilon_map" the mirrors are written into epsilon_map, which
    is saved to the HDF5 file by add_lens (or write_epsilon_map without lenses), so
    this code has to run before that.
    """
    script = ""

    # Check if mirrors are present in the data
    if "mirrors" not in data:
        return script  # Empty string since mirrors are not mandatory

    mirrors_data = data["mirrors"]
    elements = mirrors_data.get("elements", {})
    if not elements:
        raise ValueError("'mirrors' must hold its mirrors in 'elements'")
    assemble_as = mirrors_data.get("assemble_as", "geometry")
    if assemble_as not in ("geometry", "epsilon_map"):
        raise ValueError(f"Mirror assemble_as {assemble_as} not recognized: use 'geometry' or 'epsilon_map'")

    script += "\n#~ Adding Mirrors\n"
    if "surfaces" in mirrors_data:
        script += f"mirror_surfaces = {to_script_value(mirrors_data['surfaces'])}\n"
        script += "mirror_centres = dict(comp_eps.Mirror.compute_surface_centers(mirror_surfaces))\n"

    defaults = mirrors_data.get("defaults", {})
    optical_sides = {}
    # Mirror.name of each element, and the material of those that give their own
    mirror_names = {}
    materials = {}
    for mirror_name, element in elements.items():
        params = {**defaults, **element}
        params.setdefault("name", mirror_name)
        mirror_names[mirror_name] = params["name"]
        optical_sides[mirror_name] = params.pop("optical_side_zemax_mm", None)
        if "material" in params:
            materials[params["name"]] = material_script(params.pop("material"))

        params_str = kwargs_script(params, ",\n" + " " * 4)
        if "surface_index" in params:
            if "surfaces" not in mirrors_data:
                raise ValueError(f"Mirror {mirror_name} gives a surface_index, so 'mirrors' must hold the 'surfaces' table")
            constructor = "comp_eps.Mirror.from_prescription"
            params_str = "surfaces=mirror_surfaces,\n    " + params_str
        elif "excel_values" in params:
            constructor = "comp_eps.Mirror.from_excel"
        elif "A_ij" in params:
            constructor = "comp_eps.Mirror"
        else:
            raise ValueError(f"Mirror {mirror_name} must define 'surface_index', 'excel_values' or 'A_ij'")

        script += f"""
# Mirror: {mirror_name}
{mirror_name}_init = {constructor}(
    eps=epsilon_map,
    mpsat_sim=mpsat_sim,
    {params_str}
)
"""

    # What each mirror reflects towards, which sets the side its body goes on
    script += "\n"
    for mirror_name, optical_side in optical_sides.items():
        if optical_side is None:
            continue
        if isinstance(optical_side, str):
            if optical_side not in elements:
                raise ValueError(f"Mirror {mirror_name}: optical_side_zemax_mm {optical_side} is not a mirror name")
            side = f"{optical_side}_init.centre_zemax_mm"
        elif isinstance(optical_side, int):
            if "surfaces" not in mirrors_data:
                raise ValueError(f"Mirror {mirror_name} gives a surface index as optical_side_zemax_mm, "
                                 "so 'mirrors' must hold the 'surfaces' table")
            side = f"(mirror_centres[{optical_side}][1], mirror_centres[{optical_side}][2])"
        else:
            side = f"({optical_side[0]}, {optical_side[1]})"
        script += f"{mirror_name}_init.optical_side_zemax_mm = {side}\n"

    script += f"mirror_list = [{', '.join(f'{name}_init' for name in elements)}]\n"

    if "placement" in mirrors_data:
        placement = mirrors_data["placement"]
        script += f"""
# Shared ZEMAX -> MEEP offset of the mirrors
mirror_offset, mirror_need_x, mirror_need_y = comp_eps.Mirror.zemax_placement(
    mirror_list,
    extra_zemax_points={to_script_value(placement.get("extra_zemax_points", []))},
    margin_mm={placement.get("margin_mm", 0.0)})
for mirror in mirror_list:
    mirror.origin_offset_mm = (float(mirror_offset[0]), float(mirror_offset[1]))
if mirror_need_x > mirror_list[0].size_x or mirror_need_y > mirror_list[0].size_y:
    print(f"WARNING: the mirrors need a {{mirror_need_x:.1f}} x {{mirror_need_y:.1f}} mm cell, "
          f"but the simulated one is {{mirror_list[0].size_x:.1f}} x {{mirror_list[0].size_y:.1f}} mm; "
          "increase cell_size in the JSON")
"""

    if assemble_as == "geometry":
        if materials:
            # One material per mirror, the section one for those that don't give their own
            section_material = material_script(mirrors_data.get("material", "None"))
            material = "{" + ", ".join(f"'{name}': {materials.get(name, section_material)}"
                                       for name in mirror_names.values()) + "}"
        else:
            material = material_script(mirrors_data.get("material", "None"))
        script += f"""
# The mirrors as MEEP prisms
mirror_geometry = comp_eps.Mirror.assemble_prisms(mirror_list,
                                                  material={material},
                                                  n_segments={mirrors_data.get("n_segments", 200)},
                                                  verbose=True)
mpsat_sim.meep_geometry.extend(mirror_geometry)
"""
    else:
        script += """
# Writes the mirrors into epsilon_map in place
comp_eps.Mirror.assemble(mirror_list, verbose=True)
"""
    return script

#^====================================================================================================================^#
def write_epsilon_map(data):
    """
    Create code that saves epsilon_map to the HDF5 file MEEP reads

    add_lens already does this after the last lens. Use this one when the epsilon
    map is edited without any lens (feedhorns, or mirrors written into the map).
    """
    if "epsilon_h5_file" not in data.get("output", {}):
        return ""

    script = f"""
#~ Saving the Epsilon Map
# write_h5file only reads permittivity_map from the object it is called on, so it is used
# here for a map that no lens holds
from types import SimpleNamespace
comp_eps.AsphericLens.write_h5file(SimpleNamespace(permittivity_map=epsilon_map),
                                   filename= '{data["output"]["savepath"]["path"]}' + '{data["output"]["epsilon_h5_file"]["filename"]}_epsilon_map',
                                   parallel= {data["simulation"]["parallel"]})
"""
    return script

#^====================================================================================================================^#
# The JSON components in the order they are built. The feedhorns and the mirrors in
# "epsilon_map" mode are written into the epsilon map, which add_lens saves, so they
# come before the lenses. Among the MEEP geometry a later object takes precedence
# where it overlaps an earlier one, so the aperture stops sit on the wall absorbers
# and the forebaffles on both.
COMPONENT_BUILDERS = [
    ("feedhorns", add_feedhorns),
    ("mirrors", add_mirrors),
    ("wall_absorbers", add_wall_absorbers),
    ("slabs", add_slab),
    ("apertures", add_aperture),
    ("filters", add_filters),
    ("detectors", add_detectors),
    ("forebaffles", add_forebaffles),
    ("lenses", add_lens),
]

#^====================================================================================================================^#

# Add after the existing at_end_command_for_volume_monitors function

def at_every_command_for_flux_monitors(at_every_list, timestep):
    """
    Generate at_every commands specifically for flux monitors
    """
    if not at_every_list:  # Handle empty list case
        return ""
    
    command = ""
    for data in at_every_list:
        # Flux monitor functions are directly in mp_sim_run
        command += f"mp.at_every({timestep}, mp_sim_run.{data}),\n"

    return command.rstrip(",\n")  # Remove trailing comma and newline if present      

def at_end_command_for_flux_monitors(at_end_list):
    """
    Generate at_end commands specifically for flux monitors
    """
    if not at_end_list:  # Handle empty list case
        return ""
    
    command = ""
    for data in at_end_list:
        if data == "calculate_transmission_reflection":
            # Use mp_sim_run prefix for flux calculations
            command += f"mp.at_end(mp_sim_run.{data}),\n"
        else:
            command += f"mp.at_end(mp_sim_run.{data}),\n"

    return command.rstrip(",\n")  # Remove trailing comma and newline if present


def add_flux_monitor(data):
    """
    Create code for adding flux monitors defined in the JSON file
    """
    script = ""
    
    # Check if flux monitors are present in the data
    if "flux_monitor" not in data:
        # Set flux_monitor_list to None
        script += "\nflux_monitor_list=None\n"
        return script  # Empty string since flux monitors are not mandatory
    else:
        script += "\n#~ Adding Flux Monitors\nflux_monitor_list = []\n"
        
        for monitor_name, monitor_data in data["flux_monitor"].items():
            # Special handling for Python objects in the JSON
            center = monitor_data["center"]
            size = monitor_data["size"]
            direction = monitor_data.get("direction", "mp.X")
            monitor_type = monitor_data.get("monitor_type", "transmission")
            freq_min = monitor_data.get("freq_min", "None")
            freq_max = monitor_data.get("freq_max", "None")
            nfreq = monitor_data.get("nfreq", 100)
            
            # Add support for using saved flux files
            use_flux_file = monitor_data.get("use_flux_file", "None")
            
            script += f"""
# Flux Monitor: {monitor_name}
{monitor_name}_init = comp_meep.FluxMonitor(
    mpsat_sim=mpsat_sim,
    name='{monitor_name}',
    center={center},
    size={size},
    direction={direction},
    freq_min={freq_min},
    freq_max={freq_max},
    nfreq={nfreq},
    monitor_type='{monitor_type}',
    use_flux_file={use_flux_file}
)
{monitor_name}_region = {monitor_name}_init.assemble()
flux_monitor_list.append({{"{monitor_name}": [{monitor_data}, {monitor_name}_region]}})
"""
        
        # Add code to initialize the flux monitor registry
        script += "\n# Initialize the flux monitor registry\n"
        script += "mp_sim_run.set_flux_monitor_registry(flux_monitor_list, "
        
        # Add directory if specified
        if "output" in data and "flux_monitor_data_dir" in data["output"]:
            script += f"monitor_data_save_dir='{data['output']['flux_monitor_data_dir']}', "
        else:
            script += "monitor_data_save_dir=savepath + 'flux_monitor_data/', "
        
        # Add save frequency if specified
        if "output" in data and "flux_monitor_save_freq" in data["output"]:
            script += f"monitor_data_save_freq={data['output']['flux_monitor_save_freq']}, "
        else:
            script += "monitor_data_save_freq=5, "
        
        # Add reference simulation flag
        script += f"is_reference={data.get('run_reference_sim', False)}"
        
        script += ")\n"
        
        return script

#^====================================================================================================================^#
def init_meep_sim_object(data):
    """
    Create the MEEP simulation object with optional epsilon map handling
    """
    # Base parameters that are always included
    params = [
        "cell_size=mpsat_sim.cell",
        "boundary_layers=[pml]",
        "geometry=mpsat_sim.meep_geometry",
        "sources=source_list"
    ]
    
    # Add epsilon_h5_file parameter only if specified and if lenses are present
    # (since epsilon maps are primarily used with lenses)
    has_lenses = "lenses" in data
    has_epsilon_file = "epsilon_h5_file" in data.get("output", {})
    
    if has_lenses and has_epsilon_file:
        params.append(f"epsilon_h5_file=savepath + '{data['output']['epsilon_h5_file']['filename']}_epsilon_map'")
    
    # Join parameters with commas and proper indentation
    params_str = ",\n                            ".join(params)
    
    # Build the script
    script = f"""
# Initialize the meep simulation object with the different parameters
sim = mpsat_sim.meep_sim_obj({params_str})
"""
    return script

#^====================================================================================================================^#
def add_required_data(data):
    script = f"""
#~ Runtime of the simulation
run_time = {data["output"]["runtime"]}
#~ Requested data by the user
data_required= {data["output"]["data_required"]}
"""
    return script
#^====================================================================================================================^#
def add_monitor(data):
    """
    Create code for adding monitors defined in the JSON file
    
    Parameters
    ----------
    data : dict
        JSON data containing monitor definitions
        
    Returns
    -------
    script : str
        Python script code for creating monitors
    """
    script = ""
    
    # Check if monitors are present in the data
    if "monitor" not in data:
        return script  # Empty string since monitors are not mandatory
    else:
        script += "\n#~ Adding Monitors\nmonitor_list = []\n"
        
        for monitor_name, monitor_data in data["monitor"].items():
            # Special handling for Python objects in the JSON
            center = monitor_data["center"]
            size = monitor_data["size"]
            script += f"""
# Monitor: {monitor_name}
{monitor_name}_init = comp_meep.VolumeMonitor(mpsat_sim=mpsat_sim,
                                            name='{monitor_name}',
                                            center={center},
                                            size={size},
                                            data_required={monitor_data["data_required"]})
{monitor_name}_obj = {monitor_name}_init.assemble()
monitor_list.append({{"{monitor_name}": [{monitor_data}, {monitor_name}_obj]}})
"""
        # script += "\nmp_sim_run.set_volume_monitor_registry(monitor_list)\n" # set the volume monitor registry
        
        return script
    

#! THIS IS WORKING MATE BUT WITHOUT THE MEMORY EFFICIENCY CODE FOR ANIMATION:
def run_meepsat_sim(data):
    # Extract the plotting params if they exist
    plotting_params = "None"
    if "plotting_params" in data["output"]["animation_options"]:
        plotting_params = json.dumps(data["output"]["animation_options"]["plotting_params"])
    
    script = f"""
mp_sim_run.set_animation_params(anim_params= {{'image_every': {data["output"]["animation_options"]["image_every"]}, 
                                             'Nfps': {data["output"]["animation_options"]["Nfps"]}, 
                                             'anim_file_name': savepath + '{data["output"]["animation_options"]["movie_name"]}',
                                             'plotting_params': {plotting_params}}})

mpsat_sim.run_simulation(sim = sim,
                         mpsat_sim = mpsat_sim,
                         runtime = run_time,
                         get_mp4 = {str(data["output"]["animation_options"]["get_mp4"]).capitalize()},
                         image_every = {data["output"]["animation_options"]["image_every"]},
                         Nfps = {data["output"]["animation_options"]["Nfps"]},
                         savepath = savepath,
                         movie_name = '{data["output"]["animation_options"]["movie_name"]}',
                         requested_data = data_required,
                         monitor_list = monitor_list if 'monitor_list' in locals() else None,
                         save_data = False)
"""
    return script


#^====================================================================================================================^#
### JSON TO SCRIPT CONVERSION ###
def json_to_pyscript(json_file, output_dir=None, output_name=None):
    data = read_json(json_file)
    
    # Collect all script components
    script_components = [
        # Your existing components
        temporary_meepasat_path_set(data),
        import_modules(),
        import_meepsat_modules(),
        init_meepsat_sim(data),
        source_script(data),
        boundary_script(data),
        symmetry_script(data),
        init_epsilon_map(data),
        *[add_component(data) for _, add_component in COMPONENT_BUILDERS],
        # add_lens saves the epsilon map, which otherwise has to be saved here
        write_epsilon_map(data) if "lenses" not in data else "",
        add_monitor(data),
        add_flux_monitor(data),
        #! Add other components here..
        init_meep_sim_object(data),
        add_required_data(data),
        run_meepsat_sim(data)
    ]
    
    # Join with newlines between components
    script = "\n\n".join(script_components)
    
    # Determine output filename and path
    if output_name:
        script_name = output_name
    else:
        json_basename = os.path.basename(json_file)
        script_name = os.path.splitext(json_basename)[0] + "_meepsat_script.py"
    
    # Determine the output path
    if output_dir:
        output_path = os.path.join(output_dir, script_name)
    else:
        output_path = os.path.join("example_scripts", script_name)
    
    write_script(script, output_path)
    return output_path

def read_json(json_file):
    # Read the JSON file
    with open(json_file, "r") as f:
        data = json.load(f)
    return data

def write_script(script, script_file):
    # Write the script to a Python file
    with open(script_file, "w") as f:
        f.write(script)
    print(f"Script written to {script_file}")
    # return script_file

#! <-- THE END: JSON TO SCRIPT CONVERSION --> #

#! <-- THE FOLLOWING FUNCTIONS ARE USED IN simulation_2D.py to generate the sim.run script --> #
#^====================================================================================================================^#
def at_every_command(at_every_list, timestep):
    if not at_every_list:  # Handle empty list case
        return ""
    
    command = ""
    for data in at_every_list:
        command += f"mp.at_every({timestep}, mp_sim_run.{data}),\n"
    
    return command.rstrip(",\n")  # Remove trailing comma and newline if present



def at_end_command(at_end_list):
    if not at_end_list:  # Handle empty list case
        return ""
        
    command = ""
    for data in at_end_list:
        if data == "save_animation":
            command += f"mp.at_end(mp_sim_run.save_animation),\n"
        elif data == "calculate_transmission_reflection":
            command += f"mp.at_end(mp_sim_run.calculate_transmission_reflection),\n"  # Use mp_sim_run prefix
        else:
            command += f"mp.at_end(mp.{data}),\n"
    
    return command.rstrip(",\n")  # Remove trailing comma and newline if present

def data_required_script_inside_sim_run(data_required):
    script = ""
    script += at_every_command(data_required["at_every"], "image_every")
    # Add a comma and newline 
    script += ",\n"
    script += at_end_command(data_required["at_end"])
    return script
#^====================================================================================================================^#
#! <-- FUNCTIONS TO HANDLE MONITORS INSIDE sim.run --> #

def at_every_command_for_volume_monitors(at_every_list, timestep, monitor_type):
    if not at_every_list:  # Handle empty list case
        return ""
    
    command = ""
    for data in at_every_list:
        if monitor_type == "mp.Volume":
            command += f"mp.at_every({timestep}, mp_sim_run.{data}_VolumeMonitor),\n"
        # Add other monitor types here
        else:
            raise ValueError(f"Monitor type {monitor_type} not recognized.")  

    return command.rstrip(",\n")  # Remove trailing comma and newline if present      

def at_end_command_for_volume_monitors(at_end_list, monitor_type=None):
    if not at_end_list:  # Handle empty list case
        return ""
    
    command = ""
    for data in at_end_list:
        if data.startswith("output_"):
            command += f"mp.at_end({data}),\n"
        elif monitor_type == "mp.Volume":
            command += f"mp.at_end(mp_sim_run.{data}_VolumeMonitor),\n"
        # Add other monitor types here
        else:
            raise ValueError(f"Monitor type {monitor_type} not recognized.")
        
    return command.rstrip(",\n")  # Remove trailing comma and newline if present

def data_required_inside_indiv_monitors(monitor_list):
    script = ""
    for monitor_entry in monitor_list:
        # Each entry is a dict with one key (monitor name)
        monitor_name = list(monitor_entry.keys())[0]
        monitor_content = monitor_entry[monitor_name]
        # [{'monitor_1': [{'details': 'here'}, <monitor_object>]}] 
        # monitor_content is a list where:
        # monitor_content[0] is the monitor data dictionary 
        # monitor_content[1] is the monitor object
        monitor_data = monitor_content[0]
        

        # Print the entire monitor data structure for debugging
        print(f"Monitor data structure for {monitor_name}:", monitor_data)
        
        # Check if "at_every" exists before accessing
        at_every_list = monitor_data.get("data_required", {}).get("at_every", [])
        at_every_timestep = monitor_data.get("data_required", {}).get("at_every_timestep", 0)
        print(f"at_every list for {monitor_name}:", at_every_list)

        # Add at_every commands if they exist
        if at_every_list:
            script += at_every_command_for_volume_monitors(
                at_every_list, 
                at_every_timestep,
                monitor_data.get("type", None)  # Default to None if not specified
            )
            # Add a comma and newline
            script += ",\n"
        
        # Check if "at_end" exists before accessing
        at_end_list = monitor_data.get("data_required", {}).get("at_end", [])
        if at_end_list:
            script += at_end_command_for_volume_monitors(at_end_list, monitor_data.get("type", None))
    
    return script.rstrip(",\n")  # Remove trailing comma and newline if present

#! <-- THE END: FUNCTIONS TO HANDLE MONITORS INSIDE sim.run --> #
#^====================================================================================================================^#
#! EDIT THE BELOW FUNCTION TO ADD MORE SIM.RUN SCRIPT SEGMENTS IN FUTURE (IF REQUIRED):
def sims_data_requested(data_required, run_time, monitor_list):
    # Prepare the content for the run function separately
    run_args = data_required_script_inside_sim_run(data_required)
    if monitor_list is not None:
        run_args += ",\n"
        monitor_run_args = data_required_inside_indiv_monitors(monitor_list)
        run_args += monitor_run_args

    # If run_args is not empty, add a comma between commands
    if run_args:
        run_args += ",\n"
    run_args += run_time["command"]

    # Apply indentation to each line of run_args
    indented_args = "\n".join(f"    {line}" for line in run_args.split("\n"))
    
    main_script = f"""self.sim.run(
{indented_args}
)"""
    # print(main_script)
    return main_script
