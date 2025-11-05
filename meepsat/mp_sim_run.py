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



import time
import os
import h5py
import matplotlib.pyplot as plt
import meep as mp
import numpy as np
import os
from matplotlib.colors import LinearSegmentedColormap, LogNorm
from matplotlib import animation
from typing import Callable, Union, Any, Tuple, List, Optional
import meepsat.extra_functions as exf
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib
import matplotlib.colors as colors
import matplotlib.cbook as cbook
import subprocess

import warnings
warnings.filterwarnings("ignore")

# Step functions

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


# Fnction to calculate the flux array at a given timestep
def extract_flux_data_from_monitor(simulation):
    # Extract the total flux from the monitor (for normalization)
    flux_data = mp.get_fluxes(source_flux_monitor_ref)
    # Save the flux data to an npz compressed file
    np.savez_compressed(os.path.join(savepath, "source_total_flux_reference_{0}.npz".format(simulation.meep_time())),
                        flux=flux_data)    


"""
INitialising some plotting functions
"""
###! SOME global variables

def set_animation_params(anim_params: dict):
    """
    Set the global animation parameters for the simulation.
    This will be used to set the animation parameters for all animations.
    """
    global Nfps, image_every, anim_file_name, animation_plotting_params
    
    Nfps = anim_params['Nfps']
    image_every = anim_params['image_every']
    anim_file_name = anim_params['anim_file_name']
    
    # Store plotting parameters if provided
    if 'plotting_params' in anim_params:
        animation_plotting_params = anim_params['plotting_params']
    else:
        animation_plotting_params = None
    
    return


def set_plt_params(plt,
                x,
                y,
                base_factor=8):
    """
    plt: matplotlib.pyplot object
    x: array
    y: array
    base_factor: int
    """
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['savefig.dpi'] = 300
    if x or y:
        plt.rcParams['figure.figsize'] = set_figsize(x,y, base_factor)
    plt.rcParams['axes.labelsize'] = 'medium'
    plt.rcParams['axes.titlesize'] = 'large'
    plt.rcParams['xtick.labelsize'] = 'medium'
    plt.rcParams['ytick.labelsize'] = 'medium'
    plt.rcParams['legend.fontsize'] = 'medium'
    plt.rcParams['font.size'] = 14
    plt.style.use('dark_background')
    # padding between the figure and colorbar
    #plt.rcParams['axes.grid'] = True
    #plt.rcParams['grid.alpha'] = 0.5
    #plt.rcParams['grid.color'] = "#cccccc"
    # Plot everything on a dark background

    # Some custom colormaps
    global cmap_alpha
    cmap_alpha = LinearSegmentedColormap.from_list(
        'custom_alpha', [[1, 1, 1, 0], [1, 1, 1, 1]])
    global cmap_blue
    cmap_blue = LinearSegmentedColormap.from_list(
        'custom_blue', [[0, 0, 0], [0, 0.66, 1], [1, 1, 1]])
    
def set_figsize(x,y, base_factor= 8):
    factor = x/y
    if factor > 1:
        fig_x, fig_y = base_factor*factor, base_factor
    elif factor < 1:
        fig_x, fig_y = base_factor, base_factor/factor
    else:
        fig_x, fig_y = base_factor, base_factor

    return [fig_x, fig_y]

def label_plot(ax, title=None, xlabel=None, ylabel=None, elapsed=None):
    """
    Add a title and x/y labels to the plot.
    """
    if title is not None:
        ax.set_title(f'{title} at MEEP Timestep:{elapsed:0.1f}')
    if xlabel is not None:
        ax.set_xlabel('x (mm)'if xlabel is None else xlabel)
    if ylabel is not None:
        ax.set_ylabel('y (mm)'if ylabel is None else ylabel)

    return

def _cbar_label_(ax, cbar, label=None, pad=0.5):
    """
    Add a label to the colorbar.
    """
    if label is not None:
        cbar.set_label(label, pad=pad)

# Class for animating 
class Animate2DArray:
    
    def __init__(self, 
                 fps):
        
        self.fps = fps
        self._saved_frames = []  # Instance variable to store frames for this animation object

    def plot_xy(x, y, title=None, xlabel=None, ylabel=None, elapsed=None):
        """
        Plot x and y values.
        """
        fig, ax = plt.subplots()
        ax.plot(x, y)
        label_plot(ax, title, xlabel, ylabel, elapsed)
        plt.show()

    # Add all your plotting functions here:
    def plot_2d_array(self, 
                      array, 
                      eps_data=None,
                      title=None, 
                      xlabel=None, 
                      ylabel=None, 
                      extent=None,
                      x_ticks=None,
                      x_tick_labels=None,
                      y_ticks=None,
                      y_tick_labels=None,
                      elapsed=None,
                      cmap='viridis',
                      cbar_label_=None,
                      invert=False,
                      scale='linear',
                      vmin=None,
                      vmax=None):
        """
        Plot a 2D array.
        """
        fig, ax = plt.subplots()

        # Set the frame size
        self.frame_size_ = self.frame_size(fig)

        if scale == 'log':
            im = ax.imshow(array, 
                           cmap=cmap,
                           extent=extent,
                           vmin=vmin,
                           vmax=vmax)
        else:
            im = ax.imshow(array, 
                           cmap=cmap,
                           extent=extent,
                           vmin=vmin,
                           vmax=vmax)
        # ax.imshow(eps_data, cmap = cmap_alpha, origin = 'lower',
        #           alpha= 0.2)
        ax.imshow(eps_data > 1.0, cmap = cmap_alpha, origin = 'lower', 
          alpha= 0.2)
        
        label_plot(ax, title, xlabel, ylabel, elapsed)

        # Add a colorbar
        if cbar_label_ is not None:
            """
            divider = make_axes_locatable(ax)
            cax = divider.new_vertical(position='bottom', size = '5%', pad = 0.5)
            fig.add_axes(cax)
            cbar= fig.colorbar(im, cax = cax, orientation = 'horizontal')
            """
            ax_divider = make_axes_locatable(ax)
            # define size and padding of axes for colorbar
            cax = ax_divider.append_axes('top', size='5%', pad='20%')
            # you need to get a mappable obj (get_children)
            cbar= fig.colorbar(ax.get_children()[0], cax=cax, orientation='horizontal')
            # locate colorbar ticks (default is at the botttom)
            cax.xaxis.set_ticks_position('top')

            #! CHANGE THIS TO A FUNCTION
            #cbar_label_(ax= ax, cbar=cax, label=cbar_label)
            cbar.set_label(label=cbar_label_, labelpad= 0.9)

        # Set the x and y ticks
        if x_ticks is not None:
            ax.set_xticks(x_ticks)

        if x_tick_labels is not None:
            ax.set_xticklabels(x_tick_labels)

        if y_ticks is not None:
            ax.set_yticks(y_ticks)

        if y_tick_labels is not None:
            ax.set_yticklabels(y_tick_labels)

        # Don't show the plot, but capture the figure as a png.
        plt.close(fig)

        # Capture figure as a png, but store the png in memory
        # to avoid writing to disk.
        self.grab_frame(fig, 
                        ax, 
                        elapsed)

        return

    # @property --> This decorator here destroyed 2 hrs of my life
    # For cereating a frame for a particular array at a particular timestep
    def create_frame(self,
                     plot_func: str = 'plot_2d_array', 
                     kwargs: dict = {}):
        """
        Create a frame by calling a specified plotting function with given parameters.
        Parameters:
        plot_func (str): The name of the plotting function to call. Default is 'plot_2d_array'.
        kwargs (dict): A dictionary of keyword arguments to pass to the plotting function. 
                        Supported keys include:
                        - array: The data array to plot.
                        - eps_data: Additional data for plotting.
                        - title: The title of the plot.
                        - xlabel: The label for the x-axis. Default is 'x'.
                        - ylabel: The label for the y-axis. Default is 'y'.
                        - extent: The extent of the plot.
                        - x_ticks: The positions of the ticks on the x-axis.
                        - x_tick_labels: The labels for the ticks on the x-axis.
                        - y_ticks: The positions of the ticks on the y-axis.
                        - y_tick_labels: The labels for the ticks on the y-axis.
                        - elapsed: The elapsed time to display.
                        - cmap: The colormap to use. Default is 'viridis'.
                        - cbar_label_: The label for the color bar.
                        - invert: Whether to invert the plot. Default is False.
                        - scale: The scale of the plot. Default is 'linear'.
                        - vmin: The minimum value for the color scale.
                        - vmax: The maximum value for the color scale.
        """    

        # print('Content of Ez v2:', test)
        # print(f"Creating frame for {plot_func}...")
        # Call the plotting function
        getattr(self, plot_func)(array= kwargs.get('array', None),
                                 eps_data = kwargs.get('eps_data', None),
                                  title = kwargs.get('title', None),
                                  xlabel = kwargs.get('xlabel', 'x'),
                                  ylabel = kwargs.get('ylabel', 'y'),
                                  extent = kwargs.get('extent', None),
                                  x_ticks = kwargs.get('x_ticks', None),
                                  x_tick_labels = kwargs.get('x_tick_labels', None),
                                  y_ticks = kwargs.get('y_ticks', None),
                                  y_tick_labels = kwargs.get('y_tick_labels', None),
                                  elapsed = kwargs.get('elapsed', None),
                                  cmap = kwargs.get('cmap', 'viridis'),
                                  cbar_label_ = kwargs.get('cbar_label_', None),
                                  invert = kwargs.get('invert', False),
                                  scale = kwargs.get('scale', 'linear'),
                                  vmin = kwargs.get('vmin', None),
                                  vmax = kwargs.get('vmax', None))
        
    def frame_size(self, fig) -> Tuple[int, int]:
        """
        Calculate the size of a movie frame in pixels.
        Args:
            fig (matplotlib.figure.Figure): The matplotlib figure object.
        Returns:
            Tuple[int, int]: A tuple containing the width and height of the frame in pixels.
        """

        # A tuple ``(width, height)`` in pixels of a movie frame.
        # modified from matplotlib library
        w, h = fig.get_size_inches()
        return int(w * fig.dpi), int(h * fig.dpi)
    
    #! THIS IS THE OLD CODE
    # def grab_frame(self,
    #                fig= None,
    #                ax= None,
    #                elapsed= 0,
    #                frame_format= 'png'):
    #     """
    #     Captures the current frame of the given figure and saves it to memory.
    #         Parameters:
    #             fig (matplotlib.figure.Figure, optional): The figure object to capture. Defaults to None.
    #             ax (matplotlib.axes.Axes, optional): The axes object of the figure. Defaults to None.
    #             elapsed (int, optional): The elapsed time or timestep for the frame being captured. Defaults to 0.
    #         Returns:
    #             None
    #     """
    #     # Saves the figures frame to memory.
    #     # modified from matplotlib library
    #     from io import BytesIO

    #     bin_data = BytesIO()
    #     fig.savefig(bin_data, format=frame_format)
    #     print('PNG added to the bindata for timestep:', elapsed)
    #     # imgdata64 = base64.encodebytes(bin_data.getvalue()).decode('ascii')
    #     self._saved_frames.append(bin_data.getvalue())  # Use instance variable
    
    def grab_frame(self,
                fig=None,
                ax=None,
                elapsed=0,
                frame_format='png'):
        """
        Captures the current frame of the given figure and saves it to memory.
        """
        from io import BytesIO

        if fig is None:
            print("WARNING: No figure provided to grab_frame")
            return
        
        bin_data = BytesIO()
        try:
            fig.savefig(bin_data, format=frame_format, bbox_inches='tight', 
                    facecolor=fig.get_facecolor(), edgecolor='none')
            frame_data = bin_data.getvalue()
            
            if len(frame_data) == 0:
                print(f"WARNING: Empty frame data at timestep {elapsed}")
                return
            
            self._saved_frames.append(frame_data)
            print(f'PNG added to bindata for timestep: {elapsed} (size: {len(frame_data)} bytes)')
            
        except Exception as e:
            print(f"ERROR saving frame at timestep {elapsed}: {e}")
        finally:
            bin_data.close()

    # Below code is adopted from the MEEP source code:
    # https://github.com/NanoComp/meep/blob/dc27a1e0568dbb67dd9a36f8b00ce6be4ff6ea9b/python/visualization.py
    def to_gif_simple(self,
                    filename: str,
                    frame_format: str = 'png'):
        """
        Simplified memory-safe GIF creation using temporary files.
        """
        import tempfile
        import shutil
        
        if not self._saved_frames:
            print("ERROR: No frames to save!")
            return
        
        # Create temporary directory for frames
        with tempfile.TemporaryDirectory() as temp_dir:
            print(f"Writing frames to temporary directory for GIF: {temp_dir}")
            
            # Write frames to individual files
            frame_files = []
            for i, frame_data in enumerate(self._saved_frames):
                if len(frame_data) == 0:
                    continue
                    
                frame_file = os.path.join(temp_dir, f"frame_{i:06d}.png")
                try:
                    with open(frame_file, 'wb') as f:
                        f.write(frame_data)
                    frame_files.append(frame_file)
                except Exception as e:
                    print(f"Error writing frame {i}: {e}")
            
            if not frame_files:
                print("ERROR: No valid frames written!")
                return
            
            # Simple FFmpeg command for GIF with dimension fixing
            command = [
                "ffmpeg",
                "-framerate", str(self.fps),
                "-i", os.path.join(temp_dir, "frame_%06d.png"),
                "-vf", "pad=width=ceil(iw/2)*2:height=ceil(ih/2)*2",  # Pad to even dimensions
                "-r", str(self.fps),
                "-y",
                filename
            ]
            
            try:
                result = subprocess.run(command, capture_output=True, text=True, timeout=120)
                
                if result.returncode == 0:
                    print(f"GIF saved successfully to {filename}")
                    
                    # Verify output file exists and show size
                    if os.path.exists(filename):
                        file_size = os.path.getsize(filename)
                        print(f"Output file size: {file_size/1024/1024:.2f} MB")
                else:
                    print(f"FFmpeg error: {result.stderr}")
                    
            except Exception as e:
                print(f"Error running FFmpeg for GIF: {e}")
    
    def to_mp4(self, filename: str, frame_format: str = 'png', codec: str = 'h264'):
        """
        Memory-safe MP4 creation using temporary files with frame dimension fixes.
        """
        import tempfile
        import shutil
        
        if not self._saved_frames:
            print("ERROR: No frames to save!")
            return
        
        # Create temporary directory for frames
        with tempfile.TemporaryDirectory() as temp_dir:
            print(f"Writing frames to temporary directory: {temp_dir}")
            
            # Write frames to individual files
            frame_files = []
            for i, frame_data in enumerate(self._saved_frames):
                if len(frame_data) == 0:
                    continue
                    
                frame_file = os.path.join(temp_dir, f"frame_{i:06d}.png")
                try:
                    with open(frame_file, 'wb') as f:
                        f.write(frame_data)
                    frame_files.append(frame_file)
                except Exception as e:
                    print(f"Error writing frame {i}: {e}")
            
            if not frame_files:
                print("ERROR: No valid frames written!")
                return
            
            # Use FFmpeg with file input and ensure even dimensions
            command = [
                "ffmpeg",
                "-framerate", str(self.fps),
                "-i", os.path.join(temp_dir, "frame_%06d.png"),
                "-vf", "pad=width=ceil(iw/2)*2:height=ceil(ih/2)*2",  # Pad to even dimensions
                "-vcodec", codec,
                "-pix_fmt", "yuv420p",
                "-crf", "18",
                "-y",
                filename
            ]
            
            try:
                result = subprocess.run(command, capture_output=True, text=True, timeout=120)
                
                if result.returncode == 0:
                    print(f"MP4 saved successfully to {filename}")
                    
                    # Verify output file exists and show size
                    if os.path.exists(filename):
                        file_size = os.path.getsize(filename)
                        print(f"Output file size: {file_size/1024/1024:.2f} MB")
                else:
                    print(f"FFmpeg error: {result.stderr}")
                    
            except Exception as e:
                print(f"Error running FFmpeg: {e}")

#!================================================================================================
#!================================================================================================
"""
divide three: mp.at_beginning, mp.at_every, mp.at_end

All functions should accept only sim as an argument
"""

# Define all field components with custom suffixes for each type
FIELD_COMPONENTS = [ 
    {"func_name": "Ez2_dB", "component": mp.Ez, "display_name": "Ez", "suffix": "electric_field_power_anim"},
    {"func_name": "Ey2_dB", "component": mp.Ey, "display_name": "Ey", "suffix": "electric_field_power_anim"},
    {"func_name": "Ex2_dB", "component": mp.Ex, "display_name": "Ex", "suffix": "electric_field_power_anim"}
    # Add more as needed - just add new entries here!
]

# Function factory to create field visualization functions
# Function factory to create field visualization functions
def create_field_func(component, display_name, func_name):
    if component == mp.Ex or component == mp.Ey or component == mp.Ez:
        def field_func(sim):
            # Use the func_name from closure instead of trying to detect it
            E_field_power_dB(sim, component, display_name, func_name)
    
    return field_func

# Dynamically create all field visualization functions
for component_info in FIELD_COMPONENTS:
    # Create the function with proper naming
    func = create_field_func(
        component_info["component"], 
        component_info["display_name"],
        component_info["func_name"]  # Pass the func_name
    )
    func.__name__ = component_info["func_name"]
    
    # Add function to globals with appropriate name
    globals()[component_info["func_name"]] = func
    
    # Initialize animation attribute
    globals()[component_info["func_name"]].anim = None


def save_animation(sim):
    """
    Function designed to be called with mp.at_end to save all animations at the 
    end of the simulation. Uses global variables set by set_animation_params().
    """
    # Use global variable for animation filename
    global anim_file_name
    
    # Check if globals are set
    if 'anim_file_name' not in globals() or anim_file_name is None:
        raise ValueError("Animation parameters not set. Call set_animation_params() first.")
    
    # Loop through all functions defined in FIELD_COMPONENTS
    for component_info in FIELD_COMPONENTS:
        func_name = component_info["func_name"]
        display_name = component_info["display_name"]
        suffix = component_info["suffix"]
        component_func = globals()[func_name]
        
        if hasattr(component_func, 'anim') and component_func.anim is not None:
            print(f"Saving the {display_name}^2 animation...")
            try:
                component_func.anim.to_mp4(f"{anim_file_name}_{display_name}2_{suffix}.mp4")
                print(f"{display_name}^2 animation saved successfully!")
            except Exception as e:
                print(f"Error saving {display_name}^2 animation: {str(e)}")
            finally:
                component_func.anim = None
                
    print("All animations saved and memory cleaned up.")
    return

# Modify E_field_power_dB to accept the function name directly
def E_field_power_dB(sim, component, component_name, func_name=None):
    """
    Generic function to process field power in dB for any field component.
    Uses global parameters set by set_animation_params().
    """
    print("=====================================")
    print(f"{component_name}^2 field data extraction...")
    
    # Use global variables
    global Nfps, image_every, anim_file_name, animation_plotting_params
    
    # Check if globals are set
    if 'Nfps' not in globals() or Nfps is None:
        raise ValueError("Animation parameters not set. Call set_animation_params() first.")
    
    fps = Nfps
    
    # Get caller function
    if func_name is None:
        import inspect
        caller_name = inspect.currentframe().f_back.f_code.co_name
        caller_func = globals()[caller_name]
    else:
        caller_func = globals()[func_name]
    
    # Get component-specific plotting parameters if available
    plotting_params = {}
    if 'animation_plotting_params' in globals() and animation_plotting_params is not None:
        if func_name in animation_plotting_params:
            plotting_params = animation_plotting_params[func_name]
    
    # Set parameters with defaults
    _scale_ = plotting_params.get('scale', 'log')
    title = plotting_params.get('title', 'Power (dB)')
    xlabel = plotting_params.get('xlabel', 'X (mm)')
    ylabel = plotting_params.get('ylabel', 'Y (mm)')
    cbar_label = plotting_params.get('cbar_label', 'Power (dB)')
    vmin = plotting_params.get('vmin', -50)
    vmax = plotting_params.get('vmax', 0)
    invert = plotting_params.get('invert', False)
    
    # Initialize animation object if it doesn't exist
    if not hasattr(caller_func, 'anim') or caller_func.anim is None:
        caller_func.anim = Animate2DArray(fps=fps)
        print(f"Initializing {component_name}^2 animation object...")

    # Get field data and convert to power in dB
    field_arr = sim.get_array(component=component,
                              size=sim.cell_size,
                              center=mp.Vector3(),
                              cmplx= True).transpose()
    
    # # Ensure we're working with the magnitude for complex fields
    # if np.iscomplexobj(field_arr):
    #     field_arr = np.abs(field_arr)
    
    if np.iscomplexobj(field_arr):
        field_arr = np.abs(field_arr.real)**2
        # field_arr = np.angle(field_arr)

    if np.isrealobj(field_arr):
        field_arr = field_arr**2

    # # Convert to power (real values)
    # # complex conjugate multiplication
    # field_arr = field_arr**2 #np.conj(field_arr)
    
    # Ensure we have real values before taking log
    # field_arr = np.real(field_arr)
    
    # # Handle zeros to avoid log(0)
    # field_arr = np.where(field_arr <= 0, 1e-20, field_arr)
    
    # Convert to dB
    field_arr = 10*np.log10(field_arr)
    
    # # Ensure the result is real
    # field_arr = np.real(field_arr)

    # Get dielectric data for overlay
    eps_data = sim.get_array(component=mp.Dielectric,
                             size=sim.cell_size,
                             center=mp.Vector3()).transpose()

    print(f"Time step: {sim.meep_time()}")
    print(f"Creating frame for {component_name}^2 field at time {sim.meep_time()}...")
    print(f"Field array dtype: {field_arr.dtype}, shape: {field_arr.shape}")
    print(f"Field array min/max: {np.min(field_arr):.2f}/{np.max(field_arr):.2f}")

    # IMPORTANT: Call this BEFORE using cmap_blue
    set_plt_params(plt, len(field_arr[1]), len(field_arr[0]), base_factor=12)
    
    # Handle colormap selection - MOVED AFTER set_plt_params
    cmap_name = plotting_params.get('cmap', 'custom_blue')
    if cmap_name == 'custom_blue':
        # Use the predefined custom blue colormap
        plot_cmap = cmap_blue
    else:
        # Use the specified matplotlib colormap
        plot_cmap = cmap_name

    # Set up axes and ticks
    x, y, z, w = sim.get_array_metadata()
    del z, w
    x_min, x_max = min(x), max(x)
    y_min, y_max = min(y), max(y)
    x_axis, y_axis = np.meshgrid(np.linspace(x_min, x_max, len(x)), np.linspace(y_min, y_max, len(y)))
    
    # Create tick marks
    num_ticks = plotting_params.get('num_ticks', 5)

    x_ticks = np.linspace(0, len(x_axis[0]), num_ticks).astype(int)
    x_tick_labels = np.round(np.linspace(x_axis[0, 0], x_axis[0, -1], num_ticks), 1)
    y_ticks = np.linspace(0, len(y_axis[:, 0]), num_ticks).astype(int)
    y_tick_labels = np.round(np.linspace(y_axis[0, 0], y_axis[-1, 0], num_ticks), 1)

    # Create the frame
    caller_func.anim.create_frame(plot_func='plot_2d_array',
                    kwargs= {'array': field_arr,
                            'eps_data': eps_data,
                            'title': title,
                            'xlabel': xlabel,
                            'ylabel': ylabel,
                            'x_ticks': x_ticks,
                            'x_tick_labels': x_tick_labels,
                            'y_ticks': y_ticks,
                            'y_tick_labels': y_tick_labels,
                            'elapsed': sim.meep_time(),
                            'cmap': plot_cmap,
                            'cbar_label_': cbar_label,
                            'invert': invert,
                            'scale': _scale_,
                            'vmin': vmin,
                            'vmax': vmax})
    
    return

#!================================================================================================
#!================================================================================================
#!================================================================================================
# Monitor data collection functions

# Global registry to store monitor configurations
VOLUME_MONITOR_REGISTRY = {}

def set_volume_monitor_registry(monitor_list, 
                                monitor_data_save_dir= None,
                                monitor_data_save_freq = None):
    """
    Update the registry with monitor configurations from simulation_2D.py
    
    Args:
        monitor_list: List of monitor dictionaries from simulation_2D.py
    """
    global VOLUME_MONITOR_REGISTRY
    global volume_monitor_data_save_dir
    global volume_monitor_data_save_freq

    volume_monitor_data_save_dir = monitor_data_save_dir
    volume_monitor_data_save_freq = monitor_data_save_freq


    if monitor_list:
        for monitor_entry in monitor_list:
            name = list(monitor_entry.keys())[0]
            meep_monitor = monitor_entry[name][1]  # Second element contains the monitor object
            # Store monitor configuration data
            monitor_config = monitor_entry[name][0]  # First element contains the configuration
            VOLUME_MONITOR_REGISTRY[name] = monitor_config
            VOLUME_MONITOR_REGISTRY[name]["MONITOR_OBJECT"] = meep_monitor
            if monitor_data_save_dir:
                VOLUME_MONITOR_REGISTRY[name]["MONITOR_DATA_DIR"] = monitor_data_save_dir
                
            print(f"Registered monitor: {name}")

    
#!  Volume monitor components
VOLUME_MONITOR_COMPONENTS = [
    {"func_name": "Ez2_dB_VolumeMonitor", "component": mp.Ez, "display_name": "Ez"},
    {"func_name": "Ey2_dB_VolumeMonitor", "component": mp.Ey, "display_name": "Ey"},
    {"func_name": "Ex2_dB_VolumeMonitor", "component": mp.Ex, "display_name": "Ex"}
    # Add more components as needed
]
    
def create_volume_monitor_func(component, display_name):
    def volume_monitor_func(sim):
        global VOLUME_MONITOR_REGISTRY
        print("***"*10)
        print(f"Computing {display_name}² in dB at monitor locations at time {sim.meep_time()}")

        # Initialize storage directory
        if volume_monitor_data_save_dir:
            output_dir = volume_monitor_data_save_dir + '/volume_monitor_data'
            os.makedirs(output_dir, exist_ok=True)
        else:
            output_dir = 'volume_monitor_data'
            os.makedirs(output_dir, exist_ok=True)

        for monitor_name, monitor_config in VOLUME_MONITOR_REGISTRY.items():
            monitor_obj = monitor_config.get("MONITOR_OBJECT")
            # print(f"Processing monitor: {monitor_name}")
            # print(f"Monitor object: {monitor_obj}")
            # print(f"Monitor config: {monitor_config}")

            if monitor_obj is None:
                raise ValueError(f"Monitor {monitor_name} has no MEEP object")
            
            try:
                # Get field data directly from the monitor volume
                field_data = sim.get_array(component=component, 
                                            vol=monitor_obj).transpose()
                
                # print(f"Field data shape: {field_data.shape}")
                # Calculate power in dB
                power = field_data**2
                with np.errstate(divide='ignore'):  # Handle log of zero
                    power_db = 10 * np.log10(power)
                    power_db = np.where(np.isfinite(power_db), power_db, -100)
                
                # Save data directly to disk instead of keeping it in memory
                # Only save every Nth timestep to reduce storage
                current_time = sim.meep_time()
                save_frequency = volume_monitor_data_save_freq  # Adjust this number to save less frequently
                
                if current_time % save_frequency < 1:  # Save only periodically
                    timestep = int(current_time)
                    filename = f"{output_dir}/{display_name}2_{monitor_name}_step{timestep}.npz"
                    
                    # Save using compressed format with metadata
                    np.savez_compressed(
                        filename, 
                        time=current_time,
                        power_db=power_db,
                        component=display_name,
                        monitor_name=monitor_name
                    )
                    
                    print(f"Saved {display_name}² data for monitor '{monitor_name}' at time {current_time}")
                
            except Exception as e:
                print(f"Error processing monitor '{monitor_name}': {str(e)}")
            


        return # Dummy function for now
    
    return volume_monitor_func

# Dynamically create all volume monitor functions
for component_info in VOLUME_MONITOR_COMPONENTS:
    # Create the function with proper naming
    func = create_volume_monitor_func(
        component_info["component"],
        component_info["display_name"]
    )
    func.__name__ = component_info["func_name"]
    
    # Add function to globals with appropriate name
    globals()[component_info["func_name"]] = func
    
    # Initialize any needed attributes
    globals()[component_info["func_name"]].data = []  # For storing monitor data
    

#!================================================================================================ 
#!================================================================================================
#!================================================================================================

# Flux Monitor Components

# # Global registry to store flux monitor data
# FLUX_MONITOR_REGISTRY = {}

# def set_flux_monitor_registry(flux_monitor_list, 
#                               monitor_data_save_dir=None,
#                               monitor_data_save_freq=None):
#     """
#     Update the registry with flux monitor configurations
    
#     Args:
#         monitor_list: List of flux monitor dictionaries
#         monitor_data_save_dir: Directory to save monitor data
#         monitor_data_save_freq: Frequency to save monitor data
#     """
#     global FLUX_MONITOR_REGISTRY
#     global flux_monitor_data_save_dir
#     global flux_monitor_data_save_freq

#     flux_monitor_data_save_dir = monitor_data_save_dir
#     flux_monitor_data_save_freq = monitor_data_save_freq

#     if flux_monitor_list:
#         for monitor_entry in flux_monitor_list:
#             name = list(monitor_entry.keys())[0]
#             flux_region = monitor_entry[name][1]
#             monitor_config = monitor_entry[name][0]
#             FLUX_MONITOR_REGISTRY[name] = monitor_config
#             FLUX_MONITOR_REGISTRY[name]["FLUX_REGION"] = flux_region
#             FLUX_MONITOR_REGISTRY[name]["FLUX_OBJECT"] = None
#             if monitor_data_save_dir:
#                 FLUX_MONITOR_REGISTRY[name]["MONITOR_DATA_DIR"] = monitor_data_save_dir
                
#             print(f"Registered flux monitor: {name}")


# def create_flux_monitor_func(monitor_type):
#     def flux_monitor_func(sim):
#         global FLUX_MONITOR_REGISTRY
#         print("***"*10)
#         print(f"Processing {monitor_type} flux monitors at time {sim.meep_time()}")

#         # Initialize storage directory
#         if flux_monitor_data_save_dir:
#             output_dir = flux_monitor_data_save_dir + '/flux_monitor_data'
#             os.makedirs(output_dir, exist_ok=True)
#         else:
#             output_dir = 'flux_monitor_data'
#             os.makedirs(output_dir, exist_ok=True)

#         # Process appropriate monitors
#         for monitor_name, monitor_config in FLUX_MONITOR_REGISTRY.items():
#             if monitor_config.get("monitor_type", "").lower() == monitor_type.lower():
#                 try:
#                     # Get flux data from the monitor object
#                     flux_obj = monitor_config.get("FLUX_OBJECT")
#                     if flux_obj is None:
#                         raise ValueError(f"Flux monitor '{monitor_name}' has no MEEP flux object")
#                     else:
#                         print(f"Processing flux monitor '{monitor_name}'")

#                     if monitor_type == 

        



#!================================================================================================ 
#!================================================================================================
# Flux Monitor Components

# # Global registry to store flux monitor data
# FLUX_MONITOR_REGISTRY = {}
# flux_monitor_data_save_dir = None
# flux_monitor_data_save_freq = None
# flux_reference_data = {}  # Add this to store reference flux data
# is_reference_run = False  # Add this flag

# def set_flux_monitor_registry(monitor_list, 
#                               monitor_data_save_dir=None,
#                               monitor_data_save_freq=None,
#                               is_reference=False):  # Add this parameter
#     """
#     Update the registry with flux monitor configurations
    
#     Args:
#         monitor_list: List of flux monitor dictionaries
#         monitor_data_save_dir: Directory to save monitor data
#         monitor_data_save_freq: Frequency to save monitor data
#         is_reference: Whether this is a reference simulation run
#     """
#     global FLUX_MONITOR_REGISTRY
#     global flux_monitor_data_save_dir
#     global flux_monitor_data_save_freq
#     global is_reference_run

#     flux_monitor_data_save_dir = monitor_data_save_dir
#     flux_monitor_data_save_freq = monitor_data_save_freq or 10
#     is_reference_run = is_reference  # Set the global flag

#     if monitor_list:
#         for monitor_entry in monitor_list:
#             name = list(monitor_entry.keys())[0]
#             flux_region = monitor_entry[name][1]
#             monitor_config = monitor_entry[name][0]
#             FLUX_MONITOR_REGISTRY[name] = monitor_config
#             FLUX_MONITOR_REGISTRY[name]["FLUX_REGION"] = flux_region
#             FLUX_MONITOR_REGISTRY[name]["FLUX_OBJECT"] = None
#             if monitor_data_save_dir:
#                 FLUX_MONITOR_REGISTRY[name]["MONITOR_DATA_DIR"] = monitor_data_save_dir
                
#             print(f"Registered flux monitor: {name}")

# def create_flux_monitor_func(monitor_type):
#     def flux_monitor_func(sim):
#         global FLUX_MONITOR_REGISTRY, is_reference_run
#         print("***"*10)
#         print(f"Processing {monitor_type} flux monitors at time {sim.meep_time()}")

#         # Initialize storage directory
#         if flux_monitor_data_save_dir:
#             output_dir = flux_monitor_data_save_dir + '/flux_monitor_data'
#             os.makedirs(output_dir, exist_ok=True)
#         else:
#             output_dir = 'flux_monitor_data'
#             os.makedirs(output_dir, exist_ok=True)
            
#         # Create a subfolder for reference data if this is a reference run
#         if is_reference_run:
#             ref_output_dir = os.path.join(output_dir, 'reference')
#             os.makedirs(ref_output_dir, exist_ok=True)
        
#         # Process appropriate monitors
#         for monitor_name, monitor_config in FLUX_MONITOR_REGISTRY.items():
#             if monitor_config.get("monitor_type", "").lower() == monitor_type.lower():
#                 try:                    
#                     # If this is a reference run, save the flux data to a special file at the end
#                     if is_reference_run and sim.meep_time() >= sim.fields.last_source_time():
#                         flux_obj = FLUX_MONITOR_REGISTRY[monitor_name]["FLUX_OBJECT"]
#                         if flux_obj:
#                             print(f"Saving reference flux data for {monitor_name}")
#                             freqs = mp.get_flux_freqs(flux_obj)
#                             fluxes = mp.get_fluxes(flux_obj)
                            
#                             # Save reference data
#                             ref_filename = f"{ref_output_dir}/{monitor_name}_reference.npz"
#                             np.savez_compressed(
#                                 ref_filename,
#                                 freqs=freqs,
#                                 fluxes=fluxes
#                             )
                            
#                             # Also save the flux object itself using meep's save_flux
#                             flux_file = f"{ref_output_dir}/{monitor_name}_flux"
#                             mp.save_flux(flux_file, flux_obj)
#                             print(f"Reference flux saved to {ref_filename} and {flux_file}")
                
#                 except Exception as e:
#                     print(f"Error processing flux monitor '{monitor_name}': {str(e)}")
#                     import traceback
#                     traceback.print_exc()
        
#         return
    
#     return flux_monitor_func

# def calculate_transmission_reflection(sim):
#     """
#     Calculate transmission and reflection coefficients at the end of simulation
#     """
#     global FLUX_MONITOR_REGISTRY
#     print("Calculating transmission and reflection...")
    
#     # Initialize storage directory
#     if flux_monitor_data_save_dir:
#         output_dir = flux_monitor_data_save_dir + '/flux_monitor_data'
#         ref_output_dir = os.path.join(output_dir, 'reference')
#     else:
#         output_dir = 'flux_monitor_data'
#         ref_output_dir = os.path.join(output_dir, 'reference')
    
#     os.makedirs(output_dir, exist_ok=True)
    
#     # Find incident, reflection, and transmission flux monitors
#     incident_monitor = None
#     reflection_monitor = None
#     transmission_monitor = None
    
#     for monitor_name, monitor_config in FLUX_MONITOR_REGISTRY.items():
#         monitor_type = monitor_config.get("monitor_type", "").lower()
#         if monitor_type == "incident":
#             incident_monitor = monitor_name
#         elif monitor_type == "reflection":
#             reflection_monitor = monitor_name
#         elif monitor_type == "transmission":
#             transmission_monitor = monitor_name
    
#     # Calculate if we have all necessary monitors
#     if incident_monitor and (reflection_monitor or transmission_monitor):
#         try:
#             # Check for reference flux data files
#             ref_incident_file = f"{ref_output_dir}/{incident_monitor}_flux"
            
#             if os.path.exists(ref_incident_file):
#                 # Load the reference flux
#                 incident_flux = FLUX_MONITOR_REGISTRY[incident_monitor]["FLUX_OBJECT"]
#                 ref_incident_flux = mp.load_flux(ref_incident_file, incident_flux)
                
#                 # Get the frequencies
#                 freqs = mp.get_flux_freqs(incident_flux)
                
#                 # Calculate transmission if available
#                 if transmission_monitor:
#                     trans_flux = FLUX_MONITOR_REGISTRY[transmission_monitor]["FLUX_OBJECT"]
#                     trans_data = mp.get_fluxes(trans_flux)
#                     transmission = np.array(trans_data) / np.array(mp.get_fluxes(ref_incident_flux))
                    
#                     # Save transmission data
#                     filename = f"{output_dir}/transmission_coefficient.npz"
#                     np.savez_compressed(
#                         filename,
#                         freqs=freqs,
#                         transmission=transmission
#                     )
#                     print(f"Saved normalized transmission coefficient data to {filename}")
                
#                 # Calculate reflection if available
#                 if reflection_monitor:
#                     refl_flux = FLUX_MONITOR_REGISTRY[reflection_monitor]["FLUX_OBJECT"]
#                     refl_data = mp.get_fluxes(refl_flux)
#                     # For reflection, negate the reference data according to MEEP convention
#                     reflection = -np.array(refl_data) / np.array(mp.get_fluxes(ref_incident_flux))
                    
#                     # Save reflection data
#                     filename = f"{output_dir}/reflection_coefficient.npz"
#                     np.savez_compressed(
#                         filename,
#                         freqs=freqs,
#                         reflection=reflection
#                     )
#                     print(f"Saved normalized reflection coefficient data to {filename}")
                
#                 # Calculate conservation of energy
#                 if reflection_monitor and transmission_monitor:
#                     conservation = reflection + transmission
#                     filename = f"{output_dir}/flux_conservation.npz"
#                     np.savez_compressed(
#                         filename,
#                         freqs=freqs,
#                         reflection=reflection,
#                         transmission=transmission,
#                         conservation=conservation
#                     )
#                     print(f"Saved energy conservation data to {filename}")
#             else:
#                 print(f"Reference flux file not found: {ref_incident_file}")
#                 print("Run a reference simulation first using the 'run_reference_sim' option in your JSON file")
                
#                 # Fall back to non-normalized calculations
#                 incident_flux = FLUX_MONITOR_REGISTRY[incident_monitor]["FLUX_OBJECT"]
#                 freqs = mp.get_flux_freqs(incident_flux)
#                 incident_data = mp.get_fluxes(incident_flux)
                
#                 # Continue with non-normalized calculations as before...
#         except Exception as e:
#             print(f"Error calculating transmission/reflection: {str(e)}")
#             import traceback
#             traceback.print_exc()
#     else:
#         print("Cannot calculate transmission/reflection: missing required monitors")
    
#     return
# # Make flux calculation available at the end of simulation
# globals()["calculate_transmission_reflection"] = calculate_transmission_reflection