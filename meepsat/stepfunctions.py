import sys
import os
import site
from pathlib import Path
from memory_profiler import profile
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
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib
import matplotlib.colors as colors
import matplotlib.cbook as cbook
import subprocess

import meepsat.helpers as exf


import warnings
warnings.filterwarnings("ignore")

#!================================================================================================

"""
Step Functions for Animations and Monitor Data Collection 
#! IMPORTANT 
NOTE: Monitor data collection functions are at the bottom and need to be properly checked before publishing.
"""

# Initialising global variables for animation parameters
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
                fps,
                use_disk_cache=True,  # NEW: option to use disk instead of memory
                temp_dir=None):       # NEW: custom temp directory
        
        self.fps = fps
        self._saved_frames = []  # Keep for backward compatibility
        self.use_disk_cache = use_disk_cache
        self.frame_count = 0
        
        # Create temporary directory for disk caching
        if use_disk_cache:
            import tempfile
            if temp_dir is None:
                self.temp_dir = tempfile.mkdtemp(prefix='meep_anim_')
            else:
                self.temp_dir = temp_dir
                os.makedirs(self.temp_dir, exist_ok=True)
            print(f"Using disk cache for frames at: {self.temp_dir}")
        else:
            self.temp_dir = None

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
        # plt.close(fig)

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
        Captures the current frame and saves to disk or memory.
        """
        from io import BytesIO

        if fig is None:
            print("WARNING: No figure provided to grab_frame")
            return
        
        if self.use_disk_cache:
            # Write directly to disk
            frame_file = os.path.join(self.temp_dir, f"frame_{self.frame_count:06d}.png")
            try:
                fig.savefig(frame_file, format='png', bbox_inches='tight',
                        facecolor=fig.get_facecolor(), edgecolor='none')
                self.frame_count += 1
                print(f'Frame saved to disk at timestep: {elapsed} (file: {frame_file})')
            except Exception as e:
                print(f"ERROR saving frame at timestep {elapsed}: {e}")
            finally:
                # Explicitly close figure to free memory
                plt.close(fig)
        else:
            # Original memory-based approach
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
                # Explicitly close figure to free memory
                plt.close(fig)

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
        Memory-safe MP4 creation using frames from disk or memory.
        """
        import tempfile
        import shutil
        
        if self.use_disk_cache:
            # Frames already on disk
            if self.frame_count == 0:
                print("ERROR: No frames to save!")
                return
            
            print(f"Creating MP4 from {self.frame_count} frames on disk")
            
            # Use FFmpeg directly with cached frames
            command = [
                "ffmpeg",
                "-framerate", str(self.fps),
                "-i", os.path.join(self.temp_dir, "frame_%06d.png"),
                "-vf", "pad=width=ceil(iw/2)*2:height=ceil(ih/2)*2",
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
                    if os.path.exists(filename):
                        file_size = os.path.getsize(filename)
                        print(f"Output file size: {file_size/1024/1024:.2f} MB")
                else:
                    print(f"FFmpeg error: {result.stderr}")
                    
            except Exception as e:
                print(f"Error running FFmpeg: {e}")
            finally:
                # Clean up temp directory
                if os.path.exists(self.temp_dir):
                    shutil.rmtree(self.temp_dir)
                    print(f"Cleaned up temporary directory: {self.temp_dir}")
        else:
            # Frames in memory
            if not self._saved_frames:
                print("ERROR: No frames to save!")
                return
            
            with tempfile.TemporaryDirectory() as temp_dir:
                print(f"Writing frames to temporary directory: {temp_dir}")
                
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
                
                command = [
                    "ffmpeg",
                    "-framerate", str(self.fps),
                    "-i", os.path.join(temp_dir, "frame_%06d.png"),
                    "-vf", "pad=width=ceil(iw/2)*2:height=ceil(ih/2)*2",
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


# @profile
def E_field_power_dB(sim, component, component_name, func_name=None):
    """
    Generic function to process field power in dB for any field component.
    Uses global parameters set by set_animation_params().
    """
    print("=====================================")
    print(f"{component_name}^2 field data extraction...")
    
    # Use global variables
    global Nfps, image_every, anim_file_name, animation_plotting_params
    
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
    # MEMORY OPTIMIZATION: Use disk caching by default
    if not hasattr(caller_func, 'anim') or caller_func.anim is None:
        caller_func.anim = Animate2DArray(fps=fps, use_disk_cache=True)  # CHANGED
        print(f"Initializing {component_name}^2 animation object with disk caching...")

    # MEMORY OPTIMIZATION: Get field data in-place where possible
    field_arr = sim.get_array(component=component,
                              size=sim.cell_size,
                              center=mp.Vector3(),
                              cmplx=True).transpose()
    
    # MEMORY OPTIMIZATION: Process in-place to avoid copies
    if np.iscomplexobj(field_arr):
        field_arr = np.abs(field_arr.real, out=np.empty(field_arr.shape, dtype=np.float64))
        field_arr **= 2  # In-place squaring
    elif np.isrealobj(field_arr):
        field_arr **= 2  # In-place squaring
    
    # Convert to dB (in-place where possible)
    np.log10(field_arr, out=field_arr)
    field_arr *= 10
    
    # Get dielectric data for overlay
    eps_data = sim.get_array(component=mp.Dielectric,
                             size=sim.cell_size,
                             center=mp.Vector3()).transpose()

    print(f"Time step: {sim.meep_time()}")
    print(f"Creating frame for {component_name}^2 field at time {sim.meep_time()}...")

    # IMPORTANT: Call this BEFORE using cmap_blue
    set_plt_params(plt, len(field_arr[1]), len(field_arr[0]), base_factor=12)
    
    # Handle colormap selection
    cmap_name = plotting_params.get('cmap', 'custom_blue')
    if cmap_name == 'custom_blue':
        plot_cmap = cmap_blue
    else:
        plot_cmap = cmap_name

    # Set up axes and ticks
    x, y, z, w = sim.get_array_metadata()
    del z, w  # MEMORY OPTIMIZATION: Delete immediately
    x_min, x_max = min(x), max(x)
    y_min, y_max = min(y), max(y)
    
    # MEMORY OPTIMIZATION: Don't create full meshgrid if not needed
    num_ticks = plotting_params.get('num_ticks', 5)
    x_ticks = np.linspace(0, len(x), num_ticks).astype(int)
    x_tick_labels = np.round(np.linspace(x_min, x_max, num_ticks), 1)
    y_ticks = np.linspace(0, len(y), num_ticks).astype(int)
    y_tick_labels = np.round(np.linspace(y_min, y_max, num_ticks), 1)

    # Create the frame
    caller_func.anim.create_frame(plot_func='plot_2d_array',
                    kwargs={'array': field_arr,
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
    
    # MEMORY OPTIMIZATION: Explicitly delete large arrays
    del field_arr, eps_data, x, y
    
    # MEMORY OPTIMIZATION: Force garbage collection periodically
    import gc
    if int(sim.meep_time()) % 10 == 0:  # Every 10 timesteps
        gc.collect()
    
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
                                            vol=monitor_obj)
                
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


#! ================================================================================================
#! ================================================================================================
#! ================================================================================================
# Field StepFunctions

# Set global parameters for field step extraction functions
def set_field_params(field_params: dict):
    """
    Set the global field extraction parameters for the simulation.
    """
    global size_x, size_y, savepath
    global downsampling_factor_x, downsampling_factor_y
    
    if "size_x" not in field_params:
        raise ValueError("size_x must be specified in field_params")
    if "size_y" not in field_params:
        raise ValueError("size_y must be specified in field_params")
    
    size_x = field_params["size_x"]
    size_y = field_params["size_y"]
    savepath = field_params.get("savepath", ".")
    downsampling_factor_x = field_params.get("downsampling_factor_x", 1)
    downsampling_factor_y = field_params.get("downsampling_factor_y", 1)
    
    print("Field extraction parameters set:")
    print(f"  size_x: {size_x}")
    print(f"  size_y: {size_y}")
    print(f"  savepath: {savepath}")
    print(f"  downsampling_factor_x: {downsampling_factor_x}")
    print(f"  downsampling_factor_y: {downsampling_factor_y}")

    # Initialising count and global field arrays
    global count
    count = 0
    global Ex_global, Ey_global, Ez_global, Hx_global, Hy_global, Hz_global
    Ex_global = None
    Ey_global = None
    Ez_global = None
    Hx_global = None
    Hy_global = None
    Hz_global = None
    
def extract_xyzw(simulation):
    # Define a box to capture the fields in the entire simulation cell
    box = mp.Volume(center=mp.Vector3(0, 0, 0),
                    size=mp.Vector3(size_x, size_y, 0)
    )
    (x,y,z,w) = simulation.get_array_metadata(vol=box)

    #Apply downsampling to x, y, w if needed
    if downsampling_factor_x > 1:
        x = x[::downsampling_factor_x]
        y = y[::downsampling_factor_y]
        w = w[::downsampling_factor_x, ::downsampling_factor_y]

    # Save the xyzw data to an npz compressed file
    np.savez_compressed(os.path.join(savepath, "xyzw.npz"),
                        x_coords=x,
                        y_coords=y,
                        weights=w)

    return

def accumulate_efield_and_hfield(simulation):
    """
    Accumulate electric field and magnetic field over a range of timesteps with downsampling to reduce memory usage
    """
    # Downsample the arrays
    def downsample(array, downsample_x, downsample_y):
        original_shape = array.shape
        new_shape = (original_shape[0] // downsample_x,
                     original_shape[1] // downsample_y)
        
        # Truncate the array to make it divisible by downsample factors
        truncated_array = array[:new_shape[0]*downsample_x, :new_shape[1]*downsample_y]

        # Reshape and average
        downsampled_array = truncated_array.reshape(new_shape[0], downsample_x,
                                                    new_shape[1], downsample_y).mean(axis=(1, 3))
        return downsampled_array
    
    global count
    if count == 0:
        print(f"Downsampling by factors ({downsampling_factor_x}, {downsampling_factor_y}) to reduce memory usage.")

    # Define the volume for the entire simulation cell
    full_volume = mp.Volume(center=mp.Vector3(0, 0, 0),
                            size=mp.Vector3(size_x, size_y, 0))
    
    global Ex_global, Ey_global, Ez_global, Hx_global, Hy_global, Hz_global

    # Get and downsample E field components
    ex = simulation.get_array(vol=full_volume, component=mp.Ex, cmplx=True)
    ex_down = downsample(ex, downsampling_factor_x, downsampling_factor_y)
    
    # Initialize globals on first call with downsampled shape
    if Ex_global is None:
        Ex_global = np.zeros_like(ex_down, dtype=np.complex64)
        Ey_global = np.zeros_like(ex_down, dtype=np.complex64)
        Ez_global = np.zeros_like(ex_down, dtype=np.complex64)
        Hx_global = np.zeros_like(ex_down, dtype=np.complex64)
        Hy_global = np.zeros_like(ex_down, dtype=np.complex64)
        Hz_global = np.zeros_like(ex_down, dtype=np.complex64)
        print(f"Initialized global arrays with shape from {ex.shape} to {ex_down.shape} for downsampled storage.")
        
    np.add(Ex_global, ex_down, out=Ex_global)
    del ex, ex_down

    # Ey
    ey = simulation.get_array(vol=full_volume, component=mp.Ey, cmplx=True)
    ey_down = downsample(ey, downsampling_factor_x, downsampling_factor_y)
    np.add(Ey_global, ey_down, out=Ey_global)
    del ey, ey_down

    # Ez
    ez = simulation.get_array(vol=full_volume, component=mp.Ez, cmplx=True)
    ez_down = downsample(ez, downsampling_factor_x, downsampling_factor_y)
    np.add(Ez_global, ez_down, out=Ez_global)
    del ez, ez_down

    # Hx
    hx = simulation.get_array(vol=full_volume, component=mp.Hx, cmplx=True)
    hx_down = downsample(hx, downsampling_factor_x, downsampling_factor_y)
    np.add(Hx_global, hx_down, out=Hx_global)
    del hx, hx_down

    # Hy
    hy = simulation.get_array(vol=full_volume, component=mp.Hy, cmplx=True)
    hy_down = downsample(hy, downsampling_factor_x, downsampling_factor_y)
    np.add(Hy_global, hy_down, out=Hy_global)
    del hy, hy_down

    # Hz
    hz = simulation.get_array(vol=full_volume, component=mp.Hz, cmplx=True)
    hz_down = downsample(hz, downsampling_factor_x, downsampling_factor_y)
    np.add(Hz_global, hz_down, out=Hz_global)
    del hz, hz_down

    count += 1
    if count % 10 == 0:  # Print every 10 timesteps to reduce output
        print(f"Fields accumulated at t={simulation.meep_time():.2f} (count={count})")

# Calculate the average E and H fields over the simulation time
def calculate_average_fields(array, count):
    return array / count

def save_accumulated_fields(simulation):
    """
    Save the accumulated E and H fields to compressed npz files
    """
    global Ex_global, Ey_global, Ez_global, Hx_global, Hy_global, Hz_global, count
    print("Calculating average E and H fields over accumulated timesteps...")

    Ex_avg = calculate_average_fields(Ex_global, count)
    del Ex_global

    Ey_avg = calculate_average_fields(Ey_global, count)
    del Ey_global

    Ez_avg = calculate_average_fields(Ez_global, count)
    del Ez_global

    Hx_avg = calculate_average_fields(Hx_global, count)
    del Hx_global

    Hy_avg = calculate_average_fields(Hy_global, count)
    del Hy_global

    Hz_avg = calculate_average_fields(Hz_global, count)
    del Hz_global

    # Save the average E and H fields to npz files

    #Ex, Ey, Ez
    np.savez_compressed(
        os.path.join(savepath, "efield_timeavg.npz"),
        ex_real=np.real(Ex_avg),
        ex_imag=np.imag(Ex_avg),
        ey_real=np.real(Ey_avg),
        ey_imag=np.imag(Ey_avg),
        ez_real=np.real(Ez_avg),
        ez_imag=np.imag(Ez_avg),
        count=count
    )

    # Hx, Hy, Hz
    np.savez_compressed(
        os.path.join(savepath, "hfield_timeavg.npz"),
        hx_real=np.real(Hx_avg),
        hx_imag=np.imag(Hx_avg),
        hy_real=np.real(Hy_avg),
        hy_imag=np.imag(Hy_avg),
        hz_real=np.real(Hz_avg),
        hz_imag=np.imag(Hz_avg),
        count=count
    )

#! ================================================================================================
#! ================================================================================================
#! ================================================================================================
