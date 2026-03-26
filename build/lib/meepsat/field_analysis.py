import sys
import os
import site
from pathlib import Path
from memory_profiler import profile
import numpy as np
import matplotlib.pyplot as plt
import meep as mp
import h5py
import scipy.optimize as sc

@profile
def get_MEEP_ff(simulation,
                ff_distance = None,
                ff_angle = None,
                ff_npts = None,
                n2f_obj = None,
                saveplot = False,
                parallel = False,
                saveh5 = False,
                filename = None ,
                ylim = None,
                plot_title = None):
    '''
    Gets the far field using MEEP near2far function.

    Arguments
    ---------
    simulation : meep.Simulation
        The MEEP simulation object.
    ff_distance : float
        Distance from the source to the far field observation point.
    ff_angle : float
        Angle of the far field observation point in degrees.
    ff_npts : int
        Number of points in the far field observation.
    n2f_obj : meep.Near2Far
        The Near2Far object for the simulation.
    saveplot : bool
        If True, saves the far field plot.
    #! parallel : bool
    #!     If True, uses parallel file writing for the HDF5 file.
    saveh5 : bool
        If True, saves the far field data to an HDF5 file.
    filename : str
        Base name for the output files.
    ylim : float
        Y-axis limit for the plot.
    plot_title : str
        Title for the plot.

    Returns
    -------
    angles : list
        List of angles in degrees corresponding to the far field data.
    ffmeep : np.ndarray
        Far field data in dB, normalized to the maximum amplitude.
    
    '''
    ff_length = ff_distance*np.tan(np.radians(ff_angle))
    ff_res = ff_npts/ff_length

    ff = simulation.get_farfields(n2f_obj, 
        ff_res, 
        center=mp.Vector3(- ff_distance,0.5*ff_length), 
        size=mp.Vector3(y=ff_length))

    # Use the actual length of the far field data to create angles
    actual_npts = len(ff['Ez'])
    ff_lengths = np.linspace(0, ff_length, actual_npts)
    angles = [np.degrees(np.arctan(f)) for f in ff_lengths/ff_distance]

    norm = np.absolute(ff['Ez'])/np.max(np.absolute(ff['Ez'])) / (np.cos(np.radians(angles)))**2
    ff_dB = 10*np.log10(norm)  

    ffmeep = ff_dB
    angles = angles
    if saveplot : 
        plt.figure(figsize = (8,6))
        plt.plot(angles,ff_dB,'bo-')
        plt.xlim(0,ff_angle)
        plt.ylim((ylim,0))
        plt.xticks([t for t in range(0,ff_angle+1,10)])
        plt.xlabel("Angle [deg]")
        plt.ylabel("Amplitude [dB]")
        plt.grid(axis='x',linewidth=0.5,linestyle='--')
        if plot_title:
            plt.title(plot_title)
        plt.savefig(filename + '.png')
        plt.close()

    # if saveh5 :
    #     # Enable parallel parameter in function signature first
    #     if parallel:
    #         from mpi4py import MPI
    #         comm = MPI.COMM_WORLD
    #         if not h5py.get_config().mpi:
    #             raise ValueError("h5py was built without MPI support, can't use mpio driver")
            
    #         with h5py.File(filename + '.h5', 'w', driver='mpio', comm=comm) as h:
    #             h.create_dataset('deg', data=angles, dtype='float64')
    #             h.create_dataset('amplitudedB', data=ff_dB, dtype='float64')
    #     else:
    #         # Only have rank 0 create the file in non-parallel mode
    #         from mpi4py import MPI
    #         comm = MPI.COMM_WORLD
    #         rank = comm.Get_rank()
            
    #         if rank == 0:
    #             with h5py.File(filename + '.h5', 'w', libver='latest') as h:
    #                 h.create_dataset('deg', data=angles, dtype='float64', compression='gzip')
    #                 h.create_dataset('amplitudedB', data=ff_dB, dtype='float64', compression='gzip')
    #         comm.barrier()  # Ensure all processes wait for file creation

    # return angles, ffmeep

    # if saveh5:
    #     # Replace h5py with numpy compressed save
    #     if parallel:
    #         from mpi4py import MPI
    #         comm = MPI.COMM_WORLD
    #         rank = comm.Get_rank()
            
    #         if rank == 0:
    #             np.savez_compressed(filename + '.npz', 
    #                               deg=angles, 
    #                               amplitudedB=ff_dB)
    #             print(f"Far field data saved to {filename}.npz")
    #         comm.barrier()
    #     else:
    #         from mpi4py import MPI
    #         comm = MPI.COMM_WORLD
    #         rank = comm.Get_rank()
            
    #         if rank == 0:
    #             np.savez_compressed(filename + '.npz', 
    #                               deg=angles, 
    #                               amplitudedB=ff_dB)
    #             print(f"Far field data saved to {filename}.npz")
    #         comm.barrier()
    if saveh5:
        np.savez_compressed(filename + '.npz', 
                          deg=angles, 
                          amplitudedB=ff_dB)
        print(f"Far field data saved to {filename}.npz")

    return angles, ffmeep


# @profile
# def get_complex_field(sim,
#                         simres,
#                         aper_size,
#                         aper_pos_x,
#                         wvl,
#                         plot_amp = False, 
#                         saveh5 = False, 
#                         filename = 'test',
#                         parallel = False):
#         '''
#         Gets the electric field in its complex form at the aperture.
#         To that end, fits the time evolution of the field there.

#         Arguments
#         ---------
#         plot_amp : bool, optional
#             Whether to plot the amplitude of field at aperture. 
#             (default : False)
#         saveh5 : bool, optional
#             Whether to save the amplitude in an h5 file. (default : False)
#         filename : str, optional
#             Name of the plot to be saved
#         parallel : bool, optional
#             Whether the code is running in parallel

#         Returns
#         -------
#         amplitude*phase : complex array
#             Complex electric field at the aperture

#         '''

#         #Setting the timestep to a very low value, 
#         #so that MEEP uses its lowest timestep
#         timestep = .3

#         #120 steps Is roughly enough to give a 
#         #few periods for wavelengths from 1 to 10 mm
#         #Can be tweaked to save on sim time.
#         n_iter = 120

#         res = simres
#         AP_size = aper_size
#         aper_pos_x = aper_pos_x
        
#         #Get the real field at aperture
#         efield = sim.get_array(center=mp.Vector3(aper_pos_x, 0), 
#                                     size=mp.Vector3(0, AP_size), 
#                                     component=mp.Ez)

        

#         #Initializes the list containing the E field evolution
#         e_field_evol = np.ones((n_iter, len(efield)))
#         e_field_evol[0] = efield

#         #List to get the precise timestepping done by meep
#         time = np.zeros(n_iter)
#         time[0] = sim.meep_time()

#         ### Stacking the electric field evolution on the aperture
#         for k in range(n_iter):
#             sim.run(until = timestep)
#             time[k] = sim.meep_time()
#             e_field_evol[k] = sim.get_array(center=mp.Vector3(aper_pos_x, 0), 
#                                          size=mp.Vector3(0, AP_size), 
#                                          component=mp.Ez)
        
#         #Each point on the aperture is fit for a cosine with amplitude and phase
#         def f(x, amp, phase):
#             return amp*np.cos(x*2*np.pi/wvl + phase)

#         #Initialize the lists of amplitude and phase over the aperture
#         amplitude = np.zeros(int(AP_size*res))
#         phase = np.zeros(int(AP_size*res))

#         #The field is only taken on the opening of the aperture

#         #Fits amplitude and phase for each point
#         for k in range(int(AP_size*res)):
#             popt, pcov = sc.curve_fit(f, time, e_field_evol[:,k])
#             amplitude[k] = popt[0]
#             phase[k] = popt[1]
        

#         y = np.linspace(-AP_size/2,AP_size/2,len(amplitude))
#         ### Plot
#         if plot_amp :
#             norm = np.max(np.abs(amplitude))
#             amp = 10*np.log10(np.abs(amplitude)/norm)
#             plt.figure()
#             plt.plot(y, amp) 
#             plt.ylim((-60,0))
#             plt.xlim((0, AP_size/2))
#             plt.title('E field amplitude on aperture')
#             plt.xlabel('y (mm)')
#             plt.ylabel('$Amplitude [dB]$')
#             plt.savefig(filename + '.png')
#             plt.close()

#         # if saveh5 : 
#         #     if parallel :
#         #         from mpi4py import MPI
#         #         comm = MPI.COMM_WORLD
#         #         rank = comm.Get_rank()
                
#         #         if not h5py.get_config().mpi:
#         #             raise ValueError("h5py was built without MPI support, can't use mpio driver")
                
#         #         try:
#         #             with h5py.File(filename + '.h5', 'w', driver='mpio', comm=comm) as h:
#         #                 h.create_dataset('y', data=y, dtype='float64')
#         #                 h.create_dataset('amplitude', data=amplitude, dtype='float64')
#         #                 h.create_dataset('phase', data=phase, dtype='float64')
#         #         except OSError as e:
#         #             # Fallback to serial writing from rank 0 if MPI file creation fails
#         #             if rank == 0:
#         #                 print(f"MPI file creation failed: {e}")
#         #                 print("Falling back to serial file writing...")
#         #                 with h5py.File(filename + '.h5', 'w') as h:
#         #                     h.create_dataset('y', data=y, dtype='float64', compression='gzip')
#         #                     h.create_dataset('amplitude', data=amplitude, dtype='float64', compression='gzip')
#         #                     h.create_dataset('phase', data=phase, dtype='float64', compression='gzip')
#         #             comm.barrier()  # Ensure all processes wait for file creation
#         #     else: 
#         #         with h5py.File(filename + '.h5', 'w') as h:
#         #             h.create_dataset('y', data=y, dtype='float64', compression='gzip')
#         #             h.create_dataset('amplitude', data=amplitude, dtype='float64', compression='gzip')
#         #             h.create_dataset('phase', data=phase, dtype='float64', compression='gzip')
        
#         # return amplitude*np.exp(1j*phase)

#         # if saveh5:
#         #     if parallel:
#         #         from mpi4py import MPI
#         #         comm = MPI.COMM_WORLD
#         #         rank = comm.Get_rank()
                
#         #         if rank == 0:
#         #             np.savez_compressed(filename + '.npz',
#         #                             y=y,
#         #                             amplitude=amplitude,
#         #                             phase=phase)
#         #             print(f"Complex field data saved to {filename}.npz")
#         #         comm.barrier()
#         #     else:
#         #         np.savez_compressed(filename + '.npz',
#         #                         y=y,
#         #                         amplitude=amplitude,
#         #                         phase=phase)
#         #         print(f"Complex field data saved to {filename}.npz")

#         if saveh5:
#             np.savez_compressed(filename + '.npz',
#                                 y=y,
#                                 amplitude=amplitude,
#                                 phase=phase)
#             print(f"Complex field data saved to {filename}.npz")
        
#         return amplitude*np.exp(1j*phase)


@profile
def get_complex_field(sim,
                        simres,
                        aper_size,
                        aper_pos_x,
                        wvl,
                        plot_amp = False, 
                        saveh5 = False, 
                        filename = 'test',
                        parallel = False):
    '''
    Gets the electric field in its complex form at the aperture.
    Uses memory-mapped files to minimize RAM usage during field collection.
    '''
    
    import gc
    import tempfile
    import os
    
    timestep = .3
    n_iter = 60
    res = simres
    AP_size = aper_size
    
    # Get initial field to determine size
    efield = sim.get_array(center=mp.Vector3(aper_pos_x, 0), 
                                size=mp.Vector3(0, AP_size), 
                                component=mp.Ez)
    
    n_points = len(efield)
    
    # Create temporary file for memory-mapped array (deleted automatically)
    temp_dir = tempfile.mkdtemp()
    temp_file = os.path.join(temp_dir, 'e_field_evol.dat')
    
    try:
        # Create memory-mapped array on disk instead of RAM
        e_field_evol = np.memmap(temp_file, dtype='float32', mode='w+', 
                                  shape=(n_iter, n_points))
        e_field_evol[0] = efield
        
        time = np.zeros(n_iter, dtype=np.float32)
        time[0] = sim.meep_time()
        
        # Stack the electric field evolution (writes to disk, not RAM)
        for k in range(1, n_iter):
            sim.run(until = timestep)
            time[k] = sim.meep_time()
            e_field_evol[k] = sim.get_array(center=mp.Vector3(aper_pos_x, 0), 
                                         size=mp.Vector3(0, AP_size), 
                                         component=mp.Ez)
            # Flush to disk periodically to avoid RAM buildup
            if k % 10 == 0:
                e_field_evol.flush()
        
        # Cosine fitting function
        def f(x, amp, phase):
            return amp*np.cos(x*2*np.pi/wvl + phase)
        
        # Initialize amplitude and phase arrays
        amplitude = np.zeros(n_points, dtype=np.float32)
        phase = np.zeros(n_points, dtype=np.float32)
        
        # Fit amplitude and phase for each point
        # Data is loaded from disk only when accessed
        for k in range(n_points):
            try:
                popt, _ = sc.curve_fit(f, time, e_field_evol[:, k])
                amplitude[k] = popt[0]
                phase[k] = popt[1]
            except RuntimeError:
                amplitude[k] = 0
                phase[k] = 0
        
        # Explicitly delete memory-mapped array
        del e_field_evol
        del time
        gc.collect()
        
    finally:
        # Clean up temporary file
        if os.path.exists(temp_file):
            os.remove(temp_file)
        if os.path.exists(temp_dir):
            os.rmdir(temp_dir)
    
    y = np.linspace(-AP_size/2, AP_size/2, n_points, dtype=np.float32)
    
    # Plot if requested
    if plot_amp:
        norm = np.max(np.abs(amplitude))
        amp = 10*np.log10(np.abs(amplitude)/norm)
        plt.figure()
        plt.plot(y, amp) 
        plt.ylim((-60, 0))
        plt.xlim((0, AP_size/2))
        plt.title('E field amplitude on aperture')
        plt.xlabel('y (mm)')
        plt.ylabel('$Amplitude [dB]$')
        plt.savefig(filename + '.png')
        plt.close()
    
    # Save to file if requested
    if saveh5:
        np.savez_compressed(filename + '.npz',
                            y=y,
                            amplitude=amplitude,
                            phase=phase)
        print(f"Complex field data saved to {filename}.npz")
    
    # Create result and clean up
    result = amplitude * np.exp(1j * phase)
    del amplitude, phase, y
    gc.collect()
    
    return result




def custom_beam_FT(sim_res,
                    list_efields,
                    aper_size,          
                    zero_pad = 15,
                    savebeam = False,
                    parallel = False,
                    filename = None):

    '''
    Gets the Fourier Transforms of the complex electric fields at aperture.

    Arguments
    ---------
    zero_pad : float, optional
        Multiplicative factor to the length of the field list,
        which is padded with zeros in the added length.

    Returns
    -------
    freq : array
        List of the frequencies at which the FFT has been done
    FFTs : list of arrays
        Each array contains the FFT for the k-th source.
    '''

    # Ensure list_efields is a list of arrays, not a single value
    if not isinstance(list_efields, list):
        list_efields = [list_efields]
    
    # Check if each element is an array
    for i, efield in enumerate(list_efields):
        if not hasattr(efield, '__len__') or np.isscalar(efield):
            raise ValueError(f"list_efields[{i}] must be an array, got {type(efield)}")

    #Initialize the list
    FFTs = [[] for k in range(len(list_efields))]

    res = sim_res

    #List of frequencies
    freq = np.fft.fftfreq(len(list_efields[0])*zero_pad, d = 1/res)

    #Iterate over the number of sources
    for k in range(len(list_efields)):

        #FFT over the field
        fft = np.fft.fft(list_efields[k], 
                n = zero_pad*len(list_efields[k]))

        #FFT is normalized by its max
        FFTs[k] = np.abs(fft) 
        FFTs[k] = FFTs[k]/np.max(FFTs[k])

    # if savebeam:
    #     # Helper function to create dataset with appropriate parameters
    #     def create_flexible_dataset(h, name, data, dtype):
    #         # Check if data is scalar (has no length) or is a 0-dimensional array
    #         is_scalar = np.isscalar(data) or (isinstance(data, np.ndarray) and data.ndim == 0)
            
    #         if parallel:
    #             # No compression in parallel mode
    #             h.create_dataset(name, data=data, dtype=dtype)
    #         else:
    #             # Only apply compression if not scalar
    #             if is_scalar:
    #                 h.create_dataset(name, data=data, dtype=dtype)
    #             else:
    #                 h.create_dataset(name, data=data, dtype=dtype, compression='gzip')
                
    #     if parallel:
    #         from mpi4py import MPI
    #         comm = MPI.COMM_WORLD
    #         if not h5py.get_config().mpi:
    #             raise ValueError("h5py was built without MPI support, can't use mpio driver")
            
    #         with h5py.File(filename + '.h5', 'w', driver='mpio', comm=comm) as h:
    #             create_flexible_dataset(h, 'freq', freq, 'float64')
    #             create_flexible_dataset(h, 'beams', FFTs, 'float64')
    #             aper = aper_size
    #             create_flexible_dataset(h, 'aper_size', aper, 'float64')
    #     else:
    #         # Only have rank 0 create the file in non-parallel mode
    #         from mpi4py import MPI
    #         comm = MPI.COMM_WORLD
    #         rank = comm.Get_rank()
            
    #         if rank == 0:
    #             with h5py.File(filename + '.h5', 'w', libver='latest') as h:
    #                 create_flexible_dataset(h, 'freq', freq, 'float64')
    #                 create_flexible_dataset(h, 'beams', FFTs, 'float64')
    #                 aper = aper_size
    #                 create_flexible_dataset(h, 'aper_size', aper, 'float64')
    #         comm.barrier()  # Ensure all processes wait for file creation
    #     return freq, FFTs

    # if savebeam:
    #     if parallel:
    #         from mpi4py import MPI
    #         comm = MPI.COMM_WORLD
    #         rank = comm.Get_rank()
            
    #         if rank == 0:
    #             np.savez_compressed(filename + '.npz',
    #                               freq=freq,
    #                               beams=FFTs,
    #                               aper_size=aper_size)
    #             print(f"Beam FFT data saved to {filename}.npz")
    #         comm.barrier()
    #     else:
    #         np.savez_compressed(filename + '.npz',
    #                           freq=freq,
    #                           beams=FFTs,
    #                           aper_size=aper_size)
    #         print(f"Beam FFT data saved to {filename}.npz")

    if savebeam:
        np.savez_compressed(filename + '.npz',
                            freq=freq,
                            beams=FFTs,
                            aper_size=aper_size)
        print(f"Beam FFT data saved to {filename}.npz")
        
    return freq, FFTs




# Save the epsilon map
def save_epsilon_map(sim, filename, plot=True, parallel=False):
    """
    Save the epsilon map from a MEEP simulation.
    
    Parameters:
    -----------
    sim : mp.Simulation
        The MEEP simulation object
    filename : str
        Base filename for saving the epsilon map
    plot : bool
        Whether to generate and save a plot of the epsilon map
    parallel : bool
        Whether the code is running in parallel mode
    """
    print("Saving epsilon map...")
    eps_data = sim.get_epsilon()
    
    # # Save to HDF5 file
    # if parallel:
    #     from mpi4py import MPI
    #     comm = MPI.COMM_WORLD
    #     if not h5py.get_config().mpi:
    #         raise ValueError("h5py was built without MPI support, can't use mpio driver")
        
    #     with h5py.File(f"{filename}_epsilon.h5", 'w', driver='mpio', comm=comm) as f:
    #         f.create_dataset('epsilon', data=eps_data)
    # else:
    #     # Only have rank 0 create the file
    #     from mpi4py import MPI
    #     comm = MPI.COMM_WORLD
    #     rank = comm.Get_rank()
        
    #     if rank == 0:
    #         with h5py.File(f"{filename}_epsilon.h5", 'w', libver='latest') as f:
    #             f.create_dataset('epsilon', data=eps_data, compression='gzip')
    #         print(f"Epsilon data saved to {filename}_epsilon.h5")
            
    #         # Optionally plot and save the epsilon map
    #         if plot:
    #             plt.figure(figsize=(10, 8), dpi=150)
    #             plt.imshow(eps_data.transpose(), interpolation='spline36', cmap='viridis')
    #             plt.colorbar(label='ε (epsilon)')
    #             plt.title('Simulation Epsilon Map')
    #             plt.savefig(f"{filename}_epsilon.png")
    #             plt.close()
    #             print(f"Epsilon plot saved to {filename}_epsilon.png")
    #     comm.barrier()  # Ensure all processes wait for file creation
    
    # return eps_data

    # # Save to NPZ file instead of HDF5
    # if parallel:
    #     from mpi4py import MPI
    #     comm = MPI.COMM_WORLD
    #     rank = comm.Get_rank()
        
    #     if rank == 0:
    #         np.savez_compressed(f"{filename}_epsilon.npz", epsilon=eps_data)
    #         print(f"Epsilon data saved to {filename}_epsilon.npz")
            
    #         if plot:
    #             plt.figure(figsize=(10, 8), dpi=150)
    #             plt.imshow(eps_data.transpose(), interpolation='spline36', cmap='viridis')
    #             plt.colorbar(label='ε (epsilon)')
    #             plt.title('Simulation Epsilon Map')
    #             plt.savefig(f"{filename}_epsilon.png")
    #             plt.close()
    #             print(f"Epsilon plot saved to {filename}_epsilon.png")
    #     comm.barrier()
    # else:
    #     from mpi4py import MPI
    #     comm = MPI.COMM_WORLD
    #     rank = comm.Get_rank()
        
    #     if rank == 0:
    #         np.savez_compressed(f"{filename}_epsilon.npz", epsilon=eps_data)
    #         print(f"Epsilon data saved to {filename}_epsilon.npz")
            
    #         if plot:
    #             plt.figure(figsize=(10, 8), dpi=150)
    #             plt.imshow(eps_data.transpose(), interpolation='spline36', cmap='viridis')
    #             plt.colorbar(label='ε (epsilon)')
    #             plt.title('Simulation Epsilon Map')
    #             plt.savefig(f"{filename}_epsilon.png")
    #             plt.close()
    #             print(f"Epsilon plot saved to {filename}_epsilon.png")
    #     comm.barrier()

    np.savez_compressed(f"{filename}_epsilon.npz", epsilon=eps_data)
    print(f"Epsilon data saved to {filename}_epsilon.npz")
    
    return eps_data


#^###### ####### ####### ####### ####### ####### ####### ####### ####### ####### ####### ####### ####### ####### ####### ####### ####### #######
#*###### ####### ####### ####### ####### ####### ####### ####### ####### ####### ####### ####### ####### ####### ####### ####### ####### #######
#!###### ####### ####### ####### ####### ####### ####### ####### ####### ####### ####### ####### ####### ####### ####### ####### ####### #######

# Functions for post analysis with GRASP, CST, MEEPSAT!
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import h5py
import scipy.optimize as sc
import pandas as pd
import json
from scipy import ndimage


def meepsat_beam_plotting(fftfreq, 
            FFTs, 
            wvl,
            aper_size,
            deg_range = 20,
            ylim = -60, 
            symmetric_beam = True,
            legend = None,
            print_solid_angle = False,
            print_fwhm = False,
            savefig = False,
            path_name = None,
            seq_col = False):
    '''
    Plots far field beam

    Arguments
    ---------
    fftfreq : float
        Array of the frequencies of the FFT
    FFTS : float
        List of the normalized beams of the FFT
    wvl : float or list of floats
        Wavelengths of the beams 
    deg_range : float, optional
        Range in degrees of the plot (default : 20)
    ylim : float, optional
        Min amplitude of the plot, in dB (default : -60)
    symmetric_beam : bool, optional
        If the beam is symmetric, if true only plots half of the beam
        (default : True)
    legend : list of str, optional
        Legend of the various far fields plotted (default : None)
    print_solid_angle : bool, optional
        Whether to print the solid angle (default : False)
    print_fwhm : bool, optional
        Whether to print the best fit gaussian FWHM (default : False)
    savefig : bool, optional
        Whether to save the figure (default : False)
    path_name : str, optional
        Path and name of the plot to be saved 
        (default : 'plots/meep_guide_plot')
    seq_col : bool, optional
        Whether to set a sequential colormap (default : False)
    '''

    deg = np.arctan(fftfreq*wvl)*180/np.pi
    rads = np.array(deg) * np.pi/180

    col = plt.cm.jet(np.linspace(0,1,len(FFTs))) 
    
    angle_array_meepsat = []; power_dB_array_meepsat = []

    plt.figure(figsize = (8,6))
    
    def gaussian(x, stddev, mean):
        return np.exp(-(((x-mean)/4/stddev)**2))
    
    for k in range(len(FFTs)):

        fft_k = (FFTs[k] / (np.cos(rads)**2))**2
        fft_k = fft_k / np.max(fft_k)
        fft_dB = 10*np.log10(fft_k)
        middle = int(len(fft_k)/2)

        #BEAM SOLID ANGLE CALCULATION
        if print_solid_angle :
            
            
            x_span = np.append(rads, 0)
            integrand = np.append(fft_k, fft_k[0])
            integrand *= np.sin(x_span)
            right_part = np.trapz(integrand[:middle], x = x_span[:middle])
            #left_part = np.trapz(integrand[middle:], x = x_span[middle:])
            solid_angle = right_part #+ left_part
            print('Beam n.{} solid angle : {:.3e} srads'.format(k, 
                solid_angle*2*np.pi))
        
        if legend is not None : 
            plt.plot(deg[:middle], fft_dB[:middle], 
                label = '{}'.format(legend[k]), color = col[k])

        if legend is None :

            plt.plot(deg[:middle], fft_dB[:middle], color = col[k])
            #TESTING, ignore this
            #plt.plot(self.sim.angles, self.sim.ffmeep)
        
        # Append data for potential further analysis
        angle_array_meepsat.append(deg[:middle])
        power_dB_array_meepsat.append(fft_dB[:middle])

        #BEST FIT GAUSSIAN FWHM
        if print_fwhm :

            #Fit is done around the gaussian portion of the beam
            maxidx = np.argmax(fft_k)
            if maxidx == len(fft_k) - 1:
                maxidx = 0
            i = 0
            while fft_k[maxidx + i] > fft_k[maxidx + i + 1] :
                i += 1

            xdata = deg[maxidx - i : maxidx + i ]
            ydata = fft_k[maxidx - i : maxidx + i ]
            if maxidx - i <= 0:
                xdata = np.concatenate((deg[maxidx - i:], deg[:maxidx + i]))
                ydata = np.concatenate((fft_k[maxidx - i:], fft_k[:maxidx + i]))
            p0 = [1,1]
            if maxidx <=10:
                p0 = [1,0]
            popt, psig = sc.curve_fit(gaussian, xdata, ydata, p0 = p0)
            fwhm = np.abs(4*popt[0]*np.sqrt(np.log(2)))
            fwhm_th = wvl/aper_size*180/np.pi
            print('Best fit Gaussian FWHM : {:.2f}deg'.format(2*fwhm))
            print('Theoretical FWHM : {:.2f}deg'.format(fwhm_th))
            gauss = gaussian(deg[:middle], popt[0], popt[1]) + 1e-10
            y = 10*np.log10(gauss)

            plt.plot(deg[:middle], y, linestyle = '--', color = col[k])
                                    #color = 'C{}'.format(int(k)))

    plt.ylim((ylim, 0))
    plt.xlabel('Angle [deg]', fontsize = 14)
    plt.ylabel('Power [dB]', fontsize = 14)
    plt.xticks(fontsize = 12)
    plt.yticks(fontsize = 12)

    if symmetric_beam :
        plt.xlim((0,deg_range))
    if not symmetric_beam : 
        plt.xlim((-deg_range, deg_range))

    if legend is not None :
        plt.legend(loc = 'upper right', fontsize = 12)

    #Additional plotting tools
    """
    fwhm = args.wvl*0.28648

    plt.vlines([-fwhm/2, fwhm/2], -100, 0, color = 'grey', linestyle = 'dashdot')
    plt.vlines([fwhm_fft[0]/2], -100, 0, color='grey', linestyle = '--', alpha = 0.7)
    plt.annotate('Expected FWHM : {:.2f}deg'.format(fwhm), 
        xy = (.25, .9), xycoords='figure fraction', color = 'grey')
    plt.annotate('Beam FWHM : {:.2f}deg'.format(fwhm_fft[0]), 
        xy = (.25, .87), xycoords='figure fraction', color = 'grey', alpha = 0.7)
    """

    #plt.annotate('Field FWHM : {:.2f}mm'.format(fwhm_ap[0]), 
    #    xy = (.1, .84), xycoords='figure fraction')
    plt.tight_layout()
    plt.show()
    if savefig :
        plt.savefig('{}.png'.format(path_name))
    plt.close()

    return angle_array_meepsat, power_dB_array_meepsat



def calculate_grasp_resolution(y_coords):
    """
    Calculate the no of points per mm for GRASP data.
    
    Parameters:
    -----------
        y_coords (np.ndarray): Array of y-coordinates from GRASP data.
    Returns:    
        float: Resolution in points per mm.
    """
    # Get unique coordinates and sort them
    y_unique = np.sort(np.unique(y_coords.astype(float)))
    
    # Get grid dimensions
    ny =  len(y_unique)
    
    # Calculate grid spacing
    dy = y_unique[1] - y_unique[0] if ny > 1 else 1.0
    print(f"GRASP data has {ny} points with spacing {dy} mm")
    
    # Calculate the average spacing between points
    avg_spacing = np.mean(dy)
    
    # Resolution is the inverse of spacing (points per mm)
    resolution = 1 / avg_spacing
    return resolution

def calculate_CST_resolution(y_coords):
    """
    Calculate the no of points per mm for CST data.
    
    Parameters:
    -----------
        y_coords (np.ndarray): Array of y-coordinates from CST data.
    Returns:    
        float: Resolution in points per mm.
    """
    # Get unique coordinates and sort them
    y_unique = np.sort(np.unique(y_coords.astype(float)))
    
    # Get grid dimensions
    ny =  len(y_unique)
    
    # Calculate grid spacing
    dy = y_unique[1] - y_unique[0] if ny > 1 else 1.0
    print(f"CST data has {ny} points with spacing {dy} mm")
    
    # Calculate the average spacing between points
    avg_spacing = np.mean(dy)
    
    # Resolution is the inverse of spacing (points per mm)
    resolution = 1 / avg_spacing
    return resolution


def grasp_far_field_fft(grasp_data, 
                        wavelength, 
                        aper_size,
                        zero_pad_beam=15, 
                        plot_label='GRASP_FFT'):
    import numpy as np
    from scipy import fft
    y_coords = grasp_data['y']
    grasp_resolution = calculate_grasp_resolution(y_coords)
    efield = np.sqrt(np.abs(grasp_data['Ex'])**2 + np.abs(grasp_data['Ey'])**2 + np.abs(grasp_data['Ez'])**2)
    print(f"GRASP Resolution: {grasp_resolution} points per mm")

    # Filtering the data for the aperture size
    if aper_size is not None:
        half_aper_size = aper_size / 2
        center_index = len(y_coords) // 2
        center_y = y_coords[center_index]
        efield = efield[center_index, :] if efield.ndim == 2 else efield
        aperture_mask = (y_coords >= (center_y - half_aper_size)) & (y_coords <= (center_y + half_aper_size))
        efield = efield[aperture_mask]
        y_coords = y_coords[aperture_mask]
        print(f"Data filtered to aperture size of {aper_size} mm. New data length: {len(efield)}")


    #! List of frequencies
    #fft_freq = np.fft.fftfreq(len(list_efields[0])*zero_pad_beam, d = 1/grasp_resolution) 
    fft_freq = np.fft.fftfreq(len(efield)*zero_pad_beam, d = 1/grasp_resolution)
    #! Shift the zero frequency component to the center
    fft_freq = np.fft.fftshift(fft_freq)

    # Calculate angles in degrees
    theta_rad = np.arctan(fft_freq * wavelength)
    theta_deg = theta_rad * (180 / np.pi)

    #! Calculate the FFTs of efield
    fft_efield = np.fft.fft(efield, n=len(efield)*zero_pad_beam)
    fft_efield = np.fft.fftshift(fft_efield)
    fft_efield_normalized = np.abs(fft_efield) / np.max(np.abs(fft_efield))

    # Convert to power in dB
    fft_power = fft_efield_normalized**2
    fft_power_dB = 10 * np.log10(fft_power / np.max(fft_power))

    grasp_fft_dict = {
        'angle': theta_deg,
        'power_dB': fft_power_dB,
        'plot_label': plot_label
    }

    return grasp_fft_dict


# Similar to grasp_far_field_fft but for MEEP data
def meepsat_far_field_fft(y_coords,
                          efield,
                          meep_resolution,
                          wavelength,
                          aper_size,
                          far_field_distance= None,
                          zero_pad_beam=15,
                          plot_label='MEEPSAT_FFT'):
    import numpy as np
    from scipy import fft
    print(f"MEEPSAT Resolution: {meep_resolution} points per mm")
    #Filtering the data for the aperture size
    if aper_size is not None:
        half_aper_size = aper_size / 2
        center_index = len(y_coords) // 2
        center_y = y_coords[center_index]
        aperture_mask = (y_coords >= (center_y - half_aper_size)) & (y_coords <= (center_y + half_aper_size))
        efield = efield[aperture_mask]
        y_coords = y_coords[aperture_mask]
        print(f"Data filtered to aperture size of {aper_size} mm. New data length: {len(efield)}")

    #! List of frequencies
    #fft_freq = np.fft.fftfreq(len(list_efields[0])*zero_pad_beam, d = 1/grasp_resolution) 
    fft_freq = np.fft.fftfreq(len(efield)*zero_pad_beam, d = 1/meep_resolution)
    #! Shift the zero frequency component to the center
    fft_freq = np.fft.fftshift(fft_freq)

    # Calculate angles in degrees
    theta_rad = np.arctan(fft_freq * wavelength)
    theta_deg = theta_rad * (180 / np.pi)

    #! Calculate the FFTs of efield
    fft_efield = np.fft.fft(efield, n=len(efield)*zero_pad_beam)
    fft_efield = np.fft.fftshift(fft_efield)
    fft_efield_normalized = np.abs(fft_efield) / np.max(np.abs(fft_efield))

    # Convert to power in dB
    fft_power = fft_efield_normalized**2
    fft_power_dB = 10 * np.log10(fft_power / np.max(fft_power))

    # Calculate far field y-coordinates if distance is provided
    if far_field_distance is not None:
        # y_ff = far_field_distance * tan(theta)
        y_coords_ff = far_field_distance * np.tan(theta_rad)
        
        meepsat_dict = {
            'angle': theta_deg,  # Keep angles for reference
            'y_coords_ff': y_coords_ff,  # New: far field y-coordinates
            'power_dB': fft_power_dB,
            'plot_label': plot_label,
            'far_field_distance': far_field_distance
        }
    else:
        # Original behavior: return angles
        meepsat_dict = {
            'angle': theta_deg,
            'power_dB': fft_power_dB,
            'plot_label': plot_label
        }

    return meepsat_dict


# CST fft calculation
def cst_far_field_fft(y_coords,
                      efield,
                      cst_resolution,
                      wavelength,
                      aper_size,
                      zero_pad_beam=15,
                      plot_label='CST_FFT'):
    
    import numpy as np
    from scipy import fft

    print(f"CST Resolution: {cst_resolution} points per mm")
    #Filtering the data for the aperture size
    if aper_size is not None:
        half_aper_size = aper_size / 2
        center_index = len(y_coords) // 2
        center_y = y_coords[center_index]
        aperture_mask = (y_coords >= (center_y - half_aper_size)) & (y_coords <= (center_y + half_aper_size))
        efield = efield[aperture_mask]
        y_coords = y_coords[aperture_mask]
        print(f"Data filtered to aperture size of {aper_size} mm. New data length: {len(efield)}")

    #! List of frequencies
    fft_freq = np.fft.fftfreq(len(efield)*zero_pad_beam, d = 1/cst_resolution)
    #! Shift the zero frequency component to the center
    fft_freq = np.fft.fftshift(fft_freq)
    # Calculate angles in degrees
    theta_rad = np.arctan(fft_freq * wavelength)
    theta_deg = theta_rad * (180 / np.pi)
    #! Calculate the FFTs of efield
    fft_efield = np.fft.fft(efield, n=len(efield)*zero_pad_beam)
    fft_efield = np.fft.fftshift(fft_efield)
    fft_efield_normalized = np.abs(fft_efield) / np.max(np.abs(fft_efield))

    # Convert to power in dB
    fft_power = fft_efield_normalized**2
    fft_power_dB = 10 * np.log10(fft_power / np.max(fft_power))

    cst_fft_dict = {
        'angle': theta_deg,
        'power_dB': fft_power_dB,
        'plot_label': plot_label
    }

    return cst_fft_dict



def fit_gaussian_main_beam(angle, powerdB, aper_size, wvl, threshold_dB=-20):
    """
    Modified to match exactly the behavior of print_fwhm in meepsat_beam_plotting
    """
    # Convert power from dB to linear scale (matching meepsat_beam_plotting)
    power_linear = 10**(powerdB/10)
    
    # Find the maximum index
    maxidx = np.argmax(power_linear)
    if maxidx == len(power_linear) - 1:
        maxidx = 0
    
    # Find the fitting range using the same logic as meepsat_beam_plotting
    i = 0
    while (maxidx + i < len(power_linear) - 1 and 
           power_linear[maxidx + i] > power_linear[maxidx + i + 1]):
        i += 1

    # Extract data for fitting (same range logic)
    xdata = angle[maxidx - i : maxidx + i]
    ydata = power_linear[maxidx - i : maxidx + i]
    
    # Handle wraparound case (same as meepsat_beam_plotting)
    if maxidx - i <= 0:
        xdata = np.concatenate((angle[maxidx - i:], angle[:maxidx + i]))
        ydata = np.concatenate((power_linear[maxidx - i:], power_linear[:maxidx + i]))
    
    # Set initial parameters (same logic as meepsat_beam_plotting)
    p0 = [1, 1]
    if maxidx <= 10:
        p0 = [1, 0]
    
    # Define the same Gaussian function as meepsat_beam_plotting
    def gaussian(x, stddev, mean):
        return np.exp(-(((x-mean)/4/stddev)**2))
    
    try:
        # Fit the Gaussian
        popt, pcov = sc.curve_fit(gaussian, xdata, ydata, p0=p0)
        
        # Calculate FWHM using the same formula as meepsat_beam_plotting
        fwhm = np.abs(4*popt[0]*np.sqrt(np.log(2)))
        final_fwhm = 2*fwhm  # Same as meepsat_beam_plotting
        
        print(f'Best fit Gaussian FWHM: {final_fwhm:.2f}deg')
        
        # Generate fitted curve for plotting (same method)
        gauss = gaussian(angle, popt[0], popt[1]) + 1e-10
        y_fitted_dB = 10*np.log10(gauss)
        
        # Calculate R² for the fit quality
        y_predicted = gaussian(xdata, popt[0], popt[1])
        ss_res = np.sum((ydata - y_predicted) ** 2)
        ss_tot = np.sum((ydata - np.mean(ydata)) ** 2)
        r_squared = 1 - (ss_res / ss_tot)
        
        # # Plot results
        # plt.figure(figsize=(10, 6))
        # plt.plot(angle, powerdB, 'b-', label='Original Data', linewidth=2)
        # plt.plot(xdata, 10*np.log10(ydata), 'ro', label='Fitting Range', markersize=4)
        # plt.plot(angle, y_fitted_dB, 'g--', 
        #         label=f'Gaussian Fit (FWHM={final_fwhm:.2f}°, R²={r_squared:.3f})', 
        #         linewidth=2)
        
        # plt.xlabel('Angle (deg)')
        # plt.ylabel('Power (dB)')
        # plt.legend()
        # plt.title('Gaussian Fit to Main Beam (Matching meepsat_beam_plotting)')
        # plt.grid(True, alpha=0.3)
        # plt.show()

        # Find the -3 dB points on the fitted curve to calculate HPBW
        half_power_level = np.max(y_fitted_dB) - 3
        indices_above_half_power = np.where(y_fitted_dB >= half_power_level)[0]
        if len(indices_above_half_power) >= 2:
            hpbw = angle[indices_above_half_power[-1]] - angle[indices_above_half_power[0]]
            print(f"Calculated HPBW from fitted curve: {hpbw:.3f} degrees")


        # Theoretical FWHM
        fwhm_th = wvl/aper_size*180/np.pi
        print(f"Theoretical FWHM: {fwhm_th:.2f}deg")
        
        return {
            'fwhm': final_fwhm,
            'hpbw': hpbw,
            'theoretical_fwhm': fwhm_th,
            'r_squared': r_squared,
            'fitted_parameters': popt,
            'fitting_range': (xdata[0], xdata[-1]),
            'fitted_curve_angles': angle,
            'fitted_curve_dB': y_fitted_dB
        }
        
    except Exception as e:
        print(f"Gaussian fitting failed: {e}")
        return None

def mask_aperture(coords_array, aper_size):
    """
    Masks the coordinates array to only include points within the aperture size.
    
    Parameters:
    -----------
        coords_array (np.ndarray): Array of coordinates (y or z)
        aper_size (float): Aperture size in mm
    
    Returns:
    --------
        np.ndarray: Masked coordinates array indices within the aperture
    """
    half_aper_size = aper_size / 2
    center_index = len(coords_array) // 2
    center_coord = coords_array[center_index]
    aperture_mask_indices = (coords_array >= (center_coord - half_aper_size)) & (coords_array <= (center_coord + half_aper_size))
    
    return aperture_mask_indices



def summary_plots(
    simulation_resolution,
    simulation_wvl,
    efield_files_pattern,
    poynting_vector_pattern,
    xyzw_coords_file,
    x_position,
    analysis,
    frequency_label="90 GHz",
    norm_factor=1,
    contour_levels = 10,
    contour_db_range = (-60, 0),
    contour_linear_range = (0, 1),
    comparision_CST_data=None,
    comparision_GRASP_data= None,
    comparision_CST_fft_data = None,
    comparision_GRASP_fft_data = None,
    aper_size = None,
    avg_aper_lim = None,
    zero_pad_beam = 15,
    gaussian_fit_main_beam = False,
    gaussian_fit_power_threshold = -30,
    average_source_power= None,
    savefig = False,
    show_plots = True,
    savename_suffix = ''
):
    """
    Loads multiple E-field files, calculates average field, and plots comparison with last timestep.
    Also compares with CST and GRASP data if provided.

    Far field profiles are also compared if GRASP and CST data is provided.
    
    Parameters:
    -----------
        simulation_resolution (float): Simulation resolution in pixels/mm
        simulation_wvl (float): Simulation wavelength in meep units (It should be in mm since our scaling is 1 meep unit = 1 mm)   
        efield_files_pattern (str): Pattern for E-field files (e.g., 'path/single_lens_testing-e-*.h5')
        xyzw_coords_file (str): Path to the .npz file with x, y, w coordinates
        x_position (float): X position (in mm) for the slice
        analysis (module): Analysis module with readHDF5 function
        frequency_label (str): Frequency label for plot titles
        norm_factor (float): Normalization factor for power calculations
        contour_levels (int): Number of contour levels for contour plots
        comparision_CST_data (dict): Dictionary containing CST data for comparison (optional)
                                    cst_data = {'ez_magnitude', 'ez_phase', 's_magnitude'}
        comparision_GRASP_data (dict): Dictionary or list of dictionaries containing GRASP data for comparision (optional)
                                    grasp_data = {'Ex', 'Ey', 'Ez', 'x', 'y', 'plot_label'}   
        comparision_CST_fft_data (dict): Dictionary containing CST FFT data for far field comparison (optional)
                                    cst_fft_data = {'angle', 'power_dB', 'plot_label'}
        comparision_GRASP_fft_data (dict or list of dicts): Dictionary or list of dictionaries containing GRASP FFT data for far field comparison (optional)
                                    grasp_fft_data = {'angle', 'power_dB', 'plot_label'}
        aper_size (float): Aperture size in mm for GRASP data (optional)
        zero_pad_beam (int): Zero padding factor for FFT beam calculation
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import glob
    
    if savefig:
        # Create an directory named as summary_plots_results (if doesn't exist)
        if not os.path.exists('summary_plots_results'):
            os.makedirs('summary_plots_results')
            
        if frequency_label:
            freq_label_clean = frequency_label.replace(" ", "_").replace("/", "-")
            plot_path = f'summary_plots_results/{freq_label_clean}'
        else:
            plot_path = 'summary_plots_results'
        print(f"Plots will be saved to: {plot_path}")

        if savename_suffix != '':
            plot_path = os.path.join(plot_path, savename_suffix)

        if not os.path.exists(plot_path):
            os.makedirs(plot_path)

    # Get all E-field files matching the pattern
    efield_files = sorted(glob.glob(efield_files_pattern))
    print(f"Found {len(efield_files)} E-field files")
    
    if len(efield_files) == 0:
        raise ValueError(f"No files found matching pattern: {efield_files_pattern}")

    #Get all Poynting vector files matching the pattern
    poynting_vector_files = sorted(glob.glob(poynting_vector_pattern))
    print(f"Found {len(poynting_vector_files)} Poynting vector files")

    if len(poynting_vector_files) == 0:
        raise ValueError(f"No files found matching pattern: {poynting_vector_pattern}")
    
    # Load coordinates
    xyzw_data = np.load(xyzw_coords_file)
    x_coords = xyzw_data['x_coords']
    y_coords = xyzw_data['y_coords']
    
    # Initialize arrays for averaging
    ez_magnitude_sum = None
    ez_phase_sum_cos = None
    ez_phase_sum_sin = None

    # Load and process all files
    for i, file_path in enumerate(efield_files):
        print(f"Processing file {i+1}/{len(efield_files)}: {file_path}")
        
        # Load E-field data
        e_field_data = analysis.readHDF5(file_path)
        ez_real = e_field_data['ez.r']
        ez_imag = e_field_data['ez.i']

        
        # Calculate magnitude and phase
        ez_magnitude = np.sqrt(ez_real**2 + ez_imag**2)
        ez_magnitude = np.transpose(ez_magnitude)
        
        ez_phase_radians = np.arctan2(ez_imag, ez_real)
        
        # Sum for averaging
        if ez_magnitude_sum is None:
            ez_magnitude_sum = ez_magnitude
            ez_phase_sum_cos = np.cos(ez_phase_radians).T
            ez_phase_sum_sin = np.sin(ez_phase_radians).T
        else:
            ez_magnitude_sum += ez_magnitude
            ez_phase_sum_cos += np.cos(ez_phase_radians).T
            ez_phase_sum_sin += np.sin(ez_phase_radians).T
    
    # Initialize arrays for Poynting vector averaging and last timestep
    poynting_vector_sum = None

    for i, file_path in enumerate(poynting_vector_files):
        print(f"Processing Poynting vector file {i+1}/{len(poynting_vector_files)}: {file_path}")
        
        # Load Poynting vector data
        poynting_data = analysis.readHDF5(file_path)
        sx = poynting_data['sx']
        sy = poynting_data['sy']
        sz = poynting_data['sz']
        total_poynting = np.abs(sx)**2 + np.abs(sy)**2 + np.abs(sz)**2
        total_poynting = np.sqrt(total_poynting)
        total_poynting = np.transpose(total_poynting)
        # Sum for averaging
        if poynting_vector_sum is None:
            poynting_vector_sum = total_poynting
        else:
            poynting_vector_sum += total_poynting

    # Average Poynting vector
    averaged_poynting_vector = poynting_vector_sum / len(poynting_vector_files)
    # Normalize averaged Poynting vector to its maximum
    averaged_poynting_vector /= np.max(averaged_poynting_vector)

    # Load last timestep for Poynting vector
    last_poynting_file = poynting_vector_files[-1]
    poynting_last = analysis.readHDF5(last_poynting_file)
    sx_last = poynting_last['sx']
    sy_last = poynting_last['sy']
    sz_last = poynting_last['sz']
    total_poynting_last = np.abs(sx_last)**2 + np.abs(sy_last)**2 + np.abs(sz_last)**2
    total_poynting_last = np.sqrt(total_poynting_last)
    total_poynting_last = np.transpose(total_poynting_last)
    # Normalize last timestep Poynting vector to its maximum
    total_poynting_last /= np.max(total_poynting_last)

    # Average and Last timestep Poynting vector divided by norm factor
    if norm_factor is not None:
        averaged_poynting_vector_divided_source = averaged_poynting_vector / norm_factor
        print(f"Averaged Poynting vector divided by norm factor shape: {averaged_poynting_vector_divided_source.shape}")
        total_poynting_last_divided_source = total_poynting_last / norm_factor
        print(f"Last timestep Poynting vector divided by norm factor shape: {total_poynting_last_divided_source.shape}")
    
    # Calculate averages
    ez_magnitude_avg = ez_magnitude_sum / len(efield_files)
    #!!!!!
    ez_magnitude_avg = ez_magnitude_avg / np.max(ez_magnitude_avg)  # Normalize to max
    ez_phase_avg_radians = np.arctan2(ez_phase_sum_sin / len(efield_files), 
                                      ez_phase_sum_cos / len(efield_files))
    ez_phase_avg_degrees = np.degrees(ez_phase_avg_radians)

    # Load last timestep for comparison
    last_file = efield_files[-1]
    e_field_last = analysis.readHDF5(last_file)
    ez_real_last = e_field_last['ez.r']
    ez_imag_last = e_field_last['ez.i']
    
    ez_magnitude_last = np.sqrt(ez_real_last**2)# + ez_imag_last**2)
    ez_magnitude_last = np.transpose(ez_magnitude_last)
    
    ez_phase_last_radians = np.arctan2(ez_imag_last, ez_real_last)
    ez_phase_last_degrees = np.degrees(ez_phase_last_radians)
    ez_phase_last_degrees = np.transpose(ez_phase_last_degrees)

    # Find slice index
    x_index = (np.abs(x_coords - x_position)).argmin()

    #!= Checking if CST data is provided for comparison
    if comparision_CST_data is not None:
        # Creating empty lists to hold CST data
        cst_efield = []; cst_s_mag = []
        for i in range(len(comparision_CST_data)):
            cst_efield_i = comparision_CST_data[i]['efield']
            cst_s_mag_i = comparision_CST_data[i]['s_mag']

            # Calculate CST E-field magnitude and phase
            cst_efield_i['ycoords'] = cst_efield_i['Y']
            cst_efield_i['Magnitude'] = np.sqrt(cst_efield_i['Re(Ey)']**2 + cst_efield_i['Im(Ey)']**2)
            cst_efield_i['Phase_degrees'] = -np.degrees(np.arctan2(cst_efield_i['Im(Ey)'], cst_efield_i['Re(Ey)']))
            # cst_efield_i['Magnitude_dB'] = 20 * np.log10(cst_efield_i['Magnitude']) #/ np.max(cst_efield_i['Magnitude']))

            # Calculate CST S magnitude in dB
            cst_s_mag_i['S_Mag_dB'] = 10 * np.log10(cst_s_mag_i['S_Mag_linear'] / np.max(cst_s_mag_i['S_Mag_linear']))

            # # Plot CST Magnitude for verification
            # plt.figure()
            # plt.plot(cst_efield_i['ycoords'], cst_efield_i['Magnitude'], label = 'Raw CST E-field Magnitude')
            # plt.xlabel('y (mm)')
            # plt.ylabel('E-field Magnitude')
            # plt.title(f'CST E-field Magnitude Slice at x={x_position} mm ({frequency_label})')
            # plt.legend()
            # plt.grid()
            # plt.show()
            
            #* Mask out only for the aperture size 
            cst_aperture_indices = mask_aperture(cst_efield_i['ycoords'], aper_size)

            #* Then mask out and calculate the average between -avg_aper_lim to +avg_aper_lim
            cst_norm_avg_aper_lim_indices = mask_aperture(cst_efield_i['ycoords'], avg_aper_lim)
            
            # Apply mask to CST E-field data
            cst_efield_i['ycoords'] = cst_efield_i['ycoords'][cst_aperture_indices]
            cst_efield_i['Magnitude'] = cst_efield_i['Magnitude'][cst_aperture_indices]/np.mean(cst_efield_i['Magnitude'][cst_norm_avg_aper_lim_indices])
            cst_efield_i['Phase_degrees'] = cst_efield_i['Phase_degrees'][cst_aperture_indices]
            cst_efield_i['Magnitude_dB'] = 20 * np.log10(cst_efield_i['Magnitude'] + 1e-12)  # Avoid log(0)

            # Apply mask to CST S magnitude data
            cst_s_mag_i['S_Mag_dB'] = cst_s_mag_i['S_Mag_dB'][cst_aperture_indices]

            # # Plot after masking
            # plt.figure()
            # plt.plot(cst_efield_i['ycoords'], cst_efield_i['Magnitude'], label = 'Masked CST E-field Magnitude')
            # plt.xlabel('y (mm)')
            # plt.ylabel('E-field Magnitude')
            # plt.title(f'CST E-field Magnitude Slice at x={x_position} mm ({frequency_label}) - Masked')
            # plt.legend()
            # plt.grid()  
            # plt.show()

            cst_efield.append(cst_efield_i)
            cst_s_mag.append(cst_s_mag_i)
    
    #!= Checking if GRASP data is provided for comparison
    if comparision_GRASP_data is not None:
        # Creating empty lists to hold GRASP data
        ez_grasp = []; ex_grasp = []; ey_grasp = []; e_grasp = []; phase_grasp = []; 
        grasp_efield_magnitude = []; grasp_efield_magnitude_dB = []; grasp_phase_slice = []; y_grasp = []

        for comparision_GRASP_data_i in comparision_GRASP_data:
            ez_grasp_arr = comparision_GRASP_data_i['Ez']
            ex_grasp_arr = comparision_GRASP_data_i['Ex']
            ey_grasp_arr = comparision_GRASP_data_i['Ey']
            e_grasp_arr = np.abs(ex_grasp_arr) #np.sqrt(np.abs(ex_grasp_arr**2 + ey_grasp_arr**2 + ez_grasp_arr**2))
            y_grasp_arr = comparision_GRASP_data_i['x']

            phase_grasp_arr = np.angle(ex_grasp_arr)

            # Extract middle row and convert to dB
            mid_row_index = e_grasp_arr.shape[0] // 2
            grasp_efield_magnitude_arr = e_grasp_arr[mid_row_index, :]
            # grasp_efield_magnitude_dB_arr = 20 * np.log10(grasp_efield_magnitude_arr/np.max(grasp_efield_magnitude_arr) + 1e-12)  # Avoid log(0)
            grasp_phase_slice_arr = np.degrees(phase_grasp_arr[mid_row_index, :])

            # plt.plot(y_grasp_arr, grasp_efield_magnitude_arr, label=comparision_GRASP_data_i['plot_label'])
            # plt.xlabel('y (mm)')
            # plt.ylabel('E-field Magnitude (dB)')
            # plt.title(f'GRASP E-field Magnitude Slice at x={x_position} mm ({frequency_label})')
            # plt.legend()
            # plt.grid()
            # plt.show()

            #* Mask out only for the aperture size 
            grasp_aperture_indices = mask_aperture(y_grasp_arr, aper_size)

            #* Mask out and calculate the average between -avg_aper_lim to +avg_aper_lim
            grasp_norm_avg_aper_lim_indices = mask_aperture(y_grasp_arr, avg_aper_lim)

            # Normalize GRASP E-field magnitude to the average within avg_aper_lim
            grasp_efield_magnitude_arr = grasp_efield_magnitude_arr/np.mean(grasp_efield_magnitude_arr[grasp_norm_avg_aper_lim_indices])
            grasp_efield_magnitude_dB_arr = 20 * np.log10(grasp_efield_magnitude_arr + 1e-12)  # Avoid log(0)
            
            # Apply mask to all GRASP data arrays
            ez_grasp_arr = ez_grasp_arr[:, grasp_aperture_indices]
            ex_grasp_arr = ex_grasp_arr[:, grasp_aperture_indices]
            ey_grasp_arr = ey_grasp_arr[:, grasp_aperture_indices]
            e_grasp_arr = e_grasp_arr[:, grasp_aperture_indices]
            phase_grasp_arr = phase_grasp_arr[:, grasp_aperture_indices]
            grasp_efield_magnitude_arr = grasp_efield_magnitude_arr[grasp_aperture_indices]
            grasp_efield_magnitude_dB_arr = grasp_efield_magnitude_dB_arr[grasp_aperture_indices]
            grasp_phase_slice_arr = grasp_phase_slice_arr[grasp_aperture_indices]
            y_grasp_arr = y_grasp_arr[grasp_aperture_indices]

            # # Plot again after masking
            # plt.plot(y_grasp_arr, grasp_efield_magnitude_arr, label=comparision_GRASP_data_i['plot_label'] + ' (Masked)')
            # plt.xlabel('y (mm)')
            # plt.ylabel('E-field Magnitude (dB)')
            # plt.title(f'GRASP E-field Magnitude Slice at x={x_position} mm  ({frequency_label}) - Masked')
            # plt.legend()
            # plt.grid()  
            # plt.show()

            # Append everything to lists for potential multiple GRASP datasets
            ez_grasp.append(ez_grasp_arr); ex_grasp.append(ex_grasp_arr); ey_grasp.append(ey_grasp_arr); e_grasp.append(e_grasp_arr); phase_grasp.append(phase_grasp_arr)
            grasp_efield_magnitude.append(grasp_efield_magnitude_arr); grasp_efield_magnitude_dB.append(grasp_efield_magnitude_dB_arr); 
            grasp_phase_slice.append(grasp_phase_slice_arr); y_grasp.append(y_grasp_arr)
    

    #!= MEEPSAT aperture masking
    #! Extract slices for plotting and also calculate the Far field beams from those slices 
    aperture_slice_avg = ez_magnitude_avg[:, x_index]#/np.max(ez_magnitude_avg[:, x_index])
    aperture_slice_last = ez_magnitude_last[:, x_index]#/np.max(ez_magnitude_last[:, x_index])
    y_meep = y_coords

    # Mask out and calculate the average between -avg_aper_lim to +avg_aper_lim
    meep_norm_avg_aper_lim_indices = mask_aperture(y_meep, avg_aper_lim)
    aperture_slice_avg = aperture_slice_avg/np.mean(aperture_slice_avg[meep_norm_avg_aper_lim_indices])
    aperture_slice_last = aperture_slice_last/np.mean(aperture_slice_last[meep_norm_avg_aper_lim_indices])

    # Aperture masking
    meep_aperture_indices = mask_aperture(y_meep, aper_size)
    aperture_slice_avg = aperture_slice_avg[meep_aperture_indices]
    aperture_slice_last = aperture_slice_last[meep_aperture_indices]
    y_meep = y_meep[meep_aperture_indices]

    #~ CST FFB
    if comparision_CST_data is not None:
        angle_array_cst = []; power_dB_array_cst = []
        cst_resolution_list = []
        for i in range(len(comparision_CST_data)):
            # Calculate CST Resolution
            cst_resolution = calculate_CST_resolution(y_coords= cst_efield[i]['ycoords'])
            print(f"Calculated CST resolution for {comparision_CST_data[i]['plot_label']}: {cst_resolution} points/mm")

            # Use cst_far_field_fft function to calculate the far field from CST aperture slice
            cst_fft_dict = cst_far_field_fft(y_coords = cst_efield[i]['ycoords'],
                                             efield = cst_efield[i]['Magnitude'],
                                             cst_resolution = cst_resolution,
                                             wavelength = simulation_wvl,
                                             aper_size = aper_size,
                                             zero_pad_beam = zero_pad_beam,
                                             plot_label = comparision_CST_data[i]['plot_label'] + f" ({frequency_label})")
            
            angle_array_cst_i, power_dB_array_cst_i = cst_fft_dict['angle'], cst_fft_dict['power_dB']
            cst_resolution_list.append(cst_resolution)
            # Append data for plotting
            angle_array_cst.append(angle_array_cst_i)
            power_dB_array_cst.append(power_dB_array_cst_i)
    #~ GRASP FFB
    if comparision_GRASP_data is not None:
        # Initialize lists to hold the calculated far field data from GRASP
        angle_array_grasp_list = []; power_dB_array_grasp_list = []
        # Initialize the efield_data_meep_list and y_coords_meep_list, grasp_resolution_list
        # efield_data_meep_list = []; y_coords_meep_list = []; 
        grasp_resolution_list = [] 
        for i in range(len(comparision_GRASP_data)):
            # Calculate the GRASP resolution
            print(f"y_grasp_{comparision_GRASP_data[i]['plot_label']}", y_grasp[i])
            grasp_resolution = calculate_grasp_resolution(y_grasp[i])
            print(f"Calculated GRASP resolution for {comparision_GRASP_data[i]['plot_label']}: {grasp_resolution} points/mm")

            # Use grasp_far_field_fft function to calculate the far field from GRASP aperture slice
            grasp_fft_dict = grasp_far_field_fft(grasp_data = comparision_GRASP_data[i],
                                                wavelength = simulation_wvl,
                                                aper_size = aper_size,
                                                zero_pad_beam = zero_pad_beam,
                                                plot_label = comparision_GRASP_data[i]['plot_label'] + f" ({frequency_label})")
            angle_array_grasp, power_dB_array_grasp = grasp_fft_dict['angle'], grasp_fft_dict['power_dB']
            grasp_resolution_list.append(grasp_resolution)

            # Append data for plotting
            angle_array_grasp_list.append(angle_array_grasp)
            power_dB_array_grasp_list.append(power_dB_array_grasp)

    #~ MEEPSAT FFB
    #! First calculate the far field from the aperture slice avg using meepsat_far_field_fft function
    meepsat_fft_dict = meepsat_far_field_fft(y_coords = y_meep,
                                             efield = aperture_slice_avg,
                                             meep_resolution = simulation_resolution,
                                             wavelength = simulation_wvl,
                                             aper_size = aper_size,
                                             zero_pad_beam = zero_pad_beam,
                                             plot_label = f"MEEPSAT ({frequency_label})")

    angle_array_meepsat, power_dB_array_meepsat = meepsat_fft_dict['angle'], meepsat_fft_dict['power_dB']
    print("Shape of MEEPSAT far field angle array:", angle_array_meepsat.shape)
    print("Shape of MEEPSAT far field power dB array:", power_dB_array_meepsat.shape)

    #! Plot 1: Aperture Magnitude slice comparison
    plt.figure(figsize=(10, 6))

    # Plot for MEEPSAT
    plt.plot(y_meep, 10 * np.log10(aperture_slice_avg / norm_factor), 
             'b-', label='MeepSAT Time AvG', linewidth=2)
    
    # Saving the aperture slice avg as a npz file
    np.savez_compressed(f'{plot_path}/meepsat_aperture_slice_avg_{frequency_label.replace(" ", "_").replace("/", "-")}_{savename_suffix}.npz',
                        y_coords = y_meep,
                        aperture_slice_avg_dB = 10 * np.log10(aperture_slice_avg / norm_factor)
                        )

    # Plot for GRASP if provided
    if comparision_GRASP_data is not None:
        for i in range(len(comparision_GRASP_data)):
            plt.plot(y_grasp[i], grasp_efield_magnitude_dB[i], label='GRASP ({})'.format(comparision_GRASP_data[i]['plot_label']), linewidth=2)

    # Plot for CST if provided
    if comparision_CST_data is not None:
        for i in range(len(comparision_CST_data)):
            plt.plot(cst_efield[i]['Y'], cst_efield[i]['Magnitude_dB'], 'g-.', label='CST Data', linewidth=2)

    #plt.ylim(-30, 0)
    plt.xlabel('Y (mm)')
    plt.ylabel('Power (dB)')
    plt.title(f'E-field Magnitude Slice Comparison at x = {x_coords[x_index]:.1f} mm ({frequency_label})')
    plt.legend()
    plt.grid()

    if savefig:
        plt.savefig(f'{plot_path}/efield_magnitude_slice_comparison_{frequency_label.replace(" ", "_").replace("/", "-")}_{savename_suffix}.png', dpi=300)
    if show_plots:
        plt.show()

    #! Plot 1(b): Far Field comparision from the aperture_slice_avg using plotting function
    plt.figure(figsize=(10, 6))

    # Plot for MEEPSAT
    plt.plot(angle_array_meepsat, power_dB_array_meepsat, 'b-', label=meepsat_fft_dict['plot_label'], linewidth=2)

    # Plot for GRASP if provided
    if comparision_GRASP_data is not None:
        for i in range(len(angle_array_grasp_list)):
            print(f"Plotting far field comparison for {comparision_GRASP_data[i]['plot_label']}")
            plt.plot(angle_array_grasp_list[i], power_dB_array_grasp_list[i], label='{}'.format(comparision_GRASP_data[i]['plot_label']), linewidth=2, linestyle='-.')
            # plt.plot(angle_array_meepsat_list[i], power_dB_array_meepsat_list[i], 'b--', label='MeepSAT (Res: {})'.format(comparision_GRASP_data[i]['plot_label']), linewidth=2)

    # Plot for CST if provided
    if comparision_CST_data is not None:
        for i in range(len(angle_array_cst)):
            print(f"Plotting far field comparison for {comparision_CST_data[i]['plot_label']}")
            plt.plot(angle_array_cst[i], power_dB_array_cst[i], label='{}'.format(comparision_CST_data[i]['plot_label']), linewidth=2, linestyle=(0, (1, 1)))

    plt.xlabel('Angle (degrees)')
    plt.ylabel('Power (dB)')
    plt.title(f'Far Field Comparison from Aperture Slice Avg ({frequency_label})')
    plt.ylim(-60, 0)
    plt.xlim(0, 20)
    plt.legend()
    plt.grid()

    if savefig:
        plt.savefig(f'{plot_path}/far_field_comparison_aperture_slice_avg_{frequency_label.replace(" ", "_").replace("/", "-")}_{savename_suffix}.png', dpi=300)
    if show_plots:
        plt.show()
    

    #! Plot 1(c): Far Field comparision between FFT(GRASP APERTURE PROFILE)
    plt.figure(figsize=(10, 6))
    if comparision_GRASP_data is not None:
        for i in range(len(angle_array_grasp_list)):
            print(f"Plotting far field comparison for {comparision_GRASP_data[i]['plot_label']}")
            plt.plot(angle_array_grasp_list[i], power_dB_array_grasp_list[i], label='{}'.format(comparision_GRASP_data[i]['plot_label']), linewidth=2, linestyle='-.')
    
    plt.xlabel('Angle (degrees)')
    plt.ylabel('Power (dB)')
    plt.title(f'Far Field Comparison between GRASP Datasets ({frequency_label})')
    plt.ylim(-60, 0)
    plt.xlim(0, 20)
    plt.legend()
    plt.grid()

    if savefig:
        plt.savefig(f'{plot_path}/far_field_comparison_grasp_datasets_{frequency_label.replace(" ", "_").replace("/", "-")}_{savename_suffix}.png', dpi=300)
    if show_plots:
        plt.show()

    #! Plot 1(d): Far Field comparision between FFT(CST APERTURE PROFILE)
    plt.figure(figsize=(10, 6))
    if comparision_CST_data is not None:
        for i in range(len(angle_array_cst)):
            print(f"Plotting far field comparison for {comparision_CST_data[i]['plot_label']}")
            plt.plot(angle_array_cst[i], power_dB_array_cst[i], label='{}'.format(comparision_CST_data[i]['plot_label']), linewidth=2, linestyle=(0, (1, 1)))

    plt.xlabel('Angle (degrees)')
    plt.ylabel('Power (dB)')
    plt.title(f'Far Field Comparison between CST Datasets ({frequency_label})')
    plt.ylim(-60, 0)
    plt.xlim(0, 20)
    plt.legend()
    plt.grid()

    if savefig:
        plt.savefig(f'{plot_path}/far_field_comparison_cst_datasets_{frequency_label.replace(" ", "_").replace("/", "-")}_{savename_suffix}.png', dpi=300)
    if show_plots:
        plt.show()

    #! Plot 1(e): Far Field comparision between GRASP datasets of comparision_GRASP_fft_data if provided
    if comparision_GRASP_fft_data is not None:
        plt.figure(figsize=(10, 6))
        for i in range(len(comparision_GRASP_fft_data)):
            print(f"Plotting far field comparison for {comparision_GRASP_fft_data[i]['plot_label']}")
            plt.plot(comparision_GRASP_fft_data[i]['angle'], comparision_GRASP_fft_data[i]['power_dB'], label='{}'.format(comparision_GRASP_fft_data[i]['plot_label']), linewidth=2, linestyle='-.')

        plt.xlabel('Angle (degrees)')
        plt.ylabel('Power (dB)')
        plt.title(f'Far Field Comparison between FFB (calculated in GRASP) Datasets ({frequency_label})')
        plt.ylim(-60, 0)
        plt.xlim(0, 20)
        plt.legend()
        plt.grid()

        if savefig:
            plt.savefig(f'{plot_path}/far_field_comparison_grasp_ffb_datasets_{frequency_label.replace(" ", "_").replace("/", "-")}_{savename_suffix}.png', dpi=300)
        if show_plots:
            plt.show()

    #! Plot 1(f): Far Field comparision between CST datasets of comparision_CST_fft_data if provided
        if comparision_CST_fft_data is not None:
            plt.figure(figsize=(10, 6))
            for i in range(len(comparision_CST_fft_data)):
                print(f"Plotting far field comparison for {comparision_CST_fft_data[i]['plot_label']}")
                plt.plot(comparision_CST_fft_data[i]['angle'], comparision_CST_fft_data[i]['power_dB'], label='{}'.format(comparision_CST_fft_data[i]['plot_label']), linewidth=2, linestyle=(0, (1, 1)))

            plt.xlabel('Angle (degrees)')
            plt.ylabel('Power (dB)')
            plt.title(f'Far Field Comparison between FFB (calculated in CST) Datasets ({frequency_label})')
            plt.ylim(-60, 0)
            plt.xlim(0, 20)
            plt.legend()
            plt.grid()

            if savefig:
                plt.savefig(f'{plot_path}/far_field_comparison_cst_ffb_datasets_{frequency_label.replace(" ", "_").replace("/", "-")}_{savename_suffix}.png', dpi=300)
            if show_plots:
                plt.show()

    #! Plot 1(g): Far Field comparision between FFB(calculated in GRASP) and FFB(calculated in CST) datasets if ONE OF them is provided
    if (comparision_GRASP_fft_data is not None) or (comparision_CST_fft_data is not None):
        plt.figure(figsize=(10, 6))
        if comparision_GRASP_fft_data is not None:
            for i in range(len(comparision_GRASP_fft_data)):
                print(f"Plotting far field comparison for {comparision_GRASP_fft_data[i]['plot_label']}")
                plt.plot(comparision_GRASP_fft_data[i]['angle'], comparision_GRASP_fft_data[i]['power_dB'], label='{}'.format(comparision_GRASP_fft_data[i]['plot_label']), linewidth=2, linestyle='-.')
        if comparision_CST_fft_data is not None:
            for i in range(len(comparision_CST_fft_data)):
                print(f"Plotting far field comparison for {comparision_CST_fft_data[i]['plot_label']}")
                plt.plot(comparision_CST_fft_data[i]['angle'], comparision_CST_fft_data[i]['power_dB'], label='{}'.format(comparision_CST_fft_data[i]['plot_label']), linewidth=2, linestyle=(0, (1, 1)))

        plt.xlabel('Angle (degrees)')
        plt.ylabel('Power (dB)')
        plt.title(f'Far Field Comparison between FFB (calculated in GRASP and CST) Datasets ({frequency_label})')
        plt.ylim(-60, 0)
        plt.xlim(0, 20)
        plt.legend()
        plt.grid()

        if savefig:
            plt.savefig(f'{plot_path}/far_field_comparison_grasp_cst_ffb_datasets_{frequency_label.replace(" ", "_").replace("/", "-")}_{savename_suffix}.png', dpi=300)
        if show_plots:
            plt.show()

    #! Plot 1(h): Far Field comparision between FFB(calculated in GRASP), FFB(calculated in CST) and Plot 1(b) FFB calculated for all datasets
    if (comparision_GRASP_data is not None) or (comparision_CST_data is not None):
        plt.figure(figsize=(10, 6))
        # Plot for MEEPSAT
        plt.plot(angle_array_meepsat, power_dB_array_meepsat, 'b-', label=meepsat_fft_dict['plot_label'], linewidth=2)
        if comparision_GRASP_data is not None:
            for i in range(len(angle_array_grasp_list)):
                print(f"Plotting far field comparison for {comparision_GRASP_data[i]['plot_label']}")
                plt.plot(angle_array_grasp_list[i], power_dB_array_grasp_list[i], label='{}'.format(comparision_GRASP_data[i]['plot_label']), linewidth=2, linestyle='-.', marker='x')
        if comparision_CST_data is not None:
            for i in range(len(angle_array_cst)):
                print(f"Plotting far field comparison for {comparision_CST_data[i]['plot_label']}")
                plt.plot(angle_array_cst[i], power_dB_array_cst[i], label='{}'.format(comparision_CST_data[i]['plot_label']), linewidth=1.5, linestyle=(0, (1, 1)), marker='o')
        
        if comparision_GRASP_fft_data is not None:
            for i in range(len(comparision_GRASP_fft_data)):
                print(f"Plotting far field comparison for {comparision_GRASP_fft_data[i]['plot_label']}")
                plt.plot(comparision_GRASP_fft_data[i]['angle'], comparision_GRASP_fft_data[i]['power_dB'], label='{}'.format(comparision_GRASP_fft_data[i]['plot_label']), marker='x', linewidth=1, linestyle='-.')
        
        if comparision_CST_fft_data is not None:
            for i in range(len(comparision_CST_fft_data)):
                print(f"Plotting far field comparison for {comparision_CST_fft_data[i]['plot_label']}")
                plt.plot(comparision_CST_fft_data[i]['angle'], comparision_CST_fft_data[i]['power_dB'], label='{}'.format(comparision_CST_fft_data[i]['plot_label']), marker='o', linewidth=1, linestyle=(0, (1, 1)))

        plt.xlabel('Angle (degrees)')
        plt.ylabel('Power (dB)')
        plt.title(f'Far Field Comparison between FFB(GRASP and CST) and FFT(CST, GRASP, MeepSAT) Datasets ({frequency_label})')
        plt.ylim(-60, 0)
        plt.xlim(0, 20)
        plt.legend()
        plt.grid()

        if savefig:
            plt.savefig(f'{plot_path}/far_field_comparison_grasp_cst_ffb_and_all_datasets_{frequency_label.replace(" ", "_").replace("/", "-")}_{savename_suffix}.png', dpi=300)
        if show_plots:
            plt.show()

    #! Plot 1(i): Gaussian Fit to the main beam of FFB from MEEPSAT, GRASP and CST if provided
    if gaussian_fit_main_beam:
        meepsat_fitting_results = fit_gaussian_main_beam(angle=angle_array_meepsat, 
                                                         powerdB=power_dB_array_meepsat, 
                                                         aper_size= aper_size,
                                                         wvl=simulation_wvl,
                                                         threshold_dB=gaussian_fit_power_threshold)
        
        print("MEEPSAT Gaussian Fit Results:", meepsat_fitting_results)
        # Plot MEEPSAT fit
        plt.figure(figsize=(10, 6))
        plt.plot(angle_array_meepsat, power_dB_array_meepsat, 'b-', label='MEEPSAT FFB', linewidth=2)
        if meepsat_fitting_results is not None:
            plt.plot(meepsat_fitting_results['fitted_curve_angles'], meepsat_fitting_results['fitted_curve_dB'], 'r--', label=f"MeepSAT;FWHM: {meepsat_fitting_results['fwhm']:.2f} deg, FWHM(Th): {meepsat_fitting_results['theoretical_fwhm']:.2f} deg, HPBW: {meepsat_fitting_results['hpbw']:.2f} deg, R²: {meepsat_fitting_results['r_squared']:.3f}", 
                     linewidth=2, alpha=0.7)
            
        # Plot GRASP fit if provided
        if comparision_GRASP_data is not None:
            for i in range(len(angle_array_grasp_list)):
                grasp_fitting_results = fit_gaussian_main_beam(angle=angle_array_grasp_list[i], 
                                                               powerdB=power_dB_array_grasp_list[i], 
                                                               aper_size= aper_size,
                                                               wvl=simulation_wvl,
                                                               threshold_dB=gaussian_fit_power_threshold)
                # print(f"GRASP ({comparision_GRASP_data[i]['plot_label']}) Gaussian Fit Results:", grasp_fitting_results)
                plt.plot(angle_array_grasp_list[i], power_dB_array_grasp_list[i], label='GRASP ({})'.format(comparision_GRASP_data[i]['plot_label']), linewidth=2, linestyle='-.', marker='x')
                if grasp_fitting_results is not None:
                    plt.plot(grasp_fitting_results['fitted_curve_angles'], grasp_fitting_results['fitted_curve_dB'], 'r--', label=f"GRASP ({comparision_GRASP_data[i]['plot_label']});FWHM: {grasp_fitting_results['fwhm']:.2f} deg, FWHM(Th): {grasp_fitting_results['theoretical_fwhm']:.2f} deg, HPBW: {grasp_fitting_results['hpbw']:.2f} deg, R²: {grasp_fitting_results['r_squared']:.3f}", 
                             linewidth=2, alpha=0.7)
                    
        # Plot CST fit if provided
        if comparision_CST_data is not None:
            for i in range(len(angle_array_cst)):
                cst_fitting_results = fit_gaussian_main_beam(angle=angle_array_cst[i], 
                                                             powerdB=power_dB_array_cst[i], 
                                                             aper_size= aper_size,
                                                             wvl=simulation_wvl,
                                                             threshold_dB=gaussian_fit_power_threshold)
                # print(f"CST ({comparision_CST_data[i]['plot_label']}) Gaussian Fit Results:", cst_fitting_results)
                plt.plot(angle_array_cst[i], power_dB_array_cst[i], label='CST ({})'.format(comparision_CST_data[i]['plot_label']), linewidth=1.5, linestyle=(0, (1, 1)), marker='o')
                if cst_fitting_results is not None:
                    plt.plot(cst_fitting_results['fitted_curve_angles'], cst_fitting_results['fitted_curve_dB'], 'r--', label=f"CST ({comparision_CST_data[i]['plot_label']});FWHM: {cst_fitting_results['fwhm']:.2f} deg, FWHM(Th): {cst_fitting_results['theoretical_fwhm']:.2f} deg, HPBW: {cst_fitting_results['hpbw']:.2f} deg, R²: {cst_fitting_results['r_squared']:.3f}", 
                             linewidth=2, alpha=0.7)
                    
        # Plot if comparision_GRASP_fft_data is provided
        if comparision_GRASP_fft_data is not None:
            for i in range(len(comparision_GRASP_fft_data)):
                grasp_fft_fitting_results = fit_gaussian_main_beam(angle=comparision_GRASP_fft_data[i]['angle'], 
                                                                   powerdB=comparision_GRASP_fft_data[i]['power_dB'], 
                                                                   aper_size= aper_size,
                                                                   wvl=simulation_wvl,
                                                                   threshold_dB=gaussian_fit_power_threshold)
                # print(f"GRASP FFB ({comparision_GRASP_fft_data[i]['plot_label']}) Gaussian Fit Results:", grasp_fft_fitting_results)
                plt.plot(comparision_GRASP_fft_data[i]['angle'], comparision_GRASP_fft_data[i]['power_dB'], label='GRASP FFB ({})'.format(comparision_GRASP_fft_data[i]['plot_label']), linewidth=1, linestyle='-.', marker='x')
                if grasp_fft_fitting_results is not None:
                    plt.plot(grasp_fft_fitting_results['fitted_curve_angles'], grasp_fft_fitting_results['fitted_curve_dB'], 'r--', label=f"GRASP FFB ({comparision_GRASP_fft_data[i]['plot_label']});FWHM: {grasp_fft_fitting_results['fwhm']:.2f} deg, FWHM(Th): {grasp_fft_fitting_results['theoretical_fwhm']:.2f} deg, HPBW: {grasp_fft_fitting_results['hpbw']:.2f} deg, R²: {grasp_fft_fitting_results['r_squared']:.3f}", 
                             linewidth=2, alpha=0.7)
                    
        # Plot if comparision_CST_fft_data is provided
        if comparision_CST_fft_data is not None:
            for i in range(len(comparision_CST_fft_data)):
                cst_fft_fitting_results = fit_gaussian_main_beam(angle=comparision_CST_fft_data[i]['angle'], 
                                                                 powerdB=comparision_CST_fft_data[i]['power_dB'], 
                                                                 aper_size= aper_size,
                                                                 wvl=simulation_wvl,
                                                                 threshold_dB=gaussian_fit_power_threshold)
                # print(f"CST FFB ({comparision_CST_fft_data[i]['plot_label']}) Gaussian Fit Results:", cst_fft_fitting_results)
                plt.plot(comparision_CST_fft_data[i]['angle'], comparision_CST_fft_data[i]['power_dB'], label='CST FFB ({})'.format(comparision_CST_fft_data[i]['plot_label']), linewidth=1, linestyle=(0, (1, 1)), marker='o')
                if cst_fft_fitting_results is not None:
                    plt.plot(cst_fft_fitting_results['fitted_curve_angles'], cst_fft_fitting_results['fitted_curve_dB'], 'r--', label=f"CST FFB ({comparision_CST_fft_data[i]['plot_label']});FWHM: {cst_fft_fitting_results['fwhm']:.2f} deg, FWHM(Th): {cst_fft_fitting_results['theoretical_fwhm']:.2f} deg, HPBW: {cst_fft_fitting_results['hpbw']:.2f} deg, R²: {cst_fft_fitting_results['r_squared']:.3f}", 
                             linewidth=2, alpha=0.7)
                    
        plt.xlabel('Angle (degrees)')
        plt.ylabel('Power (dB)')
        plt.title(f'Far Field Gaussian Fit Comparison ({frequency_label})')
        plt.ylim(-60, 0)
        plt.xlim(0, 20)
        plt.legend()
        plt.grid()

        if savefig:
            plt.savefig(f'{plot_path}/far_field_gaussian_fit_comparison_{frequency_label.replace(" ", "_").replace("/", "-")}_{savename_suffix}.png', dpi=300)
        if show_plots:
            plt.show()

    #! Plot 2: Phase slice comparison
        phase_slice_avg = ez_phase_avg_degrees[:, x_index]
        phase_slice_last = ez_phase_last_degrees[:, x_index]
        # Subtract the centre value that corresponds to y=0
        indexy0 = (np.abs(y_coords - 0)).argmin()
        phase_slice_avg -= phase_slice_avg[indexy0]
        phase_slice_last -= phase_slice_last[indexy0]

        plt.figure(figsize=(10, 6))
        plt.plot(y_coords, phase_slice_avg, 'b-', label='MeepSAT Time AvG', linewidth=2)
        # plt.plot(y_coords, phase_slice_last, 'r--', label='Last Timestep', linewidth=2)
        if comparision_CST_data is not None:
            for i in range(len(comparision_CST_data)):
                # Find CST phase at y=0 to subtract
                indexy0_cst = (np.abs(cst_efield[i]['Y'] - 0)).argmin()
                cst_efield[i]['Phase_degrees'] -= cst_efield[i]['Phase_degrees'][indexy0_cst]
                plt.plot(cst_efield[i]['Y'], cst_efield[i]['Phase_degrees'], 'g-.', label='CST Data', linewidth=2)

        if comparision_GRASP_data is not None:
            for i in range(len(comparision_GRASP_data)):
                # Find GRASP phase at y=0 to subtract
                indexy0_grasp = (np.abs(y_grasp[i] - 0)).argmin()
                grasp_phase_slice[i] -= grasp_phase_slice[i][indexy0_grasp]
                plt.plot(y_grasp[i], grasp_phase_slice[i], label='GRASP ({})'.format(comparision_GRASP_data[i]['plot_label']), linewidth=2)
        plt.xlabel('Y (mm)')
        plt.ylabel('Phase (degrees)')
        plt.title(f'E-field Phase Slice Comparison at x = {x_coords[x_index]:.1f} mm ({frequency_label})')
        plt.legend()
        plt.grid()

        if savefig:
            plt.savefig(f'{plot_path}/efield_phase_slice_comparison_{frequency_label.replace(" ", "_").replace("/", "-")}_{savename_suffix}.png', dpi=300)
        if show_plots:
            plt.show()
        
        #! Plot 3: 2D magnitude comparison (side by side)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Time averaged
        im1 = ax1.imshow(
            10 * np.log10(np.abs(ez_magnitude_avg)**2 / norm_factor),
            extent=[np.min(x_coords), np.max(x_coords), np.min(y_coords), np.max(y_coords)],
            cmap='viridis', vmin=gaussian_fit_power_threshold, vmax=0
        )
        ax1.axvline(x=x_coords[x_index], color='r', linestyle='--', alpha=0.7)
        ax1.set_xlabel('X (mm)')
        ax1.set_ylabel('Y (mm)')
        ax1.set_title(f'Time Averaged E-field Magnitude**2 ({frequency_label})')
        ax1.grid()
        plt.colorbar(im1, ax=ax1, label='Power (dB)')

        # Save the array data as npz for future verification
        np.savez_compressed(f'{plot_path}/efield_magnitude_timeavg_2D_{frequency_label.replace(" ", "_").replace("/", "-")}_{savename_suffix}.npz', 
                            ez_magnitude_avg = ez_magnitude_avg,
                            ez_magnitude_avg_linear = np.abs(ez_magnitude_avg)**2 / norm_factor,
                            ez_power_avg_dB=10 * np.log10(np.abs(ez_magnitude_avg)**2 / norm_factor),
                            x_coords=x_coords,
                            y_coords=y_coords)

        
        # Last timestep
        im2 = ax2.imshow(
            10 * np.log10(np.abs(ez_magnitude_last)**2 / norm_factor),
            extent=[np.min(x_coords), np.max(x_coords), np.min(y_coords), np.max(y_coords)],
            cmap='viridis', vmin=gaussian_fit_power_threshold, vmax=0
        )
        ax2.axvline(x=x_coords[x_index], color='r', linestyle='--', alpha=0.7)
        ax2.set_xlabel('X (mm)')
        ax2.set_ylabel('Y (mm)')
        ax2.set_title(f'Last Timestep E-field Magnitude**2 ({frequency_label})')
        ax2.grid()
        plt.colorbar(im2, ax=ax2, label='Power (dB)')
        
        plt.tight_layout()

        if savefig:
            plt.savefig(f'{plot_path}/efield_magnitude_2D_comparison_{frequency_label.replace(" ", "_").replace("/", "-")}_{savename_suffix}.png', dpi=300)
        if show_plots:
            plt.show()
        
        #! Plot 4: 2D phase comparison (side by side)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Time averaged
        im1 = ax1.imshow(
            ez_phase_avg_degrees,
            extent=[np.min(x_coords), np.max(x_coords), np.min(y_coords), np.max(y_coords)],
            cmap='twilight', vmin=-180, vmax=180
        )
        ax1.axvline(x=x_coords[x_index], color='r', linestyle='--', alpha=0.7)
        ax1.set_xlabel('X (mm)')
        ax1.set_ylabel('Y (mm)')
        ax1.set_title(f'Time Averaged E-field Phase ({frequency_label})')
        ax1.grid()
        plt.colorbar(im1, ax=ax1, label='Phase (degrees)')

        # Save the array data as npz for future verification
        np.savez_compressed(f'{plot_path}/efield_phase_timeavg_2D_{frequency_label.replace(" ", "_").replace("/", "-")}_{savename_suffix}.npz', 
                            ez_phase_avg_degrees=ez_phase_avg_degrees,
                            x_coords=x_coords,
                            y_coords=y_coords)
        
        # Last timestep
        im2 = ax2.imshow(
            ez_phase_last_degrees,
            extent=[np.min(x_coords), np.max(x_coords), np.min(y_coords), np.max(y_coords)],
            cmap='twilight', vmin=-180, vmax=180
        )
        ax2.axvline(x=x_coords[x_index], color='r', linestyle='--', alpha=0.7)
        ax2.set_xlabel('X (mm)')
        ax2.set_ylabel('Y (mm)')
        ax2.set_title(f'Last Timestep E-field Phase ({frequency_label})')
        ax2.grid()
        plt.colorbar(im2, ax=ax2, label='Phase (degrees)')
        
        plt.tight_layout()

        if savefig:
            plt.savefig(f'{plot_path}/efield_phase_2D_comparison_{frequency_label.replace(" ", "_").replace("/", "-")}_{savename_suffix}.png', dpi=300)
        if show_plots:
            plt.show()

        #! Plot 5: 2D Poynting vector comparison (side by side)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Time averaged
        im1 = ax1.imshow(
            10 * np.log10(averaged_poynting_vector / norm_factor),
            extent=[np.min(x_coords), np.max(x_coords), np.min(y_coords), np.max(y_coords)],
            cmap='viridis', vmin=gaussian_fit_power_threshold, vmax=0
        )
        ax1.axvline(x=x_coords[x_index], color='r', linestyle='--', alpha=0.7)
        ax1.set_xlabel('X (mm)')
        ax1.set_ylabel('Y (mm)')
        ax1.set_title(f'Time Averaged Poynting Vector Magnitude ({frequency_label})')
        ax1.grid()
        plt.colorbar(im1, ax=ax1, label='Power (dB)')   

        # Save the array data as npz for future verification
        np.savez_compressed(f'{plot_path}/poynting_vector_magnitude_timeavg_2D_{frequency_label.replace(" ", "_").replace("/", "-")}_{savename_suffix}.npz', 
                            poynting_vector_value_linear = averaged_poynting_vector,
                            poynting_vector_magnitude_avg_linear = averaged_poynting_vector / norm_factor,
                            poynting_vector_magnitude_avg_dB=10 * np.log10(averaged_poynting_vector / norm_factor),
                            x_coords=x_coords,
                            y_coords=y_coords)

        # Last timestep
        im2 = ax2.imshow(
            10 * np.log10(total_poynting_last / norm_factor),
            extent=[np.min(x_coords), np.max(x_coords), np.min(y_coords), np.max(y_coords)],
            cmap='viridis', vmin=gaussian_fit_power_threshold, vmax=0
        )
        ax2.axvline(x=x_coords[x_index], color='r', linestyle='--', alpha=0.7)
        ax2.set_xlabel('X (mm)')
        ax2.set_ylabel('Y (mm)')
        ax2.set_title(f'Last Timestep Poynting Vector Magnitude ({frequency_label})')
        ax2.grid()
        plt.colorbar(im2, ax=ax2, label='Power (dB)')   
        plt.tight_layout()

        if savefig:
            plt.savefig(f'{plot_path}/poynting_vector_magnitude_2D_comparison_{frequency_label.replace(" ", "_").replace("/", "-")}_{savename_suffix}.png', dpi=300)
        if show_plots:
            plt.show()

        #! Plot 6: Poynting vector slice comparison at x_position
        poynting_slice_avg = averaged_poynting_vector[:, x_index]/np.max(averaged_poynting_vector[:, x_index])
        poynting_slice_last = total_poynting_last[:, x_index]
        plt.figure(figsize=(10, 6))
        plt.plot(y_coords, 10 * np.log10(poynting_slice_avg / norm_factor), 
                'b-', label='MeepSAT Time AvG', linewidth=2)
        # plt.plot(y_coords, 10 * np.log10(poynting_slice_last / norm_factor), 
        #          'r--', label='Last Timestep', linewidth=2)
        if comparision_CST_data is not None:
            for i in range(len(comparision_CST_data)):
                plt.plot(cst_s_mag[i]['Y'], cst_s_mag[i]['S_Mag_dB'], 'g-.', label='CST Data', linewidth=2)
        plt.ylim(-30, 0)
        plt.xlabel('Y (mm)')  
        plt.ylabel('S Magnitude (dB)')
        plt.title(f'Poynting Vector Magnitude Slice Comparison at x = {x_coords[x_index]:.1f} mm ({frequency_label})')
        plt.legend()
        plt.grid()

        if savefig:
            plt.savefig(f'{plot_path}/poynting_vector_magnitude_slice_comparison_{frequency_label.replace(" ", "_").replace("/", "-")}_{savename_suffix}.png', dpi=300)
        if show_plots:
            plt.show()

        #! Plot 7: POYNTING ANALYSIS: 
        # Extract the S field distrubution for the full box with only source
        # Run the real simulation 
        # Divide those two
        if average_source_power is not None:
            sum_source_power_dB = 10 * np.log10(np.sum(average_source_power)/np.max(average_source_power)) 
            print(f"Sum of Source Power: {np.sum(average_source_power):.4e}")
            print(f"Sum of Source Power (dB): {sum_source_power_dB:.4f} dB")
            normalized_poynting_vector_avg = averaged_poynting_vector / sum_source_power_dB

            # Plot contour levels
            contour_levels_dB = np.linspace(contour_db_range[0], 
                                            contour_db_range[1], 
                                            contour_levels)
            contour_levels_linear = np.linspace(contour_linear_range[0],
                                                contour_linear_range[1],
                                                contour_levels)

            # Plot only in dB scale for now
            plt.figure(figsize=(18, 6))
            cs = plt.contourf(
                x_coords, y_coords, 10 * np.log10(normalized_poynting_vector_avg),
                levels=contour_levels_dB, cmap='viridis'
            )
            plt.axvline(x=x_coords[x_index], color='r', linestyle='--', alpha=0.7)
            plt.xlabel('X (mm)')
            plt.ylabel('Y (mm)')
            plt.title(f'Time Avg $S$/Sun(source power) ({frequency_label})')
            plt.colorbar(cs, label='Power (dB)')
            plt.grid()

            if savefig:
                plt.savefig(f'{plot_path}/poynting_vector_divided_by_source_power_{frequency_label.replace(" ", "_").replace("/", "-")}_{savename_suffix}.png', dpi=300)
            if show_plots:
                plt.show()

            #! Plot 7(b): Poynting vector divided by integrated source power
            # Using trapz to integrate the source power over the y axis (ycoords)
            integrated_source_power = np.trapz(average_source_power, y_coords)
            print(f"Integrated Source Power: {integrated_source_power:.4e}")
            integrated_source_power_dB = 10 * np.log10(integrated_source_power/np.max(average_source_power))
            print(f"Integrated Source Power (dB): {integrated_source_power_dB:.4f} dB")

            averaged_poynting_vector_divided_source = averaged_poynting_vector / integrated_source_power_dB
            # Plot contour levels
            contour_levels_dB = np.linspace(contour_db_range[0], 
                                            contour_db_range[1], 
                                            contour_levels)
            contour_levels_linear = np.linspace(contour_linear_range[0],
                                                contour_linear_range[1],
                                                contour_levels)
            # Plot only in dB scale for now
            plt.figure(figsize=(18, 6))
            cs = plt.contourf(
                x_coords, y_coords, 10 * np.log10(averaged_poynting_vector_divided_source),
                levels=contour_levels_dB, cmap='viridis'
            )
            plt.axvline(x=x_coords[x_index], color='r', linestyle='--', alpha=0.7)
            plt.xlabel('X (mm)')
            plt.ylabel('Y (mm)')
            plt.title(f'Time Avg $S$/Integrated Source Power ({frequency_label})')
            plt.colorbar(cs, label='Power (dB)')
            plt.grid()

            if savefig:
                plt.savefig(f'{plot_path}/poynting_vector_divided_by_integrated_source_power_{frequency_label.replace(" ", "_").replace("/", "-")}_{savename_suffix}.png', dpi=300)
            if show_plots:
                plt.show()

        #! Plot 7(c): Doing the same thing, but for the average power calculated from avg efield
        if average_source_power is not None:
            sum_source_power_dB = 10 * np.log10(np.sum(average_source_power)/np.max(average_source_power))
            print(f"Sum of Source Power: {np.sum(average_source_power):.4e}")
            print(f"Sum of Source Power (dB): {sum_source_power_dB:.4f} dB")
            normalized_efieldpower_vector_avg = 10*np.log10(np.abs(ez_magnitude_avg)**2) / sum_source_power_dB
            # Plot contour levels
            contour_levels_dB = np.linspace(contour_db_range[0], 
                                            contour_db_range[1], 
                                            contour_levels)
            contour_levels_linear = np.linspace(contour_linear_range[0],
                                                contour_linear_range[1],
                                                contour_levels) 
            
            # Plot only in dB scale for now
            plt.figure(figsize=(18, 6))
            cs = plt.contourf(
                x_coords, y_coords, normalized_efieldpower_vector_avg,
                levels=contour_levels_dB, cmap='viridis'
            )
            plt.axvline(x=x_coords[x_index], color='r', linestyle='--', alpha=0.7)
            plt.xlabel('X (mm)')
            plt.ylabel('Y (mm)')
            plt.title(f'Time Avg |E|^2/Sun(source power) ({frequency_label})')
            plt.colorbar(cs, label='Power (dB)')
            plt.grid()

            if savefig:
                plt.savefig(f'{plot_path}/efield_power_divided_by_source_power_{frequency_label.replace(" ", "_").replace("/", "-")}_{savename_suffix}.png', dpi=300)
            if show_plots:
                plt.show()

            # ! Plot 7(d): Poynting vector divided by integrated source power
            # Using trapz to integrate the source power over the y axis (ycoords)
            integrated_source_power = np.trapz(average_source_power, y_coords)
            print(f"Integrated Source Power: {integrated_source_power:.4e}")
            integrated_source_power_dB = 10 * np.log10(integrated_source_power/np.max(average_source_power))
            print(f"Integrated Source Power (dB): {integrated_source_power_dB:.4f} dB")
            averaged_Efieldpower_divided_by_source = 10*np.log10(np.abs(ez_magnitude_avg)**2) / integrated_source_power_dB
            
            # Plot in dB scale for now
            plt.figure(figsize=(18, 6))
            cs = plt.contourf(
                x_coords, y_coords, averaged_Efieldpower_divided_by_source,
                levels=contour_levels_dB, cmap='viridis'
            )
            plt.axvline(x=x_coords[x_index], color='r', linestyle='--', alpha=0.7)
            plt.xlabel('X (mm)')
            plt.ylabel('Y (mm)')
            plt.title(f'Time Avg |E|^2/Integrated Source Power ({frequency_label})')
            plt.colorbar(cs, label='Power (dB)')
            plt.grid()

            if savefig:
                plt.savefig(f'{plot_path}/efield_power_divided_by_integrated_source_power_{frequency_label.replace(" ", "_").replace("/", "-")}_{savename_suffix}.png', dpi=300)
            if show_plots:
                plt.show()    
        

        # #! Plot 7: Poynting vectot divided by source (2D) comparison (side by side) in linear scale
        # if norm_factor is not None:
        #     fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        #     # Time averaged
        #     im1 = ax1.imshow(
        #         averaged_poynting_vector_divided_source,
        #         extent=[np.min(x_coords), np.max(x_coords), np.min(y_coords), np.max(y_coords)],
        #         cmap='viridis'
        #     )
        #     ax1.axvline(x=x_coords[x_index], color='r', linestyle='--', alpha=0.7)
        #     ax1.set_xlabel('X (mm)')
        #     ax1.set_ylabel('Y (mm)')
        #     ax1.set_title(f'Time Averaged Poynting Vector / Source ({frequency_label})')
        #     ax1.grid()
        #     plt.colorbar(im1, ax=ax1, label='Normalized Power')   

        #     # Last timestep
        #     im2 = ax2.imshow(
        #         total_poynting_last_divided_source,
        #         extent=[np.min(x_coords), np.max(x_coords), np.min(y_coords), np.max(y_coords)],
        #         cmap='viridis'
        #     )
        #     ax2.axvline(x=x_coords[x_index], color='r', linestyle='--', alpha=0.7)
        #     ax2.set_xlabel('X (mm)')
        #     ax2.set_ylabel('Y (mm)')
        #     ax2.set_title(f'Last Timestep Poynting Vector / Source ({frequency_label})')
        #     ax2.grid()
        #     plt.colorbar(im2, ax=ax2, label='Normalized Power')   

        #     plt.tight_layout()
        #     plt.show()

        # #! Plot 8: Same as that of Plot 7 but in contour plot
        # if norm_factor is not None:
        #     fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        #     # Time averaged
        #     cs1 = ax1.contourf(
        #         x_coords, y_coords, averaged_poynting_vector_divided_source,
        #         levels=contour_levels, cmap='viridis'
        #     )
        #     ax1.axvline(x=x_coords[x_index], color='r', linestyle='--', alpha=0.7)
        #     ax1.set_xlabel('X (mm)')
        #     ax1.set_ylabel('Y (mm)')
        #     ax1.set_title(f'Time Averaged Poynting Vector / Source ({frequency_label})')
        #     ax1.grid()
        #     plt.colorbar(cs1, ax=ax1, label='Normalized Power')   

        #     # Last timestep
        #     cs2 = ax2.contourf(
        #         x_coords, y_coords, total_poynting_last_divided_source,
        #         levels=contour_levels, cmap='viridis'
        #     )
        #     ax2.axvline(x=x_coords[x_index], color='r', linestyle='--', alpha=0.7)
        #     ax2.set_xlabel('X (mm)')
        #     ax2.set_ylabel('Y (mm)')
        #     ax2.set_title(f'Last Timestep Poynting Vector / Source ({frequency_label})')
        #     ax2.grid()
        #     plt.colorbar(cs2, ax=ax2, label='Normalized Power')   

        #     plt.tight_layout()
        #     plt.show()

        
        return {
            'ez_magnitude_avg': ez_magnitude_avg,
            'ez_phase_avg_degrees': ez_phase_avg_degrees,
            'ez_magnitude_last': ez_magnitude_last,
            'ez_phase_last_degrees': ez_phase_last_degrees,
            'x_coords': x_coords,
            'y_coords': y_coords,
            'aperture_slice_avg': aperture_slice_avg,
            'aperture_slice_last_timestep': aperture_slice_last,

        }
    



def summary_plots_forebaffles(
    simulation_resolution,
    simulation_wvl,
    efield_files_pattern,
    poynting_vector_pattern,
    xyzw_coords_file,
    x_position,
    analysis,
    frequency_label="90 GHz",
    norm_factor=1,
    contour_levels = 10,
    contour_db_range = (-60, 0),
    contour_linear_range = (0, 1),
    comparision_CST_data=None,
    comparision_GRASP_data= None,
    comparision_CST_fft_data = None,
    comparision_GRASP_fft_data = None,
    aper_size = None,
    avg_aper_lim = None,
    zero_pad_beam = 15,
    gaussian_fit_main_beam = False,
    gaussian_fit_power_threshold = -30,
    average_source_power= None,
    savefig = False,
    show_plots = True,
    savename_suffix = ''
):
    """
    Loads multiple E-field files, calculates average field, and plots comparison with last timestep.
    Also compares with CST and GRASP data if provided.

    Far field profiles are also compared if GRASP and CST data is provided.
    
    Parameters:
    -----------
        simulation_resolution (float): Simulation resolution in pixels/mm
        simulation_wvl (float): Simulation wavelength in meep units (It should be in mm since our scaling is 1 meep unit = 1 mm)   
        efield_files_pattern (str): Pattern for E-field files (e.g., 'path/single_lens_testing-e-*.h5')
        xyzw_coords_file (str): Path to the .npz file with x, y, w coordinates
        x_position (float): X position (in mm) for the slice
        analysis (module): Analysis module with readHDF5 function
        frequency_label (str): Frequency label for plot titles
        norm_factor (float): Normalization factor for power calculations
        contour_levels (int): Number of contour levels for contour plots
        comparision_CST_data (dict): Dictionary containing CST data for comparison (optional)
                                    cst_data = {'ez_magnitude', 'ez_phase', 's_magnitude'}
        comparision_GRASP_data (dict): Dictionary or list of dictionaries containing GRASP data for comparision (optional)
                                    grasp_data = {'Ex', 'Ey', 'Ez', 'x', 'y', 'plot_label'}   
        comparision_CST_fft_data (dict): Dictionary containing CST FFT data for far field comparison (optional)
                                    cst_fft_data = {'angle', 'power_dB', 'plot_label'}
        comparision_GRASP_fft_data (dict or list of dicts): Dictionary or list of dictionaries containing GRASP FFT data for far field comparison (optional)
                                    grasp_fft_data = {'angle', 'power_dB', 'plot_label'}
        aper_size (float): Aperture size in mm for GRASP data (optional)
        zero_pad_beam (int): Zero padding factor for FFT beam calculation
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import glob
    
    if savefig:
        # Create an directory named as summary_plots_results (if doesn't exist)
        if not os.path.exists('summary_plots_results'):
            os.makedirs('summary_plots_results')
            
        if frequency_label:
            freq_label_clean = frequency_label.replace(" ", "_").replace("/", "-")
            plot_path = f'summary_plots_results/{freq_label_clean}'
        else:
            plot_path = 'summary_plots_results'
        print(f"Plots will be saved to: {plot_path}")

        if savename_suffix != '':
            plot_path = os.path.join(plot_path, savename_suffix)

        if not os.path.exists(plot_path):
            os.makedirs(plot_path)

    # Get all E-field files matching the pattern
    efield_files = sorted(glob.glob(efield_files_pattern))
    print(f"Found {len(efield_files)} E-field files")
    
    if len(efield_files) == 0:
        raise ValueError(f"No files found matching pattern: {efield_files_pattern}")

    #Get all Poynting vector files matching the pattern
    poynting_vector_files = sorted(glob.glob(poynting_vector_pattern))
    print(f"Found {len(poynting_vector_files)} Poynting vector files")

    if len(poynting_vector_files) == 0:
        raise ValueError(f"No files found matching pattern: {poynting_vector_pattern}")
    
    #* Load coordinates
    xyzw_data = np.load(xyzw_coords_file)
    x_coords = xyzw_data['x_coords']
    y_coords = xyzw_data['y_coords']
    
    # Initialize arrays for averaging
    ez_magnitude_sum = None
    ez_phase_sum_cos = None
    ez_phase_sum_sin = None

    # Load and process all efield files
    for i, file_path in enumerate(efield_files):
        print(f"Processing file {i+1}/{len(efield_files)}: {file_path}")
        
        # Load E-field data
        e_field_data = analysis.readHDF5(file_path)
        ez_real = e_field_data['ez.r']
        ez_imag = e_field_data['ez.i']

        
        # Calculate magnitude and phase
        ez_magnitude = np.sqrt(ez_real**2 + ez_imag**2)
        ez_magnitude = np.transpose(ez_magnitude)
        
        ez_phase_radians = np.arctan2(ez_imag, ez_real)
        
        # Sum for averaging
        if ez_magnitude_sum is None:
            ez_magnitude_sum = ez_magnitude
            ez_phase_sum_cos = np.cos(ez_phase_radians).T
            ez_phase_sum_sin = np.sin(ez_phase_radians).T
        else:
            ez_magnitude_sum += ez_magnitude
            ez_phase_sum_cos += np.cos(ez_phase_radians).T
            ez_phase_sum_sin += np.sin(ez_phase_radians).T
    
    # Initialize arrays for Poynting vector averaging and last timestep
    poynting_vector_sum = None

    for i, file_path in enumerate(poynting_vector_files):
        print(f"Processing Poynting vector file {i+1}/{len(poynting_vector_files)}: {file_path}")
        
        # Load Poynting vector data
        poynting_data = analysis.readHDF5(file_path)
        sx = poynting_data['sx']
        sy = poynting_data['sy']
        sz = poynting_data['sz']
        total_poynting = np.abs(sx)**2 + np.abs(sy)**2 + np.abs(sz)**2
        total_poynting = np.sqrt(total_poynting)
        total_poynting = np.transpose(total_poynting)
        # Sum for averaging
        if poynting_vector_sum is None:
            poynting_vector_sum = total_poynting
        else:
            poynting_vector_sum += total_poynting

    # Average Poynting vector
    averaged_poynting_vector = poynting_vector_sum / len(poynting_vector_files)
    # Normalize averaged Poynting vector to its maximum
    averaged_poynting_vector /= np.max(averaged_poynting_vector)

    # Load last timestep for Poynting vector
    last_poynting_file = poynting_vector_files[-1]
    poynting_last = analysis.readHDF5(last_poynting_file)
    sx_last = poynting_last['sx']
    sy_last = poynting_last['sy']
    sz_last = poynting_last['sz']
    total_poynting_last = np.abs(sx_last)**2 + np.abs(sy_last)**2 + np.abs(sz_last)**2
    total_poynting_last = np.sqrt(total_poynting_last)
    total_poynting_last = np.transpose(total_poynting_last)
    # Normalize last timestep Poynting vector to its maximum
    total_poynting_last /= np.max(total_poynting_last)

    # Average and Last timestep Poynting vector divided by norm factor
    if norm_factor is not None:
        averaged_poynting_vector_divided_source = averaged_poynting_vector / norm_factor
        print(f"Averaged Poynting vector divided by norm factor shape: {averaged_poynting_vector_divided_source.shape}")
        total_poynting_last_divided_source = total_poynting_last / norm_factor
        print(f"Last timestep Poynting vector divided by norm factor shape: {total_poynting_last_divided_source.shape}")
    
    # Calculate averages
    ez_magnitude_avg = ez_magnitude_sum / len(efield_files)
    #!!!!!
    ez_magnitude_avg = ez_magnitude_avg / np.max(ez_magnitude_avg)  # Normalize to max
    ez_phase_avg_radians = np.arctan2(ez_phase_sum_sin / len(efield_files), 
                                      ez_phase_sum_cos / len(efield_files))
    ez_phase_avg_degrees = np.degrees(ez_phase_avg_radians)

    # Load last timestep for comparison
    last_file = efield_files[-1]
    e_field_last = analysis.readHDF5(last_file)
    ez_real_last = e_field_last['ez.r']
    ez_imag_last = e_field_last['ez.i']
    
    ez_magnitude_last = np.sqrt(ez_real_last**2)# + ez_imag_last**2)
    ez_magnitude_last = np.transpose(ez_magnitude_last)
    
    ez_phase_last_radians = np.arctan2(ez_imag_last, ez_real_last)
    ez_phase_last_degrees = np.degrees(ez_phase_last_radians)
    ez_phase_last_degrees = np.transpose(ez_phase_last_degrees)

    # Find slice index
    x_index = (np.abs(x_coords - x_position)).argmin()

    #!= Checking if CST data is provided for comparison
    if comparision_CST_data is not None:
        # Creating empty lists to hold CST data
        cst_efield = []; cst_s_mag = []
        for i in range(len(comparision_CST_data)):
            cst_efield_i = comparision_CST_data[i]['efield']
            cst_s_mag_i = comparision_CST_data[i]['s_mag']

            # Calculate CST E-field magnitude and phase
            cst_efield_i['ycoords'] = cst_efield_i['Y']
            cst_efield_i['Magnitude'] = np.sqrt(cst_efield_i['Re(Ey)']**2 + cst_efield_i['Im(Ey)']**2)
            cst_efield_i['Phase_degrees'] = -np.degrees(np.arctan2(cst_efield_i['Im(Ey)'], cst_efield_i['Re(Ey)']))
            # cst_efield_i['Magnitude_dB'] = 20 * np.log10(cst_efield_i['Magnitude']) #/ np.max(cst_efield_i['Magnitude']))

            # Calculate CST S magnitude in dB
            cst_s_mag_i['S_Mag_dB'] = 10 * np.log10(cst_s_mag_i['S_Mag_linear'] / np.max(cst_s_mag_i['S_Mag_linear']))

            # # Plot CST Magnitude for verification
            # plt.figure()
            # plt.plot(cst_efield_i['ycoords'], cst_efield_i['Magnitude'], label = 'Raw CST E-field Magnitude')
            # plt.xlabel('y (mm)')
            # plt.ylabel('E-field Magnitude')
            # plt.title(f'CST E-field Magnitude Slice at x={x_position} mm ({frequency_label})')
            # plt.legend()
            # plt.grid()
            # plt.show()
            
            #* Mask out only for the aperture size 
            cst_aperture_indices = mask_aperture(cst_efield_i['ycoords'], aper_size)

            #* Then mask out and calculate the average between -avg_aper_lim to +avg_aper_lim
            cst_norm_avg_aper_lim_indices = mask_aperture(cst_efield_i['ycoords'], avg_aper_lim)
            
            # Apply mask to CST E-field data
            cst_efield_i['ycoords'] = cst_efield_i['ycoords'][cst_aperture_indices]
            cst_efield_i['Magnitude'] = cst_efield_i['Magnitude'][cst_aperture_indices]/np.mean(cst_efield_i['Magnitude'][cst_norm_avg_aper_lim_indices])
            cst_efield_i['Phase_degrees'] = cst_efield_i['Phase_degrees'][cst_aperture_indices]
            cst_efield_i['Magnitude_dB'] = 20 * np.log10(cst_efield_i['Magnitude'] + 1e-12)  # Avoid log(0)

            # Apply mask to CST S magnitude data
            cst_s_mag_i['S_Mag_dB'] = cst_s_mag_i['S_Mag_dB'][cst_aperture_indices]

            # # Plot after masking
            # plt.figure()
            # plt.plot(cst_efield_i['ycoords'], cst_efield_i['Magnitude'], label = 'Masked CST E-field Magnitude')
            # plt.xlabel('y (mm)')
            # plt.ylabel('E-field Magnitude')
            # plt.title(f'CST E-field Magnitude Slice at x={x_position} mm ({frequency_label}) - Masked')
            # plt.legend()
            # plt.grid()  
            # plt.show()

            cst_efield.append(cst_efield_i)
            cst_s_mag.append(cst_s_mag_i)
    
    #!= Checking if GRASP data is provided for comparison
    if comparision_GRASP_data is not None:
        # Creating empty lists to hold GRASP data
        ez_grasp = []; ex_grasp = []; ey_grasp = []; e_grasp = []; phase_grasp = []; 
        grasp_efield_magnitude = []; grasp_efield_magnitude_dB = []; grasp_phase_slice = []; y_grasp = []

        for comparision_GRASP_data_i in comparision_GRASP_data:
            ez_grasp_arr = comparision_GRASP_data_i['Ez']
            ex_grasp_arr = comparision_GRASP_data_i['Ex']
            ey_grasp_arr = comparision_GRASP_data_i['Ey']
            e_grasp_arr = np.abs(ex_grasp_arr) #np.sqrt(np.abs(ex_grasp_arr**2 + ey_grasp_arr**2 + ez_grasp_arr**2))
            y_grasp_arr = comparision_GRASP_data_i['x']

            phase_grasp_arr = np.angle(ex_grasp_arr)

            # Extract middle row and convert to dB
            mid_row_index = e_grasp_arr.shape[0] // 2
            grasp_efield_magnitude_arr = e_grasp_arr[mid_row_index, :]
            # grasp_efield_magnitude_dB_arr = 20 * np.log10(grasp_efield_magnitude_arr/np.max(grasp_efield_magnitude_arr) + 1e-12)  # Avoid log(0)
            grasp_phase_slice_arr = np.degrees(phase_grasp_arr[mid_row_index, :])

            # plt.plot(y_grasp_arr, grasp_efield_magnitude_arr, label=comparision_GRASP_data_i['plot_label'])
            # plt.xlabel('y (mm)')
            # plt.ylabel('E-field Magnitude (dB)')
            # plt.title(f'GRASP E-field Magnitude Slice at x={x_position} mm ({frequency_label})')
            # plt.legend()
            # plt.grid()
            # plt.show()

            #* Mask out only for the aperture size 
            grasp_aperture_indices = mask_aperture(y_grasp_arr, aper_size)

            #* Mask out and calculate the average between -avg_aper_lim to +avg_aper_lim
            grasp_norm_avg_aper_lim_indices = mask_aperture(y_grasp_arr, avg_aper_lim)

            # Normalize GRASP E-field magnitude to the average within avg_aper_lim
            grasp_efield_magnitude_arr = grasp_efield_magnitude_arr/np.mean(grasp_efield_magnitude_arr[grasp_norm_avg_aper_lim_indices])
            grasp_efield_magnitude_dB_arr = 20 * np.log10(grasp_efield_magnitude_arr + 1e-12)  # Avoid log(0)
            
            # Apply mask to all GRASP data arrays
            ez_grasp_arr = ez_grasp_arr[:, grasp_aperture_indices]
            ex_grasp_arr = ex_grasp_arr[:, grasp_aperture_indices]
            ey_grasp_arr = ey_grasp_arr[:, grasp_aperture_indices]
            e_grasp_arr = e_grasp_arr[:, grasp_aperture_indices]
            phase_grasp_arr = phase_grasp_arr[:, grasp_aperture_indices]
            grasp_efield_magnitude_arr = grasp_efield_magnitude_arr[grasp_aperture_indices]
            grasp_efield_magnitude_dB_arr = grasp_efield_magnitude_dB_arr[grasp_aperture_indices]
            grasp_phase_slice_arr = grasp_phase_slice_arr[grasp_aperture_indices]
            y_grasp_arr = y_grasp_arr[grasp_aperture_indices]

            # # Plot again after masking
            # plt.plot(y_grasp_arr, grasp_efield_magnitude_arr, label=comparision_GRASP_data_i['plot_label'] + ' (Masked)')
            # plt.xlabel('y (mm)')
            # plt.ylabel('E-field Magnitude (dB)')
            # plt.title(f'GRASP E-field Magnitude Slice at x={x_position} mm  ({frequency_label}) - Masked')
            # plt.legend()
            # plt.grid()  
            # plt.show()

            # Append everything to lists for potential multiple GRASP datasets
            ez_grasp.append(ez_grasp_arr); ex_grasp.append(ex_grasp_arr); ey_grasp.append(ey_grasp_arr); e_grasp.append(e_grasp_arr); phase_grasp.append(phase_grasp_arr)
            grasp_efield_magnitude.append(grasp_efield_magnitude_arr); grasp_efield_magnitude_dB.append(grasp_efield_magnitude_dB_arr); 
            grasp_phase_slice.append(grasp_phase_slice_arr); y_grasp.append(y_grasp_arr)
    

    #!= MEEPSAT aperture masking
    #! Extract slices for plotting and also calculate the Far field beams from those slices 
    aperture_slice_avg = ez_magnitude_avg[:, x_index]#/np.max(ez_magnitude_avg[:, x_index])
    aperture_slice_last = ez_magnitude_last[:, x_index]#/np.max(ez_magnitude_last[:, x_index])
    y_meep = y_coords

    # Mask out and calculate the average between -avg_aper_lim to +avg_aper_lim
    meep_norm_avg_aper_lim_indices = mask_aperture(y_meep, avg_aper_lim)
    aperture_slice_avg = aperture_slice_avg/np.mean(aperture_slice_avg[meep_norm_avg_aper_lim_indices])
    aperture_slice_last = aperture_slice_last/np.mean(aperture_slice_last[meep_norm_avg_aper_lim_indices])

    # Aperture masking
    meep_aperture_indices = mask_aperture(y_meep, aper_size)
    aperture_slice_avg = aperture_slice_avg[meep_aperture_indices]
    aperture_slice_last = aperture_slice_last[meep_aperture_indices]
    y_meep = y_meep[meep_aperture_indices]

    #~ CST FFB
    if comparision_CST_data is not None:
        angle_array_cst = []; power_dB_array_cst = []
        cst_resolution_list = []
        for i in range(len(comparision_CST_data)):
            # Calculate CST Resolution
            cst_resolution = calculate_CST_resolution(y_coords= cst_efield[i]['ycoords'])
            print(f"Calculated CST resolution for {comparision_CST_data[i]['plot_label']}: {cst_resolution} points/mm")

            # Use cst_far_field_fft function to calculate the far field from CST aperture slice
            cst_fft_dict = cst_far_field_fft(y_coords = cst_efield[i]['ycoords'],
                                             efield = cst_efield[i]['Magnitude'],
                                             cst_resolution = cst_resolution,
                                             wavelength = simulation_wvl,
                                             aper_size = aper_size,
                                             zero_pad_beam = zero_pad_beam,
                                             plot_label = comparision_CST_data[i]['plot_label'] + f" ({frequency_label})")
            
            angle_array_cst_i, power_dB_array_cst_i = cst_fft_dict['angle'], cst_fft_dict['power_dB']
            cst_resolution_list.append(cst_resolution)
            # Append data for plotting
            angle_array_cst.append(angle_array_cst_i)
            power_dB_array_cst.append(power_dB_array_cst_i)
    #~ GRASP FFB
    if comparision_GRASP_data is not None:
        # Initialize lists to hold the calculated far field data from GRASP
        angle_array_grasp_list = []; power_dB_array_grasp_list = []
        # Initialize the efield_data_meep_list and y_coords_meep_list, grasp_resolution_list
        # efield_data_meep_list = []; y_coords_meep_list = []; 
        grasp_resolution_list = [] 
        for i in range(len(comparision_GRASP_data)):
            # Calculate the GRASP resolution
            print(f"y_grasp_{comparision_GRASP_data[i]['plot_label']}", y_grasp[i])
            grasp_resolution = calculate_grasp_resolution(y_grasp[i])
            print(f"Calculated GRASP resolution for {comparision_GRASP_data[i]['plot_label']}: {grasp_resolution} points/mm")

            # Use grasp_far_field_fft function to calculate the far field from GRASP aperture slice
            grasp_fft_dict = grasp_far_field_fft(grasp_data = comparision_GRASP_data[i],
                                                wavelength = simulation_wvl,
                                                aper_size = aper_size,
                                                zero_pad_beam = zero_pad_beam,
                                                plot_label = comparision_GRASP_data[i]['plot_label'] + f" ({frequency_label})")
            angle_array_grasp, power_dB_array_grasp = grasp_fft_dict['angle'], grasp_fft_dict['power_dB']
            grasp_resolution_list.append(grasp_resolution)

            # Append data for plotting
            angle_array_grasp_list.append(angle_array_grasp)
            power_dB_array_grasp_list.append(power_dB_array_grasp)

    #~ MEEPSAT FFB
    #! First calculate the far field from the aperture slice avg using meepsat_far_field_fft function
    meepsat_fft_dict = meepsat_far_field_fft(y_coords = y_meep,
                                             efield = aperture_slice_avg,
                                             meep_resolution = simulation_resolution,
                                             wavelength = simulation_wvl,
                                             aper_size = aper_size,
                                             zero_pad_beam = zero_pad_beam,
                                             plot_label = f"MEEPSAT ({frequency_label})")

    angle_array_meepsat, power_dB_array_meepsat = meepsat_fft_dict['angle'], meepsat_fft_dict['power_dB']
    print("Shape of MEEPSAT far field angle array:", angle_array_meepsat.shape)
    print("Shape of MEEPSAT far field power dB array:", power_dB_array_meepsat.shape)

    #! Plot 1: Magnitude slice comparison
    plt.figure(figsize=(10, 6))

    # Plot for MEEPSAT
    plt.plot(y_meep, 10 * np.log10(aperture_slice_avg / norm_factor), 
             'b-', label='MeepSAT Time AvG', linewidth=2)

    # Plot for GRASP if provided
    if comparision_GRASP_data is not None:
        for i in range(len(comparision_GRASP_data)):
            plt.plot(y_grasp[i], grasp_efield_magnitude_dB[i], label='GRASP ({})'.format(comparision_GRASP_data[i]['plot_label']), linewidth=2)

    # Plot for CST if provided
    if comparision_CST_data is not None:
        for i in range(len(comparision_CST_data)):
            plt.plot(cst_efield[i]['Y'], cst_efield[i]['Magnitude_dB'], 'g-.', label='CST Data', linewidth=2)

    #plt.ylim(-30, 0)
    plt.xlabel('Y (mm)')
    plt.ylabel('Power (dB)')
    plt.title(f'E-field Magnitude Slice Comparison at x = {x_coords[x_index]:.1f} mm ({frequency_label})')
    plt.legend()
    plt.grid()

    if savefig:
        plt.savefig(f'{plot_path}/efield_magnitude_slice_comparison_{frequency_label.replace(" ", "_").replace("/", "-")}_{savename_suffix}.png', dpi=300)
    if show_plots:
        plt.show()

    #! Plot 1(b): Far Field comparision from the aperture_slice_avg using plotting function
    plt.figure(figsize=(10, 6))

    # Plot for MEEPSAT
    plt.plot(angle_array_meepsat, power_dB_array_meepsat, 'b-', label=meepsat_fft_dict['plot_label'], linewidth=2)

    # Plot for GRASP if provided
    if comparision_GRASP_data is not None:
        for i in range(len(angle_array_grasp_list)):
            print(f"Plotting far field comparison for {comparision_GRASP_data[i]['plot_label']}")
            plt.plot(angle_array_grasp_list[i], power_dB_array_grasp_list[i], label='{}'.format(comparision_GRASP_data[i]['plot_label']), linewidth=2, linestyle='-.')
            # plt.plot(angle_array_meepsat_list[i], power_dB_array_meepsat_list[i], 'b--', label='MeepSAT (Res: {})'.format(comparision_GRASP_data[i]['plot_label']), linewidth=2)

    # Plot for CST if provided
    if comparision_CST_data is not None:
        for i in range(len(angle_array_cst)):
            print(f"Plotting far field comparison for {comparision_CST_data[i]['plot_label']}")
            plt.plot(angle_array_cst[i], power_dB_array_cst[i], label='{}'.format(comparision_CST_data[i]['plot_label']), linewidth=2, linestyle=(0, (1, 1)))

    plt.xlabel('Angle (degrees)')
    plt.ylabel('Power (dB)')
    plt.title(f'Far Field Comparison from Aperture Slice Avg ({frequency_label})')
    plt.ylim(-60, 0)
    plt.xlim(0, 20)
    plt.legend()
    plt.grid()

    if savefig:
        plt.savefig(f'{plot_path}/far_field_comparison_aperture_slice_avg_{frequency_label.replace(" ", "_").replace("/", "-")}_{savename_suffix}.png', dpi=300)
    if show_plots:
        plt.show()
    

    #! Plot 1(c): Far Field comparision between FFT(GRASP APERTURE PROFILE)
    plt.figure(figsize=(10, 6))
    if comparision_GRASP_data is not None:
        for i in range(len(angle_array_grasp_list)):
            print(f"Plotting far field comparison for {comparision_GRASP_data[i]['plot_label']}")
            plt.plot(angle_array_grasp_list[i], power_dB_array_grasp_list[i], label='{}'.format(comparision_GRASP_data[i]['plot_label']), linewidth=2, linestyle='-.')
    
    plt.xlabel('Angle (degrees)')
    plt.ylabel('Power (dB)')
    plt.title(f'Far Field Comparison between GRASP Datasets ({frequency_label})')
    plt.ylim(-60, 0)
    plt.xlim(0, 20)
    plt.legend()
    plt.grid()

    if savefig:
        plt.savefig(f'{plot_path}/far_field_comparison_grasp_datasets_{frequency_label.replace(" ", "_").replace("/", "-")}_{savename_suffix}.png', dpi=300)
    if show_plots:
        plt.show()

    #! Plot 1(d): Far Field comparision between FFT(CST APERTURE PROFILE)
    plt.figure(figsize=(10, 6))
    if comparision_CST_data is not None:
        for i in range(len(angle_array_cst)):
            print(f"Plotting far field comparison for {comparision_CST_data[i]['plot_label']}")
            plt.plot(angle_array_cst[i], power_dB_array_cst[i], label='{}'.format(comparision_CST_data[i]['plot_label']), linewidth=2, linestyle=(0, (1, 1)))

    plt.xlabel('Angle (degrees)')
    plt.ylabel('Power (dB)')
    plt.title(f'Far Field Comparison between CST Datasets ({frequency_label})')
    plt.ylim(-60, 0)
    plt.xlim(0, 20)
    plt.legend()
    plt.grid()

    if savefig:
        plt.savefig(f'{plot_path}/far_field_comparison_cst_datasets_{frequency_label.replace(" ", "_").replace("/", "-")}_{savename_suffix}.png', dpi=300)
    if show_plots:
        plt.show()

    #! Plot 1(e): Far Field comparision between GRASP datasets of comparision_GRASP_fft_data if provided
    if comparision_GRASP_fft_data is not None:
        plt.figure(figsize=(10, 6))
        for i in range(len(comparision_GRASP_fft_data)):
            print(f"Plotting far field comparison for {comparision_GRASP_fft_data[i]['plot_label']}")
            plt.plot(comparision_GRASP_fft_data[i]['angle'], comparision_GRASP_fft_data[i]['power_dB'], label='{}'.format(comparision_GRASP_fft_data[i]['plot_label']), linewidth=2, linestyle='-.')

        plt.xlabel('Angle (degrees)')
        plt.ylabel('Power (dB)')
        plt.title(f'Far Field Comparison between FFB (calculated in GRASP) Datasets ({frequency_label})')
        plt.ylim(-60, 0)
        plt.xlim(0, 20)
        plt.legend()
        plt.grid()

        if savefig:
            plt.savefig(f'{plot_path}/far_field_comparison_grasp_ffb_datasets_{frequency_label.replace(" ", "_").replace("/", "-")}_{savename_suffix}.png', dpi=300)
        if show_plots:
            plt.show()

    #! Plot 1(f): Far Field comparision between CST datasets of comparision_CST_fft_data if provided
        if comparision_CST_fft_data is not None:
            plt.figure(figsize=(10, 6))
            for i in range(len(comparision_CST_fft_data)):
                print(f"Plotting far field comparison for {comparision_CST_fft_data[i]['plot_label']}")
                plt.plot(comparision_CST_fft_data[i]['angle'], comparision_CST_fft_data[i]['power_dB'], label='{}'.format(comparision_CST_fft_data[i]['plot_label']), linewidth=2, linestyle=(0, (1, 1)))

            plt.xlabel('Angle (degrees)')
            plt.ylabel('Power (dB)')
            plt.title(f'Far Field Comparison between FFB (calculated in CST) Datasets ({frequency_label})')
            plt.ylim(-60, 0)
            plt.xlim(0, 20)
            plt.legend()
            plt.grid()

            if savefig:
                plt.savefig(f'{plot_path}/far_field_comparison_cst_ffb_datasets_{frequency_label.replace(" ", "_").replace("/", "-")}_{savename_suffix}.png', dpi=300)
            if show_plots:
                plt.show()

    #! Plot 1(g): Far Field comparision between FFB(calculated in GRASP) and FFB(calculated in CST) datasets if ONE OF them is provided
    if (comparision_GRASP_fft_data is not None) or (comparision_CST_fft_data is not None):
        plt.figure(figsize=(10, 6))
        if comparision_GRASP_fft_data is not None:
            for i in range(len(comparision_GRASP_fft_data)):
                print(f"Plotting far field comparison for {comparision_GRASP_fft_data[i]['plot_label']}")
                plt.plot(comparision_GRASP_fft_data[i]['angle'], comparision_GRASP_fft_data[i]['power_dB'], label='{}'.format(comparision_GRASP_fft_data[i]['plot_label']), linewidth=2, linestyle='-.')
        if comparision_CST_fft_data is not None:
            for i in range(len(comparision_CST_fft_data)):
                print(f"Plotting far field comparison for {comparision_CST_fft_data[i]['plot_label']}")
                plt.plot(comparision_CST_fft_data[i]['angle'], comparision_CST_fft_data[i]['power_dB'], label='{}'.format(comparision_CST_fft_data[i]['plot_label']), linewidth=2, linestyle=(0, (1, 1)))

        plt.xlabel('Angle (degrees)')
        plt.ylabel('Power (dB)')
        plt.title(f'Far Field Comparison between FFB (calculated in GRASP and CST) Datasets ({frequency_label})')
        plt.ylim(-60, 0)
        plt.xlim(0, 20)
        plt.legend()
        plt.grid()

        if savefig:
            plt.savefig(f'{plot_path}/far_field_comparison_grasp_cst_ffb_datasets_{frequency_label.replace(" ", "_").replace("/", "-")}_{savename_suffix}.png', dpi=300)
        if show_plots:
            plt.show()

    #! Plot 1(h): Far Field comparision between FFB(calculated in GRASP), FFB(calculated in CST) and Plot 1(b) FFB calculated for all datasets
    if (comparision_GRASP_data is not None) or (comparision_CST_data is not None):
        plt.figure(figsize=(10, 6))
        # Plot for MEEPSAT
        plt.plot(angle_array_meepsat, power_dB_array_meepsat, 'b-', label=meepsat_fft_dict['plot_label'], linewidth=2)
        if comparision_GRASP_data is not None:
            for i in range(len(angle_array_grasp_list)):
                print(f"Plotting far field comparison for {comparision_GRASP_data[i]['plot_label']}")
                plt.plot(angle_array_grasp_list[i], power_dB_array_grasp_list[i], label='{}'.format(comparision_GRASP_data[i]['plot_label']), linewidth=2, linestyle='-.', marker='x')
        if comparision_CST_data is not None:
            for i in range(len(angle_array_cst)):
                print(f"Plotting far field comparison for {comparision_CST_data[i]['plot_label']}")
                plt.plot(angle_array_cst[i], power_dB_array_cst[i], label='{}'.format(comparision_CST_data[i]['plot_label']), linewidth=1.5, linestyle=(0, (1, 1)), marker='o')
        
        if comparision_GRASP_fft_data is not None:
            for i in range(len(comparision_GRASP_fft_data)):
                print(f"Plotting far field comparison for {comparision_GRASP_fft_data[i]['plot_label']}")
                plt.plot(comparision_GRASP_fft_data[i]['angle'], comparision_GRASP_fft_data[i]['power_dB'], label='{}'.format(comparision_GRASP_fft_data[i]['plot_label']), marker='x', linewidth=1, linestyle='-.')
        
        if comparision_CST_fft_data is not None:
            for i in range(len(comparision_CST_fft_data)):
                print(f"Plotting far field comparison for {comparision_CST_fft_data[i]['plot_label']}")
                plt.plot(comparision_CST_fft_data[i]['angle'], comparision_CST_fft_data[i]['power_dB'], label='{}'.format(comparision_CST_fft_data[i]['plot_label']), marker='o', linewidth=1, linestyle=(0, (1, 1)))

        plt.xlabel('Angle (degrees)')
        plt.ylabel('Power (dB)')
        plt.title(f'Far Field Comparison between FFB(GRASP and CST) and FFT(CST, GRASP, MeepSAT) Datasets ({frequency_label})')
        plt.ylim(-60, 0)
        plt.xlim(0, 20)
        plt.legend()
        plt.grid()

        if savefig:
            plt.savefig(f'{plot_path}/far_field_comparison_grasp_cst_ffb_and_all_datasets_{frequency_label.replace(" ", "_").replace("/", "-")}_{savename_suffix}.png', dpi=300)
        if show_plots:
            plt.show()

    #! Plot 1(i): Gaussian Fit to the main beam of FFB from MEEPSAT, GRASP and CST if provided
    if gaussian_fit_main_beam:
        meepsat_fitting_results = fit_gaussian_main_beam(angle=angle_array_meepsat, 
                                                         powerdB=power_dB_array_meepsat, 
                                                         aper_size= aper_size,
                                                         wvl=simulation_wvl,
                                                         threshold_dB=gaussian_fit_power_threshold)
        
        print("MEEPSAT Gaussian Fit Results:", meepsat_fitting_results)
        # Plot MEEPSAT fit
        plt.figure(figsize=(10, 6))
        plt.plot(angle_array_meepsat, power_dB_array_meepsat, 'b-', label='MEEPSAT FFB', linewidth=2)
        if meepsat_fitting_results is not None:
            plt.plot(meepsat_fitting_results['fitted_curve_angles'], meepsat_fitting_results['fitted_curve_dB'], 'r--', label=f"MeepSAT;FWHM: {meepsat_fitting_results['fwhm']:.2f} deg, FWHM(Th): {meepsat_fitting_results['theoretical_fwhm']:.2f} deg, HPBW: {meepsat_fitting_results['hpbw']:.2f} deg, R²: {meepsat_fitting_results['r_squared']:.3f}", 
                     linewidth=2, alpha=0.7)
            
        # Plot GRASP fit if provided
        if comparision_GRASP_data is not None:
            for i in range(len(angle_array_grasp_list)):
                grasp_fitting_results = fit_gaussian_main_beam(angle=angle_array_grasp_list[i], 
                                                               powerdB=power_dB_array_grasp_list[i], 
                                                               aper_size= aper_size,
                                                               wvl=simulation_wvl,
                                                               threshold_dB=gaussian_fit_power_threshold)
                # print(f"GRASP ({comparision_GRASP_data[i]['plot_label']}) Gaussian Fit Results:", grasp_fitting_results)
                plt.plot(angle_array_grasp_list[i], power_dB_array_grasp_list[i], label='GRASP ({})'.format(comparision_GRASP_data[i]['plot_label']), linewidth=2, linestyle='-.', marker='x')
                if grasp_fitting_results is not None:
                    plt.plot(grasp_fitting_results['fitted_curve_angles'], grasp_fitting_results['fitted_curve_dB'], 'r--', label=f"GRASP ({comparision_GRASP_data[i]['plot_label']});FWHM: {grasp_fitting_results['fwhm']:.2f} deg, FWHM(Th): {grasp_fitting_results['theoretical_fwhm']:.2f} deg, HPBW: {grasp_fitting_results['hpbw']:.2f} deg, R²: {grasp_fitting_results['r_squared']:.3f}", 
                             linewidth=2, alpha=0.7)
                    
        # Plot CST fit if provided
        if comparision_CST_data is not None:
            for i in range(len(angle_array_cst)):
                cst_fitting_results = fit_gaussian_main_beam(angle=angle_array_cst[i], 
                                                             powerdB=power_dB_array_cst[i], 
                                                             aper_size= aper_size,
                                                             wvl=simulation_wvl,
                                                             threshold_dB=gaussian_fit_power_threshold)
                # print(f"CST ({comparision_CST_data[i]['plot_label']}) Gaussian Fit Results:", cst_fitting_results)
                plt.plot(angle_array_cst[i], power_dB_array_cst[i], label='CST ({})'.format(comparision_CST_data[i]['plot_label']), linewidth=1.5, linestyle=(0, (1, 1)), marker='o')
                if cst_fitting_results is not None:
                    plt.plot(cst_fitting_results['fitted_curve_angles'], cst_fitting_results['fitted_curve_dB'], 'r--', label=f"CST ({comparision_CST_data[i]['plot_label']});FWHM: {cst_fitting_results['fwhm']:.2f} deg, FWHM(Th): {cst_fitting_results['theoretical_fwhm']:.2f} deg, HPBW: {cst_fitting_results['hpbw']:.2f} deg, R²: {cst_fitting_results['r_squared']:.3f}", 
                             linewidth=2, alpha=0.7)
                    
        # Plot if comparision_GRASP_fft_data is provided
        if comparision_GRASP_fft_data is not None:
            for i in range(len(comparision_GRASP_fft_data)):
                grasp_fft_fitting_results = fit_gaussian_main_beam(angle=comparision_GRASP_fft_data[i]['angle'], 
                                                                   powerdB=comparision_GRASP_fft_data[i]['power_dB'], 
                                                                   aper_size= aper_size,
                                                                   wvl=simulation_wvl,
                                                                   threshold_dB=gaussian_fit_power_threshold)
                # print(f"GRASP FFB ({comparision_GRASP_fft_data[i]['plot_label']}) Gaussian Fit Results:", grasp_fft_fitting_results)
                plt.plot(comparision_GRASP_fft_data[i]['angle'], comparision_GRASP_fft_data[i]['power_dB'], label='GRASP FFB ({})'.format(comparision_GRASP_fft_data[i]['plot_label']), linewidth=1, linestyle='-.', marker='x')
                if grasp_fft_fitting_results is not None:
                    plt.plot(grasp_fft_fitting_results['fitted_curve_angles'], grasp_fft_fitting_results['fitted_curve_dB'], 'r--', label=f"GRASP FFB ({comparision_GRASP_fft_data[i]['plot_label']});FWHM: {grasp_fft_fitting_results['fwhm']:.2f} deg, FWHM(Th): {grasp_fft_fitting_results['theoretical_fwhm']:.2f} deg, HPBW: {grasp_fft_fitting_results['hpbw']:.2f} deg, R²: {grasp_fft_fitting_results['r_squared']:.3f}", 
                             linewidth=2, alpha=0.7)
                    
        # Plot if comparision_CST_fft_data is provided
        if comparision_CST_fft_data is not None:
            for i in range(len(comparision_CST_fft_data)):
                cst_fft_fitting_results = fit_gaussian_main_beam(angle=comparision_CST_fft_data[i]['angle'], 
                                                                 powerdB=comparision_CST_fft_data[i]['power_dB'], 
                                                                 aper_size= aper_size,
                                                                 wvl=simulation_wvl,
                                                                 threshold_dB=gaussian_fit_power_threshold)
                # print(f"CST FFB ({comparision_CST_fft_data[i]['plot_label']}) Gaussian Fit Results:", cst_fft_fitting_results)
                plt.plot(comparision_CST_fft_data[i]['angle'], comparision_CST_fft_data[i]['power_dB'], label='CST FFB ({})'.format(comparision_CST_fft_data[i]['plot_label']), linewidth=1, linestyle=(0, (1, 1)), marker='o')
                if cst_fft_fitting_results is not None:
                    plt.plot(cst_fft_fitting_results['fitted_curve_angles'], cst_fft_fitting_results['fitted_curve_dB'], 'r--', label=f"CST FFB ({comparision_CST_fft_data[i]['plot_label']});FWHM: {cst_fft_fitting_results['fwhm']:.2f} deg, FWHM(Th): {cst_fft_fitting_results['theoretical_fwhm']:.2f} deg, HPBW: {cst_fft_fitting_results['hpbw']:.2f} deg, R²: {cst_fft_fitting_results['r_squared']:.3f}", 
                             linewidth=2, alpha=0.7)
                    
        plt.xlabel('Angle (degrees)')
        plt.ylabel('Power (dB)')
        plt.title(f'Far Field Gaussian Fit Comparison ({frequency_label})')
        plt.ylim(-60, 0)
        plt.xlim(0, 20)
        plt.legend()
        plt.grid()

        if savefig:
            plt.savefig(f'{plot_path}/far_field_gaussian_fit_comparison_{frequency_label.replace(" ", "_").replace("/", "-")}_{savename_suffix}.png', dpi=300)
        if show_plots:
            plt.show()

    #! Plot 2: Phase slice comparison
        phase_slice_avg = ez_phase_avg_degrees[:, x_index]
        phase_slice_last = ez_phase_last_degrees[:, x_index]
        # Subtract the centre value that corresponds to y=0
        indexy0 = (np.abs(y_coords - 0)).argmin()
        phase_slice_avg -= phase_slice_avg[indexy0]
        phase_slice_last -= phase_slice_last[indexy0]

        plt.figure(figsize=(10, 6))
        plt.plot(y_coords, phase_slice_avg, 'b-', label='MeepSAT Time AvG', linewidth=2)
        # plt.plot(y_coords, phase_slice_last, 'r--', label='Last Timestep', linewidth=2)
        if comparision_CST_data is not None:
            for i in range(len(comparision_CST_data)):
                # Find CST phase at y=0 to subtract
                indexy0_cst = (np.abs(cst_efield[i]['Y'] - 0)).argmin()
                cst_efield[i]['Phase_degrees'] -= cst_efield[i]['Phase_degrees'][indexy0_cst]
                plt.plot(cst_efield[i]['Y'], cst_efield[i]['Phase_degrees'], 'g-.', label='CST Data', linewidth=2)

        if comparision_GRASP_data is not None:
            for i in range(len(comparision_GRASP_data)):
                # Find GRASP phase at y=0 to subtract
                indexy0_grasp = (np.abs(y_grasp[i] - 0)).argmin()
                grasp_phase_slice[i] -= grasp_phase_slice[i][indexy0_grasp]
                plt.plot(y_grasp[i], grasp_phase_slice[i], label='GRASP ({})'.format(comparision_GRASP_data[i]['plot_label']), linewidth=2)
        plt.xlabel('Y (mm)')
        plt.ylabel('Phase (degrees)')
        plt.title(f'E-field Phase Slice Comparison at x = {x_coords[x_index]:.1f} mm ({frequency_label})')
        plt.legend()
        plt.grid()

        if savefig:
            plt.savefig(f'{plot_path}/efield_phase_slice_comparison_{frequency_label.replace(" ", "_").replace("/", "-")}_{savename_suffix}.png', dpi=300)
        if show_plots:
            plt.show()
        
        #! Plot 3: 2D magnitude comparison (side by side)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Time averaged
        im1 = ax1.imshow(
            10 * np.log10(np.abs(ez_magnitude_avg)**2 / norm_factor),
            extent=[np.min(x_coords), np.max(x_coords), np.min(y_coords), np.max(y_coords)],
            cmap='viridis', vmin=gaussian_fit_power_threshold, vmax=0
        )
        ax1.axvline(x=x_coords[x_index], color='r', linestyle='--', alpha=0.7)
        ax1.set_xlabel('X (mm)')
        ax1.set_ylabel('Y (mm)')
        ax1.set_title(f'Time Averaged E-field Magnitude**2 ({frequency_label})')
        ax1.grid()
        plt.colorbar(im1, ax=ax1, label='Power (dB)')
        
        # Last timestep
        im2 = ax2.imshow(
            10 * np.log10(np.abs(ez_magnitude_last)**2 / norm_factor),
            extent=[np.min(x_coords), np.max(x_coords), np.min(y_coords), np.max(y_coords)],
            cmap='viridis', vmin=gaussian_fit_power_threshold, vmax=0
        )
        ax2.axvline(x=x_coords[x_index], color='r', linestyle='--', alpha=0.7)
        ax2.set_xlabel('X (mm)')
        ax2.set_ylabel('Y (mm)')
        ax2.set_title(f'Last Timestep E-field Magnitude**2 ({frequency_label})')
        ax2.grid()
        plt.colorbar(im2, ax=ax2, label='Power (dB)')
        
        plt.tight_layout()

        if savefig:
            plt.savefig(f'{plot_path}/efield_magnitude_2D_comparison_{frequency_label.replace(" ", "_").replace("/", "-")}_{savename_suffix}.png', dpi=300)
        if show_plots:
            plt.show()
        
        #! Plot 4: 2D phase comparison (side by side)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Time averaged
        im1 = ax1.imshow(
            ez_phase_avg_degrees,
            extent=[np.min(x_coords), np.max(x_coords), np.min(y_coords), np.max(y_coords)],
            cmap='twilight', vmin=-180, vmax=180
        )
        ax1.axvline(x=x_coords[x_index], color='r', linestyle='--', alpha=0.7)
        ax1.set_xlabel('X (mm)')
        ax1.set_ylabel('Y (mm)')
        ax1.set_title(f'Time Averaged E-field Phase ({frequency_label})')
        ax1.grid()
        plt.colorbar(im1, ax=ax1, label='Phase (degrees)')
        
        # Last timestep
        im2 = ax2.imshow(
            ez_phase_last_degrees,
            extent=[np.min(x_coords), np.max(x_coords), np.min(y_coords), np.max(y_coords)],
            cmap='twilight', vmin=-180, vmax=180
        )
        ax2.axvline(x=x_coords[x_index], color='r', linestyle='--', alpha=0.7)
        ax2.set_xlabel('X (mm)')
        ax2.set_ylabel('Y (mm)')
        ax2.set_title(f'Last Timestep E-field Phase ({frequency_label})')
        ax2.grid()
        plt.colorbar(im2, ax=ax2, label='Phase (degrees)')
        
        plt.tight_layout()

        if savefig:
            plt.savefig(f'{plot_path}/efield_phase_2D_comparison_{frequency_label.replace(" ", "_").replace("/", "-")}_{savename_suffix}.png', dpi=300)
        if show_plots:
            plt.show()

        #! Plot 5: 2D Poynting vector comparison (side by side)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Time averaged
        im1 = ax1.imshow(
            10 * np.log10(averaged_poynting_vector / norm_factor),
            extent=[np.min(x_coords), np.max(x_coords), np.min(y_coords), np.max(y_coords)],
            cmap='viridis', vmin=gaussian_fit_power_threshold, vmax=0
        )
        ax1.axvline(x=x_coords[x_index], color='r', linestyle='--', alpha=0.7)
        ax1.set_xlabel('X (mm)')
        ax1.set_ylabel('Y (mm)')
        ax1.set_title(f'Time Averaged Poynting Vector Magnitude ({frequency_label})')
        ax1.grid()
        plt.colorbar(im1, ax=ax1, label='Power (dB)')   

        # Last timestep
        im2 = ax2.imshow(
            10 * np.log10(total_poynting_last / norm_factor),
            extent=[np.min(x_coords), np.max(x_coords), np.min(y_coords), np.max(y_coords)],
            cmap='viridis', vmin=gaussian_fit_power_threshold, vmax=0
        )
        ax2.axvline(x=x_coords[x_index], color='r', linestyle='--', alpha=0.7)
        ax2.set_xlabel('X (mm)')
        ax2.set_ylabel('Y (mm)')
        ax2.set_title(f'Last Timestep Poynting Vector Magnitude ({frequency_label})')
        ax2.grid()
        plt.colorbar(im2, ax=ax2, label='Power (dB)')   
        plt.tight_layout()

        if savefig:
            plt.savefig(f'{plot_path}/poynting_vector_magnitude_2D_comparison_{frequency_label.replace(" ", "_").replace("/", "-")}_{savename_suffix}.png', dpi=300)
        if show_plots:
            plt.show()

        #! Plot 6: Poynting vector slice comparison at x_position
        poynting_slice_avg = averaged_poynting_vector[:, x_index]/np.max(averaged_poynting_vector[:, x_index])
        poynting_slice_last = total_poynting_last[:, x_index]
        plt.figure(figsize=(10, 6))
        plt.plot(y_coords, 10 * np.log10(poynting_slice_avg / norm_factor), 
                'b-', label='MeepSAT Time AvG', linewidth=2)
        # plt.plot(y_coords, 10 * np.log10(poynting_slice_last / norm_factor), 
        #          'r--', label='Last Timestep', linewidth=2)
        if comparision_CST_data is not None:
            for i in range(len(comparision_CST_data)):
                plt.plot(cst_s_mag[i]['Y'], cst_s_mag[i]['S_Mag_dB'], 'g-.', label='CST Data', linewidth=2)
        plt.ylim(-30, 0)
        plt.xlabel('Y (mm)')  
        plt.ylabel('S Magnitude (dB)')
        plt.title(f'Poynting Vector Magnitude Slice Comparison at x = {x_coords[x_index]:.1f} mm ({frequency_label})')
        plt.legend()
        plt.grid()

        if savefig:
            plt.savefig(f'{plot_path}/poynting_vector_magnitude_slice_comparison_{frequency_label.replace(" ", "_").replace("/", "-")}_{savename_suffix}.png', dpi=300)
        if show_plots:
            plt.show()

        #! Plot 7: POYNTING ANALYSIS: 
        # Extract the S field distrubution for the full box with only source
        # Run the real simulation 
        # Divide those two
        if average_source_power is not None:
            sum_source_power_dB = 10 * np.log10(np.sum(average_source_power)/np.max(average_source_power)) 
            print(f"Sum of Source Power: {np.sum(average_source_power):.4e}")
            print(f"Sum of Source Power (dB): {sum_source_power_dB:.4f} dB")
            normalized_poynting_vector_avg = averaged_poynting_vector / sum_source_power_dB

            # Plot contour levels
            contour_levels_dB = np.linspace(contour_db_range[0], 
                                            contour_db_range[1], 
                                            contour_levels)
            contour_levels_linear = np.linspace(contour_linear_range[0],
                                                contour_linear_range[1],
                                                contour_levels)

            # Plot only in dB scale for now
            plt.figure(figsize=(18, 6))
            cs = plt.contourf(
                x_coords, y_coords, 10 * np.log10(normalized_poynting_vector_avg),
                levels=contour_levels_dB, cmap='viridis'
            )
            plt.axvline(x=x_coords[x_index], color='r', linestyle='--', alpha=0.7)
            plt.xlabel('X (mm)')
            plt.ylabel('Y (mm)')
            plt.title(f'Time Avg $S$/Sun(source power) ({frequency_label})')
            plt.colorbar(cs, label='Power (dB)')
            plt.grid()

            if savefig:
                plt.savefig(f'{plot_path}/poynting_vector_divided_by_source_power_{frequency_label.replace(" ", "_").replace("/", "-")}_{savename_suffix}.png', dpi=300)
            if show_plots:
                plt.show()

            #! Plot 7(b): Poynting vector divided by integrated source power
            # Using trapz to integrate the source power over the y axis (ycoords)
            integrated_source_power = np.trapz(average_source_power, y_coords)
            print(f"Integrated Source Power: {integrated_source_power:.4e}")
            integrated_source_power_dB = 10 * np.log10(integrated_source_power/np.max(average_source_power))
            print(f"Integrated Source Power (dB): {integrated_source_power_dB:.4f} dB")

            averaged_poynting_vector_divided_source = averaged_poynting_vector / integrated_source_power_dB
            # Plot contour levels
            contour_levels_dB = np.linspace(contour_db_range[0], 
                                            contour_db_range[1], 
                                            contour_levels)
            contour_levels_linear = np.linspace(contour_linear_range[0],
                                                contour_linear_range[1],
                                                contour_levels)
            # Plot only in dB scale for now
            plt.figure(figsize=(18, 6))
            cs = plt.contourf(
                x_coords, y_coords, 10 * np.log10(averaged_poynting_vector_divided_source),
                levels=contour_levels_dB, cmap='viridis'
            )
            plt.axvline(x=x_coords[x_index], color='r', linestyle='--', alpha=0.7)
            plt.xlabel('X (mm)')
            plt.ylabel('Y (mm)')
            plt.title(f'Time Avg $S$/Integrated Source Power ({frequency_label})')
            plt.colorbar(cs, label='Power (dB)')
            plt.grid()

            if savefig:
                plt.savefig(f'{plot_path}/poynting_vector_divided_by_integrated_source_power_{frequency_label.replace(" ", "_").replace("/", "-")}_{savename_suffix}.png', dpi=300)
            if show_plots:
                plt.show()

        #! Plot 7(c): Doing the same thing, but for the average power calculated from avg efield
        if average_source_power is not None:
            sum_source_power_dB = 10 * np.log10(np.sum(average_source_power)/np.max(average_source_power))
            print(f"Sum of Source Power: {np.sum(average_source_power):.4e}")
            print(f"Sum of Source Power (dB): {sum_source_power_dB:.4f} dB")
            normalized_efieldpower_vector_avg = 10*np.log10(np.abs(ez_magnitude_avg)**2) / sum_source_power_dB
            # Plot contour levels
            contour_levels_dB = np.linspace(contour_db_range[0], 
                                            contour_db_range[1], 
                                            contour_levels)
            contour_levels_linear = np.linspace(contour_linear_range[0],
                                                contour_linear_range[1],
                                                contour_levels) 
            
            # Plot only in dB scale for now
            plt.figure(figsize=(18, 6))
            cs = plt.contourf(
                x_coords, y_coords, normalized_efieldpower_vector_avg,
                levels=contour_levels_dB, cmap='viridis'
            )
            plt.axvline(x=x_coords[x_index], color='r', linestyle='--', alpha=0.7)
            plt.xlabel('X (mm)')
            plt.ylabel('Y (mm)')
            plt.title(f'Time Avg |E|^2/Sun(source power) ({frequency_label})')
            plt.colorbar(cs, label='Power (dB)')
            plt.grid()

            if savefig:
                plt.savefig(f'{plot_path}/efield_power_divided_by_source_power_{frequency_label.replace(" ", "_").replace("/", "-")}_{savename_suffix}.png', dpi=300)
            if show_plots:
                plt.show()

            # ! Plot 7(d): Poynting vector divided by integrated source power
            # Using trapz to integrate the source power over the y axis (ycoords)
            integrated_source_power = np.trapz(average_source_power, y_coords)
            print(f"Integrated Source Power: {integrated_source_power:.4e}")
            integrated_source_power_dB = 10 * np.log10(integrated_source_power/np.max(average_source_power))
            print(f"Integrated Source Power (dB): {integrated_source_power_dB:.4f} dB")
            averaged_Efieldpower_divided_by_source = 10*np.log10(np.abs(ez_magnitude_avg)**2) / integrated_source_power_dB
            
            # Plot in dB scale for now
            plt.figure(figsize=(18, 6))
            cs = plt.contourf(
                x_coords, y_coords, averaged_Efieldpower_divided_by_source,
                levels=contour_levels_dB, cmap='viridis'
            )
            plt.axvline(x=x_coords[x_index], color='r', linestyle='--', alpha=0.7)
            plt.xlabel('X (mm)')
            plt.ylabel('Y (mm)')
            plt.title(f'Time Avg |E|^2/Integrated Source Power ({frequency_label})')
            plt.colorbar(cs, label='Power (dB)')
            plt.grid()

            if savefig:
                plt.savefig(f'{plot_path}/efield_power_divided_by_integrated_source_power_{frequency_label.replace(" ", "_").replace("/", "-")}_{savename_suffix}.png', dpi=300)
            if show_plots:
                plt.show()    
        

        # #! Plot 7: Poynting vectot divided by source (2D) comparison (side by side) in linear scale
        # if norm_factor is not None:
        #     fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        #     # Time averaged
        #     im1 = ax1.imshow(
        #         averaged_poynting_vector_divided_source,
        #         extent=[np.min(x_coords), np.max(x_coords), np.min(y_coords), np.max(y_coords)],
        #         cmap='viridis'
        #     )
        #     ax1.axvline(x=x_coords[x_index], color='r', linestyle='--', alpha=0.7)
        #     ax1.set_xlabel('X (mm)')
        #     ax1.set_ylabel('Y (mm)')
        #     ax1.set_title(f'Time Averaged Poynting Vector / Source ({frequency_label})')
        #     ax1.grid()
        #     plt.colorbar(im1, ax=ax1, label='Normalized Power')   

        #     # Last timestep
        #     im2 = ax2.imshow(
        #         total_poynting_last_divided_source,
        #         extent=[np.min(x_coords), np.max(x_coords), np.min(y_coords), np.max(y_coords)],
        #         cmap='viridis'
        #     )
        #     ax2.axvline(x=x_coords[x_index], color='r', linestyle='--', alpha=0.7)
        #     ax2.set_xlabel('X (mm)')
        #     ax2.set_ylabel('Y (mm)')
        #     ax2.set_title(f'Last Timestep Poynting Vector / Source ({frequency_label})')
        #     ax2.grid()
        #     plt.colorbar(im2, ax=ax2, label='Normalized Power')   

        #     plt.tight_layout()
        #     plt.show()

        # #! Plot 8: Same as that of Plot 7 but in contour plot
        # if norm_factor is not None:
        #     fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        #     # Time averaged
        #     cs1 = ax1.contourf(
        #         x_coords, y_coords, averaged_poynting_vector_divided_source,
        #         levels=contour_levels, cmap='viridis'
        #     )
        #     ax1.axvline(x=x_coords[x_index], color='r', linestyle='--', alpha=0.7)
        #     ax1.set_xlabel('X (mm)')
        #     ax1.set_ylabel('Y (mm)')
        #     ax1.set_title(f'Time Averaged Poynting Vector / Source ({frequency_label})')
        #     ax1.grid()
        #     plt.colorbar(cs1, ax=ax1, label='Normalized Power')   

        #     # Last timestep
        #     cs2 = ax2.contourf(
        #         x_coords, y_coords, total_poynting_last_divided_source,
        #         levels=contour_levels, cmap='viridis'
        #     )
        #     ax2.axvline(x=x_coords[x_index], color='r', linestyle='--', alpha=0.7)
        #     ax2.set_xlabel('X (mm)')
        #     ax2.set_ylabel('Y (mm)')
        #     ax2.set_title(f'Last Timestep Poynting Vector / Source ({frequency_label})')
        #     ax2.grid()
        #     plt.colorbar(cs2, ax=ax2, label='Normalized Power')   

        #     plt.tight_layout()
        #     plt.show()

        
        return {
            'ez_magnitude_avg': ez_magnitude_avg,
            'ez_phase_avg_degrees': ez_phase_avg_degrees,
            'ez_magnitude_last': ez_magnitude_last,
            'ez_phase_last_degrees': ez_phase_last_degrees,
            'x_coords': x_coords,
            'y_coords': y_coords,
            'aperture_slice_avg': aperture_slice_avg,
            'aperture_slice_last_timestep': aperture_slice_last,

        }
    


def average_source_power(source_power_files, aper_size, freq, efield_index = 2):
    import numpy as np
    import glob
    
    source_power_data = [np.load(f) for f in source_power_files]
    
    # Initialize sums
    mag_array = []

    for i, data in enumerate(source_power_data):
        field = data['field'][efield_index]
        #print(field)

        real = np.real(field)
        imag = np.imag(field)
        magnitude = np.sqrt(real**2+ imag**2)

        mag_array.append(magnitude)
    power_array = np.abs(np.array(mag_array))**2

    power_avg = np.mean(power_array, axis=0)

    power_dB = 10 * np.log10(power_avg / np.max(power_avg))

    # Replace -Ninf values to -50 dB
    # power_dB = np.where(power_dB < -60, -61, power_dB)

    def gauss_profile(x, A, x0, w, B):
        return A * np.exp(-2 * (x - x0)**2 / w**2) + B
    
    
    y = np.linspace(-aper_size, aper_size, len(power_avg))
    A0, x00, w0_guess, B0 = 1, 0, aper_size/4, -30 

    from scipy.optimize import curve_fit
    
    popt, pcov = curve_fit(gauss_profile, 
                           y, 
                           power_avg, 
                           p0=[A0, x00, w0_guess, B0])

    A_fit, x0_fit, w_fit, B_fit = popt
    beam_waist = np.abs(w_fit/2)
    fwhm = 2*beam_waist * np.sqrt(2 * np.log(2))
    fitted_curve = gauss_profile(y, *popt)

    # Plot the averaged source power
    plt.figure(figsize=(10, 6))
    plt.plot(y, power_avg, 'b-', linewidth=2)
    if fitted_curve is not None:
        plt.plot(y, fitted_curve, 'r--', label=f'Fitted Gaussian (Waist={beam_waist:.2f}, FWHM={fwhm:.2f})', linewidth=2)
        plt.legend()
    #plt.ylim(-30, 0)
    plt.xlabel('Y (mm)')
    plt.ylabel('Power (dB)')
    plt.title('Averaged Source Power Profile (Freq: {} GHz)'.format(freq))
    plt.grid()
    plt.show()

    return power_avg



def get_time_arrays(current_dir, freq_folder_array, resolution='10'):
    """
    For each frequency folder, finds all aperture_power_*.npz files,
    extracts the time values, sorts them numerically, and returns a list of arrays.
    """
    time_array = []
    for freq_folder in freq_folder_array:
        freq_time_array = []
        freq_output_dir = os.path.join(current_dir, 'output_files', resolution, freq_folder)
        
        if os.path.exists(freq_output_dir):
            for file in os.listdir(freq_output_dir):
                if file.startswith('aperture_power_') and file.endswith('.npz'):
                    # Extract the numeric part between 'aperture_power_' and '.npz'
                    time_str = file.replace('aperture_power_', '').replace('.npz', '')
                    freq_time_array.append(time_str)
            
            # Convert to numpy array and sort numerically
            freq_time_array = np.array(sorted(freq_time_array, key=float))
            time_array.append(freq_time_array)
        else:
            print(f"Warning: Directory {freq_output_dir} does not exist")
            time_array.append(np.array([]))
    return time_array

def extract_last_timestep_aperture_data(aperture_efield_list, power_dB_func):
    """
    Extracts the last time sample from each aperture efield data entry,
    computes power in linear and dB scale, and returns a list of dicts.
    """
    last_timestep_aperture_data = []
    for data in aperture_efield_list:
        last_efield = data['efield_list'][-1]
        last_power_dB = power_dB_func(np.abs(last_efield)**2)
        last_timestep_data = {
            'frequency': data['frequency'],
            'y_coords': data['y_coords'],
            'power_linear': np.abs(last_efield)**2,
            'power_dB': last_power_dB
        }
        last_timestep_aperture_data.append(last_timestep_data)
    return last_timestep_aperture_data

def time_average_aperture_data(aperture_efield_list, time_array, keys_to_average, time_average_efield_squared, power_dB):
    """
    Time-averages the aperture efield data for each frequency.
    Returns a list of dicts with averaged power (linear and dB).
    """
    averaged_aperture_data = []
    for i, data in enumerate(aperture_efield_list):
        averaged_data = {'frequency': data['frequency'], 'y_coords': data['y_coords']}
        for key in keys_to_average:
            power_linear = time_average_efield_squared(
                np.abs(data[key])**2,
                np.array(time_array[i], dtype=float)
            )
            averaged_data['power_avg_linear'] = power_linear
            averaged_data['power_avg_dB'] = power_dB(power_linear)
        averaged_aperture_data.append(averaged_data)
    return averaged_aperture_data

def plot_aperture_power_profiles(
    averaged_aperture_data,
    last_timestep_aperture_data,
    ylim_db=(-60, 0),
    ylim_linear=None,
    figsize=(15, 10),
    suptitle=None,
    savepath=None
):
    """
    Plots the averaged aperture power profiles and their differences for each frequency.
    Allows control over y-limits and figure size.
    """
    import matplotlib.pyplot as plt

    for data, last_timestep_data in zip(averaged_aperture_data, last_timestep_aperture_data):
        freq = data['frequency']
        y_coords = data['y_coords']

        plt.figure(figsize=figsize)

        # dB plot
        plt.subplot(2, 3, 1)
        plt.plot(y_coords, data['power_avg_dB'], label='Aperture Power (dB) - Inst. Avg over time')
        plt.plot(last_timestep_data['y_coords'], last_timestep_data['power_dB'], '--', label='Aperture Power (Last Time Sample)', color='gray')
        plt.title(f'Aperture Power (dB) at {freq}')
        plt.xlabel('Y Coordinate')
        plt.ylabel('Power (dB)')
        if ylim_db is not None:
            plt.ylim(*ylim_db)
        plt.grid()
        plt.legend()

        # Linear plot
        plt.subplot(2, 3, 2)
        plt.plot(y_coords, data['power_avg_linear'], label='Aperture Power (Linear) - Inst. Avg over time', color='orange')
        plt.plot(last_timestep_data['y_coords'], last_timestep_data['power_linear'], '--', label='Aperture Power (Last Time Sample)', color='gray')
        plt.title(f'Aperture Power (Linear) at {freq}')
        plt.xlabel('Y Coordinate')
        plt.ylabel('Power (Linear)')
        if ylim_linear is not None:
            plt.ylim(*ylim_linear)
        plt.grid()
        plt.legend()

        # Difference plot in dB
        plt.subplot(2, 3, 3)
        diff_dB = data['power_avg_dB'] - last_timestep_data['power_dB']
        plt.plot(y_coords, diff_dB, color='red', linewidth=2)
        plt.title(f'Difference (Inst. Avg - Last) dB at {freq}')
        plt.xlabel('Y Coordinate')
        plt.ylabel('Power Difference (dB)')
        plt.grid()
        plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)

        # Difference plot in linear scale
        plt.subplot(2, 3, 4)
        diff_linear = data['power_avg_linear'] - last_timestep_data['power_linear']
        plt.plot(y_coords, diff_linear, color='purple', linewidth=2)
        plt.title(f'Difference (Inst. Avg - Last) Linear at {freq}')
        plt.xlabel('Y Coordinate')
        plt.ylabel('Power Difference (Linear)')
        plt.grid()
        plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)

        # Individual plots for average profiles dB
        plt.subplot(2, 3, 5)
        plt.plot(last_timestep_data['y_coords'], last_timestep_data['power_dB'], '--', label='Aperture Power (Last Time Sample)', color='gray')
        plt.title(f'Aperture Power (dB) - At last timestep {freq}')
        plt.xlabel('Y Coordinate')
        plt.ylabel('Power (dB)')
        if ylim_db is not None:
            plt.ylim(*ylim_db)
        plt.grid()
        plt.legend()

        plt.subplot(2, 3, 6)
        plt.plot(y_coords, data['power_avg_dB'], label='Aperture Power (dB) - Inst. Avg over time', color='blue')
        plt.title(f'Aperture Power (dB) - Inst. Avg at {freq}')
        plt.xlabel('Y Coordinate')
        plt.ylabel('Power (dB)')
        if ylim_db is not None:
            plt.ylim(*ylim_db)
        plt.grid()
        plt.legend()

        plt.tight_layout()
        if suptitle:
            plt.suptitle(suptitle)
        if savepath:
            plt.savefig(savepath)
        plt.show()

def readHDF5(file_path):
    """
    Reads an HDF5 file and return the data.
    Handles both simple keys and nested group structures.
    """
    def read_recursive(group, data_dict):
        """Recursively read HDF5 groups and datasets"""
        for key in group.keys():
            item = group[key]
            if isinstance(item, h5py.Group):
                # If it's a group, create a nested dictionary
                data_dict[key] = {}
                read_recursive(item, data_dict[key])
            elif isinstance(item, h5py.Dataset):
                # If it's a dataset, read the data
                try:
                    data_dict[key] = item[()]
                except Exception as e:
                    print(f"Warning: Could not read dataset '{key}': {e}")
                    data_dict[key] = None
        return data_dict
    
    with h5py.File(file_path, 'r') as f:
        data = {}
        data = read_recursive(f, data)
    return data


def plot_grasp_meep_comparison(grasp_aperture_data, averaged_aperture_data, last_timestep_aperture_data,
                               frequency='90.0', aperture_size_mm=50, plot_type='moving_average',
                               sampling_factor=5, window_size=3, 
                               plot_params=None, show_avg=True, show_last_timestep=True):
    """
    Plot comparison between GRASP and MEEP aperture data for specified frequency.
    
    Parameters:
    -----------
    grasp_aperture_data : dict
        GRASP data loaded from HDF5 file
    averaged_aperture_data : list
        Time-averaged MEEP aperture data
    last_timestep_aperture_data : list
        Last timestep MEEP aperture data
    frequency : str
        Frequency to plot (e.g., '90.0' for 90 GHz)
    aperture_size_mm : float
        Aperture size in mm for GRASP y-axis scaling
    plot_type : str
        Type of plot: 'raw', 'sampled', 'moving_average'
    sampling_factor : int
        Factor for sampling every N points (used when plot_type='sampled')
    window_size : int
        Window size for moving average (used when plot_type='moving_average')
    plot_params : dict
        Dictionary with plotting parameters (figsize, colors, linestyles, etc.)
    show_avg : bool
        Whether to show time-averaged MEEP data
    show_last_timestep : bool
        Whether to show last timestep MEEP data
    
    Returns:
    --------
    fig, ax : matplotlib figure and axis objects
    """
    
    # Default plotting parameters
    default_params = {
        'figsize': (10, 6),
        'meep_avg_color': 'blue',
        'meep_last_color': 'cyan',
        'grasp_color': 'red',
        'meep_avg_linestyle': '-',
        'meep_last_linestyle': '--',
        'grasp_linestyle': '--',
        'ylim': (-5, 0),
        'xlabel': 'Position (mm)',
        'ylabel': 'Normalized Power (dB)',
        'title_prefix': 'Aperture Power Profile Comparison at',
        'grid': True,
        'legend': True
    }
    
    # Update default parameters with user-provided ones
    if plot_params:
        default_params.update(plot_params)
    
    # Process GRASP data
    freq_key = frequency.split('.')[0]  # Convert '90.0' to '90'
    if freq_key not in grasp_aperture_data:
        raise ValueError(f"Frequency {freq_key} not found in GRASP data. Available: {list(grasp_aperture_data.keys())}")
    
    ez_grasp = grasp_aperture_data[freq_key]['Ez']
    ex_grasp = grasp_aperture_data[freq_key]['Ex']
    ey_grasp = grasp_aperture_data[freq_key]['Ey']
    e_grasp = np.sqrt(np.abs(ez_grasp)**2 + np.abs(ex_grasp)**2 + np.abs(ey_grasp)**2)
    
    # Extract middle row and convert to dB
    mid_row_index = e_grasp.shape[0] // 2
    mid_row_e = e_grasp[mid_row_index, :]
    amplitude_e = np.abs(mid_row_e)**2
    amplitude_e_norm = amplitude_e / np.max(amplitude_e)
    amplitude_e_db = 10 * np.log10(amplitude_e_norm + 1e-12)
    y_grasp = np.linspace(-aperture_size_mm/2, aperture_size_mm/2, len(amplitude_e_db))
    
    # Create plot
    fig, ax = plt.subplots(figsize=default_params['figsize'])
    
    # Find and plot MEEP data for specified frequency
    freq_label = f'freq_{frequency}GHz'
    meep_data_found = False
    
    for data, last_timestep_data in zip(averaged_aperture_data, last_timestep_aperture_data):
        if data['frequency'] == freq_label:
            meep_data_found = True
            
            if show_avg:
                # Process averaged data based on plot type
                if plot_type == 'raw':
                    x_avg, y_avg = data['y_coords'], data['power_avg_dB']
                elif plot_type == 'sampled':
                    x_avg = data['y_coords'][::sampling_factor]
                    y_avg = data['power_avg_dB'][::sampling_factor]
                elif plot_type == 'moving_average':
                    x_avg, y_avg, _ = apply_moving_average_scipy(
                        x_coords=data['y_coords'],
                        power_data=data['power_avg_dB'],
                        window_size=window_size
                    )
                else:
                    raise ValueError("plot_type must be 'raw', 'sampled', or 'moving_average'")
                
                ax.plot(x_avg, y_avg, 
                       label=f'MEEP {frequency} GHz (Avg)', 
                       color=default_params['meep_avg_color'],
                       linestyle=default_params['meep_avg_linestyle'])
            
            if show_last_timestep:
                # Process last timestep data based on plot type
                if plot_type == 'raw':
                    x_last, y_last = last_timestep_data['y_coords'], last_timestep_data['power_dB']
                elif plot_type == 'sampled':
                    x_last = last_timestep_data['y_coords'][::sampling_factor]
                    y_last = last_timestep_data['power_dB'][::sampling_factor]
                elif plot_type == 'moving_average':
                    x_last, y_last, _ = apply_moving_average_scipy(
                        x_coords=last_timestep_data['y_coords'],
                        power_data=last_timestep_data['power_dB'],
                        window_size=window_size
                    )
                
                ax.plot(x_last, y_last, 
                       label=f'MEEP {frequency} GHz (Last Timestep)', 
                       color=default_params['meep_last_color'],
                       linestyle=default_params['meep_last_linestyle'])
            
            break
    
    if not meep_data_found:
        print(f"Warning: No MEEP data found for frequency {freq_label}")
        print(f"Available frequencies: {[data['frequency'] for data in averaged_aperture_data]}")
    
    # Plot GRASP data
    ax.plot(y_grasp, amplitude_e_db, 
           label=f'GRASP {frequency} GHz', 
           color=default_params['grasp_color'],
           linestyle=default_params['grasp_linestyle'])
    
    # Set plot properties
    ax.set_xlabel(default_params['xlabel'])
    ax.set_ylabel(default_params['ylabel'])
    ax.set_title(f"{default_params['title_prefix']} {frequency} GHz")
    ax.set_ylim(default_params['ylim'])
    
    if default_params['grid']:
        ax.grid()
    if default_params['legend']:
        ax.legend()
    
    plt.tight_layout()
    plt.show()
    
    return fig, ax

#==========================================================================================================
def apply_moving_average_scipy(x_coords, power_data, window_size):
    """
    Apply moving average using scipy's uniform filter - most efficient for uniform grids.
    """
    # Calculate window size in terms of array indices
    dx = np.mean(np.diff(x_coords))  # assuming roughly uniform spacing
    window_indices = int(window_size / dx)
    
    if window_indices < 1:
        window_indices = 1
    
    # Apply uniform filter
    power_avg = ndimage.uniform_filter1d(power_data, size=window_indices, mode='nearest')
    
    return x_coords, power_avg, np.zeros_like(power_avg)  # std not computed efficiently here

def apply_moving_average_pandas(x_coords, power_data, window_size):
    """
    Apply moving average using pandas rolling window - efficient and flexible.
    """
    df = pd.DataFrame({'x': x_coords, 'power': power_data})
    df = df.sort_values('x')  # Ensure sorted
    
    # Calculate number of points in window
    dx = np.median(np.diff(df['x']))
    window_points = max(1, int(window_size / dx))
    
    # Apply rolling window
    rolling = df['power'].rolling(window=window_points, center=True, min_periods=1)
    power_avg = rolling.mean().values
    power_std = rolling.std().fillna(0).values
    
    return df['x'].values, power_avg, power_std


def power_dB(power):
    power_dB = 10 * np.log10(power / np.max(power))
    return power_dB

def time_average_efield_squared(efield_squared_list, time_array):
    """
    Function to compute the instantaneous time average of electric fields over the provided time samples.

    Parameters:
    efield_squared_list (list of np.ndarray): List of electric field squared arrays at different time samples.
    time_array (np.ndarray): Array of time samples.
    """

    # Convert list to numpy array with proper shape (time, space)
    efield_squared_list = np.array(efield_squared_list)
    
    # # Debug print to check shapes
    # print(f"efield_array shape: {efield_array.shape}")
    # print(f"time_array shape: {time_array.shape}")
    # print(f"time_array values: {time_array}")
    

    # # Ensure time_array is 1D
    # time_array = np.array(time_array).flatten()
    
    # # Calculating the total time interval
    T = time_array[-1] - time_array[0]
    print(f"Total time interval T: {T}")

    print(f"Number of time samples: {len(time_array)}")
    N= len(efield_squared_list)
    
    # # Calculating the instantaneous time average using trapz along time axis (axis=0)
    # efield_squared_avg = np.trapz(efield_array, time_array, axis=0) / T
    efield_squared_avg = np.sum(efield_squared_list, axis=0)/N #T

    return efield_squared_avg

def time_avg(list_of_arrays):
    """
    Function to compute the average of a list of arrays.
    """
    return np.mean(list_of_arrays, axis=0)/len(list_of_arrays)


def load_npz_data(file_path):
    """
    Function to load data from a .npz file.
    """
    data = np.load(file_path)
    return data

def load_h5_data(file_path):
    """
    Function to load data from a .h5 file.
    """
    with h5py.File(file_path, 'r') as f:
        data = {key: f[key][()] for key in f.keys()}
    return data

def extract_efield_list_aperture_npz(current_dir, freq_folder_array, time_array, file_format, resolution='10'):
    """
    Function to extract electric field data from multiple .npz files based on the provided time samples.

    Parameters:
    time_array (np.ndarray): list of time samples at different frequencies
    file_format (str): Format string for the .npz files, e.g., 'absorber_power_{}.npz'.

    Returns:
    list of np.ndarray: List of electric field arrays corresponding to each time sample.
    """
    
    efield_list = []
    current_dir = os.path.join(current_dir, 'output_files', resolution)
    
    for freq, time_array in zip(freq_folder_array, time_array):
        base_dir = os.path.join(current_dir, freq)
        efield_list_for_freq = []
    
        for t in time_array:
            # Replace 'i' with the actual time sample in the file name
            file_name = file_format.replace('i', t)
            file_path = os.path.join(base_dir, file_name)
            
            if os.path.exists(file_path):
                data = load_npz_data(file_path)
                efield = data['field']
                efield_list_for_freq.append(efield)
            else:
                print(f"File {file_path} does not exist.")
        
        freq_dict = {
            'frequency': freq,
            'efield_list': efield_list_for_freq,
            'y_coords': data['y_coords']
        }
        efield_list.append(freq_dict)
        
    return efield_list

def extract_efield_list_absorber_npz(current_dir, freq_folder_array, time_array, file_format, resolution='10'):
    """
    Function to extract electric field data from multiple .npz files based on the provided time samples.

    Parameters:
    time_array (np.ndarray): list of time samples at different frequencies
    file_format (str): Format string for the .npz files, e.g., 'absorber_power_{}.npz'.

    Returns:
    list of np.ndarray: List of electric field arrays corresponding to each time sample.

    Note: the absorber data keys are:
        top_field,
        bottom_field,
        top_power,
        bottom_power,
        top_edge_field,
        bottom_edge_field,
        top_edge_power,
        bottom_edge_power,
        x_coords
    """
    
    #print("Time samples to be loaded:", time_array)
    # list of dictionaries
    data_list = []
    
    # Adding output_files to the current directory
    current_dir= os.path.join(current_dir, 'output_files', resolution)

    # Processing each frequency sample
    for freq, time_array in zip(freq_folder_array, time_array):
        #print(f"Processing frequency folder: {freq}")
        base_dir = os.path.join(current_dir, freq)
        #print(f"Base directory set to: {base_dir}")
        # Initializing lists to store data for each frequency
        top_field = []
        bottom_field = []
        top_power = []
        bottom_power = []
        top_edge_field = []
        bottom_edge_field = []
        top_edge_power = []
        bottom_edge_power = []
    
        # Processing each time sample    
        for t in time_array:
            # Replace i with the actual time sample in the file name
            file_name = file_format.replace('time_i', t)
            #print(f"Loading file: {file_name}")
            file_path = os.path.join(base_dir, file_name)
            
            if os.path.exists(file_path):
                data = load_npz_data(file_path)
                # Appending the data to respective lists
                top_field.append(data['top_field'])
                bottom_field.append(data['bottom_field'])
                top_power.append(data['top_power'])
                bottom_power.append(data['bottom_power'])
                top_edge_field.append(data['top_edge_field'])
                bottom_edge_field.append(data['bottom_edge_field'])
                top_edge_power.append(data['top_edge_power'])
                bottom_edge_power.append(data['bottom_edge_power'])

            else:
                print(f"File {file_path} does not exist.")
        
        # Creating a dictionary for the current frequency
        freq_dict = {
            'frequency': freq,
            'x_coords': data['x_coords'],
            'top_field': top_field,
            'bottom_field': bottom_field,
            'top_power': top_power,
            'bottom_power': bottom_power,
            'top_edge_field': top_edge_field,
            'bottom_edge_field': bottom_edge_field,
            'top_edge_power': top_edge_power,
            'bottom_edge_power': bottom_edge_power
        }
        data_list.append(freq_dict)

    # Returning the list of dictionaries containing data for each frequency
    return data_list



###=========================================================================================================
#^ =========================================================================================================
#! ----------------------------- BEAM ANALYSIS MODULES -----------------------------------------------------
###=========================================================================================================
#^ =========================================================================================================

def plotting_enhanced(fftfreq=None, FFTs=None, wvl=None, aper_size=None,
                     # New parameters for timestep averaging
                     fft_profiles_by_timesteps=None,
                     fftfreq_profiles_by_timesteps=None,
                     # Existing parameters
                     grasp_data=None,
                     grasp_label="GRASP",
                     grasp_methods=None,
                     deg_range=20,
                     ylim=-60, 
                     symmetric_beam=True,
                     legend=None,
                     print_solid_angle=False,
                     print_fwhm=False,
                     show_theoretical_fwhm=False,
                     show_best_fit_fwhm=False,
                     show_fwhm_in_legend=False,
                     show_inset_zoom=True,
                     inset_range=(0, 1.5),
                     show_difference_plot=True,
                     show_r_squared=True,
                     analyze_to_first_null=True,
                     analyze_grasp_nulls=True,
                     calculate_meep_grasp_r_squared=True,
                     comparison_angle_range=None,
                     fwhm_marker_color='grey',
                     fwhm_marker_style='--',
                     title=None,
                     savefig=False,
                     path_name='plots/meep_guide_plot',
                     threshold_db_MEEPSAT=-20,
                     threshold_db_GRASP=-20,
                     seq_col=False,
                     # New parameter for timestep averaging options
                     timestep_averaging='mean'):  # 'mean', 'median', 'last'
    '''
    Enhanced plotting function with MEEP-GRASP R² analysis and timestep averaging capability.
    
    Parameters:
    -----------
    fftfreq : array-like or list of arrays, optional
        If single array: frequency array shared by all FFTs
        If list: separate frequency array for each FFT
        (Used when fft_profiles_by_timesteps is None)
    FFTs : list of arrays, optional
        List of FFT arrays to plot
        (Used when fft_profiles_by_timesteps is None)
    
    New Parameters for Timestep Averaging:
    -------------------------------------
    fft_profiles_by_timesteps : list of arrays, optional
        List of FFT arrays at different timesteps for averaging
        If provided, takes precedence over FFTs parameter
    fftfreq_profiles_by_timesteps : array-like, optional
        Frequency array corresponding to fft_profiles_by_timesteps
        If provided, takes precedence over fftfreq parameter
    timestep_averaging : str, optional
        Method for averaging timesteps: 'mean', 'median', 'last' (default: 'mean')
    
    Other Parameters:
    ----------------
    wvl : float
        Wavelength for angular conversion
    aper_size : float
        Aperture size for theoretical FWHM calculation
    [... other existing parameters ...]
    '''
    from scipy.interpolate import interp1d
    from matplotlib.patches import Rectangle
    import matplotlib.patches as mpatches
    
    # Handle timestep averaging mode
    if fft_profiles_by_timesteps is not None and fftfreq_profiles_by_timesteps is not None:
        print(f"Using timestep averaging mode with {timestep_averaging} method")
        print(f"Number of timesteps: {len(fft_profiles_by_timesteps)}")
        
        # Convert to numpy array for easier manipulation
        fft_array = np.array(fft_profiles_by_timesteps)
        
        # Perform averaging based on method
        if timestep_averaging == 'mean':
            averaged_fft = np.mean(fft_array, axis=0)
            print(f"Applied mean averaging across {len(fft_profiles_by_timesteps)} timesteps")
        elif timestep_averaging == 'median':
            averaged_fft = np.median(fft_array, axis=0)
            print(f"Applied median averaging across {len(fft_profiles_by_timesteps)} timesteps")
        elif timestep_averaging == 'last':
            averaged_fft = fft_array[-1]
            print(f"Using last timestep (index {len(fft_profiles_by_timesteps)-1})")
        else:
            raise ValueError(f"Unknown timestep_averaging method: {timestep_averaging}")
        
        # Set up for standard processing
        fftfreq = fftfreq_profiles_by_timesteps
        FFTs = [averaged_fft]
        
        # Update legend if not provided
        if legend is None:
            legend = [f'MEEPSAT ({timestep_averaging} of {len(fft_profiles_by_timesteps)} timesteps)']
        
        # Update title if not provided
        if title is None:
            title = f"Far Field Pattern ({timestep_averaging.title()} of {len(fft_profiles_by_timesteps)} timesteps)"
    
    # Validate required parameters
    if fftfreq is None or FFTs is None:
        raise ValueError("Either (fftfreq, FFTs) or (fft_profiles_by_timesteps, fftfreq_profiles_by_timesteps) must be provided")
    
    if wvl is None or aper_size is None:
        raise ValueError("wvl and aper_size must be provided")
    
    # Handle both single frequency array and multiple frequency arrays
    if isinstance(fftfreq, list):
        # Multiple frequency arrays provided
        freq_arrays = fftfreq
        if len(freq_arrays) != len(FFTs):
            raise ValueError(f"Number of frequency arrays ({len(freq_arrays)}) must match number of FFTs ({len(FFTs)})")
    else:
        # Single frequency array for all FFTs
        freq_arrays = [fftfreq] * len(FFTs)
    
    # Convert frequencies to angles for each dataset
    deg_arrays = []
    rads_arrays = []
    for freq_array in freq_arrays:
        deg = np.arctan(freq_array * wvl) * 180 / np.pi
        rads = np.array(deg) * np.pi / 180
        deg_arrays.append(deg)
        rads_arrays.append(rads)

    col = plt.cm.jet(np.linspace(0,1,len(FFTs))) 
    
    # Determine if we need difference plots
    need_difference_plot = show_difference_plot and (grasp_data is not None or len(FFTs) > 1)
    
    # Create figure with subplots if showing differences
    if need_difference_plot:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), height_ratios=[2, 1])
        plt.subplots_adjust(hspace=0.3)
    else:
        fig, ax1 = plt.subplots(1, 1, figsize=(8, 6))
    
    def gaussian(x, stddev, mean):
        return np.exp(-(((x-mean)/4/stddev)**2))
    
    def calculate_r_squared(y_actual, y_predicted):
        """Calculate R² coefficient of determination"""
        ss_res = np.sum((y_actual - y_predicted) ** 2)
        ss_tot = np.sum((y_actual - np.mean(y_actual)) ** 2)
        return 1 - (ss_res / ss_tot)
    
    def calculate_meep_grasp_comparison(meep_angles, meep_power_dB, grasp_angles, grasp_power_dB, 
                                     angle_range=None, label="GRASP"):
        """
        Calculate R² and other metrics between MEEP and GRASP data.
        """
        # Find overlapping angle range
        min_angle = max(np.min(meep_angles), np.min(grasp_angles))
        max_angle = min(np.max(meep_angles), np.max(grasp_angles))
        
        # Apply user-specified range if provided
        if angle_range is not None:
            min_angle = max(min_angle, angle_range[0])
            max_angle = min(max_angle, angle_range[1])
        
        if min_angle >= max_angle:
            print(f"Warning: No overlapping range for {label}")
            return None
        
        # Filter MEEP data to comparison range
        meep_mask = (meep_angles >= min_angle) & (meep_angles <= max_angle)
        meep_angles_filtered = meep_angles[meep_mask]
        meep_power_filtered = meep_power_dB[meep_mask]
        
        if len(meep_angles_filtered) < 3:
            print(f"Warning: Insufficient MEEP data points for {label}")
            return None
        
        # Interpolate GRASP data to MEEP angles
        try:
            grasp_interp = interp1d(grasp_angles, grasp_power_dB, 
                                  kind='linear', bounds_error=False, fill_value=np.nan)
            grasp_interpolated = grasp_interp(meep_angles_filtered)
            
            # Remove NaN values
            valid_mask = ~np.isnan(grasp_interpolated)
            if np.sum(valid_mask) < 3:
                print(f"Warning: Insufficient valid interpolated points for {label}")
                return None
            
            meep_valid = meep_power_filtered[valid_mask]
            grasp_valid = grasp_interpolated[valid_mask]
            angles_valid = meep_angles_filtered[valid_mask]
            
            # Calculate metrics
            r_squared = calculate_r_squared(meep_valid, grasp_valid)
            rms_diff = np.sqrt(np.mean((meep_valid - grasp_valid)**2))
            mean_abs_diff = np.mean(np.abs(meep_valid - grasp_valid))
            max_diff = np.max(np.abs(meep_valid - grasp_valid))
            correlation = np.corrcoef(meep_valid, grasp_valid)[0, 1]
            
            return {
                'label': label,
                'r_squared': r_squared,
                'correlation': correlation,
                'rms_difference': rms_diff,
                'mean_abs_difference': mean_abs_diff,
                'max_difference': max_diff,
                'comparison_range': (min_angle, max_angle),
                'n_points': len(meep_valid),
                'meep_angles': angles_valid,
                'meep_power': meep_valid,
                'grasp_power': grasp_valid
            }
            
        except Exception as e:
            print(f"Error in comparison for {label}: {e}")
            return None
    
    def find_first_null(angles, power_linear, main_peak_idx, threshold_db=threshold_db_MEEPSAT):
        """Find the first null/minimum after the main peak"""
        for i in range(main_peak_idx + 1, len(power_linear) - 1):
            if power_linear[i] < power_linear[i-1] and power_linear[i] < power_linear[i+1]:
                if power_linear[i] < 10**(threshold_db/10) * np.max(power_linear):
                    return i, angles[i]
        return None, None
    
    def find_grasp_first_null(grasp_angles, grasp_power_dB, threshold_db=threshold_db_GRASP):
        """Find the first null for GRASP data (power already in dB)"""
        grasp_power_linear = 10**(grasp_power_dB/10)
        main_peak_idx = np.argmax(grasp_power_linear)
        null_idx, null_angle = find_first_null(grasp_angles, grasp_power_linear, main_peak_idx, threshold_db)
        return null_idx, null_angle
    
    # Store analysis results
    theoretical_fwhm_values = []
    best_fit_fwhm_values = []
    r_squared_values = []
    first_null_positions = []
    grasp_null_positions = []
    meep_grasp_comparisons = []
    
    # Store processed MEEP data for difference calculations
    meep_datasets = []
    meep_angles = None
    meep_power_dB = None
    
    # Add timestep information to results if applicable
    timestep_info = {}
    if fft_profiles_by_timesteps is not None:
        timestep_info = {
            'method': timestep_averaging,
            'n_timesteps': len(fft_profiles_by_timesteps),
            'timestep_std': np.std(fft_array, axis=0) if timestep_averaging == 'mean' else None
        }
    
    # Process MEEPSAT data
    for k in range(len(FFTs)):
        deg = deg_arrays[k]
        rads = rads_arrays[k]
        
        fft_k = (FFTs[k] / (np.cos(rads)**2))**2
        fft_k = fft_k / np.max(fft_k)
        fft_dB = 10*np.log10(fft_k)
        middle = int(len(fft_k)/2)

        # Store MEEPSAT data for comparison (use first dataset for GRASP comparison)
        if k == 0:
            meep_angles = deg[:middle]
            meep_power_dB = fft_dB[:middle]
        
        # Store all MEEP datasets for difference calculations
        meep_datasets.append({
            'angles': deg[:middle],
            'power_dB': fft_dB[:middle],
            'label': legend[k] if legend is not None and k < len(legend) else f'MEEPSAT {k}'
        })

        # BEAM SOLID ANGLE CALCULATION
        if print_solid_angle:
            x_span = np.append(rads, 0)
            integrand = np.append(fft_k, fft_k[0])
            integrand *= np.sin(x_span)
            right_part = np.trapz(integrand[:middle], x = x_span[:middle])
            solid_angle = right_part
            print('Beam n.{} solid angle : {:.3e} srads'.format(k, solid_angle*2*np.pi))
        
        # Calculate theoretical FWHM
        fwhm_th = wvl/aper_size*180/np.pi
        theoretical_fwhm_values.append(fwhm_th)
        
        # Find first null
        maxidx = np.argmax(fft_k[:middle])
        first_null_idx, first_null_angle = find_first_null(deg[:middle], fft_k[:middle], maxidx)
        first_null_positions.append(first_null_angle)
        
        # Initialize values
        best_fit_fwhm = None
        r_squared = None
        
        # FWHM fitting
        if print_fwhm or show_best_fit_fwhm or show_fwhm_in_legend or show_r_squared:
            maxidx = np.argmax(fft_k[:middle])
            if maxidx == len(fft_k[:middle]) - 1:
                maxidx = 0
            
            # Determine fitting range
            if analyze_to_first_null and first_null_idx is not None:
                fit_end_idx = first_null_idx
            else:
                # Original method: find where beam starts rising again
                i = 0
                while (maxidx + i < len(fft_k[:middle]) - 1 and 
                       fft_k[maxidx + i] > fft_k[maxidx + i + 1]):
                    i += 1
                fit_end_idx = maxidx + i

            xdata = deg[maxidx - min(maxidx, fit_end_idx - maxidx) : fit_end_idx]
            ydata = fft_k[maxidx - min(maxidx, fit_end_idx - maxidx) : fit_end_idx]
            
            if len(xdata) > 3:  # Need at least 3 points for fitting
                try:
                    p0 = [1, 0] if maxidx <= 10 else [1, 1]
                    popt, pcov = sc.curve_fit(gaussian, xdata, ydata, p0=p0)
                    fwhm = np.abs(4*popt[0]*np.sqrt(np.log(2)))
                    best_fit_fwhm = 2*fwhm
                    
                    # Calculate R²
                    y_predicted = gaussian(xdata, popt[0], popt[1])
                    r_squared = calculate_r_squared(ydata, y_predicted)
                    
                    best_fit_fwhm_values.append(best_fit_fwhm)
                    r_squared_values.append(r_squared)
                    
                    if print_fwhm:
                        print(f'Beam {k} - Best fit Gaussian FWHM: {best_fit_fwhm:.2f}deg (R²={r_squared:.4f})')
                        print(f'Beam {k} - Theoretical FWHM: {fwhm_th:.2f}deg')
                        if first_null_angle:
                            print(f'Beam {k} - First null at: {first_null_angle:.2f}deg')
                        
                        # Print timestep info if available
                        if timestep_info:
                            print(f'Beam {k} - Averaged using {timestep_info["method"]} of {timestep_info["n_timesteps"]} timesteps')
                        
                        # Plot Gaussian fit
                        gauss = gaussian(deg[:middle], popt[0], popt[1]) + 1e-10
                        y = 10*np.log10(gauss)
                        ax1.plot(deg[:middle], y, linestyle='--', color=col[k], alpha=0.7)
                        
                except Exception as e:
                    print(f"Gaussian fitting failed for beam {k}: {e}")
                    best_fit_fwhm_values.append(None)
                    r_squared_values.append(None)
            else:
                best_fit_fwhm_values.append(None)
                r_squared_values.append(None)
        else:
            best_fit_fwhm_values.append(None)
            r_squared_values.append(None)
        
        # Create legend label
        if legend is not None:
            label = legend[k]
            if show_fwhm_in_legend and best_fit_fwhm is not None:
                label += f' (Theory: {fwhm_th:.2f}°, Fit: {best_fit_fwhm:.2f}°'
                if show_r_squared and r_squared is not None:
                    label += f', R²={r_squared:.3f}'
                label += ')'
            elif show_fwhm_in_legend:
                label += f' (Theory: {fwhm_th:.2f}°)'
            
            ax1.plot(deg[:middle], fft_dB[:middle], label=label, color=col[k], linewidth=2)
        else:
            ax1.plot(deg[:middle], fft_dB[:middle], color=col[k], linewidth=2)

    # Add GRASP data and perform MEEP-GRASP comparisons
    grasp_datasets = []
    if grasp_data is not None:
        if isinstance(grasp_data, dict):
            grasp_datasets = [grasp_data]
            grasp_labels = [grasp_label]
            methods = [grasp_methods[0] if grasp_methods else "Unknown method"]
        elif isinstance(grasp_data, list):
            grasp_datasets = grasp_data
            grasp_labels = grasp_label if isinstance(grasp_label, list) else [f"{grasp_label}_{i}" for i in range(len(grasp_data))]
            methods = grasp_methods if grasp_methods else [f"Method {i}" for i in range(len(grasp_data))]
        
        colors = ['red', 'blue', 'green', 'orange', 'purple']
        linestyles = ['--', '-.', ':', '--', '-.']
        
        for i, (dataset, label, method) in enumerate(zip(grasp_datasets, grasp_labels, methods)):
            ax1.plot(dataset['angle'], dataset['power_dB'], 
                    color=colors[i % len(colors)], 
                    linestyle=linestyles[i % len(linestyles)], 
                    label=f'{label} ({method})',
                    linewidth=1.5)
            
            # Perform MEEP-GRASP comparison if requested
            if calculate_meep_grasp_r_squared and meep_angles is not None and meep_power_dB is not None:
                comparison = calculate_meep_grasp_comparison(
                    meep_angles, meep_power_dB,
                    np.array(dataset['angle']), np.array(dataset['power_dB']),
                    angle_range=comparison_angle_range,
                    label=f"{label} ({method})"
                )
                if comparison is not None:
                    meep_grasp_comparisons.append(comparison)
            
            # Find GRASP nulls
            if analyze_grasp_nulls:
                if symmetric_beam:
                    mask = np.array(dataset['angle']) >= 0
                    grasp_angles_filtered = np.array(dataset['angle'])[mask]
                    grasp_power_filtered = np.array(dataset['power_dB'])[mask]
                else:
                    grasp_angles_filtered = np.array(dataset['angle'])
                    grasp_power_filtered = np.array(dataset['power_dB'])
                
                null_idx, null_angle = find_grasp_first_null(grasp_angles_filtered, grasp_power_filtered)
                grasp_null_positions.append({
                    'label': label,
                    'method': method,
                    'null_angle': null_angle,
                    'color': colors[i % len(colors)]
                })
                
                if null_angle is not None:
                    print(f'GRASP {label} first null at: {null_angle:.2f}deg')

    # Customize main plot
    if title is not None:
        ax1.set_title(title, fontsize=16)
    
    ax1.set_ylim((ylim, 0))
    ax1.set_xlabel('Angle [deg]', fontsize=14)
    ax1.set_ylabel('Power [dB]', fontsize=14)
    ax1.tick_params(labelsize=12)
    
    if symmetric_beam:
        ax1.set_xlim((0, deg_range))
    else:
        ax1.set_xlim((-deg_range, deg_range))
    
    # Add FWHM markers
    if show_theoretical_fwhm and theoretical_fwhm_values:
        for i, fwhm_th in enumerate(theoretical_fwhm_values):
            if symmetric_beam:
                ax1.axvline(fwhm_th/2, color=fwhm_marker_color, 
                           linestyle='dashdot', alpha=0.7, 
                           label='Theoretical FWHM' if i == 0 else "")
    
    if show_best_fit_fwhm and best_fit_fwhm_values:
        for i, fwhm_bf in enumerate(best_fit_fwhm_values):
            if fwhm_bf is not None:
                if symmetric_beam:
                    ax1.axvline(fwhm_bf/2, color=fwhm_marker_color, 
                               linestyle=fwhm_marker_style, alpha=0.8,
                               label='Best Fit FWHM' if i == 0 else "")

    # Add first null markers for MEEP
    if analyze_to_first_null:
        for i, null_pos in enumerate(first_null_positions):
            if null_pos is not None:
                ax1.axvline(null_pos, color='gray', linestyle=':', alpha=0.5,
                           label=f'MEEPSAT First Null ({round(null_pos, 2)})' if i == 0 else "")

    # Add first null markers for GRASP
    if analyze_grasp_nulls and grasp_null_positions:
        for i, null_info in enumerate(grasp_null_positions):
            if null_info['null_angle'] is not None:
                ax1.axvline(null_info['null_angle'], 
                           color=null_info['color'], 
                           linestyle=':', 
                           alpha=0.8,
                           linewidth=2,
                           label=f"{null_info['label']} First Null ({round(null_info['null_angle'], 2)})" if i < 2 else "")

    ax1.legend(loc='upper right', fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Add inset zoom
    if show_inset_zoom:
        from mpl_toolkits.axes_grid1.inset_locator import inset_axes
        
        axins = inset_axes(ax1, width="40%", height="40%", loc='center right',
                          bbox_to_anchor=(0.95, 0.6, 1, 1), bbox_transform=ax1.transAxes)
        
        # Plot all MEEP datasets in inset
        for k in range(len(FFTs)):
            deg = deg_arrays[k]
            rads = rads_arrays[k]
            fft_k = (FFTs[k] / (np.cos(rads)**2))**2
            fft_k = fft_k / np.max(fft_k)
            fft_dB = 10*np.log10(fft_k)
            middle = int(len(fft_k)/2)
            
            mask = (deg[:middle] >= inset_range[0]) & (deg[:middle] <= inset_range[1])
            axins.plot(deg[:middle][mask], fft_dB[:middle][mask], color=col[k], linewidth=2)
        
        for i, dataset in enumerate(grasp_datasets):
            mask = (np.array(dataset['angle']) >= inset_range[0]) & (np.array(dataset['angle']) <= inset_range[1])
            axins.plot(np.array(dataset['angle'])[mask], np.array(dataset['power_dB'])[mask], 
                      color=colors[i % len(colors)], 
                      linestyle=linestyles[i % len(linestyles)], 
                      linewidth=1.5)
        
        axins.set_xlim(inset_range)
        axins.set_ylim(-20, 0)
        axins.grid(True, alpha=0.3)
        axins.set_title(f'Zoom: {inset_range[0]}-{inset_range[1]}°', fontsize=10)
        
        rect = Rectangle((inset_range[0], ylim), inset_range[1]-inset_range[0], -ylim, 
                        linewidth=1, edgecolor='black', facecolor='none', alpha=0.5)
        ax1.add_patch(rect)

    # Add difference plot
    if need_difference_plot:
        colors_diff = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink']
        
        # MEEP-GRASP differences (if GRASP data exists)
        if grasp_data is not None and meep_angles is not None:
            for i, dataset in enumerate(grasp_datasets):
                grasp_interp = interp1d(dataset['angle'], dataset['power_dB'], 
                                      bounds_error=False, fill_value=np.nan)
                grasp_interpolated = grasp_interp(meep_angles)
                
                difference = meep_power_dB - grasp_interpolated
                
                ax2.plot(meep_angles, difference, 
                        color=colors_diff[i % len(colors_diff)], 
                        label=f'MEEPSAT - {grasp_labels[i]}',
                        linewidth=1.5)
        
        # MEEP-MEEP differences (if multiple MEEP datasets and no GRASP data)
        elif len(meep_datasets) > 1:
            # Use first MEEP dataset as reference
            reference_dataset = meep_datasets[0]
            ref_angles = reference_dataset['angles']
            ref_power = reference_dataset['power_dB']
            
            for i, dataset in enumerate(meep_datasets[1:], 1):
                # Interpolate current dataset to reference angles
                try:
                    interp_func = interp1d(dataset['angles'], dataset['power_dB'], 
                                         bounds_error=False, fill_value=np.nan)
                    interpolated_power = interp_func(ref_angles)
                    
                    # Calculate difference
                    difference = ref_power - interpolated_power
                    
                    # Remove NaN values for plotting
                    valid_mask = ~np.isnan(difference)
                    if np.sum(valid_mask) > 0:
                        ax2.plot(ref_angles[valid_mask], difference[valid_mask], 
                                color=colors_diff[i % len(colors_diff)], 
                                label=f'{reference_dataset["label"]} - {dataset["label"]}',
                                linewidth=1.5)
                        
                        # Print some statistics
                        rms_diff = np.sqrt(np.mean(difference[valid_mask]**2))
                        mean_abs_diff = np.mean(np.abs(difference[valid_mask]))
                        print(f"Difference {reference_dataset['label']} - {dataset['label']}:")
                        print(f"  RMS Difference = {rms_diff:.3f} dB")
                        print(f"  Mean Abs Difference = {mean_abs_diff:.3f} dB")
                        
                except Exception as e:
                    print(f"Error calculating difference for {dataset['label']}: {e}")
        
        # Customize difference plot
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        ax2.set_xlabel('Angle [deg]', fontsize=14)
        ax2.set_ylabel('Difference [dB]', fontsize=14)
        
        if grasp_data is not None:
            ax2.set_title('Difference: MEEPSAT - GRASP', fontsize=14)
        else:
            ax2.set_title('Difference: MEEPSAT Datasets', fontsize=14)
            
        ax2.grid(True, alpha=0.3)
        ax2.legend(fontsize=10)
        
        if symmetric_beam:
            ax2.set_xlim((0, deg_range))
        else:
            ax2.set_xlim((-deg_range, deg_range))

    plt.tight_layout()
    
    if savefig:
        plt.savefig(f'{path_name}.png', dpi=300, bbox_inches='tight')
    
    plt.show()
    
    # Print comprehensive summary
    if show_r_squared and r_squared_values:
        print("\n=== MEEPSAT GAUSSIAN FIT QUALITY ===")
        for i, (r2, fwhm, null_pos) in enumerate(zip(r_squared_values, best_fit_fwhm_values, first_null_positions)):
            if r2 is not None:
                print(f"MEEPSAT Beam {i}: R² = {r2:.4f}, FWHM = {fwhm:.2f}°", end="")
                if null_pos:
                    print(f", First null = {null_pos:.2f}°")
                else:
                    print()
                
                # Print timestep info if available
                if timestep_info:
                    print(f"  Timestep averaging: {timestep_info['method']} of {timestep_info['n_timesteps']} samples")
    
    # Print MEEP-GRASP comparison results
    if calculate_meep_grasp_r_squared and meep_grasp_comparisons:
        print("\n=== MEEPSAT vs GRASP COMPARISON ===")
        for comp in meep_grasp_comparisons:
            print(f"\n{comp['label']}:")
            print(f"  R² = {comp['r_squared']:.4f}")
            print(f"  Correlation = {comp['correlation']:.4f}")
            print(f"  RMS Difference = {comp['rms_difference']:.3f} dB")
            print(f"  Mean Abs Difference = {comp['mean_abs_difference']:.3f} dB")
            print(f"  Max Difference = {comp['max_difference']:.3f} dB")
            print(f"  Comparison Range = {comp['comparison_range'][0]:.2f}° to {comp['comparison_range'][1]:.2f}°")
            print(f"  Number of Points = {comp['n_points']}")
    
    # Print GRASP null summary
    if analyze_grasp_nulls and grasp_null_positions:
        print("\n=== GRASP NULL POSITIONS ===")
        for null_info in grasp_null_positions:
            if null_info['null_angle'] is not None:
                print(f"{null_info['label']} ({null_info['method']}): First null = {null_info['null_angle']:.2f}°")
            else:
                print(f"{null_info['label']} ({null_info['method']}): No null found")
    
    return {
        'r_squared_values': r_squared_values,
        'best_fit_fwhm_values': best_fit_fwhm_values,
        'first_null_positions': first_null_positions,
        'theoretical_fwhm_values': theoretical_fwhm_values,
        'grasp_null_positions': grasp_null_positions,
        'meep_grasp_comparisons': meep_grasp_comparisons,
        'timestep_info': timestep_info  # New: return timestep information
    }


def calculate_grasp_far_field_using_fft(grasp_df, wavelength, window_type='hanning', zero_padding_factor=2, std= 1/6, alpha=0.25):
    """
    Calculate the far field from GRASP data using FFT with windowing and zero-padding.
    
    Parameters:
    grasp_df (DataFrame): DataFrame containing GRASP data with columns 'x', 'y', 'co'
    wavelength (float): Wavelength in same units as x,y coordinates
    window_type (str): Type of window function ('hanning', 'hamming', 'tukey', 'gaussian', None)
    zero_padding_factor (int): Factor by which to zero-pad the data (improves interpolation)
    
    Returns:
    dict: Dictionary containing the far field angles and power in dB
    """
    from scipy import fft
    from scipy.signal import windows
    
    # Get unique coordinates and sort them
    x_unique = np.sort(np.unique(grasp_df['x'].astype(float)))
    y_unique = np.sort(np.unique(grasp_df['y'].astype(float)))
    
    # Get grid dimensions
    nx, ny = len(x_unique), len(y_unique)
    
    # Calculate grid spacing
    dx = x_unique[1] - x_unique[0] if nx > 1 else 1.0
    dy = y_unique[1] - y_unique[0] if ny > 1 else 1.0
    
    # Reshape the complex field data into 2D grid
    co_polarisation_data = np.array(grasp_df['co'], dtype=complex)
    field_2d = co_polarisation_data.reshape((ny, nx))
    
    # Apply window function
    if window_type is not None:
        if window_type == 'hanning':
            window_x = windows.hann(nx)
            window_y = windows.hann(ny)
        elif window_type == 'hamming':
            window_x = windows.hamming(nx)
            window_y = windows.hamming(ny)
        elif window_type == 'tukey':
            window_x = windows.tukey(nx, alpha=alpha)  # 25% taper
            window_y = windows.tukey(ny, alpha=alpha)
        elif window_type == 'gaussian':
            window_x = windows.gaussian(nx, std=nx*std)
            window_y = windows.gaussian(ny, std=ny*std)
        else:
            window_x = np.ones(nx)
            window_y = np.ones(ny)
        
        # Create 2D window
        window_2d = np.outer(window_y, window_x)
        field_2d = field_2d * window_2d
    
    # Zero-padding for smoother interpolation
    if zero_padding_factor > 1:
        nx_padded = nx * zero_padding_factor
        ny_padded = ny * zero_padding_factor
        
        # Create padded array
        field_2d_padded = np.zeros((ny_padded, nx_padded), dtype=complex)
        
        # Place original data in center
        y_start = (ny_padded - ny) // 2
        x_start = (nx_padded - nx) // 2
        field_2d_padded[y_start:y_start+ny, x_start:x_start+nx] = field_2d
        
        field_2d = field_2d_padded
        nx, ny = nx_padded, ny_padded
    
    # Perform 2D FFT
    far_field_2d = fft.fft2(field_2d)
    far_field_2d = fft.fftshift(far_field_2d)
    
    # Calculate power
    far_field_power_2d = np.abs(far_field_2d)**2
    
    # Calculate spatial frequency coordinates (accounting for zero-padding)
    kx = fft.fftshift(fft.fftfreq(nx, dx))
    ky = fft.fftshift(fft.fftfreq(ny, dy))
    
    # Convert spatial frequencies to angles
    theta_x = kx * wavelength
    theta_y = ky * wavelength
    
    # For more accurate angles (avoiding arcsin domain issues)
    theta_x_rad = np.where(np.abs(theta_x) <= 1, np.arcsin(theta_x), np.sign(theta_x) * np.pi/2)
    
    # Extract 1D cuts through the center of the beam
    center_y, center_x = ny // 2, nx // 2
    far_field_power_center_horizontal_cut = far_field_power_2d[center_y, :]
    
    # Convert to dB
    max_power = np.max(far_field_power_center_horizontal_cut)
    far_field_power_dB = 10 * np.log10(far_field_power_center_horizontal_cut / max_power)
    
    # Use the actual theta_x corresponding to the FFT grid
    grasp_far_field = {
        'angle': theta_x_rad * 180 / np.pi,  # Convert to degrees
        'power_dB': far_field_power_dB
    }
    
    return grasp_far_field
            
# Loading the GRASP CSV data
def convert_grasp_to_dict(data):
    # Load the data
    co_polarisation_data = data['Eco']
    cross_polarisation_data = data['Ecx']
    u = np.array(data['Az'])
    v = np.array(data['El'])

    print(f"Co-pol data shape: {co_polarisation_data.shape}")
    print(f"Cross-pol data shape: {cross_polarisation_data.shape}")

    # Convert to power (magnitude squared)
    co_pol_power = np.abs(co_polarisation_data)**2
    cross_pol_power = np.abs(cross_polarisation_data)**2

    # Convert to dB
    co_pol_dB = 10 * np.log10(co_pol_power / np.max(co_pol_power))
    cross_pol_dB = 10 * np.log10(cross_pol_power / np.max(co_pol_power))  # Normalize to co-pol max

    # For 1D cuts through beam center
    center_y, center_x = np.array(co_pol_dB.shape) // 2
    co_pol_center_horizontal_cut = co_pol_dB[center_y, :]
    cross_pol_center_vertical_cut = cross_pol_dB[center_y, :]

    # Creating an dictionry for GRASP
    grasp_data = {
        'angle': u,#* 180 / np.pi,  # Convert radians to degrees
        'power_dB': co_pol_center_horizontal_cut
    }


    return grasp_data


# def custom_beam_FT(sim_res,
#                     list_efields,
#                     aper_size,          
#                     zero_pad = 15,
#                     savebeam = False,
#                     parallel = False,
#                     filename = None):

#     '''
#     Gets the Fourier Transforms of the complex electric fields at aperture.

#     Arguments
#     ---------
#     zero_pad : float, optional
#         Multiplicative factor to the length of the field list,
#         which is padded with zeros in the added length.

#     Returns
#     -------
#     freq : array
#         List of the frequencies at which the FFT has been done
#     FFTs : list of arrays
#         Each array contains the FFT for the k-th source.
#     '''

#     # Ensure list_efields is a list of arrays, not a single value
#     if not isinstance(list_efields, list):
#         list_efields = [list_efields]
    
#     # Check if each element is an array
#     for i, efield in enumerate(list_efields):
#         if not hasattr(efield, '__len__') or np.isscalar(efield):
#             raise ValueError(f"list_efields[{i}] must be an array, got {type(efield)}")

#     #Initialize the list
#     FFTs = [[] for k in range(len(list_efields))]

#     res = sim_res

#     #List of frequencies
#     freq = np.fft.fftfreq(len(list_efields[0])*zero_pad, d = 1/res)

#     #Iterate over the number of sources
#     for k in range(len(list_efields)):

#         #FFT over the field
#         fft = np.fft.fft(list_efields[k], 
#                 n = zero_pad*len(list_efields[k]))

#         #FFT is normalized by its max
#         FFTs[k] = np.abs(fft) 
#         FFTs[k] = FFTs[k]/np.max(FFTs[k])

#     if savebeam:
#         # Helper function to create dataset with appropriate parameters
#         def create_flexible_dataset(h, name, data, dtype):
#             # Check if data is scalar (has no length) or is a 0-dimensional array
#             is_scalar = np.isscalar(data) or (isinstance(data, np.ndarray) and data.ndim == 0)
            
#             if parallel:
#                 # No compression in parallel mode
#                 h.create_dataset(name, data=data, dtype=dtype)
#             else:
#                 # Only apply compression if not scalar
#                 if is_scalar:
#                     h.create_dataset(name, data=data, dtype=dtype)
#                 else:
#                     h.create_dataset(name, data=data, dtype=dtype, compression='gzip')
                
#         if parallel:
#             from mpi4py import MPI
#             comm = MPI.COMM_WORLD
#             if not h5py.get_config().mpi:
#                 raise ValueError("h5py was built without MPI support, can't use mpio driver")
            
#             with h5py.File(filename + '.h5', 'w', driver='mpio', comm=comm) as h:
#                 create_flexible_dataset(h, 'freq', freq, 'float64')
#                 create_flexible_dataset(h, 'beams', FFTs, 'float64')
#                 aper = aper_size
#                 create_flexible_dataset(h, 'aper_size', aper, 'float64')
#         else:
#             # # Only have rank 0 create the file in non-parallel mode
#             # from mpi4py import MPI
#             # comm = MPI.COMM_WORLD
#             # rank = comm.Get_rank()
            
#             # if rank == 0:
#             #     with h5py.File(filename + '.h5', 'w', libver='latest') as h:
#             #         create_flexible_dataset(h, 'freq', freq, 'float64')
#             #         create_flexible_dataset(h, 'beams', FFTs, 'float64')
#             #         aper = aper_size
#             #         create_flexible_dataset(h, 'aper_size', aper, 'float64')
#             # comm.barrier()  # Ensure all processes wait for file creation
#             # Just save without MPI
#             with h5py.File(filename + '.h5', 'w', libver='latest') as h:
#                 create_flexible_dataset(h, 'freq', freq, 'float64')
#                 create_flexible_dataset(h, 'beams', FFTs, 'float64')
#                 aper = aper_size
#                 create_flexible_dataset(h, 'aper_size', aper, 'float64')

#         return freq, FFTs
    

def efield_list_from_monitors_from_lens1(current_dir, freq_folder_array, time_array, file_format, resolution='10'):
    """
    Function to extract electric field data from multiple .npz files based on the provided time samples.

    Parameters:
    freq_folder_array (np.ndarray): Array of frequency folder names.
    time_array (list): List of time arrays, one for each frequency.
    file_format (str): Format string for the .npz files, e.g., 'lens1_1mm_power_time_i.npz'.

    Returns:
    list of dict: List of dictionaries containing electric field data for each frequency.
    """
    
    # list of dictionaries
    efield_list = []
    
    # Adding output_files to the current directory
    current_dir = os.path.join(current_dir, 'output_files', resolution)
    
    # Processing each frequency sample
    for i, freq in enumerate(freq_folder_array):
        print(f"Processing frequency folder: {freq}")
        base_dir = os.path.join(current_dir, freq)
        print(f"Base directory set to: {base_dir}")
        efield_list_for_freq = []
        
        # Get the time array for this specific frequency
        freq_time_array = time_array[i] if i < len(time_array) else []
        print(f"Time samples for {freq}: {freq_time_array}")
    
        # Processing each time sample    
        for t in freq_time_array:
            # Replace time_i with the actual time sample in the file name
            file_name = file_format.replace('time_i', t)
            #print(f"Loading file: {file_name}")
            file_path = os.path.join(base_dir, file_name)
            
            if os.path.exists(file_path):
                data = load_npz_data(file_path)
                efield = data['field']
                # Creating a list of efield arrays
                efield_list_for_freq.append(efield)
            else:
                print(f"File {file_path} does not exist.")
        
        # Creating a dictionary for the current frequency
        freq_dict = {
            'frequency': freq,
            'efield_list': efield_list_for_freq,
            'y_coords': data['y_coords'] if 'data' in locals() and 'y_coords' in data else None
        }
        efield_list.append(freq_dict)
        
    return efield_list


#==================================================================================
#==================================================================================
#==================================================================================

