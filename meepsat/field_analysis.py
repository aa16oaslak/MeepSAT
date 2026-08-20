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
import pandas as pd
import json
from scipy import ndimage





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

    if saveh5:
        np.savez_compressed(filename + '.npz', 
                          deg=angles, 
                          amplitudedB=ff_dB)
        print(f"Far field data saved to {filename}.npz")

    return angles, ffmeep


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
    np.savez_compressed(f"{filename}_epsilon.npz", epsilon=eps_data)
    print(f"Epsilon data saved to {filename}_epsilon.npz")
    
    return eps_data



# Functions for post analysis with GRASP, CST, MEEPSAT!
def meepsat_farfield(efield,
             coords,
             wavelength,
             resolution,
             zero_pad_beam=15,
             plot_label='farfield',
             window_type=None,
             alpha=0.25,
             std=1/6):

    print(f"Calculating far-field pattern for {plot_label}...")
    # Keep the field complex so the FFT uses amplitude AND phase.
    # (A magnitude-only array still works, but its far-field phase is meaningless.)
    efield = np.asarray(efield, dtype=complex)
    # coords may arrive as a pandas Series with non-positional labels
    coords = np.asarray(coords, dtype=float)

    # Apodize before the FFT so a hard aperture-edge truncation doesn't imprint
    # its own sidelobe ripple/floor on top of the physical far-field pattern.
    if window_type is not None:
        from scipy.signal import windows
        n = len(efield)
        if window_type == 'hanning':
            win = windows.hann(n)
        elif window_type == 'hamming':
            win = windows.hamming(n)
        elif window_type == 'tukey':
            win = windows.tukey(n, alpha=alpha)
        elif window_type == 'gaussian':
            win = windows.gaussian(n, std=n * std)
        else:
            raise ValueError(f"Unknown window_type: {window_type}")
        efield = efield * win

    n_fft = len(efield) * zero_pad_beam

    # List of frequencies
    fft_freq = np.fft.fftfreq(n_fft, d=1/resolution)
    # Shift the zero frequency component to the center
    fft_freq = np.fft.fftshift(fft_freq)

    # Calculate angles in degrees
    theta_rad = np.arcsin(fft_freq * wavelength) #np.arctan(fft_freq * wavelength)
    theta_deg = theta_rad * (180 / np.pi)

    # Calculate the FFTs of efield
    fft_efield = np.fft.fft(efield, n=n_fft)
    fft_efield = np.fft.fftshift(fft_efield)

    # np.fft.fft assumes the first sample sits at y = 0, but the aperture
    # actually starts at coords[0]. Remove the resulting linear phase ramp so
    # the far-field phase is referenced to the true y = 0 origin. This is a
    # pure phase factor and leaves power_dB untouched.
    fft_efield = fft_efield * np.exp(-2j * np.pi * fft_freq * coords[0])

    # Find the index of the maximum amplitude in the FFT result
    max_idx = np.argmax(np.abs(fft_efield))
    
    # Divide the FFT result by the maximum amplitude to normalize it (includes both magnitude and phase)
    fft_efield_normalized = fft_efield / fft_efield[max_idx]

    # Normalize by maximum amplitude (still complex)
    # fft_efield_normalized = fft_efield / np.max(np.abs(fft_efield))

    # Far-field phase in radians
    phase_rad = np.angle(fft_efield_normalized)

    # Convert to power in dB
    fft_power = np.abs(fft_efield_normalized)**2
    fft_power_dB = 10 * np.log10(fft_power/ np.max(fft_power))

    return {
        'angle': theta_deg,
        'power_dB': fft_power_dB,
        'phase_rad': phase_rad,
        'complex_farfield': fft_efield_normalized,
        'plot_label': plot_label
    }


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


def centre_the_beam_with_phase_correction(y_coords, efield, wavelength,incidence_angle_deg=0):
    """
    Centre beam and apply phase correction for off-axis incidence.
    
    Parameters:
    -----------
    y_coords : array
        Y coordinates
    efield : array
        Electric field
    incidence_angle_deg : float
        Incidence angle in degrees (default 0 for normal incidence)
    """
    max_index = np.argmax(np.abs(efield))
    max_y = y_coords[max_index]
    
    # Center coordinates
    y_coords_centered = y_coords - max_y
    
    # For off-axis incidence, apply phase ramp correction
    if incidence_angle_deg != 0:
        incidence_angle_rad = np.radians(incidence_angle_deg)
        # Phase ramp due to tilted wavefront
        phase_ramp = np.exp(1j * 2 * np.pi * y_coords_centered * np.sin(incidence_angle_rad) / wavelength)
        efield_centered = efield * phase_ramp
    else:
        efield_centered = efield.copy()
    
    return y_coords_centered, efield_centered


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


#==========================================================================================================

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

def load_h5_data_recursive(file_path):
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

def plot_field(simname, field_db, title, filename, xcoords, ycoords, freq,
                vmin=-80, vmax=0, 
                savepath= os.path.join('./../processed_data/'),
                    show_plots=True,
                    mark_x = None):
    
    import matplotlib.pyplot as plt
    plt.style.use('default')
    
    plt.figure(figsize=(8, 6))
    plt.imshow(field_db.T, extent=(xcoords[0], xcoords[-1], ycoords[0], ycoords[-1]),
            origin='lower', cmap='inferno', vmin=vmin, vmax=vmax)
    if mark_x is not None:
        plt.axvline(x=mark_x, color='white', linestyle='--')
    plt.colorbar(label='dB')
    plt.title(title)
    plt.xlabel('x (mm)')
    plt.ylabel('y (mm)')

    if savepath:
        # Create directory with simname and frequency subdirectories
        save_dir = savepath
        os.makedirs(save_dir, exist_ok=True)
        plt.savefig(os.path.join(save_dir, filename), dpi=300)
        # Save as a svg file as well for publication
        # plt.savefig(os.path.join(save_dir, filename).replace('.png', '.svg'), dpi=300) 
        print(f"{title} plot saved to: {os.path.join(save_dir, filename)}")
    if show_plots:
        plt.show()

def load_fields(basepath, filename):
    # Construct the full path to the file
    filepath = os.path.join(basepath, filename)
    # Load the fields stored in npz files
    data = np.load(filepath)

    return data

"""
2D near-to-far-field (NTFF) tools for the single-lens baffle sims.

Why this exists
---------------
The far field produced by the 1D FFT of a single Ez line (meepsat_farfield)
cannot represent wide angles: radiation heading toward |theta| > ~45 deg exits
the cell through the top/bottom PML and never crosses a vertical line, and the
line itself is hard-truncated inside the PML. This module instead evaluates the
frequency-domain surface-equivalence integral on a closed rectangular contour
enclosing ALL sources (source line, lens, tube, baffles), which captures the
sideways-radiated power and gives the correct obliquity behaviour.

The contour must enclose every radiating structure. A partial box around just
the baffle region would, by the equivalence principle, return only the field
scattered by whatever is inside it and miss the direct beam.

Formulation (2D, Ez/TM polarization, meep's exp(-i w t) phasor convention):

    F(theta) = sum over contour of
               [ Ez (n . rhat) - eta (nx Hy - ny Hx) ] exp(-j k rhat . r') dl

with rhat = (-cos theta, sin theta) so theta = 0 is the -x beam direction,
eta = 1 in meep units. Far-field power pattern ~ |F|^2. The saved
"time-averaged" fields are the true phasors times one global complex constant
(plain average of snapshots sampled ~10x/period), which cancels in every
normalized quantity used here.
"""

import os

import numpy as np


C_MM_S = 299792458.0 * 1000.0  # speed of light in mm/s


# ---------------------------------------------------------------------------
# Field loading and contour extraction (with per-case caching)
# ---------------------------------------------------------------------------

def load_case_fields(case_dir):
    """Load complex Ez, Hx, Hy and grid coords from a sim output directory."""
    e = np.load(os.path.join(case_dir, "efield_timeavg.npz"))
    h = np.load(os.path.join(case_dir, "hfield_timeavg.npz"))
    xyzw = np.load(os.path.join(case_dir, "xyzw.npz"))
    ez = e["ez_real"] + 1j * e["ez_imag"]
    hx = h["hx_real"] + 1j * h["hx_imag"]
    hy = h["hy_real"] + 1j * h["hy_imag"]
    return xyzw["x_coords"], xyzw["y_coords"], ez, hx, hy


def extract_box_contour(case_dir, box, #=(-240.0, 240.0, -115.0, 115.0),
                        slice_x=None, use_cache=True):
    """
    Extract Ez/Hx/Hy on the four sides of a rectangular contour, plus an
    optional extra vertical Ez line at slice_x for the 1D cross-check method.

    Decompressing the full-field npz dominates the cost, so the result is
    cached in the case directory keyed by the box/slice parameters.

    Returns a dict with points (N,2), normals (N,2), ez/hx/hy (N,), dl, and
    (if slice_x given) slice_y, slice_ez.
    """
    x0, x1, y0, y1 = box
    tag = f"x{x0:g}_{x1:g}_y{y0:g}_{y1:g}_s{'none' if slice_x is None else f'{slice_x:g}'}"
    cache_path = os.path.join(case_dir, f"ntff_contour_{tag}.npz")
    if use_cache and os.path.exists(cache_path):
        d = np.load(cache_path)
        return {k: d[k] for k in d.files}

    x, y, ez, hx, hy = load_case_fields(case_dir)
    ix0, ix1 = (np.abs(x - x0)).argmin(), (np.abs(x - x1)).argmin()
    iy0, iy1 = (np.abs(y - y0)).argmin(), (np.abs(y - y1)).argmin()
    dl = float(np.mean(np.diff(x)))

    pts, nrm, cez, chx, chy = [], [], [], [], []

    def add_side(ixs, iys, normal):
        xs, ys = np.broadcast_arrays(x[ixs], y[iys])
        pts.append(np.column_stack([xs.ravel(), ys.ravel()]))
        n = np.tile(normal, (xs.size, 1))
        nrm.append(n)
        cez.append(ez[ixs, iys].ravel())
        chx.append(hx[ixs, iys].ravel())
        chy.append(hy[ixs, iys].ravel())

    yspan = np.arange(iy0, iy1 + 1)
    xspan = np.arange(ix0 + 1, ix1)  # horizontal sides exclude the corners
    add_side(ix0, yspan, (-1.0, 0.0))            # left
    add_side(ix1, yspan, (+1.0, 0.0))            # right
    add_side(xspan, iy0, (0.0, -1.0))            # bottom
    add_side(xspan, iy1, (0.0, +1.0))            # top

    out = {
        "points": np.concatenate(pts),
        "normals": np.concatenate(nrm),
        "ez": np.concatenate(cez),
        "hx": np.concatenate(chx),
        "hy": np.concatenate(chy),
        "dl": np.array(dl),
        "box": np.array(box),
    }
    if slice_x is not None:
        isx = (np.abs(x - slice_x)).argmin()
        out["slice_y"] = y
        out["slice_ez"] = ez[isx, :]
        out["slice_x_actual"] = np.array(x[isx])

    if use_cache:
        np.savez_compressed(cache_path, **out)
    return out


# ---------------------------------------------------------------------------
# Far-field transforms using NTFF
# ---------------------------------------------------------------------------

def ntff_2d(contour, wavelength, angles_deg, eta=1.0, angle_chunk=256):
    """
    Closed-contour surface-equivalence far field.

    Returns the complex far-field amplitude F(theta) (unnormalized); the
    power pattern is |F|^2. theta is measured from the -x beam axis,
    positive toward +y.
    """
    k = 2.0 * np.pi / wavelength
    th = np.deg2rad(np.asarray(angles_deg, dtype=float))
    rhat = np.stack([-np.cos(th), np.sin(th)])  # (2, Nang)

    pts = contour["points"]
    nrm = contour["normals"]
    ez = contour["ez"]
    jz = nrm[:, 0] * contour["hy"] - nrm[:, 1] * contour["hx"]
    dl = float(contour["dl"])

    F = np.empty(th.size, dtype=complex)
    for a in range(0, th.size, angle_chunk):
        r = rhat[:, a:a + angle_chunk]
        proj = pts @ r                       # (N, chunk)
        ndotr = nrm @ r
        integrand = ez[:, None] * ndotr - eta * jz[:, None]
        F[a:a + angle_chunk] = (integrand * np.exp(-1j * k * proj)).sum(axis=0) * dl
    return F



def farfield_1d_slice(ez_line, y_coords, wavelength, y_max=None,
                      window="tukey", alpha=0.25, zero_pad=15):
    """
    Cleaned-up version of the single-line FFT method, for cross-checking:
    - trims the line to |y| <= y_max (drop PML-contaminated samples),
    - apodizes (Tukey) so the truncation doesn't imprint its own sidelobes,
    - maps spatial frequency with sin(theta) = f*lambda (not tan) and drops
      the evanescent region |f*lambda| > 1, which the FFT would otherwise
      spray across the wide-angle floor,
    - applies the cos(theta) obliquity factor.

    Returns (angles_deg, power_lin) with power ~ |cos(theta) A(k sin theta)|^2.
    """
    ez_line = np.asarray(ez_line, dtype=complex)
    y = np.asarray(y_coords, dtype=float)
    if y_max is not None:
        keep = np.abs(y) <= y_max
        ez_line, y = ez_line[keep], y[keep]

    if window == "tukey":
        from scipy.signal import windows
        ez_line = ez_line * windows.tukey(ez_line.size, alpha=alpha)
    elif window is not None:
        raise ValueError(f"Unknown window: {window}")

    dy = float(np.mean(np.diff(y)))
    n_fft = ez_line.size * zero_pad
    f = np.fft.fftshift(np.fft.fftfreq(n_fft, d=dy))
    A = np.fft.fftshift(np.fft.fft(ez_line, n=n_fft))

    s = f * wavelength                    # sin(theta)
    keep = np.abs(s) <= 1.0
    theta = np.rad2deg(np.arcsin(s[keep]))
    power = (np.cos(np.deg2rad(theta)) * np.abs(A[keep])) ** 2
    return theta, power


# ---------------------------------------------------------------------------
# Robust comparison metrics
# ---------------------------------------------------------------------------

def power_db(power_lin, ref=None):
    p = np.asarray(power_lin, dtype=float)
    ref = p.max() if ref is None else ref
    return 10.0 * np.log10(p / ref + 1e-300)


def band_power_fractions(angles_deg, power_lin, edges, #=(0.0, 5.0, 25.0, 50.0, 90.0),
                         fold=True):
    """Fraction of total radiated power per |theta| band.

    The +theta and -theta halves of a band are disjoint intervals, so they are
    integrated separately - a single trapezoid over the combined mask would
    bridge the gap across the main lobe.

    fold=True (default) sums the +theta and -theta halves together, as if the
    pattern were mirror-symmetric about theta=0 - correct only when the
    radiating geometry actually has that symmetry (e.g. a centered pixel with
    a baffle that is itself symmetric about the same axis). fold=False keeps
    the two sides separate and returns {'pos': ndarray, 'neg': ndarray} - use
    this whenever the geometry has no reason to be mirror-symmetric (e.g. an
    off-axis pixel, where the beam can strike the near and far baffle wings at
    genuinely different incidence angles).
    """
    a = np.asarray(angles_deg, dtype=float)
    p = np.asarray(power_lin)
    total = np.trapezoid(p, a)
    pos_fracs, neg_fracs = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        pos_mask = (a >= lo) & (a < hi)
        neg_mask = (a <= -lo) & (a > -hi)
        pos = np.trapezoid(p[pos_mask], a[pos_mask]) / total if pos_mask.sum() > 1 else 0.0
        neg = np.trapezoid(p[neg_mask], a[neg_mask]) / total if neg_mask.sum() > 1 else 0.0
        pos_fracs.append(pos)
        neg_fracs.append(neg)
    if fold:
        return np.array(pos_fracs) + np.array(neg_fracs)
    return {'pos': np.array(pos_fracs), 'neg': np.array(neg_fracs)}


def encircled_power(angles_deg, power_lin):
    """Cumulative fraction of power within |theta|, vs |theta|.

    Assumes theta=0 is both the pattern's center of symmetry and its peak -
    valid for a centered pixel. For an off-axis beam use
    encircled_power_centered(..., center_deg=beam_peak_angle(...)) instead.
    """
    a = np.abs(np.asarray(angles_deg))
    p = np.asarray(power_lin)
    order = np.argsort(a)
    a_sorted, p_sorted = a[order], p[order]
    cum = np.cumsum(p_sorted)
    return a_sorted, cum / cum[-1]


def encircled_power_centered(angles_deg, power_lin, center_deg=0.0):
    """Cumulative fraction of power within |theta - center_deg|.

    Generalizes encircled_power to fold about an arbitrary center instead of
    theta=0 (center_deg=0.0 reproduces encircled_power exactly). Answers "how
    tightly is this beam packed around its own chief ray" - a beam-quality
    metric that stays meaningful even when that ray doesn't point along the
    mechanical axis (pass center_deg=beam_peak_angle(angles_deg, power_lin)).
    """
    a = np.abs(np.asarray(angles_deg, dtype=float) - center_deg)
    p = np.asarray(power_lin)
    order = np.argsort(a)
    a_sorted, p_sorted = a[order], p[order]
    cum = np.cumsum(p_sorted)
    return a_sorted, cum / cum[-1]


def cumulative_power(angles_deg, power_lin):
    """Monotonic signed CDF: fraction of power at angle <= theta, swept over
    the full signed angle range - no folding or centering assumption.

    Answers "how much power has spilled past a given FIXED angle relative to
    the mechanical axis" - the quantity a baffle that is itself symmetric
    about theta=0 actually constrains, regardless of where the beam being
    baffled happens to point. Unlike encircled_power/encircled_power_centered,
    this needs no decision about where the pattern is centered.
    """
    a = np.asarray(angles_deg, dtype=float)
    p = np.asarray(power_lin, dtype=float)
    order = np.argsort(a)
    a_sorted, p_sorted = a[order], p[order]
    cum = np.cumsum(p_sorted)
    return a_sorted, cum / cum[-1]


def beam_peak_angle(angles_deg, power_lin, search_window=None, db_window=3.0):
    """Power-weighted centroid angle of the main lobe.

    search_window: optional (lo_deg, hi_deg) restricting the peak search to a
        neighborhood of the geometrically-expected chief ray, guarding against
        latching onto a distant, unrelated sidelobe. Default None searches the
        full angle array, which is safe whenever the main lobe dominates by
        tens of dB (true for both the on-axis and edge-pixel cases here).
    db_window: dB down from the local max defining the contiguous main-lobe
        mask (default 3.0 = half-power).

    The mask is grown outward from the peak index one sample at a time and
    stops at the first drop below threshold on each side, so it stays a single
    contiguous region around the true peak even if some other sidelobe
    elsewhere also happens to cross the same threshold.
    """
    a = np.asarray(angles_deg, dtype=float)
    p = np.asarray(power_lin, dtype=float)
    if search_window is not None:
        lo, hi = search_window
        mask = (a >= lo) & (a <= hi)
        idx_local = np.where(mask)[0]
        idx_pk = idx_local[np.argmax(p[idx_local])]
    else:
        idx_pk = int(np.argmax(p))

    threshold = p[idx_pk] * 10.0 ** (-db_window / 10.0)
    lo_idx = idx_pk
    while lo_idx > 0 and p[lo_idx - 1] >= threshold:
        lo_idx -= 1
    hi_idx = idx_pk
    while hi_idx < p.size - 1 and p[hi_idx + 1] >= threshold:
        hi_idx += 1

    lobe_a = a[lo_idx:hi_idx + 1]
    lobe_p = p[lo_idx:hi_idx + 1]
    return float(np.sum(lobe_a * lobe_p) / np.sum(lobe_p))


def envelope_db(angles_deg, power_lin, width_deg=1.0, q=95):
    """Percentile envelope of the power pattern in angular bins, in dB.

    Replaces raw pointwise dB (and find_peaks scatter): the deep-sidelobe
    'grass' is incoherent, so a per-bin percentile is the stable summary.
    """
    a = np.asarray(angles_deg, dtype=float)
    p = np.asarray(power_lin, dtype=float)
    edges = np.arange(a.min(), a.max() + width_deg, width_deg)
    centers, env = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (a >= lo) & (a < hi)
        if m.any():
            centers.append(0.5 * (lo + hi))
            env.append(np.percentile(p[m], q))
    return np.array(centers), power_db(np.array(env), ref=p.max())


def noise_floor_db(case_dir, wavelength, angles_deg,
                   box_a,#=(-240.0, 240.0, -115.0, 115.0),
                   box_b):#=(-230.0, 235.0, -112.0, 118.0)):
    """
    Numerical noise-floor estimate via contour independence: a correct NTFF is
    invariant under contour placement, so |F_boxA - F_boxB|^2 (relative to the
    boxA peak) measures everything that is NOT physical radiation.
    """
    fa = ntff_2d(extract_box_contour(case_dir, box=box_a), wavelength, angles_deg)
    fb = ntff_2d(extract_box_contour(case_dir, box=box_b), wavelength, angles_deg)
    ref = (np.abs(fa) ** 2).max()
    return power_db(np.abs(fa - fb) ** 2, ref=ref), fa, fb
