import sys
import os
import site
import warnings
from pathlib import Path
from typing import Callable
import numpy as np


# =====================================================================
# MPI helpers
# =====================================================================
#
# Under ``mpirun -np N python script.py`` the WHOLE script runs N times, so
# every full-cell array a MeepSAT module allocates is paid N times while MEEP's
# own FDTD grid is split 1/N per rank. That is what makes the memory footprint
# grow with the core count. These helpers are the single place the rest of the
# package asks "am I the rank that should be doing this?".
#
# The rule for using them: a MEEP call that COMMUNICATES (``get_array``,
# ``get_epsilon``, ``get_dft_array``, ``get_array_metadata``, ``init_sim``,
# ``run``) must be made by EVERY rank or the job deadlocks. Only the work that
# follows it -- the copies, the maths, the plotting, the file writing -- may be
# guarded. Measured on this MEEP build: ``get_array_metadata`` is collective, and
# ``get_epsilon_grid`` is rank-local ONLY for a structure made of geometry
# objects -- with an ``epsilon_input_file`` it re-converts the filename on every
# call and MEEP broadcasts that HDF5 read, so a master-only call hangs the job.
# When in doubt, let every rank make the MEEP call and guard what follows.


def _meep():
    """Import meep lazily, so helpers stays importable without it."""
    import meep as mp
    return mp


def is_master() -> bool:
    """
    True on the single job-wide master rank (rank 0), and in a serial run.

    Prefers ``am_really_master`` over ``am_master``: with ``divide_parallel``
    subgroups ``am_master()`` is true once *per subgroup*, so guarding a file
    write with it would still give one writer per subgroup.
    """
    try:
        mp = _meep()
    except ImportError:
        return True
    try:
        return bool(mp.am_really_master())
    except AttributeError:
        try:
            return bool(mp.am_master())
        except AttributeError:
            return True


def n_procs() -> int:
    """Number of MPI ranks in this job (1 when not running under MPI)."""
    try:
        return int(_meep().count_processors())
    except Exception:
        return 1


def mpi_rank() -> int:
    """This process's MPI rank (0 when not running under MPI)."""
    try:
        return int(_meep().my_rank())
    except Exception:
        return 0


def mprint(*args, **kwargs):
    """``print()`` that only fires on the master rank."""
    if is_master():
        print(*args, **kwargs)


def broadcast(value, root: int = 0):
    """
    Send a small picklable object from `root` to every rank.

    Used to share the *scalars* a rank-0-only computation produced (a corrected
    resolution, say) without shipping the full-cell arrays it derived them from.
    Returns `value` unchanged in a serial run.
    """
    if n_procs() == 1:
        return value
    try:
        from mpi4py import MPI
    except ImportError:
        warnings.warn(
            "Running under MPI but mpi4py is not importable, so a rank-0 result "
            "could not be broadcast to the other ranks. They will carry on with "
            "their own (stale) values. Install mpi4py, or run in serial.")
        return value
    return MPI.COMM_WORLD.bcast(value, root=root)


def barrier():
    """Block until every rank arrives. No-op in a serial run."""
    if n_procs() == 1:
        return
    try:
        from mpi4py import MPI
    except ImportError:
        return
    MPI.COMM_WORLD.barrier()

# Used to remove the elements of a dictionary (dict_to_filter) that
# don't correspond to the keyword arguments of a particular
# function (func_with_kwargs.)
# Adapted from https://stackoverflow.com/questions/26515595/how-does-one-ignore-unexpected-keyword-arguments-passed-to-a-function/44052550
def filter_dict(dict_to_filter: dict, func_with_kwargs: Callable) -> dict:
    """
    Filters a dictionary to only include keys that are parameters of a given function.
    Args:
        dict_to_filter (dict): The dictionary to filter.
        func_with_kwargs (Callable): The function whose parameter names will be used to filter the dictionary.
    Returns:
        dict: A dictionary containing only the key-value pairs from dict_to_filter where the keys are parameters of func_with_kwargs.
    Raises:
        TypeError: If func_with_kwargs is not a callable.
    """
    import inspect

    filter_keys = []
    try:
        # Python3 ...
        sig = inspect.signature(func_with_kwargs)
        filter_keys = [param.name for param in sig.parameters.values()]
    except:
        # Python2 ...
        filter_keys = inspect.getargspec(func_with_kwargs)[0]

    filtered_dict = {
        filter_key: dict_to_filter[filter_key]
        for filter_key in filter_keys
        if filter_key in dict_to_filter
    }
    return filtered_dict

# funnction to extract the xticks and yticks 
def extract_ticks(data, num_ticks, sim_box):
    """
    Generate tick positions and labels for a given simulation box.
    Parameters:
    data (array-like): The data to be plotted (not used in the current implementation).
    sim_box (list of tuples): A list containing two tuples, each representing the 
                              start and end points of the simulation box in the x 
                              and y directions, respectively. 
                              Example: [(x_start, x_end), (y_start, y_end)]
    Returns:
    tuple: A tuple containing four elements:
        - xticks (numpy.ndarray): Positions of ticks along the x-axis.
        - yticks (numpy.ndarray): Positions of ticks along the y-axis.
        - xticklabels (list of str): Labels for the ticks along the x-axis.
        - yticklabels (list of str): Labels for the ticks along the y-axis.
    """
    xticks = np.linspace(sim_box[0][0], sim_box[0][1], num_ticks)
    yticks = np.linspace(sim_box[1][0], sim_box[1][1], num_ticks)
    xticklabels = [f'{xtick:.1f}' for xtick in xticks
                    ]
    yticklabels = [f'{ytick:.1f}' for ytick in yticks
                    ] 
    
    return xticks, yticks, xticklabels, yticklabels


def sys_info(self, dist_unit, wvl=None, meep_freq=None, real_freq=None, points_per_wavelength=20):
    '''
    Writes the file that will then be 
    read within the MEEP simulation

    Arguments
    ---------
    dist_unit : float
        Chosen ratio between MEEP distances and real distance 
    wvl : float, optional
        Wavelength in MEEP units (default : None)
    meep_freq : float, optional
        Frequency in MEEP units (default : None)
    real_freq : float, optional
        Frequency in Hz (default : None)
    points_per_wavelength : int, optional
        Minimum number of grid points per wavelength (default: 20)
    '''

    c = 299792458.0
    if wvl is not None :
        meep_freq = 1/wvl
        real_freq = c*meep_freq/dist_unit

    if real_freq is not None:
        wvl = c/real_freq/dist_unit
        meep_freq = 1/wvl

    if meep_freq is not None: 
        wvl= 1/meep_freq
        real_freq = c*meep_freq/dist_unit

    # Calculate minimum resolution
    min_resolution = points_per_wavelength / wvl

    print('--- System Info ---')
    print('Real Wavelength = {:.1e}m'.format(wvl*dist_unit))
    print('MEEP Wavelength = {:.1e}'.format(wvl))
    print('System size = {:.0f} x {:.0f} wavelengths'.format(self.size_x/wvl, 
                                                    self.size_y/wvl))
    print('System size = {:.2e} x {:.2e} m'.format(self.size_x*dist_unit, 
                                                    self.size_y*dist_unit))
    print('Real frequency = {:.2e} Hz'.format(real_freq))
    print('MEEP frequency = {:.2e}'.format(meep_freq))
    if min_resolution:
        print('Minimum resolution (grid points/unit length): {:.2f}'.format(min_resolution))
        print('  (for {} points per wavelength)'.format(points_per_wavelength))
    print('------------------')


def read_json(json_file):
    """
    Reads a JSON file and returns the data as a dictionary.
    Args:
        json_file (str): Path to the JSON file.
    Returns:
        dict: The data from the JSON file.
    """
    import json
    with open(json_file, "r") as f:
        data = json.load(f)
    return data


def rot_x(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([
        [1, 0, 0],
        [0, c, -s],
        [0, s,  c]
    ])

def rot_y(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([
        [ c, 0, s],
        [ 0, 1, 0],
        [-s, 0, c]
    ])

def rot_z(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([
        [c, -s, 0],
        [s,  c, 0],
        [0,  0, 1]
    ])

def rotation_matrix(tx, ty, tz):
    """
    Zemax order: Rx -> Ry -> Rz (applied in that order)
    Equivalent to: Rz @ Ry @ Rx
    """
    return rot_z(tz) @ rot_y(ty) @ rot_x(tx)


# Helpers for FeedHorn Source in meep_geometry.py
C_MM_GHZ = 299.792458  # speed of light [mm GHz]
PHI_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]

def fit_gaussian(theta, co_dB, fit_level=-10.0):
    """Fit a boresight-centred Gaussian power beam to the main lobe.

    In dB a Gaussian is a parabola in theta^2:  P_dB = a + b*theta^2  with
    b = -10/ln(10) / (2 sigma^2), so this is a closed-form linear least-squares
    fit (no initial guess to diverge from). Returns (fwhm_deg, peak_dB, sigma_deg).
    """
    mask = main_beam_mask(co_dB, fit_level)
    if mask.sum() < 3:
        return np.nan, np.nan, np.nan
    b, a = np.polyfit(theta[mask] ** 2, co_dB[mask], 1)
    if b >= 0:
        return np.nan, a, np.nan
    sigma = np.sqrt(-DB_PER_NEPER_POWER / (2.0 * b))
    return FWHM_PER_SIGMA * sigma, a, sigma

def main_beam_mask(co_dB, level):
    """Contiguous points from boresight down to the first drop below `level` dB."""
    below = np.nonzero(co_dB < level)[0]
    stop = below[0] if below.size else co_dB.size
    mask = np.zeros(co_dB.size, bool)
    mask[:stop] = True
    return mask

def nearest_freq_indices(freqs, wanted):
    idx = [int(np.argmin(np.abs(freqs - f))) for f in wanted]
    for f, fi in zip(wanted, idx):
        if not np.isclose(freqs[fi], f):
            print(f"  requested {f} GHz -> using nearest {freqs[fi]:.2f} GHz")
    return idx

def style_axes(ax):
    ax.grid(True, color="0.88", lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)

def waist_from_fwhm(fwhm_deg, freq_ghz):
    """Waist radius w0 [mm] (1/e field, 1/e^2 power) of the Gaussian beam with this far-field FWHM.

    Far field of a Gaussian beam: P ~ exp(-2 theta^2 / theta0^2) with theta0 = lambda / (pi w0),
    so theta0 = FWHM / sqrt(2 ln 2) and w0 = lambda / (pi theta0).
    """
    wavelength = C_MM_GHZ / freq_ghz
    theta0 = np.radians(fwhm_deg) / FWHM_PER_W0
    return wavelength / (np.pi * theta0)
