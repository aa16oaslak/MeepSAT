
"""
Numeric precision of the arrays MeepSAT allocates.

MEEP's own precision is fixed when MEEP is compiled: a build configured with
``--enable-single`` stores its fields as float32, otherwise as float64. MeepSAT
cannot change that at run time. What it can control is every array it
allocates itself (epsilon maps, resolution maps, field accumulators, HDF5/npz
outputs), and those follow the precision chosen here. Set it once, through
``simulator.sim_init(precision=...)``; the rest of the package reads it back.

Two families of dtypes are exposed, and they must not be mixed up:

* ``real_dtype()`` / ``complex_dtype()`` -- the MeepSAT working precision,
  i.e. what ``sim_init(precision=...)`` asked for.
* ``meep_real_dtype()`` / ``meep_complex_dtype()`` -- what the MEEP build
  stores. A buffer handed to ``Simulation.get_array(arr=...)`` MUST use these.
  MEEP writes its raw values straight into that buffer without converting
  them, so a complex128 buffer on a single-precision build silently fills with
  garbage, and a complex64 buffer on a double-precision build corrupts the
  heap and aborts the process.
"""

import meep as mp
import numpy as np

_ALIASES = {
    "float32": "float32",
    "single": "float32",
    "float64": "float64",
    "double": "float64",
}

_STATE = {"precision": None}


def meep_precision() -> str:
    """'float32' or 'float64': the precision MEEP was compiled with."""
    return "float32" if mp.is_single_precision() else "float64"


def _normalise(precision) -> str:
    if precision is None:
        return "auto"
    if isinstance(precision, str):
        key = precision.strip().lower()
        if key == "auto":
            return "auto"
        if key in _ALIASES:
            return _ALIASES[key]
    else:
        try:
            name = np.dtype(precision).name
        except TypeError:
            name = None
        if name in ("float32", "float64"):
            return name
    raise ValueError(f"precision must be 'auto', 'float32' or 'float64', "
                     f"got {precision!r}")


def set_precision(precision="auto") -> str:
    """
    Set the MeepSAT working precision and return it as 'float32' / 'float64'.

    Parameters
    ----------
    precision : str or numpy dtype, optional
        'float32' (alias 'single'), 'float64' (alias 'double'), np.float32 /
        np.float64, or 'auto' (default) to follow the MEEP build.

    A precision that differs from the MEEP build is allowed but warned about
    (printed on the master rank): MEEP's fields keep their compiled precision
    either way.
    """
    name = _normalise(precision)
    built = meep_precision()
    if name == "auto":
        name = built
    elif name != built and mp.am_master():
        # Printed rather than warnings.warn(): meepsat.stepfunctions switches
        # all warnings off when it is imported, which would swallow this.
        if name == "float64":
            print("WARNING: precision='float64' requested, but this MEEP build "
                  "is single precision: its fields are float32 whatever MeepSAT "
                  "uses, so float64 arrays add memory without adding accuracy. "
                  "Use a double-precision MEEP build for float64 fields.",
                  flush=True)
        else:
            print("WARNING: precision='float32' requested, but this MEEP build "
                  "is double precision: MEEP's fields stay float64 and only the "
                  "arrays MeepSAT allocates are float32 (values are rounded to "
                  "float32 when stored).", flush=True)
    _STATE["precision"] = name
    return name


def get_precision() -> str:
    """The working precision; follows the MEEP build until set_precision()."""
    if _STATE["precision"] is None:
        _STATE["precision"] = meep_precision()
    return _STATE["precision"]


def real_dtype() -> np.dtype:
    """Real dtype of the MeepSAT working precision."""
    return np.dtype(get_precision())


def complex_dtype() -> np.dtype:
    """Complex dtype of the MeepSAT working precision."""
    return np.dtype("complex64" if get_precision() == "float32" else "complex128")


def meep_real_dtype() -> np.dtype:
    """Real dtype MEEP stores; use it for get_array(arr=...) buffers."""
    return np.dtype(meep_precision())


def meep_complex_dtype() -> np.dtype:
    """Complex dtype MEEP stores; use it for get_array(arr=..., cmplx=True)."""
    return np.dtype("complex64" if meep_precision() == "float32" else "complex128")
