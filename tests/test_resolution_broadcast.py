#!/usr/bin/env python
"""
`check_resolution` computes on rank 0 and broadcasts: check every rank agrees.

`simulator.check_resolution` derives the required resolution from full-cell
permittivity maps on the master rank only, then broadcasts the resulting
scalars. If that broadcast were ever dropped, the ranks would carry different
resolutions and the run would desynchronise -- far worse than the memory it
saves -- so this asserts the agreement directly, and over both sampling paths:

  * geometry objects      -> sampling is rank-local, master-only
  * epsilon_input_file    -> sampling is collective (MEEP broadcasts the file
                             read), so every rank samples and only the derived
                             maps are master-only

Run::

    mpirun -np 4 python tests/test_resolution_broadcast.py
    python tests/test_resolution_broadcast.py          # serial, also valid
"""

import os
import sys
import tempfile
import types

import numpy as np
import h5py
import meep as mp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import meepsat.simulator as sim          # noqa: E402
import meepsat.helpers as exf            # noqa: E402

FREQ = 0.5           # MEEP units
EPS = 9.0            # n = 3
RES = 10             # deliberately too low: 8 px/wavelength needs 8*3*0.5 = 12
PX_PER_WVL = 8
EXPECTED = 12


def _gather(value):
    if exf.n_procs() == 1:
        return [value]
    from mpi4py import MPI
    return MPI.COMM_WORLD.allgather(value)


def _fresh_case():
    mpsat_sim = types.SimpleNamespace(resolution=RES)
    data = {"simulation": {"primary_params": {"resolution": RES,
                                             "smallest_freq": FREQ}}}
    return mpsat_sim, data


def _check(label, simulation):
    mpsat_sim, data = _fresh_case()
    sim.check_resolution(data, mpsat_sim, simulation=simulation,
                         source_freq=FREQ,
                         min_pixels_per_wavelength=PX_PER_WVL,
                         min_pixels_per_skin_depth=0)

    seen = _gather((mpsat_sim.resolution,
                    data["simulation"]["primary_params"]["resolution"],
                    round(data["simulation"]["primary_params"]["smallest_freq"], 6)))

    assert len(set(seen)) == 1, (
        f"{label}: ranks disagree after check_resolution -> {seen}")
    res, data_res, small = seen[0]
    assert res == EXPECTED, f"{label}: expected resolution {EXPECTED}, got {res}"
    assert data_res == EXPECTED, f"{label}: data dict not updated ({data_res})"
    # max(n/lambda_0) = 3 * 0.5
    assert abs(small - 1.5) < 1e-6, f"{label}: smallest_freq {small} != 1.5"
    exf.mprint(f"  PASS {label}: every rank has resolution {res}, "
               f"smallest_freq {small}")


def main():
    exf.mprint(f"check_resolution broadcast test on {exf.n_procs()} rank(s)")

    # --- 1. geometry objects: sampling is rank-local -------------------
    geom_sim = mp.Simulation(
        cell_size=mp.Vector3(40, 20, 0), resolution=RES,
        boundary_layers=[mp.PML(2)],
        geometry=[mp.Block(center=mp.Vector3(), size=mp.Vector3(10, 10),
                           material=mp.Medium(epsilon=EPS))],
        sources=[mp.Source(mp.ContinuousSource(FREQ), component=mp.Ez,
                           center=mp.Vector3(-15, 0))])
    _check("geometry objects", geom_sim)

    # --- 2. epsilon_input_file: sampling is collective ----------------
    tmpdir = tempfile.mkdtemp(prefix="meepsat_restest_") if exf.is_master() else None
    tmpdir = exf.broadcast(tmpdir)
    path = os.path.join(tmpdir, "eps.h5")
    if exf.is_master():
        eps_map = np.ones((40 * RES + 1, 20 * RES + 1), dtype=np.float32)
        eps_map[150:250, 50:150] = EPS
        with h5py.File(path, "w") as h:
            h.create_dataset("eps", data=eps_map)
    exf.barrier()

    file_sim = mp.Simulation(
        cell_size=mp.Vector3(40, 20, 0), resolution=RES,
        boundary_layers=[mp.PML(2)],
        epsilon_input_file=path,
        sources=[mp.Source(mp.ContinuousSource(FREQ), component=mp.Ez,
                           center=mp.Vector3(-15, 0))])
    _check("epsilon_input_file", file_sim)

    exf.barrier()
    if exf.is_master():
        import shutil
        shutil.rmtree(tmpdir, ignore_errors=True)
    exf.mprint("ALL PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
