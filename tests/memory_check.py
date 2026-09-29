#!/usr/bin/env python
"""
Memory requirement of a MeepSAT run, broken down by component.

Why this script exists
----------------------
Under ``mpirun -np N python run_script.py`` the WHOLE Python script runs N
times. MEEP splits its own FDTD grid 1/N per rank, but every full-cell NumPy
array a MeepSAT module allocates is paid N times over. That is why the memory
requirement of a run grows with the core count, and it is why the interesting
question is not "how much memory does this need?" but "how much of it is
replicated per rank?".

This script answers that per component, for a cell size you give it.

Two modes
---------
1. ESTIMATE (default, serial, no simulation is built)::

       python tests/memory_check.py --size-x 600 --size-y 150 --resolution 10 \
                                    --ranks 1 4 8 16

   Prints, for every component of the pipeline, the bytes per grid cell it
   costs, whether that cost is replicated on every rank / confined to rank 0 /
   distributed by MEEP, and the resulting totals at each rank count.

2. MEASURE (real calls, real RSS)::

       mpirun -np 8 python tests/memory_check.py --measure \
              --size-x 600 --size-y 150 --resolution 10

   Builds a simulation of that cell size and walks the actual pipeline
   (``plot_and_save_epsilon`` -> ``check_resolution`` -> animation frame ->
   field accumulation -> DFT fields), reporting each stage's per-rank resident
   and peak memory. ``VmHWM`` is reset between stages, so each stage's peak is
   its own. Use it to check a change really did remove what it was meant to, and
   to re-calibrate the estimate constants below on a new machine or MEEP build.

The estimate constants come from measured runs on a 9.0 M cell 2D cell with
MEEP 1.31 (double precision). Re-measure with ``--measure`` if you change MEEP
build, dimensionality or precision; the structural conclusions (what is
replicated) hold regardless, the exact bytes/cell do not.
"""

import argparse
import os
import sys
import time
import types

import numpy as np

# Allow running straight from a checkout without installing the package.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import meepsat.precision as mpsat_precision  # noqa: E402


MB = float(2 ** 20)


# =====================================================================
# The cost model
# =====================================================================
#
# scope:
#   "rank"        replicated -- every rank pays it. THIS is what makes the
#                 footprint grow with the core count.
#   "master"      rank 0 only -- flat in the rank count.
#   "distributed" MEEP splits it across ranks -- the total is flat, the per-rank
#                 share shrinks as ranks are added.
#
# life:
#   "persistent"  held for the run (counts toward the steady-state floor)
#   "transient"   a spike while that stage executes
#
# bytes_per_cell is a callable so it can depend on precision and on how the run
# is configured (number of frequencies, components, downsampling).


def build_model(cfg):
    """
    Cost rows for the configuration in `cfg`.

    Returns ``(rows, alternative)``: `rows` is the path this configuration
    actually runs and is what the totals are computed from, `alternative` is the
    other field-extraction path, reported for comparison but never added in --
    you register one or the other, never both.
    """
    real = cfg["itemsize_real"]          # MeepSAT working precision
    mreal = cfg["itemsize_meep_real"]    # what the MEEP build stores
    mcplx = cfg["itemsize_meep_complex"]
    nfreq = cfg["freqs"]
    ncomp = cfg["components"]
    ds = max(1, cfg["downsample_x"]) * max(1, cfg["downsample_y"])
    from_file = cfg["eps_from_file"]

    rows = [
        # ---- fixed, per-process ----
        dict(name="Python + MEEP + matplotlib imports",
             where="every rank, unavoidable",
             scope="rank", life="persistent", bytes_per_cell=0.0,
             fixed_mb=cfg["baseline_mb"],
             note="one interpreter per rank; the floor a rank cannot go below"),

        # ---- setup: the epsilon map ----
        dict(name="epsilon_map (run script, np.ones)",
             where="run script / permittivity_components",
             scope="rank", life="persistent", bytes_per_cell=real,
             note="rasterisation target (pass --precision float32 if your script "
                  "hardcodes that dtype); only rank 0's copy is ever written to "
                  "the .h5. write_h5file(free_map=True) + `del epsilon_map` "
                  "releases it"),
        dict(name="MEEP copy of epsilon_input_file",
             where="MEEP (material_data.epsilon_data)",
             scope="rank", life="persistent",
             bytes_per_cell=8.0 if from_file else 0.0,
             note="float64 whatever the file precision, read on rank 0, "
                  "broadcast to all, and kept for the whole run. Only avoidable "
                  "by describing the optics as geometry objects instead of an "
                  "epsilon_input_file"),

        # ---- MEEP's own grid ----
        dict(name="MEEP structure + fields (FDTD grid)",
             where="MEEP",
             scope="distributed", life="persistent",
             bytes_per_cell=cfg["meep_grid_bytes_per_cell"],
             note="2D, complex fields; scales with PML volume and component "
                  "count. Flat in the rank count -- this is NOT the problem"),

        # ---- setup: plotting the epsilon map ----
        dict(name="plot_and_save_epsilon: get_epsilon fetch",
             where="simulator.plot_and_save_epsilon",
             scope="rank", life="transient", bytes_per_cell=2 * mreal,
             note="COLLECTIVE, so every rank receives the full cell; non-master "
                  "ranks now free it immediately"),
        dict(name="plot_and_save_epsilon: figure + PNG/H5",
             where="simulator.plot_and_save_epsilon",
             scope="master", life="transient", bytes_per_cell=62.0,
             note="spline36 imshow of the full map + HDF5 copy; was on every "
                  "rank (~70 B/cell each), now rank 0 only"),

        # ---- setup: the resolution check ----
        dict(name="check_resolution: get_epsilon_grid sample",
             where="simulator.check_resolution",
             # Rank-local for geometry materials, but collective for a file
             # material: MEEP re-converts the filename on every call and
             # broadcasts the read, so every rank has to take part.
             scope="rank" if from_file else "master", life="transient",
             bytes_per_cell=nfreq * 16.0,
             note="complex128 sample of the whole cell, one per frequency. "
                  + ("Stays on every rank because the permittivity comes from "
                     "epsilon_input_file, whose read MEEP broadcasts -- sample "
                     "at a coarser --sample-resolution, or supply epsilon_map, "
                     "to shrink it"
                     if from_file else
                     "Master-only: sampling geometry objects is rank-local")),
        dict(name="check_resolution: derived maps",
             where="simulator.check_resolution",
             scope="master", life="transient",
             bytes_per_cell=33.0 + nfreq * (5 * real + 1.0),
             note=f"n/kappa/valid/res_wavelength/res_skin_depth for {nfreq} "
                  f"frequency/ies plus the stacked maxima; was computed on every "
                  f"rank, now rank 0 only"),

        # ---- during the run: animation ----
        dict(name="animation frame fetch (Ez2_dB)",
             where="stepfunctions._fetch_power_dB",
             scope="rank", life="transient",
             bytes_per_cell=2 * (mcplx if cfg["anim_complex"] else mreal),
             note="COLLECTIVE get_array once per image_every, on every rank. "
                  "field_mode='complex' doubles it. Reduce by animating less "
                  "often or over a sub-volume"),

    ]

    # ---- during the run: field extraction, one path or the other ----
    accumulate_rows = [
        dict(name="accumulate: per-step fetch",
             where="stepfunctions.accumulate_efield_and_hfield",
             scope="rank", life="transient", bytes_per_cell=mcplx,
             note=f"COLLECTIVE, once per component ({ncomp}) per sample, on "
                  f"every rank -- inherent to get_array"),
        dict(name="accumulate: reusable full-cell buffer",
             where="stepfunctions._make_full_buffer",
             scope="master", life="persistent", bytes_per_cell=mcplx,
             note="was allocated on every rank and held all run; now rank 0 only"),
        dict(name="accumulate: downsampled memmaps",
             where="stepfunctions._acc_maps",
             scope="master", life="persistent",
             bytes_per_cell=ncomp * 2 * real / ds,
             note=f"rank-0 memmaps, {ncomp} components x (real, imag), "
                  f"downsampled by {ds}x; dirty pages count toward RSS"),
    ]

    dft_rows = [
        dict(name="DFT fields (method='dft')",
             where="stepfunctions.setup_dft_fields",
             scope="distributed", life="persistent",
             bytes_per_cell=ncomp * 16.0,
             note=f"MEEP accumulates {ncomp} components DISTRIBUTED, no "
                  f"per-step gather. Restrict with dft_where"),
        dict(name="DFT save: one component at a time",
             where="stepfunctions.save_dft_fields",
             scope="rank", life="transient", bytes_per_cell=mcplx,
             note="COLLECTIVE get_dft_array at the end of the run only"),
    ]

    if cfg["extraction"] == "dft":
        return rows + dft_rows, accumulate_rows
    return rows + accumulate_rows, dft_rows


def row_mb(row, ncells, ranks):
    """(per-rank MB, total MB across the job) for one cost row."""
    per_rank_all = row.get("fixed_mb", 0.0) + row["bytes_per_cell"] * ncells / MB
    if row["scope"] == "rank":
        return per_rank_all, per_rank_all * ranks
    if row["scope"] == "master":
        return per_rank_all, per_rank_all
    if row["scope"] == "distributed":
        return per_rank_all / ranks, per_rank_all
    raise ValueError(row["scope"])


SCOPE_LABEL = {"rank": "every rank", "master": "rank 0", "distributed": "split"}


def _wrap(text, width):
    """Minimal word wrap, so --notes stays readable in a terminal."""
    import textwrap
    return textwrap.wrap(text, width) or [""]


def print_estimate(cfg, rank_counts):
    ncells = cfg["ncells"]
    rows, alternative = build_model(cfg)

    print()
    print("=" * 108)
    print(f"MeepSAT memory estimate  --  cell {cfg['size_x']} x {cfg['size_y']} mm "
          f"at resolution {cfg['resolution']} px/mm")
    print(f"grid {cfg['nx']} x {cfg['ny']} = {ncells/1e6:.2f} M cells   "
          f"(one full-cell array: {ncells*cfg['itemsize_real']/MB:.0f} MB real, "
          f"{ncells*cfg['itemsize_meep_complex']/MB:.0f} MB complex)")
    print(f"working precision {mpsat_precision.get_precision()}, MEEP build "
          f"{mpsat_precision.meep_precision()}, {cfg['freqs']} check frequency/ies, "
          f"{cfg['components']} field components, downsample "
          f"{cfg['downsample_x']}x{cfg['downsample_y']}, "
          f"extraction path '{cfg['extraction']}'")
    print("=" * 108)

    hdr = (f"{'component':46s} {'scope':11s} {'life':11s} {'B/cell':>8s} "
           f"{'per rank':>10s}")
    hdr += "".join(f"{'tot @'+str(n):>10s}" for n in rank_counts)
    print(hdr)
    print("-" * len(hdr))

    for row in rows:
        per_rank, _ = row_mb(row, ncells, 1)
        line = (f"{row['name']:46s} {SCOPE_LABEL[row['scope']]:11s} "
                f"{row['life']:11s} {row['bytes_per_cell']:8.1f} "
                f"{per_rank:9.0f}M")
        for n in rank_counts:
            line += f"{row_mb(row, ncells, n)[1]:9.0f}M"
        print(line)

    # ---- totals -------------------------------------------------------
    print("-" * len(hdr))

    def total(pred, ranks):
        return sum(row_mb(r, ncells, ranks)[1] for r in rows if pred(r))

    def floor_mb(ranks):
        return total(lambda r: r["life"] == "persistent", ranks)

    def peak_mb(ranks):
        # The transient stages do not overlap, so the job's high-water mark is
        # the persistent floor plus the single worst spike.
        spikes = [row_mb(r, ncells, ranks)[1]
                  for r in rows if r["life"] == "transient"]
        return floor_mb(ranks) + (max(spikes) if spikes else 0.0)

    def replicated_floor(ranks):
        return total(lambda r: r["life"] == "persistent" and r["scope"] == "rank",
                     ranks)

    def worst_replicated_spike(ranks):
        spikes = [row_mb(r, ncells, ranks)[1] for r in rows
                  if r["life"] == "transient" and r["scope"] == "rank"]
        return max(spikes) if spikes else 0.0

    for label, fn in (("steady-state floor (persistent)", floor_mb),
                      ("peak (floor + largest single spike)", peak_mb),
                      ("  of the floor, replicated per rank", replicated_floor),
                      ("  largest replicated spike", worst_replicated_spike)):
        line = f"{label:46s} {'':11s} {'':11s} {'':8s} {'':10s}"
        for n in rank_counts:
            line += f"{fn(n):9.0f}M"
        print(line)

    one = peak_mb(1)
    line = f"{'peak growth vs 1 rank':46s} {'':11s} {'':11s} {'':8s} {'':10s}"
    for n in rank_counts:
        line += f"{peak_mb(n)/one:9.1f}x"
    print(line)

    if alternative:
        print("-" * len(hdr))
        other = "accumulate" if cfg["extraction"] == "dft" else "dft"
        print(f"NOT COUNTED ABOVE -- the '{other}' extraction path, for comparison "
              f"(select it with --extraction {other}):")
        for row in alternative:
            per_rank, _ = row_mb(row, ncells, 1)
            line = (f"{row['name']:46s} {SCOPE_LABEL[row['scope']]:11s} "
                    f"{row['life']:11s} {row['bytes_per_cell']:8.1f} "
                    f"{per_rank:9.0f}M")
            for n in rank_counts:
                line += f"{row_mb(row, ncells, n)[1]:9.0f}M"
            print(line)

    if cfg["notes"]:
        print("-" * len(hdr))
        print("What each cost is, and what to do about it:")
        for row in rows + alternative:
            print(f"  * {row['name']}  [{row['where']}]")
            for chunk in _wrap(row["note"], 96):
                print(f"      {chunk}")

    print("=" * 108)
    print("Reading this table:")
    print("  * 'every rank' rows are what makes the job grow with the core count;")
    print("    'rank 0' and 'split' rows are flat in it.")
    print("  * MEEP's own FDTD grid is 'split': adding ranks does not grow it, so")
    print("    it is not the reason a run gets more expensive on more cores.")
    print("  * Totals cover one extraction path only; the other is listed below")
    print("    them, uncounted. You register one or the other, never both.")
    print("  * Constants are calibrated on a 2D double-precision MEEP 1.31 run.")
    print("    Check them on your own build and cell with --measure.")
    print()


# =====================================================================
# Measure mode
# =====================================================================

def _status_mb(key):
    with open("/proc/self/status") as fh:
        for line in fh:
            if line.startswith(key):
                return int(line.split()[1]) / 1024.0
    return float("nan")


def _reset_peak():
    """Reset VmHWM so the next stage's peak is its own (Linux >= 4.0)."""
    try:
        with open("/proc/self/clear_refs", "w") as fh:
            fh.write("5")
    except OSError:
        pass


def measure(cfg, outdir):
    import meep as mp
    import h5py
    import meepsat.simulator as sim
    import meepsat.stepfunctions as stepfunctions
    import meepsat.helpers as exf

    try:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        rank, nproc = comm.Get_rank(), comm.Get_size()
    except ImportError:
        comm, rank, nproc = None, 0, 1

    outdir = os.path.join(outdir, f"np{nproc}")
    if rank == 0:
        os.makedirs(outdir, exist_ok=True)
    if comm:
        comm.barrier()

    ncells = cfg["ncells"]
    if rank == 0:
        print()
        print("=" * 123)
        print(f"MEASURED  --  {nproc} rank(s), grid {cfg['nx']} x {cfg['ny']} "
              f"= {ncells/1e6:.2f} M cells")
        print(f"one full-cell array: {ncells*cfg['itemsize_meep_real']/MB:.0f} MB "
              f"real, {ncells*cfg['itemsize_meep_complex']/MB:.0f} MB complex")
        print("=" * 123)
        print(f"{'stage':48s} {'secs':>7s} {'rank0 cur':>10s} {'rank0 pk':>9s} "
              f"{'others pk':>10s} {'SUM cur':>9s} {'SUM pk':>9s} {'B/cell pk':>10s}")
        print("-" * 123)
    # Printed as we go, so a long stage is visible while it runs rather than
    # only in a final table.
    state = {"t": time.time(), "prev_sum": None}

    def stage(name):
        if comm:
            comm.barrier()
        secs = time.time() - state["t"]
        cur, pk = _status_mb("VmRSS"), _status_mb("VmHWM")
        cur_all = comm.gather(cur, root=0) if comm else [cur]
        pk_all = comm.gather(pk, root=0) if comm else [pk]
        if rank == 0:
            others = max(pk_all[1:]) if len(pk_all) > 1 else float("nan")
            prev = state["prev_sum"]
            # Per-rank incremental peak of this stage, in bytes per grid cell:
            # the figure to compare against the estimate table.
            delta = sum(pk_all) - (prev if prev is not None else 0.0)
            bpc = delta * MB / ncells / max(nproc, 1)
            print(f"{name:48s} {secs:7.1f} {cur_all[0]:10.0f} {pk_all[0]:9.0f} "
                  f"{others:10.0f} {sum(cur_all):9.0f} {sum(pk_all):9.0f} "
                  f"{bpc:10.1f}", flush=True)
            state["prev_sum"] = sum(cur_all)
        if comm:
            comm.barrier()
        _reset_peak()
        state["t"] = time.time()

    def want(key):
        """Is this stage group selected? (--skip 3 drops the epsilon plot, etc.)"""
        return key not in cfg["skip"]

    SX, SY, RES = cfg["size_x"], cfg["size_y"], cfg["resolution"]
    dpml = cfg["dpml"]
    _reset_peak()
    stage("0 baseline (after imports)")

    # --- 1. the epsilon map, exactly as a run script builds it ---------
    epsilon_map = np.ones((cfg["nx"], cfg["ny"]),
                          dtype=mpsat_precision.real_dtype())
    lens_x0, lens_x1 = int(0.3 * cfg["nx"]), int(0.4 * cfg["nx"])
    lens_y0, lens_y1 = int(0.2 * cfg["ny"]), int(0.8 * cfg["ny"])
    epsilon_map[lens_x0:lens_x1, lens_y0:lens_y1] = 2.4
    eps_path = os.path.join(outdir, "eps_map")
    if rank == 0:
        with h5py.File(eps_path + ".h5", "w") as h:
            h.create_dataset("eps", data=epsilon_map)
    if comm:
        comm.barrier()
    stage("1 epsilon_map allocated (every rank)")

    # --- 2. the simulation ---------------------------------------------
    simulation = mp.Simulation(
        cell_size=mp.Vector3(SX, SY, 0),
        sources=[mp.Source(mp.ContinuousSource(frequency=cfg["freq"]),
                           component=mp.Ez,
                           center=mp.Vector3(-SX / 2 + 2 * dpml, 0),
                           size=mp.Vector3(0, SY / 4))],
        resolution=RES,
        boundary_layers=[mp.PML(dpml)],
        epsilon_input_file=eps_path + ".h5",
        force_complex_fields=True)
    simulation.init_sim()
    stage("2 Simulation(epsilon_input_file) + init_sim")

    # --- 3. plot_and_save_epsilon --------------------------------------
    # NB: on code without the rank-0 guard this stage has every rank create the
    # same .h5, which HDF5 file locking refuses -- use --skip 3 to A/B the rest.
    if want("3"):
        sim.plot_and_save_epsilon(simulation=simulation, savepath=outdir + "/",
                                  filename_prefix="geometry_plot",
                                  epsilon_data_name="epsilon",
                                  size_x=SX, size_y=SY, dpi=cfg["dpi"])
        stage("3 plot_and_save_epsilon")

    # --- 4. check_resolution -------------------------------------------
    if want("4"):
        mpsat_sim = types.SimpleNamespace(resolution=RES)
        data = {"simulation": {"primary_params": {"resolution": RES,
                                                 "smallest_freq": cfg["freq"]}}}
        sim.check_resolution(data, mpsat_sim, simulation=simulation,
                             source_freq=[cfg["freq"]] * cfg["freqs"],
                             min_pixels_per_wavelength=1,
                             min_pixels_per_skin_depth=0)
        stage("4 check_resolution")

    # --- 5. animation frames -------------------------------------------
    stepfunctions.set_animation_params({"image_every": 1, "Nfps": 5,
                                        "anim_file_name": os.path.join(outdir, "anim.mp4")})
    stepfunctions.set_field_params({"size_x": SX, "size_y": SY,
                                    "savepath": outdir,
                                    "downsampling_factor_x": cfg["downsample_x"],
                                    "downsampling_factor_y": cfg["downsample_y"],
                                    "method": "accumulate",
                                    "scratch_dir": outdir})
    if want("5"):
        stepfunctions.Ez2_dB(simulation)
        stage("5a animation frame (first, builds eps overlay)")
        stepfunctions.Ez2_dB(simulation)
        stage("5b animation frame (subsequent)")

    # --- 6. accumulate path --------------------------------------------
    if want("6"):
        stepfunctions.accumulate_efield_and_hfield(simulation)
        stage("6a accumulate_efield_and_hfield (first)")
        stepfunctions.accumulate_efield_and_hfield(simulation)
        stage("6b accumulate_efield_and_hfield (subsequent)")
        stepfunctions.extract_xyzw(simulation)
        stage("6c extract_xyzw")
        stepfunctions.close_frame_stores()
        stepfunctions.cleanup_accumulators()

    # --- 7. the DFT alternative ----------------------------------------
    if cfg["dft"] and want("7"):
        stepfunctions.setup_dft_fields(simulation, freq=cfg["freq"])
        stage("7a setup_dft_fields (distributed)")
        simulation.run(until=cfg["dft_until"])
        stage("7b run with DFT accumulating")
        stepfunctions.save_dft_fields(simulation)
        stage("7c save_dft_fields")

    if rank != 0:
        return

    print("-" * 123)
    print("All memory figures are MB: 'cur' is resident at the end of the stage,")
    print("'pk' the high-water mark during it (reset per stage). 'B/cell pk' is")
    print("this stage's incremental peak per rank in bytes per grid cell -- the")
    print("figure to compare against the estimate table.")
    print("Run at several -np values and watch SUM pk: what grows is replicated.")
    print()


# =====================================================================

def main(argv=None):
    p = argparse.ArgumentParser(
        description="Per-component memory requirement of a MeepSAT run.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="examples:\n"
               "  python tests/memory_check.py --size-x 600 --size-y 150 "
               "--resolution 10 --ranks 1 8 32\n"
               "  mpirun -np 8 python tests/memory_check.py --measure "
               "--size-x 600 --size-y 150 --resolution 10\n")
    p.add_argument("--size-x", type=float, default=600.0,
                   help="full cell size in x, mm, INCLUDING PML (default 600)")
    p.add_argument("--size-y", type=float, default=150.0,
                   help="full cell size in y, mm, INCLUDING PML (default 150)")
    p.add_argument("--resolution", type=float, default=10.0,
                   help="pixels per mm (default 10)")
    p.add_argument("--ranks", type=int, nargs="+", default=[1, 2, 4, 8, 16],
                   help="rank counts to tabulate in estimate mode")
    p.add_argument("--freqs", type=int, default=1,
                   help="how many frequencies check_resolution samples (default 1)")
    p.add_argument("--components", type=int, default=3,
                   help="field components extracted; 3 for a 2D TM/TE run (default 3)")
    p.add_argument("--downsample", type=int, nargs=2, default=(2, 2),
                   metavar=("DX", "DY"), help="field downsampling factors (default 2 2)")
    p.add_argument("--anim-complex", action="store_true",
                   help="animation uses field_mode='complex' (twice the fetch)")
    p.add_argument("--extraction", default="accumulate",
                   choices=("accumulate", "dft"),
                   help="which field-extraction path the run registers "
                        "(default accumulate, as the example notebooks do)")
    p.add_argument("--precision", default="auto",
                   choices=("auto", "float32", "float64"),
                   help="MeepSAT working precision (default auto)")
    p.add_argument("--baseline-mb", type=float, default=180.0,
                   help="per-rank interpreter+libraries floor, MB (default 180)")
    p.add_argument("--meep-grid-bytes", type=float, default=42.0,
                   help="MEEP FDTD grid, bytes per cell, TOTAL across ranks "
                        "(default 42, measured for 2D complex fields)")
    p.add_argument("--measure", action="store_true",
                   help="measure real RSS stage by stage (run under mpirun)")
    p.add_argument("--outdir", default=None,
                   help="scratch directory for measure mode "
                        "(default ./memory_check_out)")
    p.add_argument("--dpml", type=float, default=10.0,
                   help="PML thickness for measure mode (default 10)")
    p.add_argument("--freq", type=float, default=0.5,
                   help="source frequency in MEEP units for measure mode")
    p.add_argument("--dpi", type=int, default=150,
                   help="figure dpi for measure mode (default 150)")
    p.add_argument("--no-dft", dest="dft", action="store_false",
                   help="skip the DFT stages in measure mode")
    p.add_argument("--geometry-materials", action="store_true",
                   help="optics are geometry objects rather than an "
                        "epsilon_input_file (drops MEEP's replicated copy of the "
                        "map and makes the resolution sampling rank-local)")
    p.add_argument("--notes", action="store_true",
                   help="also print what each cost is and how to reduce it")
    p.add_argument("--skip", nargs="*", default=(), metavar="STAGE",
                   help="measure-mode stage groups to skip: 3 (epsilon plot), "
                        "4 (resolution check), 5 (animation), 6 (accumulate), "
                        "7 (DFT). Useful for A/B-ing one component at a time")
    p.add_argument("--dft-until", type=float, default=0.2,
                   help="MEEP time to run with DFT accumulating (default 0.2)")
    args = p.parse_args(argv)

    mpsat_precision.set_precision(args.precision)

    nx = int(args.size_x * args.resolution + 1)
    ny = int(args.size_y * args.resolution + 1)
    cfg = {
        "size_x": args.size_x, "size_y": args.size_y,
        "resolution": args.resolution,
        "nx": nx, "ny": ny, "ncells": float(nx * ny),
        "freqs": max(1, args.freqs),
        "components": max(1, args.components),
        "downsample_x": args.downsample[0], "downsample_y": args.downsample[1],
        "anim_complex": args.anim_complex,
        "extraction": args.extraction,
        "itemsize_real": np.dtype(mpsat_precision.real_dtype()).itemsize,
        "itemsize_meep_real": np.dtype(mpsat_precision.meep_real_dtype()).itemsize,
        "itemsize_meep_complex": np.dtype(mpsat_precision.meep_complex_dtype()).itemsize,
        "baseline_mb": args.baseline_mb,
        "meep_grid_bytes_per_cell": args.meep_grid_bytes,
        "dpml": args.dpml, "freq": args.freq, "dpi": args.dpi,
        "dft": args.dft, "dft_until": args.dft_until,
        "skip": {str(s) for s in args.skip},
        "notes": args.notes,
        "eps_from_file": not args.geometry_materials,
    }

    if args.measure:
        measure(cfg, args.outdir or os.path.join(os.getcwd(), "memory_check_out"))
    else:
        print_estimate(cfg, args.ranks)
    return 0


if __name__ == "__main__":
    sys.exit(main())
