# MeepSAT tests

## `memory_check.py` — where a run's memory actually goes

Under `mpirun -np N python run_script.py` the whole Python script runs N times.
MEEP splits its own FDTD grid 1/N per rank, but every full-cell NumPy array a
MeepSAT module allocates is paid N times over. So the question this script
answers is not "how much memory does this need?" but **"how much of it is
replicated on every rank?"** — that replicated part is what makes a run more
expensive on more cores.

### Estimate mode (serial, nothing is simulated)

```bash
python tests/memory_check.py --size-x 600 --size-y 150 --resolution 10 \
                            --ranks 1 8 32 --precision float32
```

Prints one row per component with its cost in bytes per grid cell, its scope
(`every rank` / `rank 0` / `split` across ranks), whether it is persistent or a
spike, and the totals at each rank count. Use it to size a job before queueing
it, and to see which knob is worth turning.

Useful flags: `--freqs` (how many frequencies `check_resolution` samples),
`--components`, `--downsample DX DY`, `--anim-complex`, `--extraction
{accumulate,dft}` to pick the field-extraction path, `--precision` to match what
your run script allocates.

### Measure mode (real calls, real RSS)

```bash
mpirun -np 8 python tests/memory_check.py --measure \
       --size-x 600 --size-y 150 --resolution 10 --outdir /tmp/memcheck
```

Builds a simulation of that cell and walks the real pipeline — `epsilon_map` →
`Simulation(epsilon_input_file=...)` → `plot_and_save_epsilon` →
`check_resolution` → animation frame → field accumulation → the DFT path —
printing each stage's per-rank resident and peak memory as it completes.
`VmHWM` is reset between stages, so each peak belongs to its own stage. `B/cell
pk` is the figure to compare against the estimate table.

Run it at several `-np` values and watch `SUM pk`: whatever grows is replicated.
`--skip 3 7` drops stage groups, which is how you A/B a single component.

Requires Linux (`/proc/self/status`, `/proc/self/clear_refs`).

### Re-calibrating

The estimate constants were measured on a 9.0 M cell 2D cell with MEEP 1.31
(double precision). `--baseline-mb` and `--meep-grid-bytes` set the two
machine-dependent ones. If you move to another MEEP build, precision or
dimensionality, re-measure and update them. The structural conclusions (which
costs are replicated) hold regardless; the exact bytes per cell do not.

### What was measured on this build

At 8 ranks on a 9.0 M cell cell, per rank: the `epsilon_map` costs 4–8 B/cell,
MEEP's own float64 copy of `epsilon_input_file` another 8 B/cell (it is read on
rank 0 and broadcast, and kept for the whole run), an animation frame fetch
~16 B/cell, and a field-accumulation sample ~16 B/cell. MEEP's FDTD grid itself
is ~42 B/cell *in total* and does not grow with the rank count — it is not the
reason a run gets more expensive on more cores.

Two MEEP facts worth knowing before adding a rank guard anywhere, both measured
rather than assumed:

* `get_array`, `get_epsilon`, `get_dft_array` and `get_array_metadata` are
  **collective** — every rank must call them or the job deadlocks. Only the work
  that *follows* them can be guarded.
* `get_epsilon_grid` is rank-local for a structure made of geometry objects, but
  **not** when the permittivity comes from `epsilon_input_file`: the filename is
  re-converted on every call and the HDF5 read is broadcast from rank 0. A
  master-only call in that case hangs the job.
