"""
Quick-look plot of a MeepSAT model JSON, for checking the geometry before a run.

The model is built at a low resolution with the same json_to_script code a run uses,
the permittivity is sampled at the source frequency and plotted with the PML, the
sources, the monitors and the component names on top. Nothing is run.

From the shell (the meepsat-plot command, or python -m meepsat.visualization):

    meepsat-plot model.json                          # interactive window
    meepsat-plot model.json --save                   # model_geometry.png next to the JSON
    meepsat-plot model.json --save fig.pdf --show    # both
    meepsat-plot model.json --quantity both --cmap magma --xlim -100 900
    meepsat-plot model.json --set forebaffles.forebaffle_top.hypotenuse=700

From Python:

    import meepsat.visualization as mpsat_plot
    model = mpsat_plot.build_model(mpsat_plot.load_model("model.json"), resolution=1)
    fig = mpsat_plot.plot_system(model, quantity="both")
"""
import argparse
import contextlib
import copy
import io
import json
import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

from meepsat.json_to_script import COMPONENT_BUILDERS

# The component sections a model JSON can hold, in the order they are built
COMPONENT_SECTIONS = [section for section, _ in COMPONENT_BUILDERS]

# MEEP reports a perfect conductor (mp.metal, mp.perfect_electric_conductor) as
# epsilon = -1e20; MeepSAT's epsilon maps use large negative values for PEC as well
PEC_EPS_THRESHOLD = -1e3

# Default material of the sampling simulation: marks the points no geometry object
# covers, where the epsilon map is used instead
_UNCOVERED_EPS = 0.123456789


# ================================ Loading the JSON ================================
def parse_override(text):
    """
    "a.b.c=value" -> (["a", "b", "c"], value), with the value read as JSON when it
    parses (numbers, true/false, null, lists) and as a string otherwise
    """
    if "=" not in text:
        raise ValueError(f"--set expects key.path=value, got {text!r}")
    key, value = text.split("=", 1)
    try:
        value = json.loads(value)
    except json.JSONDecodeError:
        pass
    return key.strip().split("."), value


def load_model(json_file, overrides=()):
    """
    Reads a model JSON and applies "key.path=value" overrides to it
    """
    with open(json_file) as f:
        data = json.load(f)

    for override in overrides:
        keys, value = parse_override(override)
        target = data
        for key in keys[:-1]:
            if key not in target:
                raise KeyError(f"--set {override}: '{key}' is not in the JSON")
            target = target[key]
        target[keys[-1]] = value
    return data


def source_frequency(data):
    """
    Frequency the permittivity is sampled at: the first source's, else the smallest
    frequency of the simulation
    """
    import meepsat.json_to_script as json_to_script

    freq = json_to_script.source_frequency(data)
    if freq is None:
        freq = data["simulation"]["primary_params"].get("smallest_freq")
    return float(freq)


# ================================ Building the model ================================
def build_model(data, resolution=1, freq=None, skip=(), base_dir=None, workdir=None, verbose=False):
    """
    Builds the components of a model JSON at `resolution` and samples its permittivity

    Arguments
    ---------
    data : dict
        Model JSON (see load_model); it is copied, not modified
    resolution : int
        Pixels per unit length used for the epsilon map, the meshed components
        and the sampled permittivity (an integer, as in init_epsilon_map)
    freq : float, optional
        Frequency (MEEP units) the permittivity is sampled at, so that conductive
        materials show their loss (default: the source frequency)
    skip : iterable of str
        JSON sections not to build, e.g. ["wall_absorbers"]
    base_dir : str, optional
        Directory relative paths in the JSON (feedhorn profiles, ...) are read
        from (default: the current directory)
    workdir : str, optional
        Where the components may write their files (default: a temporary
        directory, removed afterwards), so that the JSON's output directory is
        never written to
    verbose : bool
        Show the output of the component code (default: only on errors)

    Returns
    -------
    dict with 'eps' (complex permittivity, indexed [x, y]), 'extent', 'freq',
    'resolution', 'cell_size' (with PML), 'boundary_layers', 'namespace' (the
    objects the component code made) and 'data' (the JSON as it was built)
    """
    import meep as mp
    import meepsat.simulator as sim
    import meepsat.meep_geometry as comp_meep
    import meepsat.permittivity_components as comp_eps
    import meepsat.json_to_script as json_to_script

    resolution = int(resolution)
    if resolution < 1:
        raise ValueError("resolution must be an integer of at least 1 pixel per unit length")

    data = copy.deepcopy(data)
    params = data["simulation"]["primary_params"]
    params["resolution"] = resolution
    freq = source_frequency(data) if freq is None else float(freq)

    unknown = set(skip) - set(COMPONENT_SECTIONS)
    if unknown:
        raise ValueError(f"Cannot skip {sorted(unknown)}: the sections are {COMPONENT_SECTIONS}")
    for section in skip:
        data.pop(section, None)
    if "absorbers" in data:
        print("WARNING: the 'absorbers' section needs PyramidalAbsorbers, which is not in "
              "meepsat yet, so it is not plotted")

    own_workdir = workdir is None
    if own_workdir:
        workdir = tempfile.mkdtemp(prefix="meepsat_plot_")
    data.setdefault("output", {})["savepath"] = {"path": str(Path(workdir).resolve()) + "/"}
    # The lenses write the epsilon map only when an HDF5 file is named
    data["output"].setdefault("epsilon_h5_file", {"filename": "meepsat_plot"})

    log = io.StringIO()
    previous_dir = os.getcwd()
    start = time.time()
    try:
        if base_dir is not None:
            os.chdir(base_dir)
        # The components print a lot; keep it unless something fails
        with contextlib.redirect_stdout(sys.stdout if verbose else log):
            mp.verbosity(1 if verbose else 0)
            boundary = data["boundary_layers"]["boundary"]
            mpsat_sim = sim.sim_init(sim_name=str(data["simulation"].get("name", "model")),
                                     cell_size=[params["cell_size"]["x"], params["cell_size"]["y"],
                                                params["cell_size"]["z"]],
                                     smallest_freq=params.get("smallest_freq", freq),
                                     resolution=resolution,
                                     boundary_layer_type=boundary["type"],
                                     boundary_layer_size=boundary["size"],
                                     factor_dpml=boundary["factor_dpml"])

            size_x, size_y = mpsat_sim.cell_size[0], mpsat_sim.cell_size[1]
            epsilon_map = np.ones((int(size_x * resolution + 1), int(size_y * resolution + 1)),
                                  dtype=mpsat_sim.real_dtype)
            namespace = {"mp": mp, "np": np, "comp_meep": comp_meep, "comp_eps": comp_eps,
                         "json_to_script": json_to_script, "data": data,
                         "mpsat_sim": mpsat_sim, "epsilon_map": epsilon_map}

            exec(json_to_script.boundary_script(data), namespace)
            for section, add_component in COMPONENT_BUILDERS:
                if section in data:
                    exec(add_component(data), namespace)

            # Sample the geometry objects on the epsilon map's own grid. Points that no
            # object covers come back as the sentinel default material and take the
            # epsilon map's value (lenses, feedhorns), as MEEP does with
            # geometry on top of an epsilon_input_file.
            xtics = -size_x / 2 + np.arange(epsilon_map.shape[0]) / resolution
            ytics = -size_y / 2 + np.arange(epsilon_map.shape[1]) / resolution
            simulation = mp.Simulation(cell_size=mpsat_sim.cell, resolution=resolution,
                                       geometry=mpsat_sim.meep_geometry,
                                       default_material=mp.Medium(epsilon=_UNCOVERED_EPS))
            eps = np.asarray(simulation.get_epsilon_grid(xtics, ytics, np.array([0.0]), frequency=freq))
            eps = eps.reshape(epsilon_map.shape)
            uncovered = np.isclose(eps.real, _UNCOVERED_EPS) & (eps.imag == 0)
            eps[uncovered] = epsilon_map[uncovered]
    except Exception:
        if not verbose:
            sys.stdout.write(log.getvalue())
        raise
    finally:
        os.chdir(previous_dir)
        if own_workdir:
            import shutil
            shutil.rmtree(workdir, ignore_errors=True)

    return {
        "eps": eps,
        "extent": [xtics[0], xtics[-1], ytics[0], ytics[-1]],
        "freq": freq,
        "resolution": resolution,
        "cell_size": (size_x, size_y),
        "boundary_layers": namespace.get("boundary_layers", []),
        "namespace": namespace,
        "data": data,
        "n_geometry": len(mpsat_sim.meep_geometry),
        "build_time": time.time() - start,
    }


# ================================ Overlays ================================
def _vector(value):
    """JSON vector ([x, y, z], {"x":..}, or "mp.Vector3(...)") -> (x, y)"""
    import meep as mp

    if isinstance(value, str):
        value = eval(value, {"mp": mp, "np": np})
    if isinstance(value, dict):
        return float(value.get("x", 0)), float(value.get("y", 0))
    if isinstance(value, mp.Vector3):
        return float(value.x), float(value.y)
    return float(value[0]), float(value[1])


def _label(ax, x, y, text, color, **kwargs):
    ax.annotate(text, (x, y), fontsize=7, color=color, ha="center", va="bottom",
                xytext=(0, 3), textcoords="offset points",
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.7), **kwargs)


def draw_pml(ax, model, color="tab:red", alpha=0.15):
    """Shades the PML layers of model['boundary_layers']"""
    import meep as mp
    from matplotlib.patches import Rectangle

    x0, x1, y0, y1 = model["extent"]
    for layer in model["boundary_layers"]:
        t = layer.thickness
        directions = [mp.X, mp.Y] if layer.direction == mp.ALL else [layer.direction]
        sides = [mp.Low, mp.High] if layer.side == mp.ALL else [layer.side]
        for direction in directions:
            for side in sides:
                if direction == mp.X:
                    xy, w, h = ((x0, y0) if side == mp.Low else (x1 - t, y0)), t, y1 - y0
                else:
                    xy, w, h = ((x0, y0) if side == mp.Low else (x0, y1 - t)), x1 - x0, t
                ax.add_patch(Rectangle(xy, w, h, fc=color, ec="none", alpha=alpha, hatch="//",
                                       zorder=3))


def draw_sources(ax, data, color="tab:orange", labels=True):
    """Each source as its line, with an arrow along its direction"""
    span = max(data["simulation"]["primary_params"]["cell_size"]["x"],
               data["simulation"]["primary_params"]["cell_size"]["y"])
    for name, src in data.get("sources", {}).items():
        cx, cy = src["center_x"], src["center_y"]
        sx, sy = src.get("size_x", 0), src.get("size_y", 0)
        ax.plot([cx - sx / 2, cx + sx / 2], [cy - sy / 2, cy + sy / 2], color=color, lw=2, zorder=5)
        angle = np.radians(src.get("extra_args", {}).get("angle", 0))
        length = 0.06 * span
        ax.annotate("", (cx + length * np.cos(angle), cy + length * np.sin(angle)), (cx, cy),
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=1.5), zorder=5)
        if labels:
            ax.annotate(f"{name} ({src['type']})", (cx - sx / 2, cy - sy / 2), fontsize=7, color=color,
                        ha="center", va="top", xytext=(0, -3), textcoords="offset points",
                        bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.7))


def draw_monitors(ax, data, color="tab:cyan", labels=True):
    """Volume and flux monitors as dashed outlines (or lines)"""
    from matplotlib.patches import Rectangle

    for section in ("monitor", "flux_monitor"):
        for name, monitor in data.get(section, {}).items():
            cx, cy = _vector(monitor["center"])
            sx, sy = _vector(monitor["size"])
            ax.add_patch(Rectangle((cx - sx / 2, cy - sy / 2), sx, sy, fc="none", ec=color,
                                   ls="--", lw=1.2, zorder=5))
            if labels:
                _label(ax, cx, cy + sy / 2, name, color)


def _edge_label(ax, x, text, color, top):
    """A vertical name at x along the top or bottom edge of the axes, off the optics"""
    ax.text(x, 0.99 if top else 0.01, text, transform=ax.get_xaxis_transform(), rotation=90,
            fontsize=7, color=color, ha="center", va="top" if top else "bottom", clip_on=True,
            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.7), zorder=6)


def draw_component_labels(ax, data, color="black"):
    """
    Names of the lenses (along the bottom edge), apertures (along the top edge),
    forebaffles (half-way along them), detectors and filters
    """
    for name, lens in data.get("lenses", {}).items():
        _edge_label(ax, lens["x"] + lens.get("thick", 0) / 2, lens.get("name", name), color, top=False)
    for name, aperture in data.get("apertures", {}).items():
        if "pos_x" in aperture:
            _edge_label(ax, aperture["pos_x"], name, color, top=True)
    for name, forebaffle in data.get("forebaffles", {}).items():
        angle = np.radians(forebaffle.get("angle_degrees", 0))
        half = forebaffle.get("hypotenuse", 0) / 2
        x = forebaffle.get("x_vertex", 0) + half * np.cos(angle)
        y = forebaffle.get("y_vertex", 0) + half * np.sin(angle)
        _label(ax, x, y, forebaffle.get("name", name), color)
    # Detectors and filters are usually thin and close together, so their names stand
    # upright above them
    for name, detector in data.get("detectors", {}).items():
        _label(ax, detector["pos_x"], detector["diameter"] / 2, name, color, rotation=90)
    for name, filt in data.get("filters", {}).items():
        if "center" in filt:
            x, y = _vector(filt["center"])
            half_height = _vector(filt["size"])[1] / 2 if "size" in filt else 0
            _label(ax, x, y + half_height, name, color, rotation=90)


# ================================ Plotting ================================
QUANTITIES = {
    "real": ("Re(ε)", lambda eps: eps.real),
    "imag": ("Im(ε) (loss)", lambda eps: eps.imag),
    "index": ("n = Re√ε", lambda eps: np.sqrt(eps).real),
}


def plot_system(model, quantity="real", cmap="viridis", loss_cmap="magma", vmin=None, vmax=None,
                log=False, metal_color="0.25", show_pml=True, show_sources=True,
                show_monitors=True, show_labels=True, xlim=None, ylim=None,
                figsize=None, title=None):
    """
    Plots the permittivity of a model from build_model, with the PML, the sources,
    the monitors and the component names on top

    quantity : "real", "imag", "index" or "both" (real and imag, one above the other)
    cmap, loss_cmap : colormaps of the real part/index and of the loss
    vmin, vmax : colour limits (default: the range of the non-metal points)
    log : logarithmic colour scale (non-positive values are left blank)
    metal_color : colour of the perfect conductors (PEC, mp.metal)

    Returns the matplotlib figure
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap, LogNorm, Normalize
    from matplotlib.patches import Patch

    quantities = ["real", "imag"] if quantity == "both" else [quantity]
    for q in quantities:
        if q not in QUANTITIES:
            raise ValueError(f"quantity must be one of {list(QUANTITIES) + ['both']}, got {q!r}")

    eps = model["eps"]
    metal = eps.real < PEC_EPS_THRESHOLD
    extent = model["extent"]
    data = model["data"]

    if figsize is None:
        x_span = (xlim[1] - xlim[0]) if xlim else extent[1] - extent[0]
        y_span = (ylim[1] - ylim[0]) if ylim else extent[3] - extent[2]
        width = 11
        height = np.clip(width * y_span / x_span, 2.5, 9) * len(quantities) + 0.8
        figsize = (width, height)
    fig, axes = plt.subplots(len(quantities), 1, figsize=figsize, sharex=True, sharey=True,
                             squeeze=False, constrained_layout=True)

    for ax, q in zip(axes[:, 0], quantities):
        name, func = QUANTITIES[q]
        with np.errstate(invalid="ignore"):
            values = func(np.where(metal, np.nan, eps))
        finite = values[np.isfinite(values)]
        if log:
            finite = finite[finite > 0]
            values = np.where(values > 0, values, np.nan)
        lo = vmin if vmin is not None else (finite.min() if finite.size else 0)
        hi = vmax if vmax is not None else (finite.max() if finite.size else 1)
        if hi <= lo:
            hi = lo + (abs(lo) * 1e-3 or 1)
        norm = LogNorm(lo, hi) if log else Normalize(lo, hi)

        im = ax.imshow(values.T, origin="lower", extent=extent, cmap=loss_cmap if q == "imag" else cmap,
                       norm=norm, interpolation="nearest")
        ax.imshow(np.where(metal, 1.0, np.nan).T, origin="lower", extent=extent,
                  cmap=ListedColormap([metal_color]), interpolation="nearest")
        fig.colorbar(im, ax=ax, label=name, shrink=0.9, pad=0.01)

        handles = []
        if metal.any():
            handles.append(Patch(color=metal_color, label="PEC / metal"))
        if show_pml and model["boundary_layers"]:
            draw_pml(ax, model)
            handles.append(Patch(fc="tab:red", alpha=0.3, hatch="//", label="PML"))
        if show_sources and data.get("sources"):
            draw_sources(ax, data, labels=show_labels)
            handles.append(Patch(color="tab:orange", label="source"))
        if show_monitors and (data.get("monitor") or data.get("flux_monitor")):
            draw_monitors(ax, data, labels=show_labels)
            handles.append(Patch(fc="none", ec="tab:cyan", ls="--", label="monitor"))
        if show_labels:
            draw_component_labels(ax, data)
        if handles:
            ax.legend(handles=handles, loc="upper right", fontsize=7, framealpha=0.8)

        ax.set_xlim(xlim if xlim else extent[:2])
        ax.set_ylim(ylim if ylim else extent[2:])
        ax.set_aspect("equal")
        ax.set_ylabel("y (mm)")
    axes[-1, 0].set_xlabel("x (mm)")

    if title is None:
        title = (f"{data['simulation'].get('name', 'model')} — {model['resolution']} px/mm, "
                 f"ε at f = {model['freq']:.4g} (MEEP units)")
    fig.suptitle(title, fontsize=10)
    return fig


# ================================ Command line ================================
def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        prog="meepsat-plot",
        description="Plot the system described by a MeepSAT model JSON at a low resolution, "
                    "for checking the geometry before a run.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("json_file", help="model JSON")

    out = parser.add_argument_group("output")
    out.add_argument("--save", nargs="?", const="auto", default=None, metavar="PATH",
                     help="save the figure; without PATH it is <json name>_geometry.png next to the JSON")
    out.add_argument("--show", action="store_true",
                     help="open an interactive window (the default when --save is not given)")
    out.add_argument("--dpi", type=int, default=200, help="resolution of the saved figure")
    out.add_argument("--figsize", type=float, nargs=2, metavar=("W", "H"), default=None,
                     help="figure size in inches (default: from the cell's aspect ratio)")
    out.add_argument("--title", default=None, help="figure title")

    model = parser.add_argument_group("model")
    model.add_argument("-r", "--resolution", type=int, default=1,
                       help="pixels per unit length (mm) of the quick-look build")
    model.add_argument("--freq", type=float, default=None,
                       help="frequency (MEEP units) the permittivity is sampled at "
                            "(default: the source frequency)")
    model.add_argument("--skip", nargs="+", default=[], choices=COMPONENT_SECTIONS, metavar="SECTION",
                       help=f"JSON sections not to build: {', '.join(COMPONENT_SECTIONS)}")
    model.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY.PATH=VALUE",
                       help="change a JSON value before building, e.g. "
                            "forebaffles.forebaffle_top.hypotenuse=700 (repeatable)")
    model.add_argument("--base-dir", default=None,
                       help="directory relative paths in the JSON are read from (default: the JSON's)")
    model.add_argument("-v", "--verbose", action="store_true", help="show the component build output")

    look = parser.add_argument_group("appearance")
    look.add_argument("-q", "--quantity", choices=list(QUANTITIES) + ["both"], default="real",
                      help="what to plot: Re(ε), Im(ε), the refractive index, or Re and Im")
    look.add_argument("--cmap", default="viridis", help="colormap of Re(ε) / the index")
    look.add_argument("--loss-cmap", default="magma", help="colormap of Im(ε)")
    look.add_argument("--vmin", type=float, default=None, help="lower colour limit")
    look.add_argument("--vmax", type=float, default=None, help="upper colour limit")
    look.add_argument("--log", action="store_true", help="logarithmic colour scale")
    look.add_argument("--metal-color", default="0.25", help="colour of the perfect conductors")
    look.add_argument("--xlim", type=float, nargs=2, metavar=("MIN", "MAX"), default=None, help="zoom in x (mm)")
    look.add_argument("--ylim", type=float, nargs=2, metavar=("MIN", "MAX"), default=None, help="zoom in y (mm)")
    look.add_argument("--no-pml", action="store_true", help="do not shade the PML")
    look.add_argument("--no-sources", action="store_true", help="do not draw the sources")
    look.add_argument("--no-monitors", action="store_true", help="do not draw the monitors")
    look.add_argument("--no-labels", action="store_true", help="do not name the components")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    json_file = Path(args.json_file).resolve()
    try:
        data = load_model(json_file, args.overrides)
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise SystemExit(f"meepsat-plot: {error}")
    show = args.show or args.save is None

    import matplotlib
    if not show:
        # Saving only: no window, so no display is needed (e.g. on a cluster node)
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import meepsat.helpers as mpsat_helpers

    base_dir = args.base_dir if args.base_dir is not None else json_file.parent

    model = build_model(data, resolution=args.resolution, freq=args.freq, skip=args.skip,
                        base_dir=base_dir, verbose=args.verbose)
    if not mpsat_helpers.is_master():
        return

    eps = model["eps"]
    print(f"Built {json_file.name} at {model['resolution']} px/mm in {model['build_time']:.1f} s: "
          f"{model['n_geometry']} geometry objects, {eps.shape[0]} x {eps.shape[1]} samples, "
          f"{int((eps.real < PEC_EPS_THRESHOLD).sum())} PEC, "
          f"{int((eps.imag > 0).sum())} lossy")

    fig = plot_system(model, quantity=args.quantity, cmap=args.cmap, loss_cmap=args.loss_cmap,
                      vmin=args.vmin, vmax=args.vmax, log=args.log, metal_color=args.metal_color,
                      show_pml=not args.no_pml, show_sources=not args.no_sources,
                      show_monitors=not args.no_monitors, show_labels=not args.no_labels,
                      xlim=args.xlim, ylim=args.ylim, figsize=args.figsize, title=args.title)

    if args.save is not None:
        save_path = (json_file.parent / f"{json_file.stem}_geometry.png" if args.save == "auto"
                     else Path(args.save))
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=args.dpi)
        print(f"Saved {save_path}")
    if show:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    main()
