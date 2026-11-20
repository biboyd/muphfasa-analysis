import argparse
import os

import numpy as np
import yt

from tgv_plots.plot_vort_slice import setup_vel_gradients
from muphfasa_yt import load_mks


def _q_criterion(field, data):
    """Q = 0.5 (|Omega|^2 - |S|^2) = -0.5 sum_ij (du_i/dx_j)(du_j/dx_i)."""
    g = {(v, x): data["boxlib", f"v{v}_gradient_{x}"] for v in "xyz" for x in "xyz"}
    q = 0
    for i in "xyz":
        for j in "xyz":
            q = q - 0.5 * g[i, j] * g[j, i]
    return q


def add_q_field(ds):
    """Add the Q-criterion field to *ds* (idempotent)."""
    setup_vel_gradients(ds)
    if ("boxlib", "Q") not in ds.derived_field_list:
        ds.add_field(("boxlib", "Q"), function=_q_criterion, units="1/s**2",
                     sampling_type="local", force_override=True)


def default_bounds(ds, lo_frac=0.1, hi_frac=0.8):
    """Isosurface range as fractions of the snapshot's maximum Q."""
    qmax = float(ds.all_data().max(("boxlib", "Q")))
    return lo_frac * qmax, hi_frac * qmax


def render_q(ds, bounds=None, n_levels=4, width_frac=0.01, opacity=30.0,
             cmap="viridis", view=(1.0, 1.0, 1.0), resolution=1024, zoom=1.0):
    """Volume render of Q isosurfaces; returns the yt Scene."""
    if bounds is None:
        bounds = default_bounds(ds)
    qmin, qmax = bounds

    sc = yt.create_scene(ds, ("boxlib", "Q"), lens_type="plane-parallel")
    source = sc[0]
    source.set_log(False)
    source.set_use_ghost_zones(True)

    # Each layer is a gaussian in Q-space, so it renders as a thin shell at one
    # Q value; w is in units of (field range)^2. yt's default alpha ramps up
    # log-spaced across layers, which hides the low-Q shells, so use a uniform one.
    tf = yt.ColorTransferFunction((qmin, qmax))
    tf.add_layers(n_levels, w=width_frac * (qmax - qmin) ** 2 / n_levels,
                  mi=qmin, ma=qmax, colormap=cmap,
                  alpha=np.full(n_levels, float(opacity)))
    source.set_transfer_function(tf)

    center = 0.5 * (ds.domain_left_edge + ds.domain_right_edge)
    box = ds.domain_width.max()
    cam = sc.camera
    cam.resolution = (resolution, resolution)
    cam.set_width(np.sqrt(3.0) * 1.05 * box / zoom)
    direction = np.asarray(view, dtype=float)
    direction /= np.linalg.norm(direction)
    cam.focus = center
    cam.position = center + 2.0 * box * direction
    cam.switch_orientation(normal_vector=-direction, north_vector=[0, 0, 1])

    # the parallel image reduction sums every rank's image, so draw the outline on one rank only
    if yt.is_root():
        sc.annotate_domain(ds, color=[1, 1, 1, 0.3])
    return sc


def main():
    p = argparse.ArgumentParser(description="Volume render of Q-criterion isosurfaces.")
    p.add_argument("plotfile", help="path to an amrex plotfile")
    p.add_argument("--qmin", type=float, help="lowest isosurface value in 1/s^2 (default: 0.1*max(Q))")
    p.add_argument("--qmax", type=float, help="highest isosurface value in 1/s^2 (default: 0.8*max(Q))")
    p.add_argument("-n", "--n-levels", type=int, default=4, help="number of isosurfaces")
    p.add_argument("--width", type=float, default=0.01,
                   help="shell thickness as a fraction of the Q range (larger = thicker/more opaque)")
    p.add_argument("--opacity", type=float, default=30.0,
                   help="opacity of each shell; raise if surfaces look faint, lower if they wash out")
    p.add_argument("--cmap", default="viridis")
    p.add_argument("--view", type=float, nargs=3, default=(1.0, 1.0, 1.0),
                   metavar=("X", "Y", "Z"), help="direction from the box center to the camera")
    p.add_argument("--resolution", type=int, default=1024, help="image size in pixels (square)")
    p.add_argument("--zoom", type=float, default=1.0)
    p.add_argument("--sigma-clip", type=float, default=0.0,
                   help="clip the image at this many sigma above the mean to set brightness (0 = off)")
    p.add_argument("--outdir", default="plots_Q")
    args = p.parse_args()

    yt.enable_parallelism()
    ds = load_mks(args.plotfile)
    add_q_field(ds)

    lo, hi = default_bounds(ds)
    bounds = (args.qmin if args.qmin is not None else lo,
              args.qmax if args.qmax is not None else hi)

    sc = render_q(ds, bounds=bounds, n_levels=args.n_levels, width_frac=args.width,
                  opacity=args.opacity, cmap=args.cmap, view=args.view, resolution=args.resolution, zoom=args.zoom)

    # render is collective across ranks; every rank then holds the full composited image
    sc.render()

    if yt.is_root():
        os.makedirs(args.outdir, exist_ok=True)
        name = os.path.basename(os.path.normpath(args.plotfile))
        out = os.path.join(args.outdir, f"{name}_Q.png")
        label = f"t = {float(ds.current_time.to('s')):.3f} s    Q in [{bounds[0]:.3g}, {bounds[1]:.3g}] 1/s$^2$"
        sc.save_annotated(out, sigma_clip=args.sigma_clip, render=False,
                          text_annotate=[[(0.03, 0.03), label,
                                          dict(color="white", fontsize=14, horizontalalignment="left")]])
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
