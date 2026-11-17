import numpy as np
import matplotlib.pyplot as plt
import yt
import unyt
from sys import argv


def _ke(field, data):
    return 0.5 * (data['boxlib', 'vx']*data['boxlib', 'vx'] + 
                  data['boxlib', 'vy']*data['boxlib', 'vy'] + 
                  data['boxlib', 'vz']*data['boxlib', 'vz']) * unyt.m**2/unyt.s**2

def _vorticity_x(field, data):
    dvz_dy = data["boxlib", "vz_gradient_y"] * unyt.m / unyt.s
    dvy_dz = data["boxlib", "vy_gradient_z"] * unyt.m / unyt.s
    vort = dvz_dy - dvy_dz

    return vort

def _vorticity_y(field, data):
    dvx_dz = data["boxlib", "vx_gradient_z"] * unyt.m / unyt.s
    dvz_dx = data["boxlib", "vz_gradient_x"] * unyt.m / unyt.s
    vort = dvx_dz - dvz_dx

    return vort

def _vorticity_z(field, data):
    dvy_dx = data["boxlib", "vy_gradient_x"] * unyt.m / unyt.s
    dvx_dy = data["boxlib", "vx_gradient_y"] * unyt.m / unyt.s
    vort = dvy_dx - dvx_dy

    return vort

def _vorticity_magnitude(field, data):
    vort_x = data["boxlib", "vorticity_x"] 
    vort_y = data["boxlib", "vorticity_y"] 
    vort_z = data["boxlib", "vorticity_z"] 

    return np.sqrt(vort_x**2 + vort_y**2 + vort_z**2)

def _enstrophy(field, data):
    return 0.5 * data["boxlib", "vorticity"]**2

def _grad_squared_vel(field, data):
    du_dx = data["boxlib", "vx_gradient_x"] * unyt.m / unyt.s
    du_dy = data["boxlib", "vx_gradient_y"] * unyt.m / unyt.s
    du_dz = data["boxlib", "vx_gradient_z"] * unyt.m / unyt.s

    dv_dx = data["boxlib", "vy_gradient_x"] * unyt.m / unyt.s
    dv_dy = data["boxlib", "vy_gradient_y"] * unyt.m / unyt.s
    dv_dz = data["boxlib", "vy_gradient_z"] * unyt.m / unyt.s

    dw_dx = data["boxlib", "vz_gradient_x"] * unyt.m / unyt.s
    dw_dy = data["boxlib", "vz_gradient_y"] * unyt.m / unyt.s
    dw_dz = data["boxlib", "vz_gradient_z"] * unyt.m / unyt.s

    g2 = du_dx**2 + du_dy**2 + du_dz**2 + \
         dv_dx**2 + dv_dy**2 + dv_dz**2 + \
         dw_dx**2 + dw_dy**2 + dw_dz**2
    return g2

def setup_vel_gradients(ds):
    """Register gradient + vorticity fields on *ds* (idempotent)."""
    if ("boxlib", "vorticity") in ds.derived_field_list:
        return
    # create vx_gradient_{x,y,z} and vy_gradient_{x,y,z}
    # (yt handles AMR level boundaries / ghost zones internally)
    ds.force_periodicity()
    ds.add_gradient_fields(("boxlib", "vx"))
    ds.add_gradient_fields(("boxlib", "vy"))
    ds.add_gradient_fields(("boxlib", "vz"))

def add_new_fields(ds):
    """Add derived fields to *ds* ."""
    setup_vel_gradients(ds)
    if ("boxlib", "KE") not in ds.derived_field_list:
        ds.add_field(("boxlib", "KE"), function=_ke, units="m**2/s**2", sampling_type='local')
    if ("boxlib", "vorticity_x") not in ds.derived_field_list:
        ds.add_field(("boxlib", "vorticity_x"), function=_vorticity_x, units="1/s", sampling_type='local')
    if ("boxlib", "vorticity_y") not in ds.derived_field_list:
        ds.add_field(("boxlib", "vorticity_y"), function=_vorticity_y, units="1/s", sampling_type='local')
    if ("boxlib", "vorticity_z") not in ds.derived_field_list:
        ds.add_field(("boxlib", "vorticity_z"), function=_vorticity_z, units="1/s", sampling_type='local')
    if ("boxlib", "vorticity") not in ds.derived_field_list:
        ds.add_field(("boxlib", "vorticity"), function=_vorticity_magnitude, units="1/s", sampling_type='local')
    if ("boxlib", "enstrophy") not in ds.derived_field_list:
        ds.add_field(("boxlib", "enstrophy"), function=_enstrophy, units="1/s**2", sampling_type='local')
    if ("boxlib", "grad_squared_vel") not in ds.derived_field_list:
        ds.add_field(("boxlib", "grad_squared_vel"), function=_grad_squared_vel, units="1/s**2", sampling_type='local')

def plot_vorticity(ds, axis='z', bounds=None):
    """SlicePlot of omega_z. Calls setup_vorticity(ds) if needed."""
    s = yt.SlicePlot(ds, axis, ('boxlib', 'vorticity'))
    s.set_cmap('all', 'viridis')
    if bounds is not None:
        s.set_zlim('all', bounds[0], bounds[1])

    s.annotate_time()
    return s

if __name__ == "__main__":
    curr_file = argv[1]

    if len(argv) > 2:
        bounds = [float(argv[2]), float(argv[3])]

    ds = yt.load(curr_file)
    add_new_fields(ds)

    for axis in ('x', 'y', 'z'):
        plot = plot_vorticity(ds, axis=axis, bounds=bounds)
        plot.save(f"plots_vort_{axis}/", mpl_kwargs={'dpi': 300})
