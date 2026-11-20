import numpy as np
import matplotlib.pyplot as plt
import yt
from os import listdir

from muphfasa_yt import load_mks


def _vorticity_magnitude(field, data):
    # omega_z = d(vy)/dx - d(vx)/dy  [1/s]
    # add_gradient_fields creates ("boxlib", "vx_gradient_x"), etc.
    dvy_dx = data["boxlib", "vy_gradient_x"]
    dvx_dy = data["boxlib", "vx_gradient_y"]
    # grad fields carry a leading NaN/zero at domain edge from
    # one-sided differencing — clean for display
    vort = dvy_dx - dvx_dy
    return vort


def setup_vorticity(ds):
    """Register gradient + vorticity fields on *ds* (idempotent)."""
    if ("boxlib", "vorticity") in ds.derived_field_list:
        return
    # create vx_gradient_{x,y,z} and vy_gradient_{x,y,z}
    # (yt handles AMR level boundaries / ghost zones internally)
    ds.add_gradient_fields(("boxlib", "vx"))
    ds.add_gradient_fields(("boxlib", "vy"))

    ds.add_field(
        name=("boxlib", "vorticity"),
        function=_vorticity_magnitude,
        sampling_type="local",
        units="1/s",
        display_name=r"$\omega_z$",
        take_log=False,
    )

def plot_vorticity(ds):
    """SlicePlot of omega_z. Calls setup_vorticity(ds) if needed."""
    setup_vorticity(ds)
    s = yt.SlicePlot(ds, 'z', ('boxlib', 'vorticity'))
    s.set_cmap('all', 'RdBu_r')
    s.set_zlim('all', -5, 5)  # uncomment & tune to your data

    s.annotate_contour('density', levels=1, factor=1, take_log=False,
                       clim=(0.5,0.5))
    #s.annotate_streamlines('vx', 'vy',
    #                       density=0.8, color='k')
    return s

mydir = 'cylinder2d'
allfiles = np.sort(listdir(mydir))
pfiles = []
ds_list = []

for f in allfiles:
    if ('plt' == f[:3]):
        pfiles.append(f)
        ds_list.append(load_mks(f"{mydir}/{f}"))

minmax_list = []
for ds in ds_list:
    ds.force_periodicity()
    sv = plot_vorticity(ds)
    sv.save()
    minmax_list.append(ds.all_data().max("vorticity"))
    minmax_list.append(-ds.all_data().min("vorticity"))

print(max(minmax_list))
