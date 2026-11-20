import numpy as np
import matplotlib.pyplot as plt
import yt
import unyt
from os import listdir

from muphfasa_yt import load_mks


def plot_vel_stream(ds):

    ds.add_field(name=("boxlib", "Fluid"),
            function=lambda field, data: data['density']*unyt.m**3 /unyt.kg,
            take_log=False,
            display_name="Fluid",
            sampling_type="local")

    s = yt.SlicePlot(ds, 'z', 'Fluid')
    s.set_cmap('all', 'magma')
    s.annotate_streamlines('vx', 'vy',
                           density=0.8, color='k')

    return s

mydir = 'cylinder2d'
allfiles = np.sort(listdir(mydir))
pfiles = []
ds_list = []

for f in allfiles:
    if ('plt' == f[:3]):
        pfiles.append(f)
        ds_list.append(load_mks(f"{mydir}/{f}"))

for ds in ds_list:
    s = plot_vel_stream(ds)
    s.save()
