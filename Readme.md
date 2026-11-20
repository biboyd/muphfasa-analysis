# MuPhFASa Analysis
A collection of analysis and visualization scripts for analyzing muphfasa output.

Dependencies:

* python 3.12: numpy, matplotlib, pandas, yt, mpi4py

Install (editable) from the repo root so the scripts can import the shared `muphfasa_yt` helper
(`load_mks`, which loads plotfiles with MKS units) from anywhere:

```
pip install -e .
```

## Parallel volume rendering

`tgv_plots/plot_q_render.py` runs under MPI (needs `mpi4py`):

```
OMP_NUM_THREADS=1 mpirun -np 8 python tgv_plots/plot_q_render.py plotfiles/plt0002000
```

yt's ray sampler is already OpenMP-threaded, so on a single node a plain serial run uses every core
and MPI gives no extra speedup. Under `mpirun`, set `OMP_NUM_THREADS` to (cores / ranks) so ranks
don't oversubscribe the node. MPI is for multi-node runs or datasets too large for one rank's memory.

