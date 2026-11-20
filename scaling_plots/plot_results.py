import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import csv

import sys
#parse the results.txt and create csv
#!/usr/bin/env python3


input_file = "results.txt"
output_file = "results.csv"

pattern = re.compile(
    r'_(\d+)nodes_.*?MLUPS=([0-9.]+)\s+ms/step=([0-9.]+)'
)

with open(input_file) as fin, open(output_file, "w", newline="") as fout:
    writer = csv.writer(fout)
    writer.writerow(["nodes", "MLUPS", "ms_per_step"])

    for line in fin:
        m = pattern.search(line)
        if m:
            writer.writerow([
                int(m.group(1)),
                float(m.group(2)),
                float(m.group(3))
            ])

df = pd.read_csv("results.csv", header=0)#, names=['nodes', 'MLUPS', 'mstep/ms'])
print(df)
fig, ax = plt.subplots(1, 1)

ax.plot(df.nodes, df.MLUPS/1e3, 'o-')

lin_nodes = np.linspace(0, df.nodes.max(), 20)
lin_scale = (2*df.MLUPS.iloc[0] - df.MLUPS.iloc[0])/1e3/(2*df.nodes.iloc[0] - df.nodes.iloc[0]) *lin_nodes
ax.plot(lin_nodes, lin_scale, 'k--', label='Perfect strong scaling')
ax.fill_between(lin_nodes, lin_scale*0.8, lin_scale*0.9, color='k', alpha=0.2,label='80%-90% strong scaling')

ax.set_xlim(0)
ax.set_ylim(0)
ax.grid()
#ax.set_xticks(np.arange(0, df.nodes.max()*12, 12))
ax.set_xlabel("# of nodes (12 GPU Tiles per node)")
ax.set_ylabel("GLUPS\n(Billion lattice units per second)")
ax.legend()

ax.set_title(sys.argv[1])

fig.savefig("strong_scale.png", dpi=300)
