import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sys import argv

def load_and_add_fields(filename):
    """
    load history file and add GLUPS per node field
    """
    df = pd.read_json(filename, lines=True)
    # cut on steps
    df['GLUPS_per_node'] = 12*df['mlups_per_rank']/1e3

    return df

def filter(df, np_filter):
    df = df.iloc[np_filter]
    # sort
    df.sort_values(['Q', 'nnodes'], inplace=True)
    return df

def gen_title(df):
    run_info = df.iloc[0]
    title = f"{run_info.test_problem}," 
    title += f" Re={run_info.re:0d}," 
    title += f" N={run_info.ncells.split(' ')[0]}^{run_info.dim}\n"
    title += f"tau={run_info.tau:0.2f}," 
    title += f" U_lat={run_info.u_lat:0.2f}"
    return title

def plot_scaling(df, title=None):
    fig, ax = plt.subplots(1,1, figsize=(8, 4.5))
    gb = df.groupby('lattice')

    for lat in gb.groups.keys():
        curr_df = gb.get_group(lat)
        ax.plot(curr_df['nnodes'], curr_df['GLUPS_per_node'], '-o', label=lat)

    ax.set_xlabel("# Nodes (w/ 12 rank/node)")
    ax.set_ylabel("GLUPS/Node")
    ax.legend()
    ax.grid()
    run_info = df.iloc[0]
    ax.set_title(title)
    fig.savefig("all_lat_scaling.png", dpi=300)

if __name__ == "__main__":
    filename = argv[1]
    df = load_and_add_fields(filename)
    
    # filter on n steps
    np_filter = np.where((df['steps'].to_numpy() > 30000) & 
                         (df['git_sha'] == 'a3d25a5095f0455ca749e5e12138e1caec6dc6ef'))
    df = filter(df, np_filter)

    # plot scaling
    title = gen_title(df)
    plot_scaling(df, title)