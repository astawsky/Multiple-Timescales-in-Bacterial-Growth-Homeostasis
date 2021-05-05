#!/usr/bin/env bash

from AnalysisCode.global_variables import dataset_names, retrieve_dataframe_directory
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from mpl_toolkits.mplot3d import axes3d, Axes3D
    
    
def plot_manifold(ax):
    gt_domain = np.linspace(2.5, 0)
    gr_domain = np.linspace(0, 3)
    
    X, Z = np.meshgrid(
        gt_domain,
        gr_domain
    )

    Y = 1 / np.exp(X * Z)  # In the case of ln(f)
    
    ax.plot_wireframe(Z, X, Y, alpha=.5, linewidth=.3)  # plot_surface , alpha=.5
    
    
def plot_perfect_line(ax):
    # gr_domain = np.linspace(0, 3)
    gt_domain = np.linspace(0.235, 2.5)
    # gr_domain = np.linspace(df['growth_rate'].min(), df['growth_rate'].max())

    gr_domain = np.array([np.log(2) / z for z in gt_domain])

    y = np.array([1 / np.exp(gr * gt) for gr, gt in zip(gr_domain, gt_domain)])
    ax.plot3D(gr_domain, gt_domain, y, color='black', alpha=1, zorder=200)  # , zorder=10000
    
    
def script(variables, ax, points_are='pu'):  # can be ta
    ax.set_xlim([0, 3])
    ax.set_ylim([2.5, 0])
    # ax.set_zlim([1, 0])
    
    plot_manifold(ax)  # Line of r=1
    plot_perfect_line(ax)  # Line of f=1/2 and \phi=ln(2)
    
    # Pool all the different experiments together and categorize them by experiment
    for ds in dataset_names:
        if ds == 'Pooled_SM':  # For each experiment independently
            continue
        
        # For each type of values append
        if points_are == 'pu':
            # Import the dataframe
            df = pd.read_csv(retrieve_dataframe_directory(ds, 'pu', False))[variables]
            df = df.sample(n=min(500, len(df)), replace=False)
        else:
            # Import the dataframe
            df = pd.read_csv(retrieve_dataframe_directory(ds, 'ta', False))[variables].drop_duplicates()
            
        # Data for a three-dimensional line
        ax.scatter3D(df['growth_rate'].values, df['generationtime'].values, df['division_ratio'].values, alpha=.2)
    
    ax.set_ylabel(r'$\tau$')
    ax.set_zlabel(r'$f$')
    ax.set_xlabel(r'$\alpha$')
    

# The variables of the 3D manifold
variables = ['growth_rate', 'generationtime', 'division_ratio']

# stylistic parameters and actions
scale = 1
sns.set_context('paper', font_scale=scale)
sns.set_style("ticks", {'axes.grid': True})

fig = plt.figure(figsize=[6.5 * scale, 3 * scale])  # , tight_layout=True
ax = fig.add_subplot(1, 2, 1, projection='3d')
script(variables, points_are='pu', ax=ax)

# set up the axes for the second plot
ax = fig.add_subplot(1, 2, 2, projection='3d')

script(variables, points_are='ta', ax=ax)

plt.tight_layout()
# plt.savefig('figure_manifold.png', dpi=300)
plt.show()
plt.close()
