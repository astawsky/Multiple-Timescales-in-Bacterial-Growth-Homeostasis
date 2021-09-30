#!/usr/bin/env bash

from AnalysisCode.global_variables import (
    symbols, units, cmap, slash
)
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.stats import linregress


def main(**kwargs):

    # Use the extra long lineages from Susman et al. 2018
    pu = pd.read_csv(kwargs['without_outliers']('MG1655_inLB_LongTraces') + 'physical_units_without_outliers.csv').sort_values(['lineage_ID', 'generation'])
    # pu = pd.read_csv(retrieve_dataframe_directory('MG1655_inLB_LongTraces', 'pu', False)).sort_values(['lineage_ID', 'generation'])

    lin_id = 14  # The illustrative lineage we will use

    # Figure preferences
    scale = 1

    # stylistic reasons
    sns.set_context('paper', font_scale=scale)
    sns.set_style("ticks", {'axes.grid': True})

    fig, axes = plt.subplots(4, 1, tight_layout=True, figsize=[6.5 * scale, 4.2 * scale], sharex=True)

    for ax, variable in zip(axes, ['length_birth', 'growth_rate', 'generationtime', 'fold_growth']):
        to_plot = pu[pu['lineage_ID'] == lin_id][variable].values

        print(len(to_plot))

        if len(to_plot) < 50:
            continue

        slope, intercept = linregress(np.arange(len(pu[pu['lineage_ID'] == lin_id][variable].dropna().values)),
                                      pu[pu['lineage_ID'] == lin_id][variable].dropna().values)[:2]
        ax.set_ylabel(symbols['physical_units'][variable] + ' ' + units[variable])
        ax.axhline(np.nanmean(to_plot), ls='-', c='k')
        ax.set_ylim([np.nanmean(to_plot) + 1.5 * np.nanstd(to_plot), np.nanmean(to_plot) - 1.5 * np.nanstd(to_plot)])
        ax.plot(to_plot, marker='o', color=cmap[0], markersize=.8)
        ax.plot(intercept + (slope * np.arange(len(to_plot))), ls='--', color=cmap[1])
    axes[-1].set_xlabel('Generation')
    plt.savefig(f'TrendForDFA{slash}dfa_trend_illustration{kwargs["noise_index"]}.png', dpi=300)
    # plt.show()
    plt.close()

