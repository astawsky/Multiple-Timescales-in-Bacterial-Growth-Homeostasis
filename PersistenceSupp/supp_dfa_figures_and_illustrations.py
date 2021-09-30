#!/usr/bin/env bash

from AnalysisCode.global_variables import (
    cmap, cgsc_6300_wang_exps, lexA3_wang_exps, mm_datasets, slash
)
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os


def main(**kwargs):


    se = pd.DataFrame()  # Where to save all the scaling exponents for the different experiment groups

    for name, group in zip(['Wang 2010, CGSC 6300', 'Wang 2010, lexA3', 'Susman and Kohram', 'Vashistha 2021', 'Tanouchi 25C', 'Tanouchi 27C', 'Tanouchi 37C'],
                           [cgsc_6300_wang_exps, lexA3_wang_exps, mm_datasets, ['Pooled_SM'], ['MC4100_25C (Tanouchi 2015)'], ['MC4100_27C (Tanouchi 2015)'], ['MC4100_37C (Tanouchi 2015)']]):

        for ds in group:
            # import the scaling exponents already calculated
            scaling_exponents = pd.read_csv(f'PersistenceMainText{slash}{ds}{slash}scaling_exponents{kwargs["noise_index"]}.csv')
            # scaling_exponents = pd.read_csv(os.path.dirname(os.path.abspath(__file__)) + f'{slash}{ds}{slash}scaling_exponents.csv')

            # put the experiment category
            scaling_exponents['experiment'] = name

            # Append it to the one contains all experiments
            se = se.append(scaling_exponents, ignore_index=True)

    condition = (se['dataset'] == 'Trace') & (se['kind'] == 'dfa (short)')  # Only plot the trace lineages
    # condition2 = (se['dataset'] == 'Shuffled') & (se['kind'] == 'dfa (short)')  # Only plot the artificial lineages

    scale = 1.5

    sns.set_context('paper', font_scale=scale)
    sns.set_style("ticks", {'axes.grid': True})
    fig, ax = plt.subplots(figsize=[6.5, 6.5], tight_layout=True)
    plt.axhline(0.5, ls='-', c='k')
    sns.pointplot(data=se[condition], x='variable', y='slope', hue='experiment', join=False, dodge=True, palette=cmap, ci="sd", linewidth=.1)
    plt.ylabel(r'$\gamma$')
    plt.xlabel('')
    plt.legend(title='')
    # ax.get_legend().remove()
    plt.ylim([0, 1.2])
    plt.savefig(f'PersistenceSupp{slash}dfa_figures_supp{kwargs["noise_index"]}.png', dpi=300)
    # plt.show()
    plt.close()
