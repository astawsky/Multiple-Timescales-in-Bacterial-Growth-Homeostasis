#!/usr/bin/env bash

import pandas as pd
import numpy as np
import glob
import matplotlib.pyplot as plt
import os
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression
from AnalysisCode.global_variables import (
    phenotypic_variables, create_folder, dataset_names, sm_datasets, wang_datasets, tanouchi_datasets,
    get_time_averages_df, check_the_division, symbols, cut_uneven_pairs, boot_pearson
)
import seaborn as sns


def script_to_see_cross_correlation():
    pu = pd.read_csv(
        r'/Users/alestawsky/PycharmProjects/Multiple Timescales in Bacterial Growth Homeostasis/Datasets/Pooled_SM/ProcessedData/physical_units.csv')
    pu = pu[pu['dataset'] == 'SL'].copy().reset_index(drop=True)
    # tc = pd.read_csv(r'/Users/alestawsky/PycharmProjects/Multiple Timescales in Bacterial Growth Homeostasis/Datasets/Pooled_SM/ProcessedData/z_score_under_3/trace_centered_without_outliers.csv')
    
    variables = ['growth_rate', 'generationtime', 'fold_growth', 'division_ratio']
    
    normalize_and_cumsum = lambda x: ((x - x.mean()) / x.std()).cumsum()
    
    for trap_id in pu.trap_ID.unique():
        
        # rel = pu[(pu['trap_ID'] == trap_id)].dropna()
        
        lin_a = pu[(pu['trap_ID'] == trap_id) & (pu['trace'] == 'A')].copy().dropna()
        lin_b = pu[(pu['trap_ID'] == trap_id) & (pu['trace'] == 'B')].copy().dropna()
        min_gen = min(len(lin_a), len(lin_b))
        
        if (min_gen < 30):
            continue
        else:
            print('Trap ID', trap_id)  # 9 is interesting
            lin_a = lin_a[lin_a['generation'] < min_gen].copy()
            lin_b = lin_b[lin_b['generation'] < min_gen].copy()
            print(lin_a, lin_b)
        
        fig, axes = plt.subplots(len(variables), 1, tight_layout=True)
        
        for varr, ax in zip(variables, axes):
            print('variable', varr)
            
            ax.axhline(0, ls='-', c='black')
            
            # for dist in np.arange(0, min_gen - 9):
            #     print(dist)
            #     if dist != 0:
            #         print('dist is not zreo!')
            #         print(lin_a[varr].values[:dist], lin_b[varr].values[dist:])
            #         print(dist, pearsonr(lin_a[varr].values[:-dist], lin_b[varr].values[dist:])[0])
            #     else:
            #         print('dist is zero')
            #         print(dist, pearsonr(lin_a[varr].values, lin_b[varr].values)[0])
            
            ax.plot([pearsonr(lin_a[varr].values[:-dist], lin_b[varr].values[dist:])[0] if dist != 0 else pearsonr(lin_a[varr].values, lin_b[varr].values)[0] for dist in np.arange(0, min_gen - 9)],
                    color='black', label=symbols['physical_units'][varr])
            ax.plot([pearsonr(lin_a[varr].values[dist:], lin_b[varr].values[:-dist])[0] if dist != 0 else pearsonr(lin_a[varr].values, lin_b[varr].values)[0] for dist in np.arange(0, min_gen - 9)],
                    color='grey', label=symbols['physical_units'][varr])
        #
        for ax in axes:
            ax.legend()
        plt.show()
        plt.close()
        
        
def script_to_see_over_all_gens():
    pu = pd.read_csv(
        r'/Users/alestawsky/PycharmProjects/Multiple Timescales in Bacterial Growth Homeostasis/Datasets/Pooled_SM/ProcessedData/physical_units.csv')
    pu = pu[pu['dataset'] == 'SL'].copy().reset_index(drop=True)
    # tc = pd.read_csv(r'/Users/alestawsky/PycharmProjects/Multiple Timescales in Bacterial Growth Homeostasis/Datasets/Pooled_SM/ProcessedData/z_score_under_3/trace_centered_without_outliers.csv')
    
    variables = ['growth_rate', 'generationtime', 'fold_growth', 'division_ratio']
    
    pu = cut_uneven_pairs(pu)
    
    pu_a = pu[pu['trace'] == 'A'].sort_values(['trap_ID', 'generation']).copy()
    pu_b = pu[pu['trace'] == 'B'].sort_values(['trap_ID', 'generation']).copy()
    
    fig, axes = plt.subplots(1, len(variables), figsize=[11, 6], tight_layout=True)
    
    for varr, ax in zip(variables, axes):
        x = np.append(pu_a[varr].dropna().values, pu_b[varr].dropna().values)
        y = np.append(pu_b[varr].dropna().values, pu_a[varr].dropna().values)
        ax.scatter(x, y, label=symbols['physical_units'][varr] + f': {np.round(pearsonr(x, y)[0], 2)}')
        ax.legend()
    
    plt.show()
    plt.close()
    
    
def script_for_gens_specific():
    pu = pd.read_csv(
        r'/Users/alestawsky/PycharmProjects/Multiple Timescales in Bacterial Growth Homeostasis/Datasets/Pooled_SM/ProcessedData/trace_centered.csv')
    pu = pu[pu['dataset'] == 'SL'].copy().reset_index(drop=True)
    # tc = pd.read_csv(r'/Users/alestawsky/PycharmProjects/Multiple Timescales in Bacterial Growth Homeostasis/Datasets/Pooled_SM/ProcessedData/z_score_under_3/trace_centered_without_outliers.csv')
    
    variables = ['growth_rate', 'generationtime', 'fold_growth', 'division_ratio']
    
    pu = cut_uneven_pairs(pu)

    fig, axes = plt.subplots(1, len(variables), figsize=[14, 5], tight_layout=True)

    for varr, ax in zip(variables, axes):
        print(varr)
        to_add = pd.DataFrame()
        for gen in np.arange(15):
            
            if (varr == 'division_ratio'):
                print('necessary continue')
                gen += 1
                print(gen)
            else:
                print(gen)
        
            pu_a = pu[(pu['trace'] == 'A') & (pu['generation'] == gen)].sort_values(['trap_ID']).copy()
            pu_b = pu[(pu['trace'] == 'B') & (pu['generation'] == gen)].sort_values(['trap_ID']).copy()
            
            # print(pu_a['generation'])
        
            x = np.append(pu_a[varr].dropna().values, pu_b[varr].dropna().values)
            y = np.append(pu_b[varr].dropna().values, pu_a[varr].dropna().values)
            
            for bs in boot_pearson(x, y, 250):
                to_add = to_add.append({'generation': gen, 'correlation': bs}, ignore_index=True)
            
        print(to_add)

        ax.axhline(0, color='black')
        sns.pointplot(x='generation', y='correlation', data=to_add, label=symbols['physical_units'][varr], ci='sd', ax=ax)
        handles, labels = ax.get_legend_handles_labels()
        plt.legend(handles, labels, title='')
        ax.set_xticklabels(np.arange(15, dtype=int))
    plt.show()
    plt.close()


script_for_gens_specific()
exit()

pu = pd.read_csv(
    r'/Users/alestawsky/PycharmProjects/Multiple Timescales in Bacterial Growth Homeostasis/Datasets/Pooled_SM/ProcessedData/physical_units.csv')
pu = pu[pu['dataset'] == 'SL'].copy().reset_index(drop=True)
# tc = pd.read_csv(r'/Users/alestawsky/PycharmProjects/Multiple Timescales in Bacterial Growth Homeostasis/Datasets/Pooled_SM/ProcessedData/z_score_under_3/trace_centered_without_outliers.csv')

variables = ['growth_rate', 'generationtime', 'fold_growth', 'division_ratio']

pu = cut_uneven_pairs(pu)

pu_a = pu[pu['trace'] == 'A'].sort_values(['trap_ID', 'generation']).copy()
pu_b = pu[pu['trace'] == 'B'].sort_values(['trap_ID', 'generation']).copy()

# print(pu_a['generation'])

fig, axes = plt.subplots(1, len(variables), figsize=[11, 6], tight_layout=True)

for varr, ax in zip(variables, axes):
    
    x = pu_a.dropna().append(pu_b.dropna(), ignore_index=True).reset_index(drop=True)
    print(x)
    exit()
    y = np.append(pu_b[varr].dropna().values, pu_a[varr].dropna().values)
    ax.plot(np.arange(9), y, label=symbols['physical_units'][varr] + f': {np.round(pearsonr(x, y)[0], 2)}')
    ax.legend()

# for varr, ax in zip(variables, axes):
#     x = np.append(pu_a[varr].dropna().values, pu_b[varr].dropna().values)
#     y = np.append(pu_b[varr].dropna().values, pu_a[varr].dropna().values)
#     ax.scatter(x, y, label=symbols['physical_units'][varr] + f': {np.round(pearsonr(x, y)[0], 2)}')
#     ax.legend()
    
plt.show()
plt.close()

exit()

normalize_and_cumsum = lambda x: ((x - x.mean()) / x.std()).cumsum()

for trap_id in pu.trap_ID.unique():
    
    # rel = pu[(pu['trap_ID'] == trap_id)].dropna()
    
    lin_a = pu[(pu['trap_ID'] == trap_id) & (pu['trace'] == 'A')].copy().dropna()
    lin_b = pu[(pu['trap_ID'] == trap_id) & (pu['trace'] == 'B')].copy().dropna()
    min_gen = min(len(lin_a), len(lin_b))
    
    if (min_gen < 30):
        continue
    else:
        print('Trap ID', trap_id)  # 9 is interesting
        lin_a = lin_a[lin_a['generation'] < min_gen].copy()
        lin_b = lin_b[lin_b['generation'] < min_gen].copy()
        print(lin_a, lin_b)
        
    fig, axes = plt.subplots(1, len(variables), tight_layout=True)

    for varr, ax in zip(variables, axes):
        print('variable', varr)
        
        ax.axhline(0, ls='-', c='black')

        # for dist in np.arange(0, min_gen - 9):
        #     print(dist)
        #     if dist != 0:
        #         print('dist is not zreo!')
        #         print(lin_a[varr].values[:dist], lin_b[varr].values[dist:])
        #         print(dist, pearsonr(lin_a[varr].values[:-dist], lin_b[varr].values[dist:])[0])
        #     else:
        #         print('dist is zero')
        #         print(dist, pearsonr(lin_a[varr].values, lin_b[varr].values)[0])
        
        ax.scatter(lin)
        
        ax.plot([pearsonr(lin_a[varr].values[:-dist], lin_b[varr].values[dist:])[0] if dist != 0 else pearsonr(lin_a[varr].values, lin_b[varr].values)[0] for dist in np.arange(0, min_gen - 9)], color='black', label=symbols['physical_units'][varr])
        ax.plot([pearsonr(lin_a[varr].values[dist:], lin_b[varr].values[:-dist])[0] if dist != 0 else pearsonr(lin_a[varr].values, lin_b[varr].values)[0] for dist in np.arange(0, min_gen - 9)], color='grey', label=symbols['physical_units'][varr])
    #
    for ax in axes:
        ax.legend()
    plt.show()
    plt.close()
