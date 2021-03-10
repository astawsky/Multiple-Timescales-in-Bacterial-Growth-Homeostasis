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


def average_time_series(linlin):
    # np.arange(0, len(ts), a)
    # print('ts', ts)
    
    to_output = {}
    for a in np.arange(1, 11):
        to_output.update({a: {}})
        
        to_output[a].update({'initial_size': np.mean(linlin['length_birth'].values[0])})
        # to_output[a].update({'initial_size': np.mean(linlin['length_birth'].values[:a])})
        
        for varvar in ['generationtime', 'growth_rate', 'division_ratio']:
            ts = linlin[varvar].values
            # print('a', a)
            
            if varvar == 'division_ratio':
                starts = np.arange(1, len(ts), a)
                ends = starts + a
            else:
                starts = np.arange(0, len(ts), a)
                ends = starts + a
            
            # print(starts, ends, sep='\n\n')
            
            assert len(starts) == len(ends)
            
            new_ts = np.array([np.mean(ts[start:end]) for start, end in zip(starts, ends)])
            
            # print('new-ts', new_ts)
            
            to_output[a].update({varvar: new_ts})
        
    return to_output


pu = pd.read_csv(
    r'/Users/alestawsky/PycharmProjects/Multiple Timescales in Bacterial Growth Homeostasis/Datasets/MG1655_inLB_LongTraces/ProcessedData/physical_units.csv')

variables = ['growth_rate', 'generationtime', 'fold_growth', 'division_ratio']

normalize_and_cumsum = lambda x: ((x - x.mean()) / x.std()).cumsum()
normalize = lambda x: ((x - x.mean()) / x.std())

for lin_id in pu.lineage_ID.unique():
    print('lin ID', lin_id)  # 9 is interesting
    
    lineage = pu[pu['lineage_ID'] == lin_id].copy()
    
    dict_avg = average_time_series(lineage)
    
    print(dict_avg)
    
    size_cv = []
    
    fig, axes = plt.subplots(2, 1, tight_layout=True)
    
    for a in dict_avg.keys():
        print(a)
        
        size_ts = []
        
        lb = dict_avg[a]['initial_size']
        size_ts.append(lb)
        for tau, alpha, f in zip(dict_avg[a]['generationtime'], dict_avg[a]['growth_rate'], dict_avg[a]['division_ratio']):
            lb = lb * np.exp(tau * alpha) * f
            size_ts.append(lb)
            
        size_cv.append(np.std(size_ts))
        
        axes[1].plot(np.arange(0, len(size_ts)) * a, size_ts, alpha=1/a, color='black', marker='x')
    
    axes[0].plot(list(dict_avg.keys()), size_cv)
    
    plt.show()
    plt.close()
    
    # exit()
    #
    # for ax in axes:
    #     ax.axhline(0, ls='-', c='black')
