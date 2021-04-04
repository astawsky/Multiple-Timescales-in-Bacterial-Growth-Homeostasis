#!/usr/bin/env bash

from AnalysisCode.global_variables import (
    symbols, create_folder, phenotypic_variables, cmap, seaborn_preamble, sm_datasets, cgsc_6300_wang_exps, lexA3_wang_exps, mm_datasets, dataset_names, shuffle_lineage_generations, tanouchi_datasets
)
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.stats import linregress, pearsonr
from itertools import combinations
import os


for ds in ['Pooled_SM']:  # dataset_names:
    print(ds)
    # import the data
    pu = pd.read_csv(f'/Users/alestawsky/PycharmProjects/Thesis/Datasets/{ds}/ProcessedData/z_score_under_3/physical_units_without_outliers.csv')
    
    # stds, gentime = [], []
    #
    # for linid in pu.lineage_ID.unique():
    #     # print(linid)
    #     trace = pu[pu['lineage_ID'] == linid].copy()
    #
    #     # print(trace)
    #     # exit()
    #
    #     stds.append(trace['length_birth'].std() / (trace['length_birth'].mean() ** 2))
    #     gentime.append(trace['generationtime'].mean())
    #
    # sns.regplot(x=gentime, y=stds, label=np.round(pearsonr(gentime, stds)[0], 2))
    # plt.legend()
    # # plt.xlabel(r'$\tau / \langle \tau \rangle$')
    # # plt.ylabel(r'$\sigma(x_0)$')
    # plt.show()
    # plt.close()
    
    var1 = 'growth_rate'
    var2 = 'division_ratio'
    
    for tid in pu.trap_ID.unique():
        trace_a = pu[(pu['lineage_ID'] == tid) & (pu['trace'] == 'A')].sort_values('generation')[[var1, var2]].dropna().copy()
        trace_b = pu[(pu['lineage_ID'] == tid) & (pu['trace'] == 'B')].sort_values('generation').copy()
        
        [trace_a for gen in trace_a.generation.values]
