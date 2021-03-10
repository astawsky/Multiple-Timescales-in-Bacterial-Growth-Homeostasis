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

pu = pd.read_csv(
    r'/Users/alestawsky/PycharmProjects/Multiple Timescales in Bacterial Growth Homeostasis/Datasets/MG1655_inLB_LongTraces/ProcessedData/physical_units.csv')
# tc = pd.read_csv(r'/Users/alestawsky/PycharmProjects/Multiple Timescales in Bacterial Growth Homeostasis/Datasets/Pooled_SM/ProcessedData/z_score_under_3/trace_centered_without_outliers.csv')

variables = ['growth_rate', 'generationtime', 'fold_growth', 'division_ratio']

normalize_and_cumsum = lambda x: ((x - x.mean()) / x.std()).cumsum()
normalize = lambda x: ((x - x.mean()) / x.std())

for lin_id in pu.lineage_ID.unique():
    print(lin_id)  # 9 is interesting
    
    lineage = pu[pu['lineage_ID'] == lin_id].copy()
    
    fig, axes = plt.subplots(2, 1, tight_layout=True)
    
    for ax in axes:
        ax.axhline(0, ls='-', c='black')
    
    for varr, ax in zip(variables, [axes[0], axes[0], axes[1], axes[1]]):
        print(varr)
        print(lineage['start_time'].values, normalize(lineage[varr]).values)
        ax.plot(lineage['start_time'].values, normalize(lineage[varr]).values, label=symbols['physical_units'][varr], marker='x')
    
    for ax in axes:
        ax.legend()
    plt.show()
    plt.close()
