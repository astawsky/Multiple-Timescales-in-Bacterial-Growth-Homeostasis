#!/usr/bin/env bash

import pandas as pd
import numpy as np
import glob
import matplotlib.pyplot as plt
import os
from scipy.stats import pearsonr, linregress
from sklearn.linear_model import LinearRegression
from AnalysisCode.global_variables import (
    phenotypic_variables, create_folder, dataset_names, sm_datasets, wang_datasets, tanouchi_datasets,
    get_time_averages_df, check_the_division, symbols, cut_uneven_pairs, boot_pearson
)
import seaborn as sns


def get_linear_regression(lin, intput):
    pred_df = lin[intput + ['start_time']].dropna().copy()
    
    return LinearRegression().fit(pred_df[intput].values[:-1], pred_df[intput].values[1:])

#
# def scheme(lin, regression_vars, other_vars):
#     # main_vars = ['length_birth', 'generationtime', 'growth_rate']
#     # new_series = {v: lin[lin['generation'] == 0][v].values for v in main_vars}
#     new_df = pd.DataFrame(columns=phenotypic_variables)
#     new_df = new_df.append({v: lin[lin['generation'] == 1][v].values for v in phenotypic_variables}, ignore_index=True)
#     # size_initial = [lin[lin['generation'] == 0]['length_birth']]
#     # tau_initial = [lin[lin['generation'] == 0]['generationtime']]
#     # alpha_initial = [lin[lin['generation'] == 0]['growth_rate']]
#
#     reg = get_linear_regression(lin, regression_vars)
#
#     for gen in np.arange(1, len(lin), dtype=int):
#         print(gen)
#
#         predictions = reg.predict(new_df[regression_vars].values[-1, :])
#
#         print(predictions)
#
#         to_add = {v: l for v, l in zip(regression_vars, predictions)}
#         to_add.update({
#             'div_and_fold': to_add['division_ratio'] * np.exp(to_add['generationtime'] * to_add['generationtime'])
#             'fold_growth': to_add['generationtime'] * to_add['generationtime'],
#             'length_birth': to_add
#                       })
#
#         # new_df = new_df.append({: lin[lin['generation'] == 0][v].values for v,  in main_vars}, ignore_index=True)
#
#         # predictions = reg.predict(lin[regression_vars].dropna().values[:-1, :])
#
#         for count, v in enumerate(regression_vars):
#             reg.predict(lin[regression_vars].dropna().values[:-1, :])
#
#             new_series[v] = np.append(predictions[:, 0])
#
#         exit()
#
#         for v in main_vars:
#             print(v)
#
#             # np.append(new_series[v], np.array([])
#     pass


pu = pd.read_csv(
    r'/Users/alestawsky/PycharmProjects/Multiple Timescales in Bacterial Growth Homeostasis/Datasets/MG1655_inLB_LongTraces/ProcessedData/physical_units.csv')
# tc = pd.read_csv(r'/Users/alestawsky/PycharmProjects/Multiple Timescales in Bacterial Growth Homeostasis/Datasets/Pooled_SM/ProcessedData/z_score_under_3/trace_centered_without_outliers.csv')

for id in pu.lineage_ID.unique()[1:3]:  # 'division_ratio', 'growth_rate', 'div_and_fold', 'length_birth'
    for v in ['division_ratio', 'growth_rate']:
        
        ac = np.array([])
        
        lin = pu[pu['lineage_ID'] == id][v].dropna().copy()  # div_and_fold, growth_rate
        
        print(lin)
        
        if len(lin) < 20:
            continue
        else:
            print(id)
        
        for dist in np.arange(len(lin)-10):
            ac = np.append(ac, pearsonr(lin.values[dist:], lin.values[:-dist])[0] if dist != 0 else 1)
        # plt.plot(lin['div_and_fold'].values)
        plt.plot(ac, ls='-', label=v)

# plt.xlim([0, 50])
    plt.legend()
    plt.show()
    plt.close()
exit()

variables = ['growth_rate', 'generationtime', 'division_ratio']  # , 'fold_growth'

normalize_and_cumsum = lambda x: ((x - x.mean()) / x.std()).cumsum()
normalize = lambda x: ((x - x.mean()) / x.std())

for lin_id in pu.lineage_ID.unique():
    print(lin_id)  # 9 is interesting
    
    lineage = pu[pu['lineage_ID'] == lin_id].copy()

    scheme(lineage, variables, [])
    exit()
    
    if len(lineage) < 50:
        continue
    
    fig, ax = plt.subplots(1, 1, tight_layout=True)

    rel = lineage[['growth_rate', 'division_ratio', 'start_time', 'length_birth', 'generationtime']].dropna()

    # reg = LinearRegression().fit(rel[['growth_rate', 'generationtime']].values[:-1],
    #                              rel[['growth_rate', 'generationtime']].values[1:])
    
    reg = LinearRegression().fit(rel[['growth_rate', 'division_ratio']].values[:-1],
                                 rel[['growth_rate', 'division_ratio']].values[1:])

    # print(reg.intercept_)
    # print(reg.coef_)
    # print(reg.predict(rel[['growth_rate', 'division_ratio']].values[:-1]))
    
    # slope, intercept = linregress(lineage['growth_rate'].values[:-1], lineage['growth_rate'].values[1:])[:2]
    
    # ax.plot(lineage['start_time'].values, [-np.log(f) / a for f, a in zip(lineage['division_ratio'].values, lineage['growth_rate'].values)],
    #         label='app:'+symbols['physical_units']['generationtime'], marker='x')
    # ax.plot(lineage['start_time'].values, lineage['generationtime'].values, label=symbols['physical_units']['generationtime'], marker='x')
    
    new_alpha = np.append(np.array([rel['growth_rate'].values[0]]), reg.predict(rel[['growth_rate', 'division_ratio']].values[:-1])[:, 0])
    
    # new_alpha = np.append(np.array([rel['growth_rate'].values[0]]), reg.predict(rel[['growth_rate', 'generationtime']].values[:-1])[:, 0])
    # new_tau = np.append(np.array([rel['generationtime'].values[0]]), reg.predict(rel[['growth_rate', 'generationtime']].values[:-1])[:, 1])
    # new_f = np.array([np.exp(-(t * a)) for t, a in zip(new_tau, new_alpha)]) + np.random.normal(0, rel['division_ratio'].std(), size=len(new_tau))
    
    new_f = np.append(np.array([rel['division_ratio'].values[0]]), reg.predict(rel[['growth_rate', 'division_ratio']].values[:-1])[:, 1])
    new_tau = np.array([-np.log(f) / a for f, a in zip(new_f, new_alpha)]) + np.random.normal(0, rel['generationtime'].std(), size=len(new_f))
    # new_tau = [-np.log(f) / a for f, a in zip(rel['division_ratio'].values, rel['growth_rate'].values)]
    new_size = np.array([rel['length_birth'].values[0]])
    for a, f, t in zip(new_alpha, new_f, new_tau):
        # print(new_size[-1], f, np.exp(t * a))
        # exit()
        new_size = np.append(new_size, new_size[-1] * f * np.exp(t * a))
    
    ax.plot(new_size, label='new')
    ax.plot(rel['length_birth'].values, label='data')
    
    # ax.plot(rel['start_time'].values, rel['growth_rate'].values)
    # ax.plot(rel['start_time'].values, np.append(np.array([rel['growth_rate'].values[0]]), reg.predict(rel[['growth_rate', 'division_ratio']].values[:-1])[:, 0]))

    # ax.plot(rel['start_time'].values, rel['division_ratio'].values)
    # ax.plot(rel['start_time'].values, np.append(np.array([rel['division_ratio'].values[0]]), reg.predict(rel[['growth_rate', 'division_ratio']].values[:-1])[:, 1]))

    plt.legend()
    plt.show()
    plt.close()


