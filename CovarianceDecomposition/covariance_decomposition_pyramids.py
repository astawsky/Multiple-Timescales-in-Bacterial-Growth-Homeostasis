#!/usr/bin/env bash

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from AnalysisCode.global_variables import phenotypic_variables, symbols, get_time_averages_df, trace_center_a_dataframe


def pyramid_of_pairwise_covariances(pu, ta, tc, fig, axes, variables=phenotypic_variables, figurename='covariance decomposition, main variables', annot=True):
    # Normalize the trace-centered or time-averages correlation by the pooled ensemble standard deviations
    def normalize_correlation(phys, variables):
        # The pooled mean
        pooled_pu_mean = phys[variables].mean()

        pu_corr_df = phys.cov(ddof=1)
        ta_corr_df = pd.DataFrame(columns=variables, index=variables, dtype=float)
        tc_corr_df = pd.DataFrame(columns=variables, index=variables, dtype=float)
        
        memory = []

        for param_1 in variables:
            for param_2 in variables:
                if param_1 in memory:
                    continue
                else:
                    print(param_1, param_2)
                
                normalization = (pu[param_1].std(ddof=1) * pu[param_2].std(ddof=1))
                
                # The two components in the decomposition
                delta = []
                line = []
                
                
                for lin_id in phys.lineage_ID.unique():
                    l_cond = (phys['lineage_ID'] == lin_id)  # Condition that they are in the same experiment and lineage
                    lin = phys[l_cond].copy()[[param_1, param_2]].dropna().reset_index(drop=True)  # The masked dataframe that contains bacteria in the same lineage and experiment
                
                    # Add the components
                    line.append(len(lin) * ((lin[param_1].mean() - pooled_pu_mean[param_1]) * (lin[param_2].mean() - pooled_pu_mean[param_2])))
                    delta.append((lin[param_1] - lin[param_1].mean()) * (lin[param_2] - lin[param_2].mean()))
    
                # Get the variance of each one
                delta_cov = (np.sum(delta) / (len(phys[[param_1, param_2]].dropna()) - 1)) / normalization
                lin_cov = (np.sum(line) / (len(phys[[param_1, param_2]].dropna()) - 1)) / normalization
    
                print(delta_cov, lin_cov, (delta_cov + lin_cov), pu_corr_df.loc[param_1, param_2])
                
                # Make sure it is a true decomposition
                assert (np.abs(pu_corr_df.loc[param_1, param_2] - (delta_cov + lin_cov)) < .0000001).all()

                ta_corr_df.loc[param_1, param_2] = lin_cov
                tc_corr_df.loc[param_1, param_2] = delta_cov
                
            memory.append(param_1)
                
        return [pu_corr_df.rename(columns=symbols['physical_units'], index=symbols['physical_units']),
                ta_corr_df.rename(columns=symbols['time_averages'], index=symbols['time_averages']),
                tc_corr_df.rename(columns=symbols['trace_centered'], index=symbols['trace_centered'])]
    
    # # Normalize the trace-centered or time-averages correlation by the pooled ensemble standard deviations
    # def normalize_correlation(df, variables):
    #     cov = df.cov(ddof=0)
    #     # cov = df.cov()
    #     corr_df = pd.DataFrame(columns=variables, index=variables, dtype=float)
    #     for param_1 in variables:
    #         for param_2 in variables:
    #             normalization = (pu[param_1].std(ddof=0) * pu[param_2].std(ddof=0))
    #
    #             # # If the value lies in the noise range then make it zero
    #             # if -.1 < cov.loc[param_1, param_2] / normalization < .1:
    #             #     corr_df.loc[param_1, param_2] = float('nan')  # 0
    #             # else:
    #             #     corr_df.loc[param_1, param_2] = cov.loc[param_1, param_2] / normalization
    #             corr_df.loc[param_1, param_2] = cov.loc[param_1, param_2] / normalization
    #             # corr_df.loc[param_1, param_2] = np.round(cov.loc[param_1, param_2] / normalization, 2)
    #
    #     return corr_df
    
    # So that we get a lower diagonal matrix
    mask = np.ones_like(normalize_correlation(pu, variables))
    mask[np.tril_indices_from(mask)] = False
    
    # The bounds for the color in the heatmap. Given from the bounds of the pearson correlation
    vmax, vmin = 1, -1

    npu, nta, ntc = normalize_correlation(pu, variables)
    
    # # Different normalized correlations corresponding to the different pyramids. Change their column names to the latex version for plotting.
    # npu = normalize_correlation(pu, variables).rename(columns=symbols['physical_units'], index=symbols['physical_units'])
    # nta = normalize_correlation(ta, variables).rename(columns=symbols['time_averages'], index=symbols['time_averages'])
    # ntc = normalize_correlation(tc, variables).rename(columns=symbols['trace_centered'], index=symbols['trace_centered'])
    
    print(npu.loc[symbols['physical_units']['growth_rate'], symbols['physical_units']['generationtime']])
    print(nta.loc[symbols['time_averages']['growth_rate'], symbols['time_averages']['generationtime']] + ntc.loc[symbols['trace_centered']['growth_rate'], symbols['trace_centered']['generationtime']])
    print(nta.loc[symbols['time_averages']['growth_rate'], symbols['time_averages']['generationtime']])
    print(ntc.loc[symbols['trace_centered']['growth_rate'], symbols['trace_centered']['generationtime']])
    exit()
    
    # Plot the first pyramid
    sns.heatmap(npu, annot=annot, center=0, vmax=vmax,
                vmin=vmin, cbar=False, ax=axes[0], mask=mask, square=True)  # , fmt='.2f'
    axes[0].set_title('A', x=-.2, fontsize='xx-large')

    # Plot the second pyramid
    sns.heatmap(nta, annot=annot, center=0, vmax=vmax, vmin=vmin,
                cbar=False, ax=axes[1], mask=mask, square=True)
    axes[1].set_title('B', x=-.2, fontsize='xx-large')
    
    cbar_ax = fig.add_axes([.91, .1, .03, .8])

    # Plot the third pyramid
    sns.heatmap(ntc, annot=annot, center=0, vmax=vmax,
                vmin=vmin, cbar=True, ax=axes[2], mask=mask, cbar_kws={"orientation": "vertical"}, square=True, cbar_ax=cbar_ax)
    axes[2].set_title('C', x=-.2, fontsize='xx-large')
    
    fig.tight_layout(rect=[0, 0, .9, 1])
    # plt.savefig(f'{figurename}.png', dpi=300)
    plt.show()
    plt.close()

    
# The variables we want to plot
variables = ['div_and_fold', 'fold_growth', 'division_ratio', 'generationtime', 'length_birth', 'growth_rate']

# The pooled sister machine data in physical units, time-averages and trace-centered
pu = pd.read_csv(r'/Users/alestawsky/PycharmProjects/Thesis/Datasets/Pooled_SM/ProcessedData/z_score_under_3/physical_units_without_outliers.csv')
ta = get_time_averages_df(pu, phenotypic_variables)
tc = pd.read_csv(r'/Users/alestawsky/PycharmProjects/Thesis/Datasets/Pooled_SM/ProcessedData/z_score_under_3/trace_centered_without_outliers.csv')

# Graphical preferences
scale = 1.5
sns.set_context('paper', font_scale=.7 * scale)
sns.set_style("ticks", {'axes.grid': False})
fig, axes = plt.subplots(1, 3, figsize=[6.5 * scale, 2.1 * scale])

# Plot the pyramid covariances
pyramid_of_pairwise_covariances(pu, ta, tc, fig, axes, variables=variables, figurename='covariance decomposition', annot=True)

