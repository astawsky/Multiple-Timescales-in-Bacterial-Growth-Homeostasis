#!/usr/bin/env bash

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from AnalysisCode.global_variables import phenotypic_variables, symbols, get_time_averages_df, slash


def pyramid_of_pairwise_covariances(pu, ta, tc, fig, axes, variables=phenotypic_variables, figurename='covariance decomposition, main variables', annot=True, **kwargs):
    # Normalize the trace-centered or time-averages correlation by the pooled ensemble standard deviations
    def normalize_correlation(phys, variables):
        # The pooled mean
        pooled_pu_mean = phys[variables].mean()

        pu_corr_df = pd.DataFrame(columns=variables, index=variables, dtype=float)
        ta_corr_df = pd.DataFrame(columns=variables, index=variables, dtype=float)
        tc_corr_df = pd.DataFrame(columns=variables, index=variables, dtype=float)
        
        memory = []

        for param_1 in variables:
            for param_2 in variables:
                if param_1 in memory:
                    continue
                # else:
                #     print(param_1, param_2)
                
                normalization = (phys[param_1].std(ddof=1) * phys[param_2].std(ddof=1))
                
                # The two components in the decomposition
                total = np.array([])
                delta = np.array([])
                line = np.array([])
                
                for lin_id in phys.lineage_ID.unique():
                    l_cond = (phys['lineage_ID'] == lin_id)  # Condition that they are in the same experiment and lineage
                    # The masked dataframe that contains bacteria in the same lineage and experiment
                    lin = phys[l_cond].copy()[[param_1, param_2]].dropna().reset_index(drop=True) if param_1 != param_2 else phys[l_cond].copy()[param_1].dropna().reset_index(drop=True)

                    # Add the components
                    if param_1 != param_2:
                        total = np.append(total, (lin[param_1] - pooled_pu_mean[param_1]).values * (lin[param_2] - pooled_pu_mean[param_2]))
                        line = np.append(line, len(lin) * ((lin[param_1].mean() - pooled_pu_mean[param_1]) * (lin[param_2].mean() - pooled_pu_mean[param_2])))
                        delta = np.append(delta, (lin[param_1] - lin[param_1].mean()).values * (lin[param_2] - lin[param_2].mean()).values)
                    else:
                        total = np.append(total, (lin - pooled_pu_mean[param_1]) ** 2)
                        line = np.append(line, np.array(len(lin) * ((lin.mean() - pooled_pu_mean[param_1]) ** 2)))
                        delta = np.append(delta, np.array((lin - lin.mean()) ** 2))
    
                # Get the variance of each one
                total_cov = (np.sum(total) / (len(phys[[param_1, param_2]].dropna()) - 1))
                delta_cov = (np.sum(delta) / (len(phys[[param_1, param_2]].dropna()) - 1))
                lin_cov = (np.sum(line) / (len(phys[[param_1, param_2]].dropna()) - 1))
                
                # Make sure it is a true decomposition
                assert (np.abs(total_cov / normalization - (delta_cov + lin_cov) / normalization) < .0000001).all()
                
                lin_cov = np.round(lin_cov / normalization, 2)
                delta_cov = np.round(delta_cov / normalization, 2)
                total_cov = np.round(total_cov / normalization, 2)
                
                if lin_cov + delta_cov != total_cov:
                    delta_cov += total_cov - (delta_cov + lin_cov)  # Change the short term correlations accordingly so that they all add up to one

                ta_corr_df.loc[param_1, param_2] = lin_cov
                tc_corr_df.loc[param_1, param_2] = delta_cov
                pu_corr_df.loc[param_1, param_2] = total_cov
                
            memory.append(param_1)
                
        return [pu_corr_df.rename(columns=symbols['physical_units'], index=symbols['physical_units']),
                ta_corr_df.rename(columns=symbols['time_averages'], index=symbols['time_averages']),
                tc_corr_df.rename(columns=symbols['trace_centered'], index=symbols['trace_centered'])]
    
    # The bounds for the color in the heatmap. Given from the bounds of the pearson correlation
    vmax, vmin = 1, -1

    npu, nta, ntc = normalize_correlation(pu, variables)
    
    # So that we get a lower diagonal matrix
    mask = np.ones_like(npu)
    mask[np.tril_indices_from(mask)] = False
    
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
    plt.savefig(f'CovarianceDecomposition{slash}{figurename}{kwargs["noise_index"]}.png', dpi=300)
    # plt.show()
    plt.close()


def main(**kwargs):
    # The variables we want to plot
    variables = ['div_and_fold', 'fold_growth', 'division_ratio', 'generationtime', 'length_birth', 'growth_rate']

    # The pooled sister machine data in physical units, time-averages and trace-centered
    pu = pd.read_csv(kwargs['without_outliers']('Pooled_SM') + '/physical_units_without_outliers.csv')
    ta = get_time_averages_df(pu, phenotypic_variables)
    tc = pd.read_csv(kwargs['without_outliers']('Pooled_SM') + '/trace_centered_without_outliers.csv')

    # Graphical preferences
    scale = 1.5
    sns.set_context('paper', font_scale=.7 * scale)
    sns.set_style("ticks", {'axes.grid': False})
    fig, axes = plt.subplots(1, 3, figsize=[6.5 * scale, 2.1 * scale])

    # Plot the pyramid covariances
    pyramid_of_pairwise_covariances(pu, ta, tc, fig, axes, variables=variables, figurename='covariance decomposition', annot=True, **kwargs)
