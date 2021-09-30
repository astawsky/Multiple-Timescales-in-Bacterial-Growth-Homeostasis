#!/usr/bin/env bash

from AnalysisCode.global_variables import (
    symbols, units, create_folder, phenotypic_variables, shuffle_lineage_generations, cmap, seaborn_preamble, sm_datasets,
    wang_datasets, get_time_averages_df, trace_center_a_dataframe, cgsc_6300_wang_exps, shuffle_info, lexA3_wang_exps, mm_datasets, tanouchi_datasets, dataset_names, shuffle_lineage_generations,
    retrieve_dataframe_directory, slash
)
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.stats import linregress, pearsonr
from itertools import combinations
import os


"""
create a dataframe that contains all the intergenerational correlations between
"""


def create_the_dataframe(**kwargs):
    pu = pd.read_csv(kwargs['without_outliers']('Pooled_SM') + "physical_units_without_outliers.csv").sort_values(['lineage_ID', 'generation'])
    tc = pd.read_csv(kwargs['without_outliers']('Pooled_SM') + "trace_centered_without_outliers.csv").sort_values(['lineage_ID', 'generation'])

    corr_df = pd.DataFrame()
    
    for row, var_old in enumerate(phenotypic_variables):  # We have a combination of variables
        for col, var_young in enumerate(phenotypic_variables):  # The combination is one young and one old variable
            
            print(var_old, var_young, sep=' , ')
            
            for centered in [True, False]:  # For trace-centered values and not
                
                # Looking at trace centered or physical units and their categorical values of lineage and generation
                if centered:
                    relevant = pu[[var_old, var_young, 'lineage_ID', 'generation']].copy().dropna()
                else:
                    relevant = tc[[var_old, var_young, 'lineage_ID', 'generation']].copy().dropna()
                    
                relevant = relevant.loc[:, ~relevant.columns.duplicated()]  # No duplicate columns, just in case
                
                plotting_x = np.arange(11)  # The number of intergenerational distances between two bacteria
                
                for distance in plotting_x:  # Per inter-generational distance...
                    
                    old_indices = np.array([])  # Indices for the older bacteria
                    young_indices = np.array([])  # Indices for the younger bacteria
                    
                    # Collect the old and young samples
                    for lin_id in relevant.lineage_ID.unique():
                        generations_possible = relevant[(relevant['lineage_ID'] == lin_id)].copy()
                        
                        valuable_gens = np.array([g for g in generations_possible['generation'] if g + distance in generations_possible['generation'].values])
                        
                        old_indices = np.append(old_indices, generations_possible[generations_possible['generation'].isin(valuable_gens)].index)
                        young_indices = np.append(young_indices, generations_possible[generations_possible['generation'].isin(valuable_gens + distance)].index)
                        
                        # for gen in generations_possible:
                        #     # We did not measure or include the progeny
                        #     if gen+distance not in generations_possible:
                        #         # print('Was not possible.')
                        #         continue
                        #
                        #     if centered:  # Do we center the values or not?
                        #         old_value = pu[(pu['lineage_ID'] == lin_id) & (pu['generation'] == gen)][var_old].values - pu[(pu['lineage_ID'] == lin_id)][var_old].mean()
                        #         young_value = pu[(pu['lineage_ID'] == lin_id) & (pu['generation'] == gen + distance)][var_young].values - pu[(pu['lineage_ID'] == lin_id)][var_young].mean()
                        #     else:
                        #         old_value = pu[(pu['lineage_ID'] == lin_id) & (pu['generation'] == gen)][var_old].values
                        #         young_value = pu[(pu['lineage_ID'] == lin_id) & (pu['generation'] == gen + distance)][var_young].values
                        #
                        #     # Make sure neither value is a NaN
                        #     if not np.isnan(old_value) and not np.isnan(young_value):
                        #         old_array.append(old_value[0])
                        #         young_array.append(young_value[0])
                        #
                        #     # print(gen)
                        #     # print(old_array)
                        #     # print(young_array)
                        #     # print('---'*200)
                    
                    old_indices = old_indices.flatten()
                    young_indices = young_indices.flatten()

                    # Calculate the pearson correlation
                    r = 1 if (distance == 0) and (var_old == var_young) else pearsonr(relevant[var_old].loc[old_indices], relevant[var_young].loc[young_indices])[0]
                    
                    corr_df = corr_df.append({
                        'var_old': var_old,
                        'var_young': var_young,
                        'distance': distance,
                        'centered': centered,
                        'r': r
                    }, ignore_index=True)  # Add it to the dataframe we will save
    
    corr_df.to_csv(f'IntergenerationalCorrelations{slash}intergenerational_df{kwargs["noise_index"]}.csv', index=False)  # Save it so we do not do it again


def main(**kwargs):
    create_the_dataframe(**kwargs)  # This creation takes some time! We will use the whole pooled ensemble, have patience :)

    print('dataframe created')

    corr_df = pd.read_csv(f'IntergenerationalCorrelations{slash}intergenerational_df{kwargs["noise_index"]}.csv')

    scale = 1.5

    sns.set_context('paper', font_scale=scale/2)
    sns.set_style("ticks", {'axes.grid': True})

    fig, axes = plt.subplots(len(phenotypic_variables), len(phenotypic_variables), tight_layout=True, figsize=[6.5 * scale, 6.5 * scale])

    for row, var_old in enumerate(phenotypic_variables):  # We have a combination of variables
        for col, var_young in enumerate(phenotypic_variables):  # The combination is one young and one old variable
            for centered in [True, False]:  # For trace-centered values and not
                # The intergenerational correlations sorted by inter-gen distance
                relevant = corr_df[(corr_df['centered'] == centered) & (corr_df['var_old'] == var_old) & (corr_df['var_young'] == var_young)].copy().sort_values('distance')

                # Plot the correlation per inter-gen distance
                axes[row, col].plot(relevant['distance'].values, relevant['r'].values, color=cmap[1] if centered else cmap[0], marker='o')

                # Graphical preferences
                axes[row, col].axhline(0, ls='-', c='k')  # corr=0
                axes[row, col].set_ylim([-1, 1])  # via pearson correlation
                axes[row, col].set_xticks(np.arange(0, len(relevant), 2))
                axes[row, col].set_yticks([-1, -.75, -.5, -.25, 0, .25, .5, .75, 1])
                if col != 0:
                    axes[row, col].set_yticklabels('')
                else:
                    axes[row, col].set_yticklabels([-1, -.75, -.5, -.25, 0, .25, .5, .75, 1])
                if row != len(phenotypic_variables)-1:
                    axes[row, col].set_xticklabels('')
                else:
                    axes[row, col].set_xticklabels(np.arange(0, len(relevant), 2))

    # Graphical preferences
    for ax, variable in zip(axes[len(phenotypic_variables)-1, :], phenotypic_variables):
        ax.set_xlabel(symbols['physical_units'][variable]+r'$_{n+k}$' if variable not in ['length_birth', 'length_final'] else symbols['physical_units'][variable]+r'$_{, \, n+k}$', size='xx-large')
    for ax, variable in zip(axes[:, 0], phenotypic_variables):
        ax.set_ylabel(symbols['physical_units'][variable]+r'$_{n}$' if variable not in ['length_birth', 'length_final'] else symbols['physical_units'][variable]+r'$_{, \, n}$', size='xx-large')

    plt.savefig(f'IntergenerationalCorrelations{slash}intergenerational_figure{kwargs["noise_index"]}.png', dpi=300)
    # plt.show()
    plt.close()
