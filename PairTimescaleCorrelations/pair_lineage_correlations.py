#!/usr/bin/env bash

import pandas as pd
import numpy as np
from scipy.stats import linregress, pearsonr
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import seaborn as sns
from AnalysisCode.global_variables import (
    phenotypic_variables, create_folder, symbols, units, dataset_names, get_time_averages_df, sm_datasets, wang_datasets, cmap, cut_uneven_pairs, trace_center_a_dataframe
)
import os


def main(args):
    def put_all_graphs_into_a_big_grid(df, label, a_lins, b_lins, variables=phenotypic_variables, suffix=''):
        
        # Graphical Preferences
        sns.set_context('paper')
        sns.set_style("ticks", {'axes.grid': True})
        
        df[variables] = df[variables].where(
            np.abs(df[variables] - df[variables].mean()) < (3 * df[variables].std()),
            other=np.nan
        )  # Take out the outliers
        
        # More graphical preferences
        fig, axes = plt.subplots(nrows=len(variables), ncols=len(variables), figsize=[7, 7])
        
        # Go through all possible symmetric variable pairings, ie. combinations of 2
        for row, row_var in zip(range(axes.shape[0]), variables):
            for col, col_var in zip(range(axes.shape[1]), variables):
                ax = axes[row, col]  # define the axis we will be plotting on
                
                # So it looks presentable
                ax.xaxis.set_major_formatter(FormatStrFormatter('%.2g'))
                ax.yaxis.set_major_formatter(FormatStrFormatter('%.2g'))
                # ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
                # ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
                
                if col > row:  # Upper diagonal entries are ignored
                    ax.set_frame_on(False)
                    ax.set_xticks([])
                    ax.set_yticks([])
                else:  # Lower diagonal entries are plotted
                    
                    # We will group what we need and drop the NaN values
                    temp = pd.DataFrame()
                    temp[col_var + '_A'] = df[df['lineage_ID'].isin(a_lins)][col_var].values
                    temp[row_var + '_B'] = df[df['lineage_ID'].isin(b_lins)][row_var].values
                    temp = temp.dropna().copy()  # Take out the NaN values

                    # This is because the choice of A and B lineage in the trap is arbitrary, we symmetrize
                    temp1 = pd.DataFrame()
                    temp1[row_var + '_A'] = df[df['lineage_ID'].isin(a_lins)][row_var].values
                    temp1[col_var + '_B'] = df[df['lineage_ID'].isin(b_lins)][col_var].values
                    temp1 = temp1.dropna().copy()  # Take out the NaN values
                    
                    col_array = np.append(temp[col_var + '_A'].values, temp1[col_var + '_B'].values)  # The x-values symmetrized
                    row_array = np.append(temp[row_var + '_B'].values, temp1[row_var + '_A'].values)  # The y-values symmetrized
                    
                    pcorr = str(pearsonr(col_array, row_array)[0])[:4]  # The pearson correlation into a string with a precision of two decimal points
                    slope, intercept, _, _, std_err = linregress(col_array, row_array)  # The linear regression
                    
                    print(f'number of points in the scatter plot {row_var}^A {col_var}^B: {len(col_array)}')  # For the caption in the paper
                    
                    ax.scatter(col_array, row_array, color=cmap[0], marker='o', alpha=.3)  # plot the points
                    ax.plot(np.unique(col_array), [intercept + slope * vel for vel in np.unique(col_array)], c='k', ls='--')  # Plot the best fit line
                    ax.annotate(r'$\rho = $' + pcorr + '\n' + r'$\beta = {:.2}$'.format(np.round(slope, 2)), xy=(.5, .8), xycoords=ax.transAxes, fontsize=7, ha='center', va='bottom', color='red',
                                bbox=dict(boxstyle='round,pad=0.2', fc='yellow', alpha=0.7))  # Show the slope as well as the Pearson correlation coefficient
                
                if row == axes.shape[0] - 1:  # If we are at the last row, plot the x label latex variables
                    ax.set_xlabel(symbols[label][col_var] + r'$^{A}$')
                else:
                    ax.set_xlabel('')
                    ax.set_xticklabels([])
                
                if col == 0:  # If we are at the first column, plot the y label latex variables
                    ax.set_ylabel(symbols[label][row_var] + r'$^{B}$')
                else:
                    ax.set_ylabel('')
                    ax.set_yticklabels([])
        
        plt.tight_layout(pad=.5)
        plt.savefig('{}{}.png'.format(label, suffix), dpi=300)
        # plt.show()
        plt.close()
    
    physical_units = cut_uneven_pairs(pd.read_csv(args['pu']))  # import the pooled sister machine lineages and make them have the same generations in pair lineages
    physical_units = physical_units[physical_units['dataset'] == 'NL'].copy()  # Only use neighbor lineages for micro-environmental effects
    
    a_lin_ids = physical_units[(physical_units['trace'] == 'A')]['lineage_ID'].unique()  # All the lineage IDs that correspond to the "A" lineage of the trap
    b_lin_ids = physical_units[(physical_units['trace'] == 'B')]['lineage_ID'].unique()  # All the lineage IDs that correspond to the "B" lineage of the trap
    
    trace_centered = trace_center_a_dataframe(physical_units, False)  # Get the trace-centered values from the physical units
    
    time_averages = get_time_averages_df(physical_units, phenotypic_variables)[['lineage_ID', 'max_gen'] + phenotypic_variables].drop_duplicates().sort_values(['lineage_ID'])  # Get the time-averages
    time_averages = time_averages[time_averages['max_gen'] > 15]  # Only use lineages with more than 15 generations each
    
    # Plot all the variables -- Scatter Regression Plot
    print('pu')
    put_all_graphs_into_a_big_grid(physical_units, 'physical_units', a_lin_ids, b_lin_ids, variables=phenotypic_variables, suffix='')
    print('tc')
    put_all_graphs_into_a_big_grid(trace_centered, 'trace_centered', a_lin_ids, b_lin_ids, variables=phenotypic_variables, suffix='')
    print('unique ta')
    put_all_graphs_into_a_big_grid(time_averages, 'time_averages', a_lin_ids, b_lin_ids, variables=phenotypic_variables, suffix='')

    
data_origin = 'Pooled_SM'  # Only sister machine data has the format for this analysis

filepath = os.path.dirname(os.path.abspath(__file__))

processed_data = os.path.dirname(filepath) + '/Datasets/' + data_origin + '/ProcessedData/'

"""
data_origin ==> Name of the dataset we are analysing
raw_data ==> Where the folder containing the raw data for this dataset is
processed_data ==> The folder we will put the processed data in
"""
args = {
    'data_origin': data_origin,
    'MM': False if data_origin in sm_datasets else True,
    # Data singularities, long traces with significant filamentation, sudden drop-offs
    'Figures': filepath + '/Figures',
    'pu': processed_data + 'z_score_under_3/physical_units_without_outliers.csv' if data_origin in wang_datasets else processed_data + 'physical_units.csv',
    'tc': processed_data + 'z_score_under_3/trace_centered_without_outliers.csv' if data_origin in wang_datasets else processed_data + 'trace_centered.csv'
}

main(args)

print('*' * 200)
