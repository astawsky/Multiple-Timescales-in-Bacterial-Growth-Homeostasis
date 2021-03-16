#!/usr/bin/env bash

import pandas as pd
import numpy as np
from scipy.stats import linregress, pearsonr
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import seaborn as sns
from AnalysisCode.global_variables import (
    phenotypic_variables, create_folder, symbols, units, dataset_names, get_time_averages_df, sm_datasets, wang_datasets, cmap
)
import os


def main(args):
    def put_all_graphs_into_a_big_grid(df, label, variables=phenotypic_variables, suffix=''):
        
        create_folder(label)  # This is where we will save it
        
        # Graphical Preferences
        sns.set_context('paper')
        sns.set_style("ticks", {'axes.grid': True})
        
        latex_symbols = {variable: symbols[label][variable] for variable in variables}  # For the variabels we want to show
        unit_symbols = {variable: units[variable] if label != 'trace_centered' else '' for variable in variables}  # For the variabels we want to show
        
        df[variables] = df[variables].where(
            np.abs(df[variables] - df[variables].mean()) < (3 * df[variables].std()),
            other=np.nan
        )  # Take out the outliers
        
        df = df[variables].copy().rename(columns=latex_symbols)  # Replace the column names of the phenotypic variables with their latex counterparts
        
        if len(variables) == 2:  # In case we just want to see one scatterplot and not all of them
    
            # More graphical preferences
            sns.set_context('talk')
            fig, axes = plt.subplots(figsize=[5, 5], tight_layout=True)
            
            # Latex name of both variables
            sym1 = list(latex_symbols.values())[0]
            sym2 = list(latex_symbols.values())[1]
            
            relevant = df[[sym2, sym1]].dropna()  # Only keep these variables in the dataframe and delete any NaNs
            
            pcorr = str(pearsonr(relevant[sym2].values, relevant[sym1].values)[0])[:4]  # The pearson correlation into a string with a precision of two decimal points
            slope, intercept, r_value, _, std_err = linregress(relevant[sym2], relevant[sym1])  # The linear regression
            
            sns.regplot(data=relevant, x=sym2, y=sym1, line_kws={'color': 'black', 'ls': '--', 'lw': 2})  # Plot the regression and scatterplot
            
            axes.annotate(r'$\rho = $' + pcorr + ', ' + r'$\beta = {}$'.format(str(slope)[:4]), xy=(.5, .72), xycoords=axes.transAxes, fontsize=13, ha='center', va='bottom',
                          color='red',
                          bbox=dict(boxstyle='round,pad=0.2', fc='yellow', alpha=0.7))
            axes.set_ylabel(sym1 + ' ' + list(unit_symbols.values())[0])
            axes.set_xlabel(sym2 + ' ' + list(unit_symbols.values())[1])

            plt.tight_layout(pad=.5)
            plt.savefig('{}/{}_{}{}.png'.format(label, variables[0], variables[1], suffix), dpi=300)
            # plt.show()
            plt.close()
        else:  # If we want the matrix of scatterplots
            
            # More graphical preferences
            fig, axes = plt.subplots(nrows=len(variables) - 1, ncols=len(variables) - 1, figsize=[7, 7])
            
            # Go through all possible symmetric variable pairings, ie. combinations of 2
            for row, row_var in zip(range(axes.shape[0]), list(latex_symbols.values())[1:]):
                for col, col_var in zip(range(axes.shape[1]), list(latex_symbols.values())[:-1]):
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
                        # index_without_nans = [index for index in df.index if ~np.isnan(df[[col_var, row_var]].loc[index].values).any()]
                        index_without_nans = df[[col_var, row_var]].dropna().index  # Take out the NaN values
                        
                        col_array = df[col_var].loc[index_without_nans].values  # The x-values
                        row_array = df[row_var].loc[index_without_nans].values  # The y-values
                        
                        pcorr = str(pearsonr(col_array, row_array)[0])[:4]  # The pearson correlation into a string with a precision of two decimal points
                        slope, intercept, _, _, std_err = linregress(col_array, row_array)  # The linear regression
                        
                        ax.scatter(col_array, row_array, color=cmap[0], marker='o', alpha=.3)  # plot the points
                        ax.plot(np.unique(col_array), [intercept + slope * vel for vel in np.unique(col_array)], c='k', ls='--')  # Plot the best fit line
                        ax.annotate(r'$\rho = $' + pcorr + '\n' + r'$\beta = {:.2}$'.format(np.round(slope, 2)), xy=(.5, .8), xycoords=ax.transAxes, fontsize=7, ha='center', va='bottom', color='red',
                                    bbox=dict(boxstyle='round,pad=0.2', fc='yellow', alpha=0.7))  # Show the slope as well as the Pearson correlation coefficient
                    
                    if row == axes.shape[0] - 1:  # If we are at the last row, plot the x label latex variables
                        ax.set_xlabel(col_var)
                    else:
                        ax.set_xlabel('')
                        ax.set_xticklabels([])
                        
                    if col == 0:  # If we are at the first column, plot the y label latex variables
                        ax.set_ylabel(row_var)
                    else:
                        ax.set_ylabel('')
                        ax.set_yticklabels([])

            plt.tight_layout(pad=.5)
            plt.savefig('{}/{}{}.png'.format(label, args['data_origin'], suffix), dpi=300)
            # plt.show()
            plt.close()
            
    # import the labeled measured bacteria in trace-centered units
    physical_units = pd.read_csv(args['pu'])
    trace_centered = pd.read_csv(args['tc'])
    
    time_averages = get_time_averages_df(physical_units, phenotypic_variables)[['lineage_ID', 'max_gen'] + phenotypic_variables].drop_duplicates().sort_values(['lineage_ID'])  # Get the time-averages
    time_averages = time_averages[time_averages['max_gen'] > 15]  # Only use lineages with more than 15 generations each
    
    # Plot all the variables -- Scatter Regression Plot
    # print('pu')
    # put_all_graphs_into_a_big_grid(physical_units, 'physical_units', variables=phenotypic_variables, suffix='')
    print('tc')
    put_all_graphs_into_a_big_grid(trace_centered, 'trace_centered', variables=phenotypic_variables, suffix='')
    # print('unique ta')
    # put_all_graphs_into_a_big_grid(time_averages, 'time_averages', variables=phenotypic_variables, suffix='')


# Do all the Mother and Sister Machine data
for data_origin in dataset_names:
    print(data_origin)

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
