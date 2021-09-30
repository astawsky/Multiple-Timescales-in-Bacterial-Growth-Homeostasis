#!/usr/bin/env bash

import pandas as pd
import numpy as np
from scipy.stats import linregress, pearsonr
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import seaborn as sns
from AnalysisCode.global_variables import (
    phenotypic_variables, create_folder, symbols, units, dataset_names, get_time_averages_df, sm_datasets,
    wang_datasets, cmap, slash
)
import os


def primary(args, **kwargs):
    def put_all_graphs_into_a_big_grid(df, label, variables=phenotypic_variables, suffix='', **kwargs):
        
        create_folder(f'TimescaleCorrelations{slash}' + label)  # This is where we will save it
        
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
            plt.savefig(f'TimescaleCorrelations{slash}{label}{slash}{variables[0]}_{variables[1]}{suffix}{kwargs["noise_index"]}.png', dpi=300)
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
            plt.savefig(f'TimescaleCorrelations{slash}{label}{slash}{args["data_origin"]}{suffix}{kwargs["noise_index"]}.png', dpi=300)
            # plt.show()
            plt.close()
            
    # import the labeled measured bacteria in trace-centered units
    physical_units = pd.read_csv(args['pu'])
    trace_centered = pd.read_csv(args['tc'])
    
    time_averages = get_time_averages_df(physical_units, phenotypic_variables)[['lineage_ID', 'max_gen'] + phenotypic_variables].drop_duplicates().sort_values(['lineage_ID'])  # Get the time-averages
    time_averages = time_averages[time_averages['max_gen'] > 15]  # Only use lineages with more than 15 generations each
    
    # Plot all the variables -- Scatter Regression Plot
    # print('pu')
    # put_all_graphs_into_a_big_grid(physical_units, 'physical_units', variables=phenotypic_variables, suffix='', **kwargs)
    print('tc')
    put_all_graphs_into_a_big_grid(trace_centered, 'trace_centered', variables=phenotypic_variables, suffix='', **kwargs)
    print('unique ta')
    put_all_graphs_into_a_big_grid(time_averages, 'time_averages', variables=phenotypic_variables, suffix='', **kwargs)
    
    
def other(args):
    def get_df(df, label, ds, variables=phenotypic_variables):
        output_df = pd.DataFrame()
        
        memory = []
        
        # Go through all possible symmetric variable pairings, ie. combinations of 2
        for var1 in variables:
            
            memory.append(var1)
            
            for var2 in variables:
                if var2 in memory:
                    continue
                # index_without_nans = [index for index in df.index if ~np.isnan(df[[col_var, row_var]].loc[index].values).any()]
                index_without_nans = df[[var1, var2]].dropna().index  # Take out the NaN values
                
                col_array = df[var1].loc[index_without_nans].values  # The x-values
                row_array = df[var2].loc[index_without_nans].values  # The y-values

                # pcorr = str(pearsonr(col_array, row_array)[0])[:4]  # The pearson correlation into a string with a precision of two decimal points
                pcorr = np.round(pearsonr(col_array, row_array)[0], 2)  # The pearson correlation into a string with a precision of two decimal points
                slope, intercept, _, _, std_err = linregress(col_array, row_array)  # The linear regression
                
                output_df = output_df.append({
                    'var1': var1,
                    'var2': var2,
                    'timescale': label,
                    'correlation': pcorr,
                    'slope': slope,
                    'intercept': intercept,
                    'std_error': std_err,
                    'dataset': ds
                }, ignore_index=True)
        
        return output_df
    
    # import the labeled measured bacteria in trace-centered units
    physical_units = pd.read_csv(args['pu'])
    trace_centered = pd.read_csv(args['tc'])
    
    time_averages = get_time_averages_df(physical_units, phenotypic_variables)[['lineage_ID', 'max_gen'] + phenotypic_variables].drop_duplicates().sort_values(['lineage_ID'])  # Get the time-averages
    time_averages = time_averages[time_averages['max_gen'] > 15]  # Only use lineages with more than 15 generations each
    
    output_df = get_df(physical_units, 'physical units', args['data_origin'], phenotypic_variables)
    output_df = output_df.append(get_df(time_averages, 'long', args['data_origin'], phenotypic_variables), ignore_index=True).reset_index(drop=True)
    output_df = output_df.append(get_df(trace_centered, 'short', args['data_origin'], phenotypic_variables), ignore_index=True).reset_index(drop=True)
    
    return output_df


def initial_run(**kwargs):
    ds_df = pd.DataFrame()

    # Do all the Mother and Sister Machine data
    for data_origin in dataset_names:
        print(data_origin)

        # filepath = os.path.dirname(os.path.abspath(__file__))

        # processed_data = os.path.dirname(filepath) + '/Datasets/' + data_origin + '/ProcessedData/'

        """
        data_origin ==> Name of the dataset we are analysing
        raw_data ==> Where the folder containing the raw data for this dataset is
        processed_data ==> The folder we will put the processed data in
        """
        args = {
            'data_origin': data_origin,
            'MM': False if data_origin in sm_datasets else True,
            # Data singularities, long traces with significant filamentation, sudden drop-offs
            # 'Figures': 'Figures',
            'pu': kwargs['without_outliers'](data_origin) + 'physical_units_without_outliers.csv',
            # if data_origin in wang_datasets else processed_data + 'physical_units.csv'
            'tc': kwargs['without_outliers'](data_origin) + 'trace_centered_without_outliers.csv'
            # if data_origin in wang_datasets else processed_data + 'physical_units.csv'
        }

        primary(args, **kwargs)

        ds_df = ds_df.append(other(args), ignore_index=True)

        print('*' * 200)

    ds_df.to_csv(f'TimescaleCorrelations{slash}dataset_long_short{kwargs["noise_index"]}.csv', index=False)


def second_run(**kwargs):

    ds_df = pd.read_csv(f'TimescaleCorrelations{slash}dataset_long_short{kwargs["noise_index"]}.csv').reset_index(drop=True)

    # ds_df['combo'] = ds_df['var1'] + '_' + ds_df['var2']
    #
    # print(ds_df['combo'])
    # print(phenotypic_variables)
    # exit()

    for timescale in ['long', 'short', 'physical units']:
        df = ds_df[ds_df['timescale'] == timescale].copy()

        print(timescale)

        if timescale == 'long':
            latex_symbols = {variable: symbols['time_averages'][variable] for variable in
                             phenotypic_variables}  # For the variables we want to show
        elif timescale == 'short':
            latex_symbols = {variable: symbols['trace_centered'][variable] for variable in
                             phenotypic_variables}  # For the variables we want to show
        elif timescale == 'physical units':
            latex_symbols = {variable: symbols['physical_units'][variable] for variable in
                             phenotypic_variables}  # For the variables we want to show
        else:
            raise IOError('wrong')

        df['combo'] = [f"({latex_symbols[ds_df['var1'].iloc[ind]]}, {latex_symbols[ds_df['var2'].iloc[ind]]})" for ind
                       in np.arange(len(df))]

        growth_pairs = [
            ['fold_growth', 'division_ratio'], ['division_ratio', 'generationtime'], ['division_ratio', 'growth_rate'],
            ['generationtime', 'growth_rate']
        ]

        latex_growth_pairs = [f"({latex_symbols[gp[0]]}, {latex_symbols[gp[1]]})" for gp in growth_pairs]

        composite_pairs = [
            ['div_and_fold', 'fold_growth'], ['div_and_fold', 'division_ratio'], ['div_and_fold', 'growth_rate'],
            ['div_and_fold', 'generationtime'], ['fold_growth', 'generationtime'],
            ['fold_growth', 'growth_rate']
        ]

        latex_composite_pairs = [f"({latex_symbols[cp[0]]}, {latex_symbols[cp[1]]})" for cp in composite_pairs]

        latex_size_pairs = [
            f"({latex_symbols[variable]}, {latex_symbols['length_birth']})" if variable not in ['length_final',
                                                                                                'growth_rate']
            else f"({latex_symbols['length_birth']}, {latex_symbols[variable]})" for variable in phenotypic_variables]

        # Graphical Preferences
        sns.set_context('paper')
        sns.set_style("ticks", {'axes.grid': True})

        fig, axes = plt.subplots(nrows=3, ncols=1, tight_layout=True, figsize=[9, 6.5])

        # # So it looks presentable
        # ax.xaxis.set_major_formatter(FormatStrFormatter('%.2g'))
        # ax.yaxis.set_major_formatter(FormatStrFormatter('%.2g'))

        for row in np.arange(len(axes)):
            ax = axes[row]
            ax.axhline(0, c='k')
            ax.set_ylim([-1, 1])
            ax.set_yticks([-1, -.75, -.5, -.25, 0, .25, .5, .75, 1])
            if row == 0:
                sns.stripplot(data=df[df['combo'].isin(latex_size_pairs)], x='combo', y='correlation', hue='dataset',
                              ax=ax)
                ax.get_legend().remove()
                ax.set_ylabel('Size Correlations')
                # Put the legend out of the figure
                # ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)  # , fontsize='small'
            elif row == 1:
                sns.stripplot(data=df[df['combo'].isin(latex_growth_pairs)], x='combo', y='correlation', hue='dataset',
                              ax=ax)
                ax.get_legend().remove()
                ax.set_ylabel('Growth Correlations')
            else:
                sns.stripplot(data=df[df['combo'].isin(latex_composite_pairs)], x='combo', y='correlation',
                              hue='dataset', ax=ax)
                ax.get_legend().remove()
                ax.set_ylabel('Composite Correlations')

            ax.set_xlabel('')
        # # Put the legend out of the figure
        # axes[0].legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize='small')

        plt.savefig(f'TimescaleCorrelations{slash}{timescale}_all_datasets{kwargs["noise_index"]}.png', dpi=300)
        # plt.show()
        plt.close()


def main(**kwargs):
    initial_run(**kwargs)

    second_run(**kwargs)  # Plots it
