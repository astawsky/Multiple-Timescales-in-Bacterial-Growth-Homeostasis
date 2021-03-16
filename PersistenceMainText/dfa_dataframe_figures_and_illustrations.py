#!/usr/bin/env bash

from AnalysisCode.global_variables import (
    symbols, create_folder, phenotypic_variables, cmap, seaborn_preamble, sm_datasets, cgsc_6300_wang_exps, lexA3_wang_exps, mm_datasets, dataset_names, shuffle_lineage_generations
)
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.stats import linregress
from itertools import combinations
import os


def dfa_short(lineage, min_ws, max_ws=None, window_size_steps=5, steps_between_windows=2):
    """ This is following the notation and equations on Wikipedia """
    
    # How many cycles/generations in the lineages
    total_length = len(lineage)
    
    # Maximum window size possible
    if not max_ws:
        max_ws = total_length / 2
    
    # What are the window sizes? In order for it not to take a long time we skip two
    window_sizes = np.arange(min_ws, max_ws, window_size_steps, dtype=int)
    
    # Convert it into a random walk of some persistence
    total_walk = ((lineage - lineage.mean()) / lineage.std()).cumsum()
    
    # This is where we keep the rescaled ranges for all partial time-series
    mse_array = []
    
    for ws in window_sizes:
        # The windowed/partial time-series
        partial_time_series = [total_walk.iloc[starting_point:starting_point + ws].values for starting_point in np.arange(0, total_length, steps_between_windows, dtype=int)[:-1]]
        
        # Where we will keep the mean squared error for each window we have
        window_fs = []
        
        # Go through all the partial time-series
        for ts in partial_time_series:
            
            # Get the linear regression parameters
            slope, intercept, _, _, _ = linregress(np.arange(len(ts)), ts)
            
            # Linear approximation
            line = [intercept + slope * dom for dom in np.arange(len(ts))]
            
            # Necessary
            assert len(ts) == len(line)
            
            # Mean Squared Deviation of the linear approximation
            f = np.sqrt(((ts - line) ** 2).sum() / len(ts))
            
            # Add it to the Mean Squared Error of the window size ws
            window_fs.append(f)
        
        # Add it to the Mean Squared Error of the window size ws
        mse_array.append(np.mean(window_fs))
    
    # So that we can regress successfully
    assert len(window_sizes) == len(mse_array)
    
    # if np.isnan(np.log(window_sizes)).any():
    #     print(window_sizes)
    #     print(np.log(window_sizes))
    #     exit()
    #
    # if np.isnan(np.log(mse_array)).any():
    #     print(mse_array)
    #     print(np.log(mse_array))
    #     exit()
    
    # Get the linear regression parameters
    slope, intercept, _, _, std_err = linregress(np.log(window_sizes), np.log(mse_array))
    
    # # See what it looks like
    # plt.scatter(window_sizes, mse_array)
    # plt.plot(window_sizes, [np.exp(intercept) * (l ** slope) for l in window_sizes], label='{:.2}'.format(slope))
    # plt.yscale('log')
    # plt.xscale('log')
    # plt.tight_layout()
    # plt.legend()
    # plt.show()
    # plt.close()
    # # exit()
    
    return [window_sizes, mse_array, slope, intercept, std_err]


def plot_histograms_of_scaling_exponents(figure_folder, histogram_of_regression):
    # Histogram of H of every kind
    for kind in histogram_of_regression.kind.unique():
        seaborn_preamble()
        fig, ax = plt.subplots(tight_layout=True, figsize=[5, 5])
        plt.axhline(.5, ls='-', c='k', linewidth=1, zorder=1)
        
        sns.pointplot(data=histogram_of_regression[histogram_of_regression['kind'] == kind], x='variable', y='slope', hue='dataset', order=[symbols['physical_units'][v] for v in phenotypic_variables],
                      join=False, dodge=True, capsize=.5)  # , capthick=1
        # sns.boxplot(data=histogram_of_regression[histogram_of_regression['kind'] == kind], x='variable', y='slope', hue='dataset', order=[symbols['physical_units'][v] for v in phenotypic_variables],
        #             showfliers=False)
        plt.legend(title='')
        # plt.title(kind)
        plt.ylabel('')
        plt.xlabel('')
        plt.savefig('{}/{}.png'.format(figure_folder, kind), dpi=300)
        # plt.show()
        plt.close()


def recreate_loglog_plots(loglog_plot_recreation, figure_folder, individuals=False):
    # Create the folders where we will save the figures
    if individuals:
        create_folder('{}/Individual Scalings'.format(figure_folder))
        for kind in loglog_plot_recreation.kind.unique():
            create_folder('{}/Individual Scalings/{}'.format(figure_folder, kind))
            for dataset in loglog_plot_recreation.dataset.unique():
                create_folder('{}/Individual Scalings/{}/{}'.format(figure_folder, kind, dataset))
    else:
        for kind in loglog_plot_recreation.kind.unique():
            create_folder('{}/{}'.format(figure_folder, kind))
    
    # For every kind of loglog plot
    for kind in loglog_plot_recreation.kind.unique():
        print(kind)
        for variable in loglog_plot_recreation.variable.unique():
            if individuals:
                for dataset in loglog_plot_recreation.dataset.unique():
                    for lin_id in loglog_plot_recreation.lineage_ID.unique():
                        relevant = loglog_plot_recreation[
                            (loglog_plot_recreation['kind'] == kind) & (loglog_plot_recreation['lineage_ID'] == lin_id) & (loglog_plot_recreation['variable'] == variable) & (
                                    loglog_plot_recreation['dataset'] == dataset)].copy()
                        
                        fig, ax = plt.subplots(tight_layout=True)
                        plt.scatter(relevant['window_sizes'].values, relevant['y_to_regress'].values)
                        plt.plot(relevant['window_sizes'].values, relevant['best_fit_line'].values,
                                 label=r'log(y)$= {:.2} + {:.2} \,$log(x)'.format(relevant.intercept.unique()[0], relevant.slope.unique()[0]))
                        plt.ylabel('Variation')
                        plt.xlabel('Window Size')
                        plt.yscale('log')
                        plt.xscale('log')
                        plt.legend(title='')
                        plt.tight_layout()
                        plt.show()
                        # plt.savefig('{}/Individual Scalings/{}/{}/{}.png'.format(figure_folder, kind, dataset, lin_id), dpi=300)
                        plt.close()
                        exit()
            else:
                relevant = loglog_plot_recreation[
                    (loglog_plot_recreation['kind'] == kind) & (loglog_plot_recreation['variable'] == variable) & (loglog_plot_recreation['dataset'].isin(['Trace', 'Shuffled']))].copy()
                
                seaborn_preamble()
                
                set_trace_legend, set_shuffled_legend = False, False
                
                wss_array = {'Trace': np.array([]), 'Shuffled': np.array([])}
                ytr_array = wss_array.copy()
                
                # For each lineage separately
                for ind in relevant.index:
                    # Convert the window sizes and mean squared error per windows size from the dataframe from a string to an array of integers or floats
                    wss = relevant['window_sizes'].loc[ind].strip('][').split(' ')
                    wss = [int(r.split('\n')[0]) for r in wss if r != '']
                    ytr = relevant['y_to_regress'].loc[ind].strip('][').split(', ')
                    ytr = [float(r) for r in ytr if r != '']
                    
                    # If it is a trace lineage plot it and include it in the legend
                    if relevant.loc[ind]['dataset'] == 'Trace':
                        # The constant shuffled color
                        color = cmap[0]
                        
                        # Add the analysis curve to the array of its dataset for the population regression
                        wss_array['Trace'] = np.append(wss_array['Trace'], wss)
                        ytr_array['Trace'] = np.append(ytr_array['Trace'], ytr)
                        
                        # Include it in the legend but not more than once
                        if set_trace_legend:
                            plt.plot(wss, ytr, color=color, marker='x')
                        else:
                            plt.plot(wss, ytr, color=color, marker='x', label='Trace')
                            set_trace_legend = True
                    # If it is a shuffled lineage plot it and include it in the legend
                    elif relevant.loc[ind]['dataset'] == 'Shuffled':
                        # The constant shuffled color
                        color = cmap[1]
                        
                        # Add the analysis curve to the array of its dataset for the population regression
                        wss_array['Shuffled'] = np.append(wss_array['Shuffled'], wss)
                        ytr_array['Shuffled'] = np.append(ytr_array['Shuffled'], ytr)
                        
                        # Include it in the legend but not more than once
                        if set_shuffled_legend:
                            plt.plot(wss, ytr, color=color, marker='x')
                            pass
                        else:
                            plt.plot(wss, ytr, color=color, marker='x', label='Shuffled')
                            set_shuffled_legend = True
                    # We do not want to see the white noise
                    else:
                        continue
                
                # Get the linear regression of all the trajectories pooled for each dataset
                slope_trace, intercept_trace, _, _, std_err_trace = linregress(np.log(np.array(wss_array['Trace']).flatten()), np.log(np.array(ytr_array['Trace']).flatten()))
                slope_art, intercept_art, _, _, std_err_art = linregress(np.log(np.array(wss_array['Shuffled']).flatten()), np.log(np.array(ytr_array['Shuffled']).flatten()))
                
                # Plot the best fit line and its parameters
                plt.plot(np.unique(wss_array['Trace']), [np.exp(intercept_trace) * (l ** slope_trace) for l in np.unique(wss_array['Trace'])], ls='--', color='blue', linewidth=3,
                         label=r'$' + str(np.round(intercept_trace, 2)) + r'n^{' + str(np.round(slope_trace, 2)) + r'\pm' + str(np.round(std_err_trace, 3)) + r'}$')
                plt.plot(np.unique(wss_array['Shuffled']), [np.exp(intercept_art) * (l ** slope_art) for l in np.unique(wss_array['Shuffled'])], ls='--', color='red', linewidth=3,
                         label=r'$' + str(np.round(intercept_art, 2)) + r'n^{' + str(np.round(slope_art, 2)) + r'\pm' + str(np.round(std_err_art, 3)) + r'}$')
                
                plt.title(variable)
                plt.ylabel(r'$F(n)$')
                plt.xlabel('n')
                plt.yscale('log')
                plt.xscale('log')
                plt.legend(title='')
                plt.tight_layout()
                # plt.show()
                plt.savefig('{}/{}/{}.png'.format(figure_folder, kind, variable), dpi=300)
                plt.close()
    # plt.scatter(window_sizes, y_to_regress, color='blue')


def get_the_dfa_dataframe(args):
    # Hyper parameters
    min_ws = 5
    max_ws = None
    window_size_steps = 4  # 2
    steps_between_windows = 3
    
    physical_units = pd.read_csv(args['pu'])  # Phenotypic variables in physical units
    shuffled_lineages = shuffle_lineage_generations(physical_units, mm=True if args['data_origin'] not in sm_datasets else False)  # The generations shuffled
    
    # This is to see the differences between the slope or intercepts of different lineages from different datasets and using different scaling analyses
    histogram_of_regression = pd.DataFrame(columns=['dataset', 'lineage_ID', 'slope', 'intercept', 'std_err', 'variable', 'kind'])
    
    # This is to recreate the loglog plots whose slopes give us the scalings of lineages based on some scaling analysis
    loglog_plot_recreation = pd.DataFrame(columns=['dataset', 'lineage_ID', 'window_sizes', 'y_to_regress', 'best_fit_line', 'variable', 'kind'])
    
    for variable in phenotypic_variables:
        
        print(variable)
        
        for lin_id in physical_units.lineage_ID.unique():
            
            # The two types of lineages with this ID; they share the same distribution but in different generational order
            trace = physical_units[physical_units['lineage_ID'] == lin_id][variable].dropna()
            shuffled = shuffled_lineages[shuffled_lineages['lineage_ID'] == lin_id][variable].dropna()
            # white_noise = pd.Series(np.random.normal(0, 1, len(trace)))
            
            # If we have enough points to get a slope using this lineage
            if len(np.arange(min_ws, len(trace) / 2, window_size_steps, dtype=int)) < 3:
                continue
            
            # What type of lineages do we want to look at?
            types_of_lineages, names_types_of_lineages = [trace, shuffled], ["Trace", "Shuffled"]
            
            for lineage, dataset in zip(types_of_lineages, names_types_of_lineages):
                for scaling_analysis, kind in zip([dfa_short], ['dfa (short)']):
                    
                    # Calculate the scaling of this "lineage" using this "kind" of analysis
                    window_sizes, y_to_regress, slope, intercept, std_err = scaling_analysis(lineage, min_ws=min_ws, max_ws=max_ws, window_size_steps=window_size_steps,
                                                                                             steps_between_windows=steps_between_windows)
                    
                    # Append the regression statistics to a dataframe
                    histogram_of_regression = histogram_of_regression.append(
                        {'dataset': dataset, 'lineage_ID': lin_id, 'slope': slope, 'intercept': intercept, 'std_err': std_err, 'variable': symbols['physical_units'][variable], 'kind': kind},
                        ignore_index=True)
                    
                    # Append the loglog regression graph to a dataframe
                    loglog_plot_recreation = loglog_plot_recreation.append(
                        {
                            'dataset': dataset, 'lineage_ID': lin_id, 'window_sizes': window_sizes, 'y_to_regress': y_to_regress,
                            'best_fit_line': np.array([np.exp(intercept) * (ws ** slope) for ws in window_sizes]),
                            'variable': symbols['physical_units'][variable], 'kind': kind
                        }, ignore_index=True)
                    
                    # Make sure there are no NaNs
                    if histogram_of_regression.isnull().values.any():
                        print(kind)
                        print({'dataset': dataset, 'lineage_ID': lin_id, 'slope': slope, 'intercept': intercept, 'std_err': std_err, 'variable': symbols['physical_units'][variable], 'kind': kind})
                        raise IOError('histogram_of_regression.isnull().values.any()')
                    if loglog_plot_recreation.isnull().values.any():
                        print(kind)
                        print({
                            'dataset': dataset, 'lineage_ID': lin_id, 'window_sizes': window_sizes, 'y_to_regress': y_to_regress,
                            'best_fit_line': np.array([np.exp(intercept) * (ws ** slope) for ws in window_sizes]),
                            'variable': symbols['physical_units'][variable], 'kind': kind
                        })
                        raise IOError('loglog_plot_recreation.isnull().values.any()')
    
    # Save them
    histogram_of_regression.to_csv('{}/scaling_exponents.csv'.format(args['Scaling_Exponents']), index=False)
    loglog_plot_recreation.to_csv('{}/loglog_scaling_recreation.csv'.format(args['LogLog_Recreation']), index=False)


def create_dfa_dataframe():
    
    # experimental_groups = {
    #     # '(Tanouchi 2015) 37C': pool_experiments(['MC4100_37C (Tanouchi 2015)'], '(Tanouchi 2015) 37C', outliers=True),
    #     # '(Tanouchi 2015) 27C': pool_experiments(['MC4100_27C (Tanouchi 2015)'], '(Tanouchi 2015) 27C', outliers=True),
    #     # '(Tanouchi 2015) 25C': pool_experiments(['MC4100_25C (Tanouchi 2015)'], '(Tanouchi 2015) 25C', outliers=True),
    #     '(Wang et al 2010) CGSC_6300': pool_experiments(cgsc_6300_wang_exps, '(Wang et al 2010) CGSC_6300', outliers=True),
    #     '(Wang et al 2010) lexA3': pool_experiments(lexA3_wang_exps, '(Wang et al 2010) lexA3', outliers=True),
    #     '(Kohram et al 2020, Susman et al 2018)': pool_experiments(mm_datasets, '(Kohram et al 2020, Susman et al 2018)', outliers=True),
    #     '(Vashistha et al 2021)': pool_experiments(sm_datasets, '(Vashistha et al 2021)', outliers=True)
    # }
    #
    # for data_origin, pooled in experimental_groups.items():
    #     print(data_origin, len(pooled.lineage_ID.unique()), len(pooled.experiment_name.unique()), sep='\n\n')
    
    # Do all the Mother and Sister Machine data
    for data_origin in dataset_names:
        
        print(data_origin)
        
        current_dir = os.path.dirname(os.path.abspath(__file__))
        
        create_folder(current_dir + '/' + data_origin)
        
        processed_data = os.path.dirname(current_dir) + '/Datasets/' + data_origin + '/ProcessedData/'
        
        """
        data_origin ==> Name of the dataset we are analysing
        raw_data ==> Where the folder containing the raw data for this dataset is
        processed_data ==> The folder we will put the processed data in
        """
        
        args = {
            'data_origin': data_origin,
            # Data singularities, long traces with significant filamentation, sudden drop-offs
            'Scaling_Exponents': current_dir + '/' + data_origin,
            'LogLog_Recreation': current_dir + '/' + data_origin,
            # 'dataframes': current_dir + '/Dataframes/' + data_origin,
            'pu': processed_data + '/physical_units.csv'
        }
        
        get_the_dfa_dataframe(args)
        
        df = pd.read_csv('{}/scaling_exponents.csv'.format(args['Scaling_Exponents']))
        
        plot_histograms_of_scaling_exponents(args['Scaling_Exponents'], df)
        
        # df = pd.read_csv('{}/loglog_scaling_recreation.csv'.format(args['LogLog_Recreation']))
        #
        # recreate_loglog_plots(df, args['LogLog_Recreation'], individuals=False)

    
    
def plot_dfa_cumsum_illustration(ax):
    c = cmap[3]  # color
    
    # Physical Units dataframe
    pu = pd.read_csv(f'/Users/alestawsky/PycharmProjects/Thesis/Datasets/Pooled_SM/ProcessedData/z_score_under_3/physical_units_without_outliers.csv')
    
    lin_id = 15  # ID of illustrative lineage
    rel = pu[pu['lineage_ID'] == lin_id].copy().sort_values('generation')  # lineage in generational order
    
    r = rel['length_birth']  # inital size
    r = (r - r.mean()).cumsum().values  # normalized cumulative sum ("walk")
    
    ws = 5  # window size
    
    starting = np.arange(0, len(r) - ws, 3)  # window indices
    ending = starting + ws - 1  # window indices
    
    ax.axhline(0, c='k')  # lineage mean baseline

    ax.plot(np.arange(1, len(r) + 1), r, label=r'$Z$')  # plot the initial size normalized cumulative sum

    # Plot the linear trends in the windows
    for start, end in zip(starting, ending):
        slope, intercept = linregress(np.arange(start + 1, end + 2), r[start:end + 1])[:2]
        ax.plot(np.arange(start + 1, end + 2), intercept + slope * np.arange(start + 1, end + 2), color=c, ls='--', label='' if start != starting[-1] else r'$\hat{Z}$')
    
    # graphics
    ax.set_xlim([1, len(r)])
    ax.set_xlabel('Generation')
    ax.legend(title='')


def plot_loglog_lines(loglog_plot_recreation, ax):
    
    for variable in [r'$r$', r'$x_0$']:
        
        print(variable)
        
        # For every kind of loglog plot
        relevant = loglog_plot_recreation[(loglog_plot_recreation['variable'] == variable) & (loglog_plot_recreation['dataset'].isin(['Trace', 'Shuffled']))].copy()
        
        set_trace_legend, set_shuffled_legend = False, False
        
        wss_array = {'Trace': np.array([]), 'Shuffled': np.array([])}
        ytr_array = wss_array.copy()
        
        # For each lineage separately
        for ind in relevant.index:
            # Convert the window sizes and mean squared error per windows size from the dataframe from a string to an array of integers or floats
            wss = relevant['window_sizes'].loc[ind].strip('][').split(' ')
            wss = [int(r.split('\n')[0]) for r in wss if r != '']
            ytr = relevant['y_to_regress'].loc[ind].strip('][').split(', ')
            ytr = [float(r) for r in ytr if r != '']
            
            # If it is a trace lineage plot it and include it in the legend
            if relevant.loc[ind]['dataset'] == 'Trace':
                # The constant shuffled color
                color = cmap[0]
                
                # Add the analysis curve to the array of its dataset for the population regression
                wss_array['Trace'] = np.append(wss_array['Trace'], wss)
                ytr_array['Trace'] = np.append(ytr_array['Trace'], ytr)
                
                # Include it in the legend but not more than once
                if set_trace_legend:
                    ax.plot(wss, ytr, color=color, marker='', alpha=.09)
                else:
                    ax.plot(wss, ytr, color=color, marker='', alpha=.09)  # , label='Trace'
                    set_trace_legend = True
            # If it is a shuffled lineage plot it and include it in the legend
            elif relevant.loc[ind]['dataset'] == 'Shuffled':
                # The constant shuffled color
                color = cmap[1]
                
                # Add the analysis curve to the array of its dataset for the population regression
                wss_array['Shuffled'] = np.append(wss_array['Shuffled'], wss)
                ytr_array['Shuffled'] = np.append(ytr_array['Shuffled'], ytr)
                
                # # Include it in the legend but not more than once
                # if set_shuffled_legend:
                #     ax.plot(wss, ytr, color=color, marker='')
                # else:
                #     ax.plot(wss, ytr, color=color, marker='')  # , label='Shuffled'
                #     set_shuffled_legend = True
            # We do not want to see the white noise
            else:
                continue
        
        # Get the linear regression of all the trajectories pooled for each dataset
        slope_trace, intercept_trace, _, _, std_err_trace = linregress(np.log(np.array(wss_array['Trace']).flatten()), np.log(np.array(ytr_array['Trace']).flatten()))
        slope_art, intercept_art, _, _, std_err_art = linregress(np.log(np.array(wss_array['Shuffled']).flatten()), np.log(np.array(ytr_array['Shuffled']).flatten()))
        
        print(variable, slope_trace, intercept_trace, std_err_trace)
        
        # Plot the best fit line and its parameters
        ax.plot(np.unique(wss_array['Trace']), [np.exp(intercept_trace) * (l ** slope_trace) for l in np.unique(wss_array['Trace'])], ls='--', color='blue',
                linewidth=1, zorder=500, label=variable+f': ${np.round(slope_trace, 2)}$')  # label=r'$' + str(np.round(intercept_trace, 2)) + r'\, k^{' + str(np.round(slope_trace, 2)) + r'\pm' + str(np.round(std_err_trace, 3)) + r'}$'   ,    label='Trace'

    ax.plot(np.arange(5, 83, dtype='int'), (.47/(5**.5)) * (np.arange(5, 83) ** .5), label=r'random: $.5$', ls='--', c='k')
    # ax.plot(np.unique(wss_array['Shuffled']), [np.exp(intercept_art) * (l ** .5) for l in np.unique(wss_array['Shuffled'])], ls='--', color='red', linewidth=3,
    #         label='Shuffled', zorder=100)  # label=r'$' + str(np.round(intercept_art, 2)) + r'\, k^{' + str(np.round(slope_art, 2)) + r'\pm' + str(np.round(std_err_art, 3)) + r'}$'
    # ax.plot(np.unique(wss_array['Shuffled']), [np.exp(intercept_art) * (l ** slope_art) for l in np.unique(wss_array['Shuffled'])], ls='--', color='red', linewidth=3,
    #         label='Shuffled', zorder=100)  # label=r'$' + str(np.round(intercept_art, 2)) + r'\, k^{' + str(np.round(slope_art, 2)) + r'\pm' + str(np.round(std_err_art, 3)) + r'}$'
    
    # ax.title(variable)
    ax.set_ylabel(r'$F(k)$')
    ax.set_xlabel('k (window size)')
    ax.set_yscale('log')
    ax.set_xscale('log')
    # ax.legend(title='', loc='upper left')


def plot_dfa_slopes(ax):
    se = pd.DataFrame()  # Where we will put all the scaling exponents
    
    for ds in ['Pooled_SM', 'Maryam_LongTraces', 'Lambda_LB', 'MG1655_inLB_LongTraces']:  # Pool all these datasets into one dataframe
        
        print(ds)
        scaling_exponents = pd.read_csv(f'/Users/alestawsky/PycharmProjects/Thesis/DFA/Scaling Exponents/{ds}/scaling_exponents.csv')  # Import
        
        scaling_exponents['experiment'] = ds  # Add their dataset name
        
        se = se.append(scaling_exponents, ignore_index=True)  # Append them
    
    se = se.replace({'$f_n e^{\phi_{n}}$': r'$r$'})  # Use the latex variable names
    
    plt.axhline(0.5, ls='-', c='k')  # Plot a black like to show what random processes should be like
    sns.pointplot(data=se, x='variable', y='slope', hue='dataset', join=False, dodge=True, palette=cmap, capsize=.1, ax=ax, ci="sd", zorder=100)  # The error bars
    ax.set_ylabel(r'$\gamma$')
    ax.set_xlabel('')
    # ax.legend(title='', fontsize='small', markerscale=.5)
    ax.get_legend().remove()
    plt.ylim([0, 1.2])
    

# # To calculate the dataframe with the correct analysis
# create_dfa_dataframe()

current_dir = os.path.dirname(os.path.abspath(__file__))

recreation = pd.read_csv(current_dir + r'/Pooled_SM/loglog_scaling_recreation.csv')
recreation = recreation.replace(to_replace={r'$f_n e^{\phi_{n}}$': r'$r$'})

sns.set_context('paper', font_scale=1)
sns.set_style("ticks", {'axes.grid': True})
fig, axes = plt.subplots(1, 3, figsize=[6.5, 2.9], tight_layout=True)

axes[0].set_title('A', x=-.1, fontsize='xx-large')
axes[1].set_title('B', x=-.12, fontsize='xx-large')
axes[2].set_title('C', x=-.18, fontsize='xx-large')

plot_dfa_cumsum_illustration(axes[0])
plot_loglog_lines(recreation[recreation['kind'] == 'dfa (short)'], axes[1])
plot_dfa_slopes(axes[2])
plt.savefig('dfa_figure.png', dpi=500)
# plt.show()
plt.close()
