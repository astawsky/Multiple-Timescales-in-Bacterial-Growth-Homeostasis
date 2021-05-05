#!/usr/bin/env bash

from AnalysisCode.global_variables import (
    symbols, phenotypic_variables, cmap, shuffle_info, check_the_division, tanouchi_datasets
)
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os

""" Create all information dataframe where the lineage lengths are kept constant but the cells in the trace itself are randomly sampled from the population without replacement """


def shuffle_info_sm(info):
    # Give it a name, contains S, NL
    new_info = pd.DataFrame(columns=info.columns)
    
    # what is the trace length of each trace? This is the only thing that stays the same
    sizes = {'{} {} {}'.format(dataset, trap_ID, trace): len(info[(info['trap_ID'] == trap_ID) & (info['trace'] == trace) & (info['dataset'] == dataset)]) for dataset in np.unique(info['dataset'])
             for trap_ID in np.unique(info[info['dataset'] == dataset]['trap_ID']) for trace in ['A', 'B']}
    
    lineage_id = 0  # To rename each lineage
    
    for dataset in np.unique(info['dataset']):  # For Sister and Neighbor lineages
        for trap_ID in np.unique(info[info['dataset'] == dataset]['trap_ID']):  # For each trap in each type of lineage
            for trace in ['A', 'B']:  # For each lineage in each trap
                # trace length
                size = sizes['{} {} {}'.format(dataset, trap_ID, trace)]
                
                # sample from the old dataframe
                samples = info[info['dataset'] == dataset].sample(replace=False, n=size)
                
                # drop what we sampled
                info = info.drop(index=samples.index)
                
                # add some correct labels even though they don't matter so that the dataframe structure is still intact
                samples['dataset'] = dataset
                samples['trap_ID'] = trap_ID
                samples['trace'] = trace
                samples['lineage_ID'] = lineage_id
                samples['generation'] = np.arange(size)
                
                # add them to the new, shuffled dataframe
                new_info = new_info.append(samples, ignore_index=True)
                
                # Go on to the next ID
                lineage_id += 1
    
    return new_info


""" variance decomposition for SM data conditioning on trap and lineage """


def vd_with_trap_and_lineage(variables, lin_type, ax):
    # The dataframe with all the experiments in it
    total_df = pd.read_csv(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/Datasets/Pooled_SM/ProcessedData/z_score_under_3/physical_units_without_outliers.csv')
    
    # Check what type of pair lineages we want to look at
    if lin_type == 'NL':
        type_of_lineages = ['NL']
    elif lin_type == 'SL':
        type_of_lineages = ['SL']
    else:
        type_of_lineages = ['SL', 'NL']
    
    # Gives us the choice to use either neighbor or sister lineages. Or Both.
    total_df = total_df[total_df['dataset'].isin(type_of_lineages)]
    
    print('type of pair lineages we are using:', lin_type)
    output_df = pd.DataFrame()
    
    # What kind of lineage we want to look at, Artificial for noise estimation and Trace for the results
    for kind in ['Trace', 'Artificial']:
        if kind == 'Trace':
            pu = total_df.copy()
            
            # The pooled mean
            pooled_pu_mean = pu[variables].mean()
        else:
            pu = shuffle_info_sm(total_df)  # shuffle the information to get an ergodic set of the same statistics
            
            pooled_pu_mean = pu[variables].mean()  # The pooled mean for the analysis later
        
        # These represent the terms in the equation of the variance decomposition. Here we will store their values inside the sums of the equation of variance for each phenotypic variable seperately.
        delta = pd.DataFrame(columns=variables)
        lin_spec = pd.DataFrame(columns=variables)
        trap = pd.DataFrame(columns=variables)
        
        for trap_id in pu.trap_ID.unique():
            
            trap_ensemble = pu[(pu['trap_ID'] == trap_id)].copy()  # All the bacteria in this experiment
            
            trap_mean = trap_ensemble[variables].mean()  # The average value of all bacteria in this experiment
            
            for trace in ['A', 'B']:
                
                lin = trap_ensemble[(trap_ensemble['trace'] == trace)].copy()  # All the bacteria in this trap
                
                # Add the deviation, for all generations in this lineage, of the mean of this experiment from the mean of the pooled ensemble
                trap = trap.append(lin[variables].count() * ((trap_mean - pooled_pu_mean) ** 2), ignore_index=True)
                
                # Add the deviation, for all generations in this lineage, of the mean of this trap from the mean of this experiment
                lin_spec = lin_spec.append(lin[variables].count() * ((lin[variables].mean() - trap_mean) ** 2), ignore_index=True)
                
                # Add the deviation of all the generation-specific values around the lineage mean
                delta = delta.append(((lin[variables] - lin[variables].mean()) ** 2).sum(), ignore_index=True)
        
        # Calculate the variance for each term with one degree of freedom
        trap_var = trap.sum() / (pu[variables].count() - 1)
        delta_var = delta.sum() / (pu[variables].count() - 1)
        lin_var = lin_spec.sum() / (pu[variables].count() - 1)
        
        # Make sure it is a true decomposition, ie. that the variances of the terms add up to the variance of the pooled ensemble
        assert (np.abs(pu[variables].var() - (trap_var[variables] + delta_var[variables] + lin_var[variables])) < .0000001).all()
        
        # Now save them to the output dataframe
        for variable in variables:
            
            # For every phenotypic variable, add its variance decomposition to the dataframe
            output_df = output_df.append({
                'variable': symbols['physical_units'][variable],  # Latex of the variable (Useful for plotting)
                'Trap+Lin': (trap_var[variable] + lin_var[variable]) / pu[variable].var(),  # Variance explained by the trap and experiment
                'Trap': (trap_var[variable]) / pu[variable].var(),  # Variance explained by the experiment
                'kind': kind
            }, ignore_index=True)
    
    for spacer in ['', ' ', '  ', '   ', '    ']:  # For graphical purposes to distinguish each type of variable (size, growth and dimensionless)
        output_df = output_df.append({
            'variable': spacer,
            'Trap+Lin': 0,
            # This is so we can graph it nicely
            'Trap': 0,
            'kind': 'NA'
        }, ignore_index=True)

    # The order we want them to appear in
    real_order = np.array(['', symbols['physical_units']['growth_rate'], symbols['physical_units']['generationtime'], symbols['physical_units']['fold_growth'],
                           symbols['physical_units']['division_ratio'], symbols['physical_units']['div_and_fold'], ' ', symbols['physical_units']['length_birth'],
                           symbols['physical_units']['added_length'], '  ', '   '])
    
    # # The order they appear in
    # real_order = ['', symbols['physical_units']['div_and_fold'], symbols['physical_units']['division_ratio'], symbols['physical_units']['fold_growth'], ' ',
    #               symbols['physical_units']['added_length'], symbols['physical_units']['length_birth'], '  ', symbols['physical_units']['generationtime'],
    #               symbols['physical_units']['growth_rate'], '   ']
    
    # Fill in the grey area of the noise
    plt.fill_between(real_order,
                     [output_df[output_df['kind'] == 'Artificial']['Trap+Lin'].mean() for _ in range(len(real_order))],
                     [0 for _ in range(len(real_order))], color='lightgrey')
    
    # Plot the barplots
    for color, y, label in zip([cmap[0], cmap[1]], ['Trap+Lin', 'Trap'], [r'$\Gamma_{Lin}$', r'$\Gamma_{Env}$']):
        # palette = {"Trace": color, "Artificial": 'red'}
        sns.barplot(x='variable', y=y, data=output_df[output_df['kind'] == 'Trace'], color=color, edgecolor='black', label=label, order=real_order[1:-2])
    
    # The legend
    handles, labels = ax.get_legend_handles_labels()
    plt.legend(handles, labels, title='')

    # Graphical things
    ax.set_xlabel('')
    ax.set_ylabel('')
    plt.ylim([0, .45])


""" variance decomposition for MM data conditioning on traps """


def mm_traps(chosen_datasets, ax):
    
    # The dataframe with all the experiments in it
    total_df = pd.DataFrame()
    
    # Pool the data from the chosen experiments
    for data_origin in chosen_datasets:
        print(data_origin)
        pu = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/Datasets/' + data_origin + '/ProcessedData/z_score_under_3/physical_units_without_outliers.csv'
        pu = pd.read_csv(pu)
        
        # Give them the experiment category
        pu['experiment'] = data_origin
        
        # Append the dataframe with this experiment's dataframe
        total_df = total_df.append(pu, ignore_index=True)
    
    # The dataframe that will contain the components of each decomposition inside the "pooled ensemble" sum
    output_df = pd.DataFrame(columns=['variable', 'intrinsic', 'environment', 'lineage', 'kind'])
    
    # For graphical purposes
    output_df = output_df.append({
        'variable': '',
        # This is so we can graph it nicely
        'Lin': 0,
        'kind': 'Trace'
    }, ignore_index=True)
    
    for kind in ['Trace', 'Artificial']:  # Trace or artificial lineages that are randomly sampled from the pooled distribution
        print(kind)
        if kind == 'Trace':
            df = total_df.copy()
            
            # The pooled mean
            pooled_pu_mean = df[phenotypic_variables].mean()
        else:
            df = shuffle_info(total_df, mm=True)
            # Shuffle this one manually
            df['experiment'] = total_df['experiment'].copy().sort_values().values
            df = df.copy()
            
            # The pooled mean
            pooled_pu_mean = df[phenotypic_variables].mean()
        
        # The two components in the decomposition
        delta = pd.DataFrame(columns=phenotypic_variables)
        line = pd.DataFrame(columns=phenotypic_variables)
        
        for exp in df.experiment.unique():
            e_cond = (df['experiment'] == exp)  # Condition that they are in the same experiment
            for lin_id in df[df['experiment'] == exp].lineage_ID.unique():
                l_cond = (df['lineage_ID'] == lin_id) & e_cond  # Condition that they are in the same experiment and lineage
                lin = df[l_cond].copy()  # The masked dataframe that contains bacteria in the same lineage and experiment
                
                # Add the components
                line = line.append(lin[phenotypic_variables].count() * ((lin[phenotypic_variables].mean() - pooled_pu_mean) ** 2), ignore_index=True)
                delta = delta.append(((lin[phenotypic_variables] - lin[phenotypic_variables].mean()) ** 2).sum(), ignore_index=True)
        
        # Get the variance of each one
        delta_var = delta.sum() / (df[phenotypic_variables].count() - 1)
        lin_var = line.sum() / (df[phenotypic_variables].count() - 1)
        
        # Make sure it is a true decomposition
        assert (np.abs(df[phenotypic_variables].var() - (delta_var[phenotypic_variables] + lin_var[phenotypic_variables])) < .0000001).all()
        
        # Add it to the dataframe of final components
        for variable in phenotypic_variables:
            output_df = output_df.append({
                'variable': symbols['physical_units'][variable],
                'Lin': (lin_var[variable]) / df[variable].var(),
                'kind': kind
            }, ignore_index=True)
    
    # For graphical purposes
    for kind in ['Trace', 'Artificial']:
        output_df = output_df.append({
            'variable': ' ',
            # This is so we can graph it nicely
            'Lin': 0,
            'kind': kind
        }, ignore_index=True)
        
        output_df = output_df.append({
            'variable': '  ',
            # This is so we can graph it nicely
            'Lin': 0,
            'kind': kind
        }, ignore_index=True)
    output_df = output_df.append({
        'variable': '   ',
        # This is so we can graph it nicely
        'Lin': 0,
        'kind': 'Trace'
    }, ignore_index=True)
    
    # The order we want them to appear in
    real_order = np.array(['', symbols['physical_units']['growth_rate'], symbols['physical_units']['generationtime'], symbols['physical_units']['fold_growth'],
                  symbols['physical_units']['division_ratio'], symbols['physical_units']['div_and_fold'], ' ', symbols['physical_units']['length_birth'],
                  symbols['physical_units']['added_length'], '  ', '   '])
    
    # # The order we want them to appear in
    # real_order = ['', symbols['physical_units']['div_and_fold'], symbols['physical_units']['division_ratio'], symbols['physical_units']['fold_growth'], ' ',
    #               symbols['physical_units']['added_length'], symbols['physical_units']['length_birth'], '  ', symbols['physical_units']['generationtime'],
    #               symbols['physical_units']['growth_rate'], '   ']
    
    # Conditions for the grey noise area from the variance decomposition of the artificial lineages
    conds = (output_df['kind'] == 'Artificial') & (~output_df['variable'].isin(['', ' ', '  ', '   ']))
    
    # Plot the grey noise line
    ax.fill_between(real_order,
                    [output_df[conds]['Lin'].mean() for _ in range(len(real_order))],
                    [0 for _ in range(len(real_order))], color='lightgrey')
    
    # Plot the barplots
    sns.barplot(x='variable', y='Lin', data=output_df[output_df['kind'] == 'Trace'], color=cmap[0], edgecolor='black', label='Lineage', order=real_order[1:-2], ax=ax)
    
    # The legend in the plot
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, [r'$\Gamma_{Trap}$'], title='')
    
    # Graphical things
    ax.set_xlabel('')
    ax.set_ylabel('Variance Decomposition')
    ax.set_ylim([0, .45])
    
    
""" Illustration of cell cycle physiological variables (Needs extra work outside python) """


def cell_cycle_illustration(ax):
    data_origin = 'Pooled_SM'  # We will use cycles from this dataset
    args = {
        'data_origin': data_origin,
        'raw_data': os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/Datasets/' + data_origin + '/RawData/',
        'processed_data': os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/Datasets/' + data_origin + '/ProcessedData/'
    }
    
    # From the process_rawdata.py file where it is used to visually confirm the regression as well as the division events
    check_the_division(args, lineages=[1], raw_lineages=[], raw_indices=[], pu=[], ax=ax)


# Graphical preferences
scale = 2
# scale = 1.5
sns.set_context('paper', font_scale=scale)
sns.set_style("ticks", {'axes.grid': True})

fig, axes = plt.subplots(1, 3, figsize=[6.5 * scale, 2.5 * scale], tight_layout=True)
# fig, axes = plt.subplots(1, 3, figsize=[6.5 * scale, 3.5 * scale], tight_layout=True)

axes[0].set_title('A', x=-.2, fontsize='xx-large')
axes[1].set_title('B', x=-.3, fontsize='xx-large')
axes[2].set_title('C', x=-.3, fontsize='xx-large')

cell_cycle_illustration(axes[0])
axes[0].set_xlim([1.48, 2.4])
axes[0].set_ylim([2.8, 6.5])
axes[0].set_ylabel(r'length $(\mu$m$)$')
axes[0].set_xlabel(r'time $($hr$)$')

# For the experiments that are in the main text
mm_traps(['lambda_lb', 'Maryam_LongTraces'], ax=axes[1])
vd_with_trap_and_lineage(phenotypic_variables, lin_type='NL', ax=axes[2])

# # For experiments that were not in the main text to show consistency
# mm_traps([tanouchi_datasets[0]], ax=axes[0])
# mm_traps([tanouchi_datasets[1]], ax=axes[1])
# mm_traps([tanouchi_datasets[2]], ax=axes[2])
# for ax in axes:
#     ax.set_ylim([0, .08])
# axes[1].set_ylabel('')
# axes[2].set_ylabel('')
# axes[1].get_legend().remove()
# axes[2].get_legend().remove()

plt.tight_layout()
plt.show()  # has to be manually adjusted to some degree
# plt.savefig('vd_square.png', dpi=300)
plt.close()
