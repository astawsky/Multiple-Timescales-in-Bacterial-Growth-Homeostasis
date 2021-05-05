#!/usr/bin/env bash

from AnalysisCode.global_variables import (
    symbols, phenotypic_variables, cmap, sm_datasets, shuffle_info, cgsc_6300_wang_exps, lexA3_wang_exps, tanouchi_datasets
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
    
    lineage_id = 0
    for dataset in np.unique(info['dataset']):
        for trap_ID in np.unique(info[info['dataset'] == dataset]['trap_ID']):
            for trace in ['A', 'B']:
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


""" vd conditioning on experiments and lineage """


def mm_lineage_experiment(chosen_datasets, ax, loc='upper left'):
    total_df = pd.DataFrame()  # pool all the experiments
    for data_origin in chosen_datasets:
        pu = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/Datasets/' + data_origin + '/ProcessedData/z_score_under_3/physical_units_without_outliers.csv'
        
        pu = pd.read_csv(pu)
        pu['experiment'] = data_origin
        
        total_df = total_df.append(pu, ignore_index=True)
    
    # The dataframe that contains all the values inside the summation of the decomposition
    output_df = pd.DataFrame(columns=['variable', 'intrinsic', 'environment', 'lineage', 'kind'])
    
    # For graphical purposes
    output_df = output_df.append({
        'variable': '',
        'Exp+Env': 0,
        'Exp+Env+Lin': 0,
        # This is so we can graph it nicely
        'Exp': 0,
        'kind': 'Trace'
    }, ignore_index=True)
    
    # For both kinds of lineages
    for kind in ['Trace', 'Artificial']:
        if kind == 'Trace':
            # Use the pooled experiment dataframe
            pu = total_df.copy()
            
            # The pooled mean of all experiments chosen
            pooled_pu_mean = pu[phenotypic_variables].mean()
        else:
            # Sample from pooled ensemble without replacement
            pu = shuffle_info(total_df, mm=True)
            
            # Change this one manually
            pu['experiment'] = total_df['experiment'].copy().sort_values().values
            pu = pu.copy()
            
            # The pooled mean
            pooled_pu_mean = pu[phenotypic_variables].mean()
        
        # The different values in the summation of the decomposition
        delta = pd.DataFrame(columns=phenotypic_variables)
        line = pd.DataFrame(columns=phenotypic_variables)
        expert = pd.DataFrame(columns=phenotypic_variables)
        
        # For each experiment and each lineage in each experiment...
        for exp in pu.experiment.unique():
            e_cond = (pu['experiment'] == exp)  # Masking
            exp_lins = pu[e_cond].copy()  # The masked dataframe
            exp_mean = exp_lins[phenotypic_variables].mean()  # The average of the experiment values
            for lin_id in pu[pu['experiment'] == exp].lineage_ID.unique():
                l_cond = (pu['lineage_ID'] == lin_id) & e_cond  # Masking
                lin = pu[l_cond].copy()  # The masked dataframe
                
                # Calculate what is inside the decomposition
                expert = expert.append(lin[phenotypic_variables].count() * ((exp_mean - pooled_pu_mean) ** 2), ignore_index=True)
                line = line.append(lin[phenotypic_variables].count() * ((lin[phenotypic_variables].mean() - exp_mean) ** 2), ignore_index=True)
                delta = delta.append(((lin[phenotypic_variables] - lin[phenotypic_variables].mean()) ** 2).sum(), ignore_index=True)
        
        # Average over all instances with degree of freedom 1
        exp_var = expert.sum() / (pu[phenotypic_variables].count() - 1)
        delta_var = delta.sum() / (pu[phenotypic_variables].count() - 1)
        lin_var = line.sum() / (pu[phenotypic_variables].count() - 1)
        
        # Make sure it is a true decomposition
        assert (np.abs(pu[phenotypic_variables].var() - (exp_var[phenotypic_variables] + delta_var[phenotypic_variables] + lin_var[phenotypic_variables])) < .0000001).all()
        
        # Add it to the thing
        for variable in phenotypic_variables:
            output_df = output_df.append({
                'variable': symbols['physical_units'][variable],
                'Exp+Lin': (exp_var[variable] + lin_var[variable]) / pu[variable].var(),
                # This is so we can graph it nicely
                'Exp': (exp_var[variable]) / pu[variable].var() if kind == 'Trace' else (exp_var[variable] + lin_var[variable]) / pu[variable].var(),
                'kind': kind
            }, ignore_index=True)

    # For graphical purposes
    for kind in ['Trace', 'Artificial']:
        output_df = output_df.append({
            'variable': ' ',
            'Exp+Env': 0,
            'Exp+Env+Lin': 0,
            # This is so we can graph it nicely
            'Exp': 0,
            'kind': kind
        }, ignore_index=True)
        
        output_df = output_df.append({
            'variable': '  ',
            'Exp+Env': 0,
            'Exp+Env+Lin': 0,
            # This is so we can graph it nicely
            'Exp': 0,
            'kind': kind
        }, ignore_index=True)

    # For graphical purposes
    output_df = output_df.append({
        'variable': '   ',
        'Exp+Lin': 0,
        # This is so we can graph it nicely
        'Exp': 0,
        'kind': 'Trace'
    }, ignore_index=True)

    # The order we want them to appear in
    real_order = np.array(['', symbols['physical_units']['growth_rate'], symbols['physical_units']['generationtime'], symbols['physical_units']['fold_growth'],
                  symbols['physical_units']['division_ratio'], symbols['physical_units']['div_and_fold'], ' ', symbols['physical_units']['length_birth'],
                  symbols['physical_units']['added_length'], '  ', '   '])
    
    # Condition for the gray filled line
    conds = (output_df['kind'] == 'Artificial') & (~output_df['variable'].isin(['', ' ', '  ', '   ']))

    # Plot the grey noise line
    ax.fill_between(real_order,  # output_df.variable.unique()
                    [output_df[conds]['Exp+Lin'].mean() for _ in range(len(real_order))],
                    [0 for _ in range(len(real_order))], color='lightgrey')
    
    # Plot the barplots for the trace lineages
    for color, y, label in zip([cmap[0], cmap[2]], ['Exp+Lin', 'Exp'], [r'$\Gamma_{Trap}$', r'$\Gamma_{Exp}$']):
        sns.barplot(x='variable', y=y, data=output_df[output_df['kind'] == 'Trace'], color=color, edgecolor='black', label=label, order=real_order[1:-2], ax=ax)
    
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, title='', loc=loc)
    ax.set_xlabel('')
    ax.set_ylabel('Variance Decomposition')
    ax.set_ylim([0, .45])
    
    
""" vd conditioning on experiment, trap and lineage """


def vd_with_trap_lineage_and_experiments(sm_ds, variables, lin_type, ax, loc='upper right'):
    total_df = pd.DataFrame()  # The dataframe with all the experiments in it

    # Check what type of pair lineages we want to look at
    if lin_type == 'NL':
        type_of_lineages = ['NL']
    elif lin_type == 'SL':
        type_of_lineages = ['SL']
    else:
        type_of_lineages = ['SL', 'NL']
    
    # Pool all the experiments and add a category label
    for data_origin in sm_ds[:-1]:
        print(data_origin)
        
        # pu = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/Datasets/' + data_origin + '/ProcessedData/z_score_under_3/physical_units_without_outliers.csv'
        
        pu = pd.read_csv(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/Datasets/' + data_origin + '/ProcessedData/z_score_under_3/physical_units_without_outliers.csv')
        
        pu = pu[pu['dataset'].isin(type_of_lineages)]  # use only the pair lineages we are interested in
        
        if len(total_df) != 0:
            pu['trap_ID'] = pu['trap_ID'] + total_df['trap_ID'].max()
        
        pu['experiment'] = data_origin  # category variable of the experiment
        
        total_df = total_df.append(pu, ignore_index=True)  # Add them to the dataframe with all the experiments
    
    print('type of pair lineages we are using:', lin_type)
    output_df = pd.DataFrame()
    
    # What kind of lineage we want to look at, Artificial for noise estimation and Trace for the results
    for kind in ['Trace', 'Artificial']:
        if kind == 'Trace':
            # pu = total_df[total_df['dataset'].isin(type_of_lineages)].copy()
            pu = total_df.copy()
            
            # The pooled mean
            pooled_pu_mean = pu[variables].mean()
        else:
            pu = shuffle_info_sm(total_df)  # shuffle the information to get an ergodic set of the same statistics
            pu['experiment'] = total_df['experiment'].copy().sort_values().values  # Randomize the experiments as well
            
            pooled_pu_mean = pu[variables].mean()  # The pooled mean for the analysis later
        
        # These represent the terms in the equation of the variance decomposition. Here we will store their values inside the sums of the equation of variance for each phenotypic variable seperately.
        delta = pd.DataFrame(columns=variables)
        diff = pd.DataFrame(columns=variables)
        trap = pd.DataFrame(columns=variables)
        expert = pd.DataFrame(columns=variables)
        
        for exp in pu.experiment.unique():
            
            exp_lins = pu[(pu['experiment'] == exp)].copy()  # All the bacteria in this experiment
            
            exp_mean = exp_lins[variables].mean()  # The average value of all bacteria in this experiment
            
            for trap_id in exp_lins.trap_ID.unique():
                
                trap_lins = exp_lins[(exp_lins['trap_ID'] == trap_id)].copy()  # All the bacteria in this trap
                
                trap_means = trap_lins[variables].mean()  # The average value of all the bacteria in this trap
                
                for trace in ['A', 'B']:
                    
                    lin = trap_lins[(trap_lins['trace'] == trace)].copy()  # All the bacteria in this lineage (channel)
                    
                    # Add the deviation, for all generations in this lineage, of the mean of this experiment from the mean of the pooled ensemble
                    expert = expert.append(lin[variables].count() * ((exp_mean - pooled_pu_mean) ** 2), ignore_index=True)
                    
                    # Add the deviation, for all generations in this lineage, of the mean of this trap from the mean of this experiment
                    trap = trap.append(lin[variables].count() * ((trap_means - exp_mean) ** 2), ignore_index=True)
                    
                    # Add the deviation, for all generations in this lineage, of the mean of this lineage (channel) from the mean of this trap
                    diff = diff.append(lin[variables].count() * ((lin[variables].mean() - trap_means) ** 2), ignore_index=True)
                    
                    # Add the deviation of all the generation-specific values around the lineage mean
                    delta = delta.append(((lin[variables] - lin[variables].mean()) ** 2).sum(), ignore_index=True)
        
        # Calculate the variance for each term with one degree of freedom
        exp_var = expert.sum() / (pu[variables].count() - 1)
        delta_var = delta.sum() / (pu[variables].count() - 1)
        diff_var = diff.sum() / (pu[variables].count() - 1)
        tmean_var = trap.sum() / (pu[variables].count() - 1)
        
        # Make sure it is a true decomposition, ie. that the variances of the terms add up to the variance of the pooled ensemble
        assert (np.abs(pu[variables].var() - (exp_var[variables] + delta_var[variables] + diff_var[variables] + tmean_var[variables])) < .0000001).all()
        
        # Now save them to the output dataframe
        for variable in variables:
    
            # For every phenotypic variable, add its variance decomposition to the dataframe
            output_df = output_df.append({
                'variable': symbols['physical_units'][variable],  # Latex of the variable (Useful for plotting)
                'Exp+Trap': (exp_var[variable] + tmean_var[variable]) / pu[variable].var(),  # Variance explained by the trap and experiment
                'Exp+Trap+Lin': (exp_var[variable] + tmean_var[variable] + diff_var[variable]) / pu[variable].var(),  # Variance explained by the experiment, trap and lineage
                'Exp': (exp_var[variable]) / pu[variable].var(),  # Variance explained by the experiment
                'kind': kind
            }, ignore_index=True)

    for spacer in ['', ' ', '  ', '   ', '    ']:
        output_df = output_df.append({
            'variable': spacer,
            'Exp+Trap': 0,
            'Exp+Trap+Lin': 0,
            # This is so we can graph it nicely
            'Exp': 0,
            'kind': 'NA'
        }, ignore_index=True)

    # output_df = output_df.append({
    #     'variable': '',
    #     'Exp+Trap': 0,
    #     'Exp+Trap+Lin': 0,
    #     # This is so we can graph it nicely
    #     'Exp': 0,
    #     'kind': 'NA'
    # }, ignore_index=True)
    #
    # output_df = output_df.append({
    #     'variable': ' ',
    #     'Exp+Trap': 0,
    #     'Exp+Trap+Lin': 0,
    #     # This is so we can graph it nicely
    #     'Exp': 0,
    #     'kind': 'NA'
    # }, ignore_index=True)
    #
    # output_df = output_df.append({
    #     'variable': '  ',
    #     'Exp+Trap': 0,
    #     'Exp+Trap+Lin': 0,
    #     # This is so we can graph it nicely
    #     'Exp': 0,
    #     'kind': 'NA'
    # }, ignore_index=True)
    #
    # output_df = output_df.append({
    #     'variable': '   ',
    #     'Exp+Trap': 0,
    #     'Exp+Trap+Lin': 0,
    #     # This is so we can graph it nicely
    #     'Exp': 0,
    #     'kind': 'NA'
    # }, ignore_index=True)
    #
    # output_df = output_df.append({
    #     'variable': '    ',
    #     'Exp+Trap': 0,
    #     'Exp+Trap+Lin': 0,
    #     # This is so we can graph it nicely
    #     'Exp': 0,
    #     'kind': 'NA'
    # }, ignore_index=True)
    
    # The order we want them to appear in
    real_order = np.array(['', symbols['physical_units']['growth_rate'], symbols['physical_units']['generationtime'], symbols['physical_units']['fold_growth'],
                  symbols['physical_units']['division_ratio'], symbols['physical_units']['div_and_fold'], ' ', symbols['physical_units']['length_birth'],
                  symbols['physical_units']['added_length'], '  ', '   '])
    
    # Plot the grey noise line
    ax.fill_between(real_order,  # output_df.variable.unique()
                    [output_df[output_df['kind'] == 'Artificial']['Exp+Trap+Lin'].mean() for _ in range(len(real_order))],
                    [0 for _ in range(len(real_order))], color='lightgrey')
    
    for color, y, label in zip([cmap[0], cmap[1], cmap[2]], ['Exp+Trap+Lin', 'Exp+Trap', 'Exp'], [r'$\Gamma_{Lin}$', r'$\Gamma_{Env}$', r'$\Gamma_{Exp}$']):
        # palette = {"Trace": color, "Artificial": 'red'}
        sns.barplot(x='variable', y=y, data=output_df[output_df['kind'] == 'Trace'], color=color, edgecolor='black', label=label, order=real_order[1:-2])

    ax.set_yticklabels('')
    ax.set_ylabel('')
    ax.set_xlabel('')
    ax.set_ylabel('')
    plt.ylim([0, .45])
    
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, title='', loc=loc)


""" variance decomposition for MM data conditioning on traps """


def mm_traps(chosen_datasets, ax, loc='upper left'):
    
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
    real_order = ['', symbols['physical_units']['growth_rate'], symbols['physical_units']['generationtime'], symbols['physical_units']['fold_growth'],
                  symbols['physical_units']['division_ratio'], symbols['physical_units']['div_and_fold'], ' ', symbols['physical_units']['length_birth'],
                  symbols['physical_units']['added_length'], '  ', '   ']
    
    # # The order we want them to appear in
    # real_order = ['', symbols['physical_units']['div_and_fold'], symbols['physical_units']['division_ratio'], symbols['physical_units']['fold_growth'], ' ',
    #               symbols['physical_units']['added_length'], symbols['physical_units']['length_birth'], '  ', symbols['physical_units']['generationtime'],
    #               symbols['physical_units']['growth_rate'], '   ']
    
    # Conditions for the grey noise area from the variance decomposition of the artificial lineages
    conds = (output_df['kind'] == 'Artificial') & (~output_df['variable'].isin(['', ' ', '  ', '   ']))
    
    # Plot the grey noise line
    ax.fill_between(real_order,  # output_df.variable.unique()
                    [output_df[conds]['Lin'].mean() for _ in range(len(real_order))],
                    [0 for _ in range(len(real_order))], color='lightgrey')
    
    # Plot the barplots
    sns.barplot(x='variable', y='Lin', data=output_df[output_df['kind'] == 'Trace'], color=cmap[0], edgecolor='black', label='Lineage', order=real_order[1:-2], ax=ax)
    
    # The legend in the plot
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, [r'$\Gamma_{Trap}$'], title='', loc=loc)
    
    # Graphical things
    ax.set_xlabel('')
    ax.set_ylabel('Variance Decomposition')
    ax.set_ylim([0, .45])


# The order we want them to appear in
real_order = np.array(['', symbols['physical_units']['growth_rate'], symbols['physical_units']['generationtime'], symbols['physical_units']['fold_growth'],
              symbols['physical_units']['division_ratio'], symbols['physical_units']['div_and_fold'], ' ', symbols['physical_units']['length_birth'],
              symbols['physical_units']['added_length'], '  ', '   '])

scale = 1.5
sns.set_context('paper', font_scale=scale)
sns.set_style("ticks", {'axes.grid': True})
fig, axes = plt.subplots(1, 3, figsize=[6.5 * scale, 2.5 * scale], tight_layout=True)

axes[0].set_title('A', x=-.2, fontsize='xx-large')
axes[1].set_title('B', x=-.05, fontsize='xx-large')
axes[2].set_title('C', x=-.05, fontsize='xx-large')

# For experiments that were not in the main text to show consistency
mm_traps([tanouchi_datasets[0]], ax=axes[0])
mm_traps([tanouchi_datasets[1]], ax=axes[1])
mm_traps([tanouchi_datasets[2]], ax=axes[2])
for ax in axes:
    ax.set_ylim([0, .08])
    # ax.set_xlim([real_order[1], real_order[-2]])
axes[1].set_ylabel('')
axes[2].set_ylabel('')
axes[1].get_legend().remove()
axes[2].get_legend().remove()

plt.savefig('tanouchi_variance_decomposition.png', dpi=300)
plt.show()
plt.close()

scale = 1.5
sns.set_context('paper', font_scale=scale)
sns.set_style("ticks", {'axes.grid': True})
fig, axes = plt.subplots(1, 2, figsize=[6.5 * scale, 3.5 * scale], tight_layout=True)

axes[0].set_title('A', x=-.2, fontsize='xx-large')
axes[1].set_title('B', x=-.05, fontsize='xx-large')

# For experiments that were not in the main text to show consistency
mm_lineage_experiment(cgsc_6300_wang_exps, ax=axes[0])
mm_lineage_experiment(lexA3_wang_exps, ax=axes[1])
axes[1].get_legend().remove()  # Repetative

plt.savefig('wang_variance_decomposition.png', dpi=300)
plt.show()
plt.close()

scale = 1.5
sns.set_context('paper', font_scale=scale)
sns.set_style("ticks", {'axes.grid': True})
fig, axes = plt.subplots(1, 2, figsize=[6.5 * scale, 3.5 * scale], tight_layout=True)

axes[0].set_title('A', x=-.2, fontsize='xx-large')
axes[1].set_title('B', x=-.05, fontsize='xx-large')

# For the MM and SM experiments in the main text
mm_lineage_experiment(['lambda_lb', 'Maryam_LongTraces'], ax=axes[0], loc='upper right')
vd_with_trap_lineage_and_experiments(sm_datasets, phenotypic_variables, 'NL', axes[1])

plt.savefig('mm_and_sm_decompositions.png', dpi=300)
plt.show()
plt.close()
