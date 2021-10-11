#!/usr/bin/env bash

from AnalysisCode.global_variables import (
    symbols, phenotypic_variables, cmap, shuffle_info, check_the_division, tanouchi_datasets, slash, seaborn_preamble
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


# def plot_neighbor_raw_data(ax, **kwargs):
#     raw_lineages = pd.read_csv(kwargs['raw_data']('Pooled_SM') + 'raw_data_all_in_one.csv')
#     a_trace = raw_lineages[raw_lineages['lineage_ID'] == 1].copy().sort_values('time').reset_index(drop=True)
#     b_trace = raw_lineages[raw_lineages['lineage_ID'] == 2].copy().sort_values('time').reset_index(drop=True)
#
#     ax.plot(a_trace['time'].values, a_trace['length'].values, label='A')
#     ax.plot(b_trace['time'].values, b_trace['length'].values, label='B')
#     ax.legend()


def possible_nl_lineages(**kwargs):
    # 133 ( GOOD COORDINATION ), 315, 91, 341, 311, 111 --> Possible NL lineages

    ta = pd.read_csv(kwargs['processed_data']('Pooled_SM') + 'time_averages.csv').drop(
        columns=['generation']).drop_duplicates()
    pu = pd.read_csv(kwargs['processed_data']('Pooled_SM') + 'physical_units.csv')

    mask = (ta['max_gen'] >= 30)

    print('min\n', ta[mask].sort_values('length_birth', ascending=True)[['lineage_ID', 'max_gen']])
    print('max\n', ta[mask].sort_values('length_birth', ascending=False)[['lineage_ID', 'max_gen']])
    # exit()
    #
    # print(ta.columns)
    # print(pu.columns)
    # mask = ta['lineage_ID'].isin(pu[pu['dataset'] == 'NL'].lineage_ID.values)
    # print(len(ta[(ta['lineage_ID'] % 2 == 0) & mask].max_gen.values))
    #
    # updated = ta[(ta['lineage_ID'] % 2 == 0) & mask]
    #
    # print(updated.sort_values('max_gen').lineage_ID.values)
    # updated1 = ta[(ta['lineage_ID'] % 2 == 1) & mask]
    # print(updated1.sort_values('max_gen').lineage_ID.values)


def plot_nl_length_birth_in_gentime_and_avg(num, ax, **kwargs):
    ta = pd.read_csv(kwargs['processed_data']('Pooled_SM') + 'time_averages.csv').drop(
        columns=['generation']).drop_duplicates()
    pu = pd.read_csv(kwargs['processed_data']('Pooled_SM') + 'physical_units.csv')

    ax.plot(pu[pu['lineage_ID'] == num].generation.values, pu[pu['lineage_ID'] == num].length_birth.values, c='green',
            marker='.')
    ax.plot(pu[pu['lineage_ID'] == num + 1].generation.values, pu[pu['lineage_ID'] == num + 1].length_birth.values,
            c='red', marker='.')
    ax.axhline(ta[ta['lineage_ID'] == num].length_birth.values, c='green')
    ax.axhline(ta[ta['lineage_ID'] == num + 1].length_birth.values, c='red')
    ax.set_xlabel('n [Gen]')
    ax.set_ylabel(r'$x_0(n) \, [\mu m]$')
    # plt.show()
    # plt.close()


def the_hists(num, ax, **kwargs):
    pu = pd.read_csv(kwargs['processed_data']('Pooled_SM') + 'physical_units.csv')

    sns.histplot(data=pu, x='length_birth', color='grey', ax=ax, stat='density', fill=True, kde=True)
    sns.histplot(data=pu[pu.lineage_ID == num], x='length_birth', color='green', ax=ax, stat='density', fill=True, kde=False)
    sns.histplot(data=pu[pu.lineage_ID == num+1], x='length_birth', color='red', ax=ax, stat='density', fill=True, kde=False)

    # sns.histplot(data=pu[pu.lineage_ID == 337], x='length_birth', color='green', ax=ax, stat='density', fill=True, kde=False)
    # sns.histplot(data=pu[pu.lineage_ID == 296], x='length_birth', color='red', ax=ax, stat='density', fill=True, kde=False)
    ax.set_xlim([1, 5])
    ax.set_xlabel(r'$x_0 \, [\mu m]$')


""" variance decomposition for SM data conditioning on trap and lineage """


def vd_with_trap_and_lineage(variables, lin_type, ax, **kwargs):
    # The dataframe with all the experiments in it
    total_df = pd.read_csv(kwargs['without_outliers']('Pooled_SM') + 'physical_units_without_outliers.csv')
    # total_df = pd.read_csv(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + f'/Datasets/Pooled_SM/{kwargs["noise_index"].split("_").join("_")}ProcessedData/z_score_under_3/physical_units_without_outliers.csv')

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
    plt.legend(handles, labels, title='', loc='upper right')

    # Graphical things
    ax.set_xlabel('')
    ax.set_ylabel('Variance Decomposition')
    ax.set_ylim([0, .45])


def main(**kwargs):
    # possible_nl_lineages(**kwargs)

    seaborn_preamble()

    scale = 1

    fig, axes = plt.subplots(2, 2, figsize=(6.5 * scale, 6.5 * scale))  # figsize=()

    num = 133  # This is the best one cause it shows coordination and seperate averages but close enough

    # # From the process_rawdata.py file where it is used to visually confirm the regression as well as the division events
    # check_the_division('Pooled_SM', lineages=[1], raw_lineages=[], raw_indices=[], pu=[], ax=axes[0, 1], **kwargs)  # A trace
    # check_the_division('Pooled_SM', lineages=[2], raw_lineages=[], raw_indices=[], pu=[], ax=axes[0, 1], **kwargs)  # B trace

    # plot_neighbor_raw_data(axes[0, 1], **kwargs)
    axes[0, 0].set_title('A', x=-.3, fontsize='xx-large')
    axes[0, 1].set_title('B', x=-.3, fontsize='xx-large')
    axes[1, 0].set_title('C', x=-.3, fontsize='xx-large')
    axes[1, 1].set_title('D', x=-.3, fontsize='xx-large')
    axes[0, 0].set_frame_on(False)
    axes[0, 0].set_xticks([])
    axes[0, 0].set_yticks([])
    plot_nl_length_birth_in_gentime_and_avg(num, axes[0, 1], **kwargs)
    the_hists(num, axes[1, 0], **kwargs)
    vd_with_trap_and_lineage(phenotypic_variables, lin_type='NL', ax=axes[1, 1], **kwargs)

    plt.tight_layout()
    plt.savefig(f'NewIntroductoryFigure{slash}fig4.png', dpi=300)
    # plt.show()
    plt.close()


if __name__ == '__main__':
    main()
