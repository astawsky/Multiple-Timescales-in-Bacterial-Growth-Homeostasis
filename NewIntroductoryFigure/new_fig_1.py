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


def plot_neighbor_raw_data(ax, num1, num2, **kwargs):
    raw_lineages = pd.read_csv(kwargs['raw_data']('Maryam_LongTraces') + 'raw_data_all_in_one.csv')
    a_trace = raw_lineages[raw_lineages['lineage_ID'] == num1].copy().sort_values('time').reset_index(drop=True)
    b_trace = raw_lineages[raw_lineages['lineage_ID'] == num2].copy().sort_values('time').reset_index(drop=True)

    ax.plot(a_trace['time'].values, a_trace['length'].values, label='A')
    ax.plot(b_trace['time'].values, b_trace['length'].values, label='B')
    # ax.legend()


def plot_nl_length_birth_in_gentime_and_avg(num1, num2, ax, **kwargs):
    ta = pd.read_csv(kwargs['processed_data']('Maryam_LongTraces') + 'time_averages.csv').drop(
        columns=['generation']).drop_duplicates()
    pu = pd.read_csv(kwargs['processed_data']('Maryam_LongTraces') + 'physical_units.csv')

    ax.plot(pu[pu['lineage_ID'] == num1].generation.values, pu[pu['lineage_ID'] == num1].length_birth.values, c='blue')
    ax.plot(pu[pu['lineage_ID'] == num2].generation.values, pu[pu['lineage_ID'] == num2].length_birth.values,
             c='orange')
    ax.axhline(ta[ta['lineage_ID'] == num1].length_birth.values, c='blue')
    ax.axhline(ta[ta['lineage_ID'] == num2].length_birth.values, c='orange')
    ax.set_xlabel('n [Gen]')
    ax.set_ylabel(r'$x_0(n) \, [\mu m]$')
    # plt.show()
    # plt.close()


def the_hists(num1, num2, ax, **kwargs):
    pu = pd.read_csv(kwargs['processed_data']('Maryam_LongTraces') + 'physical_units.csv')

    sns.histplot(data=pu, x='length_birth', color='grey', ax=ax, stat='density', fill=True, kde=True)
    sns.histplot(data=pu[pu.lineage_ID == num1], x='length_birth', color='blue', ax=ax, stat='density', fill=1, kde=True)
    sns.histplot(data=pu[pu.lineage_ID == num2], x='length_birth', color='orange', ax=ax, stat='density', fill=1, kde=True)
    ax.set_xlim([1, 5])
    ax.set_xlabel(r'$x_0$')


def main(**kwargs):
    # ta = pd.read_csv(kwargs['processed_data']('Maryam_LongTraces') + 'time_averages.csv').drop(
    #     columns=['generation']).drop_duplicates()
    # pu = pd.read_csv(kwargs['processed_data']('Maryam_LongTraces') + 'physical_units.csv')
    #
    # print(pu.columns)
    # print(ta.columns)
    #
    # sns.scatterplot(data=ta, x='max_gen', y='lineage_ID')
    # plt.show()
    # plt.close()
    #
    # exit()

    seaborn_preamble()

    scale = 1

    fig, axes = plt.subplots(2, 2, figsize=(6.5 * scale, 6.5 * scale))  # figsize=()

    num1 = 5  # 5 & 28 or 29
    num2 = 29

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
    plot_neighbor_raw_data(axes[0, 1], num1, num2, **kwargs)
    plot_nl_length_birth_in_gentime_and_avg(num1, num2, axes[1, 0], **kwargs)
    the_hists(num1, num2, axes[1, 1], **kwargs)

    plt.tight_layout()
    plt.savefig(f'NewIntroductoryFigure{slash}fig1.png', dpi=300)
    # plt.show()
    plt.close()


if __name__ == '__main__':
    main()
