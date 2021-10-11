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
    raw_indices = pd.read_csv(kwargs['processed_data']('Maryam_LongTraces') + 'raw_indices_processing.csv')

    a_last_index = raw_indices[(raw_indices.type == 'end') & (raw_indices.lineage_ID == num1)].value.values[30]

    cut_mask = (raw_lineages.time <= a_last_index)

    a_trace = raw_lineages[(raw_lineages['lineage_ID'] == num1) & cut_mask].copy().sort_values('time').reset_index(drop=True)
    b_trace = raw_lineages[(raw_lineages['lineage_ID'] == num2) & cut_mask].copy().sort_values('time').reset_index(drop=True)
    # print(a_trace)
    # print(raw_indices)
    # exit()

    ax.plot(a_trace['time'].values, a_trace['length'].values, label='A', color='green')

    # ax.plot(b_trace['time'].values, b_trace['length'].values, label='B', color='red')
    # ax.legend()


def possible_mm_lineages(**kwargs):
    # 133 ( GOOD COORDINATION ), 315, 91, 341, 311, 111 --> Possible NL lineages

    ta = pd.read_csv(kwargs['processed_data']('Maryam_LongTraces') + 'time_averages.csv').drop(
        columns=['generation']).drop_duplicates()
    pu = pd.read_csv(kwargs['processed_data']('Maryam_LongTraces') + 'physical_units.csv')

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

    print('min', ta[ta['length_birth'] == ta['length_birth'].min()].lineage_ID)
    print('max', ta[ta['length_birth'] == ta['length_birth'].max()].lineage_ID)
    exit()


def plot_nl_length_birth_in_gentime_and_avg(num1, num2, ax, **kwargs):
    ta = pd.read_csv(kwargs['processed_data']('Maryam_LongTraces') + 'time_averages.csv').drop(
        columns=['generation']).drop_duplicates()
    pu = pd.read_csv(kwargs['processed_data']('Maryam_LongTraces') + 'physical_units.csv')

    ax.plot(pu[pu['lineage_ID'] == num1].generation.values, pu[pu['lineage_ID'] == num1].length_birth.values,
            c='green', marker='.')
    ax.plot(pu[pu['lineage_ID'] == num2].generation.values, pu[pu['lineage_ID'] == num2].length_birth.values,
             c='red', marker='.')
    ax.axhline(ta[ta['lineage_ID'] == num1].length_birth.values, c='green')
    ax.axhline(ta[ta['lineage_ID'] == num2].length_birth.values, c='red')
    ax.set_xlabel('n [Gen]')
    ax.set_ylabel(r'$x_0(n) \, [\mu m]$')
    # plt.show()
    # plt.close()


def the_hists(num1, num2, ax, **kwargs):
    pu = pd.read_csv(kwargs['processed_data']('Maryam_LongTraces') + 'physical_units.csv')

    sns.histplot(data=pu, x='length_birth', color='grey', ax=ax, stat='density', fill=True, kde=True)
    sns.histplot(data=pu[pu.lineage_ID == num1], x='length_birth', color='green', ax=ax, stat='density', fill=True, kde=False)
    sns.histplot(data=pu[pu.lineage_ID == num2], x='length_birth', color='red', ax=ax, stat='density', fill=True, kde=False)
    ax.set_xlim([1.3, 4])
    ax.set_xlabel(r'$x_0 \, [\mu m]$')


def raw_data_plot(num, ax, **kwargs):
    raw_lineages = pd.read_csv(kwargs['raw_data']('Maryam_LongTraces') + 'raw_data_all_in_one.csv')
    rl = raw_lineages[raw_lineages['lineage_ID'] == num].copy().sort_values('time').reset_index(drop=True)
    sns.lineplot(data=rl, x='time', y='length', ax=ax, color=cmap[0])
    sns.scatterplot(data=rl, x='time', y='length', ax=ax, alpha=.5, color=cmap[0])
    # ax.plot(rl['time'], rl['length'], markerfacecolor=(1, 1, 0, 0.5))  # , marker='o', alpha=0.3)
    ax.set_xlabel('Time [Mins.]')
    ax.set_ylabel(r'Length $[\mu m]$')
    ax.set_xlim([0, 10])
    # ax.plot()


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

    num1 = 23 # 5  # 5 & 28 or 29
    num2 = 10 # 29

    # # From the process_rawdata.py file where it is used to visually confirm the regression as well as the division events
    # check_the_division('Pooled_SM', lineages=[1], raw_lineages=[], raw_indices=[], pu=[], ax=axes[0, 1], **kwargs)  # A trace
    # check_the_division('Pooled_SM', lineages=[2], raw_lineages=[], raw_indices=[], pu=[], ax=axes[0, 1], **kwargs)  # B trace

    # possible_mm_lineages(**kwargs)

    # plot_neighbor_raw_data(axes[0, 1], **kwargs)
    axes[0, 0].set_title('A', x=-.3, fontsize='xx-large')
    axes[0, 1].set_title('B', x=-.3, fontsize='xx-large')
    axes[1, 0].set_title('C', x=-.3, fontsize='xx-large')
    axes[1, 1].set_title('D', x=-.3, fontsize='xx-large')
    axes[0, 0].set_frame_on(False)
    axes[0, 0].set_xticks([])
    axes[0, 0].set_yticks([])

    # check_the_division('Maryam_LongTraces', lineages=[num1], raw_lineages=[], raw_indices=[], pu=[], ax=axes[0, 1], **kwargs)
    raw_data_plot(num1, axes[0, 1], **kwargs)
    # axes[0, 1].legend()
    # axes[0, 1].set_xlabel('Absolute Time (Mins.)')
    # axes[0, 1].set_ylabel(r'Length $[\mu m]$')
    # axes[0, 1].set_xlim([0, 20])

    # plot_neighbor_raw_data(axes[0, 1], num1, num2, **kwargs)
    plot_nl_length_birth_in_gentime_and_avg(num1, num2, axes[1, 0], **kwargs)
    the_hists(23, 10, axes[1, 1], **kwargs)

    plt.tight_layout()
    plt.savefig(f'NewIntroductoryFigure{slash}fig1.png', dpi=300)
    # plt.show()
    plt.close()


if __name__ == '__main__':
    main()
