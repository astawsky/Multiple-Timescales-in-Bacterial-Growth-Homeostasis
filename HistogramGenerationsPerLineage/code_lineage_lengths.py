#!/usr/bin/env bash

from AnalysisCode.global_variables import dataset_names, create_folder, wang_datasets, slash
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os


def run_it(args, **kwargs):
    physical_units = pd.read_csv(args['pu'])
    
    lengths = []
    for lineage_id in physical_units.lineage_ID.unique():
        trace = physical_units[physical_units['lineage_ID'] == lineage_id].copy()
        lengths.append(len(trace))
    
    # stylistic reasons
    sns.set_context('paper')
    sns.set_style("ticks", {'axes.grid': True})
    
    # fig, ax = plt.subplots(tight_layout=True, figsize=[3, 3])
    
    sns.displot(data=lengths, label=r'${}$ lineages'.format(len(lengths)) + '\n' + r'$\sim {} \pm {}$ long'.format(np.int(np.mean(lengths)), np.int(np.std(lengths))))
    plt.xlabel('lineage lengths')
    plt.ylabel('PDF')
    plt.title(args['data_origin'])
    plt.legend(title='')
    plt.tight_layout()
    plt.savefig('PDF{}{}{}.png'.format(slash, args['data_origin'], kwargs['noise_index']), dpi=300)
    plt.close()
    
    sns.displot(data=lengths, label=r'${}$ lineages'.format(len(lengths)) + '\n' + r'$\sim {} \pm {}$ long'.format(np.int(np.mean(lengths)), np.int(np.std(lengths))), kind="ecdf")
    plt.xlabel('lineage lengths')
    plt.ylabel('PDF')
    plt.title(args['data_origin'])
    plt.legend(title='')
    plt.tight_layout()
    plt.savefig('CDF{}{}{}.png'.format(slash, args['data_origin'], kwargs['noise_index']), dpi=300)
    plt.close()
    

def main(**kwargs):

    create_folder('PDF')
    create_folder('CDF')

    for data_origin in dataset_names:

        """
                data_origin ==> Name of the dataset we are analysing
                raw_data ==> Where the folder containing the raw data for this dataset is
                processed_data ==> The folder we will put the processed data in
                """
        arguments = {
            'data_origin': data_origin,
            'pu': kwargs["without_outliers"](data_origin) + f'{slash}physical_units_without_outliers.csv' if data_origin in wang_datasets else kwargs["without_outliers"](data_origin) + f'{slash}physical_units.csv'
        }

        run_it(arguments, **kwargs)
