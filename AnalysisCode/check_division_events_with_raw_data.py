#!/usr/bin/env bash

import pandas as pd
import numpy as np
import glob
import matplotlib.pyplot as plt
import os
from scipy.stats import zscore
from sklearn.linear_model import LinearRegression
from AnalysisCode.global_variables import phenotypic_variables, create_folder, dataset_names, sm_datasets, mm_datasets, wang_datasets, tanouchi_datasets


def main():
    import argparse
    
    # Create the arguments for this function
    parser = argparse.ArgumentParser(description='Decide which datasets to process Mother Machine and Sister Machine Raw Data for.')
    
    parser.add_argument('-dataset_names', '--dataset_names', metavar='', nargs="+", help='What is the label for this data for the Data and Figures folders?', required=False,
                        default=dataset_names)
    
    # Finalize the arguments
    input_args = parser.parse_args()
    
    # Do all the Mother and Sister Machine data
    for data_origin in wang_datasets:  # input_args.dataset_names:
        print(data_origin)
        
        """
        data_origin ==> Name of the dataset we are analysing
        raw_data ==> Where the folder containing the raw data for this dataset is
        processed_data ==> The folder we will put the processed data in
        """
        args = {
            'data_origin': data_origin,
            'raw_data': os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/Datasets/' + data_origin + '/RawData/',
            'processed_data': os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/Datasets/' + data_origin + '/ProcessedData/'
        }
        
        raw_data = pd.read_csv(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/Datasets/' + data_origin + '/raw_data_all_in_one.csv')
        pu = pd.read_csv(args['processed_data'] + 'physical_units.csv')
        
        # Are there extra?
        if not np.array_equal(raw_data.lineage_ID.unique(), pu.lineage_ID.unique()):
            print(raw_data.lineage_ID.unique(), pu.lineage_ID.unique())
            print(len(raw_data.lineage_ID.unique()), len(pu.lineage_ID.unique()))
            exit()
        
        for lin_id in pu.lineage_ID.unique():
            print(lin_id)
            plt.plot(raw_data[raw_data['lineage_ID'] == lin_id].sort_values('time')['length'].values)
            gts = pu[pu['lineage_ID'] == lin_id].sort_values('generation')['generationtime'].copy() * 60 + 1
            for gt in gts.cumsum().values:
                plt.axvline(gt, color='green')
            plt.title(data_origin + ' --- ' + str(lin_id))
            plt.show()
            plt.close()


if __name__ == '__main__':
    main()
