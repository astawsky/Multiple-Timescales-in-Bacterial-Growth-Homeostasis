#!/usr/bin/env bash

import pandas as pd
import numpy as np
import glob
import matplotlib.pyplot as plt
import os
from scipy.signal import find_peaks
from sklearn.linear_model import LinearRegression
from AnalysisCode.global_variables import (
    phenotypic_variables, create_folder, dataset_names, sm_datasets, wang_datasets, tanouchi_datasets,
    get_time_averages_df, check_the_division, slash
)

""" Input a dataframe with the phenotypic variables we want to include in the outgoing dataframe and center each lineage by its time-average """


def center_the_lineages(info, the_variables):
    tc = info.copy()
    for lineage_id in info['lineage_ID'].unique():
        trace = info[(info['lineage_ID'] == lineage_id)].copy()
        tc.loc[trace.index, the_variables] = trace[the_variables] - trace[the_variables].mean()
    
    tc = tc.sort_values(['lineage_ID', 'generation']).reset_index(drop=True)  # Put it in the right order
    
    return tc


""" Gets rid of the machine error from the signal as well as possible """


def clean_up(lineage, window_size_of_interest=200):
    """ Gets rid of the remaining discrepancies """
    
    def recursive_rise_exclusion(lin, totals, rises):
        diff = np.log(lin['length'].values[:-1]) - np.log(
            lin['length'].values[1:])  # What is the difference between the natural logs of two consecutive length measurements across one unit of time?
        
        new_rises = np.where(diff <= -.5)[0]  # Here is where we check if it went up dramatically
        
        new_new = []
        for rise in new_rises:  # for each rise, check if it is followed by an outlier in the next point or not, in which case delete the point before the rise
            # window of interest to see if it is an outlier in that window
            if len(lin) <= window_size_of_interest:  # The raw lineage is too small
                woi = [0, len(lin)]
            elif rise <= window_size_of_interest:  # The raw lineage is big enough but rise is too close to the start of the lineage
                woi = [0, rise + (window_size_of_interest - rise)]
            elif len(lin) - rise <= window_size_of_interest:  # The raw lineage is big enough but rise is too close to the end of the lineage
                woi = [len(lin) - window_size_of_interest, len(lin)]
            else:  # No complications with window size
                woi = [rise - window_size_of_interest, rise + window_size_of_interest]
            
            if lin['length'].iloc[rise + 1] > (lin['length'].iloc[woi[0]:woi[1]].mean() + 2.5 * lin['length'].std()):  # If the next point is abnormally large
                new_new.append(rise + 1)  # replace that point with the one after it, ie. the outlier
            else:
                new_new.append(rise)
        
        new_rises = np.array(new_new)
        
        # if len(new_new) > 0:
        #     plt.plot(lin['length'].values)
        #     plt.scatter(new_rises, lin['length'].values[new_rises], color='brown')
        #     # plt.xlim([2500, 3200])
        #     plt.tight_layout()
        #     plt.show()
        #     plt.close()
        
        rises = np.append(rises, [int(l + np.sum([l >= old for old in totals])) for l in new_rises]).flatten()  # Re-align these point to the raw_data
        
        if len(new_rises) > 0:  # Say we find some abnormal rise
            # Fix the lineage
            new_lin = lin[~lin.index.isin(new_rises)].reset_index(drop=True)
            
            # add to the total indices that are excluded
            new_totals = np.append(totals,
                                   [int(l + np.sum([l >= old for old in totals])) for l in new_rises]).flatten()  # Add the rises indices to the total indices we will ignore for analysis
            
            # Recursion
            lin, totals, rises = recursive_rise_exclusion(new_lin, new_totals, rises)
        
        return [lin, totals, rises]
    
    """ Get rid of the non-positive lengths first """
    total_nans = np.array([])  # Where we will keep all the indices that will be taken out of the corrected lineage
    non_positives = np.array(lineage[lineage['length'] <= 0].index)  # Where the length is non-positive, which is impossible
    
    if len(non_positives) > 0:  # Notify the user and take the non-positives out
        # print('NOTE: This lineage has a length lower than or equal to 0.')
        lineage = lineage[~lineage.index.isin(non_positives)].reset_index(drop=True)  # Goes without saying but there was an instance of this in the Wang Data
    
    total_nans = np.append(total_nans, non_positives).flatten()  # Add the non-positives to the total indices we will ignore for analysis
    
    lineage = lineage.reset_index(drop=True)  # for good-practice
    
    """ Get rid of the single-point singularities first """
    diff = np.log(lineage['length'].values[:-1]) - np.log(
        lineage['length'].values[1:])  # What is the difference between the natural logs of two consecutive length measurements across one unit of time?
    
    # Take out all the singularities
    straight_up = np.where(diff <= -.4)[0]  # These are the points whose next point in time shoot up abnormally
    straight_down = np.where(diff >= .4)[0]  # These are the points whose next point in time fall down abnormally
    
    singularities = np.array([int(down) for up in straight_up for down in straight_down if (down - up == 1)])
    # singularities = np.array([int(down) for up in straight_up for down in straight_down for sep in np.arange(1, 6) if (down - up == sep)])  # Singularities that fall
    singularities = np.append(singularities, np.array([int(up) for up in straight_up for down in straight_down if (up - down == 1)])).flatten()  # Singularities that rise
    # singularities = np.append(singularities, np.array([int(up) + 1 for up in straight_up for down in straight_down if (up - down == 2)])).flatten()  # Singularities that rise
    
    if len(singularities) > 0:  # Notify the user and take the non-positives out
        # print('NOTE: This lineage has singularities that either rise or fall abnormally rapidly.')
        lineage = lineage[~lineage.index.isin(singularities)].reset_index(drop=True)  # Goes without saying but there was an instance of this in the Wang Data
    
    singularities = np.array([int(l + np.sum([l >= old for old in total_nans])) for l in singularities])
    
    total_nans = np.append(total_nans, singularities).flatten()  # Add the singularities to the total indices we will ignore for analysis
    
    """ Get rid of the remaining singularities recursively """
    new_lineage, new_total_nans, failures = recursive_rise_exclusion(lineage, total_nans, rises=np.array([]))
    
    assert len(lineage) >= len(new_lineage)
    assert len(total_nans) <= len(new_total_nans)
    assert (len(singularities) + len(non_positives) + len(failures)) == len(new_total_nans)
    
    return [new_lineage, new_total_nans, failures, singularities, non_positives]


""" Recognizes where division events occur """


def get_division_indices(raw_trace):
    # From the raw data, we see where the difference between two values of length
    # falls drastically, suggesting a division has occurred.
    diffs = -np.diff(np.log(raw_trace[np.where(~np.isnan(raw_trace))]))
    
    peaks, _ = find_peaks(diffs, threshold=np.log(1.3))
    
    start_indices = np.append([0], peaks[:-1] + 1)
    end_indices = peaks
    
    # If the first cycle is too small to be a cycle
    if start_indices[1] - start_indices[0] < 5:
        start_indices = start_indices[1:]
        end_indices = end_indices[1:]
    
    # Make sure they are the same size
    assert len(start_indices) == len(end_indices)
    
    # plt.plot(raw_trace[np.where(~np.isnan(raw_trace))])
    # plt.scatter(start_indices, raw_trace[np.where(~np.isnan(raw_trace))][start_indices], color='green')
    # plt.scatter(end_indices, raw_trace[np.where(~np.isnan(raw_trace))][end_indices], color='red')
    # # plt.axvline()
    # plt.show()
    # plt.close()
    
    return [start_indices, end_indices]


"""We use this for linear regression in the cycle parameter process """


def get_the_physiological_variables(clean_lin, start_indices, end_indices, lin_id, fit_the_lengths):
    # the dataframe for our variables
    cycle_variables_lineage = pd.DataFrame(columns=phenotypic_variables + ['lineage_ID', 'generation'])
    
    # Check that there are no nan values in our data
    if clean_lin.isnull().values.any():
        print('NaNs found!')
        print(clean_lin[clean_lin.isnull()])
        exit()
    
    # pick out the time and length of the clean data
    rl = clean_lin[['time', 'length']].copy().dropna()
    
    assert len(start_indices) == len(end_indices)
    
    # plt.plot(clean_lin['time'], clean_lin['length'], color='green')
    # For every growth cycle fit an exponential
    for start, end, gen in zip(start_indices, end_indices, np.arange(len(start_indices), dtype=int)):
        
        assert end <= len(rl)  # Make sure the end index is within the bounds of the time-series
        
        # domain = x, image = y, for the linear regression y=ax+b
        domain = (clean_lin['time'].iloc[start: end + 1].copy().values - clean_lin['time'].iloc[start]).reshape(-1, 1)
        image = np.log(rl['length'].iloc[start:end + 1].values).reshape(-1, 1)  # the end+1 is due to indexing
        
        # If there are no points to regress over say it and skip this generation
        if len(domain) == 0:
            print('Only NaNs')
            # Take this out
            start_indices = start_indices[np.where(start_indices != start)]
            end_indices = end_indices[np.where(end_indices != end)]
            continue
        
        # Make sure they are the same size for the regression!
        if len(domain) != len(image):
            print(len(clean_lin[['time', 'length']].iloc[start: end]))
            print(len(domain), len(image))
            exit()
        
        # If there is a negative value show it! Gross Error in that case!
        if (rl['length'].iloc[start:end].values <= 0).any():
            print(rl['length'].iloc[start:end])
            plt.plot(np.arange(start, end), rl['length'].iloc[start:end], ls='--')
            plt.plot(rl['length'])
            plt.show()
            plt.close()
        
        # our regression
        try:
            reg = LinearRegression().fit(domain, image)
        except:
            print('exception to the linear regression!')
            print(domain)
            print(image)
            exit()
        
        # If the growth rate is non-positive then it is obviously not a credible cycle, probably a glitch
        if (reg.coef_[0][0] <= 0):
            print('Negative Growth rate in cycle! Not counted! {}'.format(reg.coef_[0][0]))
            
            # Take this out of the start, end indices
            start_indices = start_indices[np.where(start_indices != start)]
            end_indices = end_indices[np.where(end_indices != end)]
            continue
        
        # If there are too many NaNs (more than 2/3rds) in the cycle then skip this generation
        if (clean_lin.iloc[start:end + 1].count()['length'] <= ((2 / 3) * len(clean_lin.iloc[start:end + 1]))):
            print('A lot of NaNs! Not counted!')
            # Take this out
            start_indices = start_indices[np.where(start_indices != start)]
            end_indices = end_indices[np.where(end_indices != end)]
            continue
        
        # the phenotypic variables of a cycle
        cycle = pd.Series()
        
        # Define the cycle phenotypic variables
        cycle['generationtime'] = rl['time'].iloc[end] - rl['time'].iloc[start]  # The inter-division time
        cycle['growth_rate'] = reg.coef_[0][0]  # The slope, single-cell growth rate
        cycle['fold_growth'] = cycle['growth_rate'] * cycle['generationtime']
        
        # Categorical labels to identify lineage and a cycle in it
        cycle['lineage_ID'] = int(lin_id)
        cycle['generation'] = gen
        
        # Do the length at birth and length at division come straight from the data or from the regression?
        if fit_the_lengths:
            # phenotypic variables
            cycle['length_birth'] = np.exp(reg.predict(domain[0].reshape(-1, 1))[0][0])
            cycle['length_final'] = np.exp(reg.predict(domain[-1].reshape(-1, 1))[0][0])
        else:
            # phenotypic variables
            cycle['length_birth'] = np.exp(image[0][0])
            cycle['length_final'] = np.exp(image[-1][0])
        
        # We define the division ratio as how much length a cell received from its mother,
        # since the first generation has no recorded mother the value will be a NaN.
        if len(cycle_variables_lineage) == 0:
            cycle['division_ratio'] = np.nan
        else:
            cycle['division_ratio'] = cycle['length_birth'] / cycle_variables_lineage['length_final'].iloc[-1]
        
        # After defining the lengths, get the added length
        cycle['added_length'] = cycle['length_final'] - cycle['length_birth']
        
        # Add them to the cycle variables of the lineage
        cycle_variables_lineage = cycle_variables_lineage.append(cycle, ignore_index=True)
    
    # The experimental size variables
    cycle_variables_lineage['div_and_fold'] = cycle_variables_lineage['division_ratio'] * np.exp(cycle_variables_lineage['fold_growth'])
    
    cycle_variables_lineage = cycle_variables_lineage.sort_values('generation')  # Good practice
    
    without_nans = cycle_variables_lineage.copy().reset_index(drop=True)  # Without throwing away outliers
    
    # Are YOU inside 3 standard deviations inside your distribution?
    insider_condition = np.abs(cycle_variables_lineage[phenotypic_variables] - cycle_variables_lineage[phenotypic_variables].mean()) < (3 * cycle_variables_lineage[phenotypic_variables].std())
    
    # Throwing away (lineage context) outliers
    cycle_variables_lineage[phenotypic_variables] = cycle_variables_lineage[phenotypic_variables].where(
        insider_condition,
        other=np.nan
    )
    
    # Start and end indices that belong to outliers
    outlier_start_indices = start_indices[[True if row.any() else False for row in ~insider_condition.values]]
    outlier_end_indices = end_indices[[True if row.any() else False for row in ~insider_condition.values]]  # [~insider_condition]
    
    # make the lineage ID and the generation ID integers
    cycle_variables_lineage['lineage_ID'] = int(lin_id)
    cycle_variables_lineage['generation'] = np.arange(len(cycle_variables_lineage), dtype=int)
    
    cycle_variables_lineage = cycle_variables_lineage.sort_values('generation').reset_index(drop=True)  # Good practice
    
    return [cycle_variables_lineage, without_nans, start_indices, end_indices, outlier_start_indices, outlier_end_indices]  # start_indices, end_indices


"""
1. Detect mistakes in the measurements of the machine, ie. extremely sharp falls or rises.
2. Create a dataframe with physiological variables for each generation in the lineage
3. Create a dataframe that contains all the division indices and the indices of measurement mistakes
"""


def deal_with_indices(lineage, lineage_id, indices_and_mistakes, check):
    raw_lineage = lineage.copy()  # Copy the lineage before we edit it
    
    lineage, total_nans, rises, singularities, non_positives = clean_up(lineage)  # Detect the mistakes within the cycle
    
    start_indices, end_indices = get_division_indices(lineage['length'].values)  # Figure out the indices for the division events
    
    # add the cycle variables to the overall dataframe
    cycle_variables_lineage, with_outliers, start_indices, end_indices, outsider_start_indices, outsider_end_indices = \
        get_the_physiological_variables(
            lineage, start_indices, end_indices, int(lineage_id), fit_the_lengths=True
        )
    
    # The outliers start and end growth cycle indices
    outsider_start_indices = [int(
        start + np.sum([start >= l for l in rises]) + np.sum([start >= l for l in singularities]) + np.sum([start >= l for l in non_positives])
    ) for start in outsider_start_indices]
    outsider_end_indices = [int(
        end + np.sum([end >= l for l in rises]) + np.sum([end >= l for l in singularities]) + np.sum([end >= l for l in non_positives])
    ) for end in outsider_end_indices]
    
    # The start and end growth cycle indices
    raw_start = [int(
        start + np.sum([start >= l for l in rises]) + np.sum([start >= l for l in singularities]) + np.sum([start >= l for l in non_positives])
    ) for start in start_indices]
    raw_end = [int(
        end + np.sum([end >= l for l in rises]) + np.sum([end >= l for l in singularities]) + np.sum([end >= l for l in non_positives])
    ) for end in end_indices]
    
    # Add the start and end times
    cycle_variables_lineage['start_time'], with_outliers['start_time'] = raw_lineage['time'].iloc[raw_start].values, raw_lineage['time'].iloc[raw_start].values
    cycle_variables_lineage['end_time'], with_outliers['end_time'] = raw_lineage['time'].iloc[raw_end].values, raw_lineage['time'].iloc[raw_end].values
    
    # Add the indices to their dataframe
    for value, label in zip([outsider_start_indices, outsider_end_indices, raw_start, raw_end, rises, singularities, non_positives],
                            ['outsider', 'outsider', 'start', 'end', 'rises', 'singularities', 'non_positives']):
        for v in value:
            indices_and_mistakes = indices_and_mistakes.append({
                'value': v,
                'type': label,
                'lineage_ID': lineage_id
            }, ignore_index=True)
    
    if check:
        if np.array([l != 0 for l in [len(total_nans), len(non_positives), len(singularities), len(rises)]]).any():
            print(f'{len(total_nans)} number of ignored points,'
                  f'\n{len(non_positives)} number of non-positives,'
                  f'\n{len(singularities)} number of singularities,'
                  f'\n{len(rises)} number of long-failures')
    
    return [cycle_variables_lineage, with_outliers, indices_and_mistakes]


"""
Saves the dataframes with and without outliers to a .csv file for physical units, trace-centered, and time-averaged quantities
"""


def save_dataframes(some_class):
    some_class.raw_data.to_csv(os.path.dirname(os.path.dirname(some_class.args['raw_data'])) + '/raw_data_all_in_one.csv', index=False)  # Save the raw data
    
    some_class.cycle_variables.to_csv(some_class.args['without_outliers'] + '/physical_units_without_outliers.csv', index=False)  # Save the physiological variables
    
    trace_centered = center_the_lineages(some_class.cycle_variables, phenotypic_variables)
    trace_centered.to_csv(some_class.args['without_outliers'] + '/trace_centered_without_outliers.csv', index=False)  # Save the trace centered dataframe
    
    time_averages = get_time_averages_df(some_class.cycle_variables, phenotypic_variables)
    time_averages.to_csv(some_class.args['without_outliers'] + '/time_averages_without_outliers.csv', index=False)  # Save the time averaged dataframe
    
    # Now do the ones with filamentation and outliers #
    
    some_class.with_outliers_cycle_variables.to_csv(some_class.args['processed_data'] + '/physical_units.csv', index=False)  # Save the physiological variables
    
    trace_centered = center_the_lineages(some_class.with_outliers_cycle_variables, phenotypic_variables).reset_index(drop=True)
    trace_centered.to_csv(some_class.args['processed_data'] + '/trace_centered.csv', index=False)  # Save the trace centered dataframe
    
    time_averages = get_time_averages_df(some_class.with_outliers_cycle_variables.sort_values(['lineage_ID', 'generation']).reset_index(drop=True), phenotypic_variables)
    time_averages.to_csv(some_class.args['processed_data'] + '/time_averages.csv', index=False)  # Save the time averaged dataframe
    
    # Save the raw indices that are supposed to be highlighted
    some_class.raw_indices.reset_index(drop=True).sort_values(['lineage_ID']).to_csv(some_class.args['processed_data'] + '/raw_indices_processing.csv', index=False)


""" Create the csv files for physical, trace-centered, and trap-centered units for SM data """


class MM:
    def __init__(self, args):
        
        def get_raw_lineage_and_time_step():
            # Import the data in a file into a pandas dataframe with the correct extension and pandas function
            if self.args['data_origin'] == 'MG1655_inLB_LongTraces':
                time_step = 5 / 60
                # The only file that has an extra column
                if file == 'pos4-4':
                    print('In this particular case the lineage divides after the first time-point and it has an extra column.')
                    # Extra column when importing
                    lineage = pd.read_csv(file, delimiter='\t', names=['_', 'time', 'length', 'something similar to length', 'something protein', 'other protein'])[['time', 'length']]
                    # lineage divides after the first time-point
                    lineage = lineage.iloc[1:]
                else:  # All the rest don't have these problems
                    lineage = pd.read_csv(file, delimiter='\t', names=['time', 'length', 'something similar to length', 'something protein', 'other protein'])[['time', 'length']]
                lineage['time'] = lineage['time'] * (5 / 3)  # Incorrect labelling
            elif self.args['data_origin'] == 'Maryam_LongTraces':
                time_step = 3 / 60
                # This is because some the data is in .xls format while others are in .csv
                if extension == 'csv':
                    lineage = pd.read_csv(file, names=['time', 'length'])[['time', 'length']].dropna(axis=0)
                elif extension == 'xls':
                    lineage = pd.read_excel(file, names=['time', 'length'])[['time', 'length']].dropna(axis=0)
                else:
                    raise IOError('For MaryamLongTraces dataset non-xls/csv files have not been inspected!')
            elif self.args['data_origin'] in tanouchi_datasets:
                lineage = pd.read_csv(file, delimiter=',', names=['time', 'division_flag', 'length', 'fluor', 'avg_fluor'])
                lineage['time'] = (lineage['time'] - 1) / 60  # Because we map the index to the correct time-step-size which is 1 minute
                time_step = 1 / 60  # one-minute measurements!
            elif self.args['data_origin'] == 'lambda_lb':
                # There are quite a lot of files with an extra column at the beginning  Rawdata/pos17-1, pos17-1
                if filename in extra_column:
                    lineage = pd.read_csv(file, delimiter='\t', names=['_', 'time', 'length', 'something similar to length', 'something protein', 'other protein'])[['time', 'length']]
                elif filename == 'pos15':
                    print('This one is special.')
                    lineage = pd.read_csv(file, delimiter='\t', names=['_', 'time', 'length', 'something similar to length', 'something protein', 'other protein', '__', '___', '___1'])[
                        ['time', 'length']]
                else:
                    lineage = pd.read_csv(file, delimiter='\t', names=['time', 'length', 'something similar to length', 'something protein', 'other protein'])[['time', 'length']]
                
                if filename in correct_timestamp:
                    lineage['time'] = lineage['time'] * 2
                
                time_step = max(np.unique(np.diff(lineage['time'])), key=list(np.diff(lineage['time'])).count)
                
                print('step size', time_step)
            elif self.args['data_origin'] in wang_datasets:
                lineage = pd.read_csv(file, delimiter=' ')  # names=['time', 'division_flag', 'length', 'width', 'area', 'yfp_intensity', 'CMx', 'CMy']
                
                # Sometimes the time index is called time and sometimes its called index
                if 'time' in lineage.columns:
                    lineage = lineage.rename(columns={'division': 'division_flag'})[['time', 'division_flag', 'length']]
                elif 'index' in lineage.columns:
                    lineage = lineage.rename(columns={'index': 'time', 'division': 'division_flag'})[['time', 'division_flag', 'length']]
                
                # We have to make sure that the indices are monotonically increasing by 1 in order to trust the time axis
                if len(np.unique(np.diff(lineage['time']))) != 1 or np.unique(np.diff(lineage['time']))[0] != 1:
                    # print(np.unique(np.diff(lineage['time'])))
                    # raise IOError('time given in Wang data is not monotonous and increasing.')
                    print('the time given in the data file is not increasing always by 1')  # so we do not know how to measure time for this lineage we will not use it.')
                    check = True
                    # continue
                
                lineage['time'] = (lineage['time']) / 60  # Because we map the index to the correct time-step-size which is 1 minute
                
                lineage['length'] = (lineage['length']) * 0.0645  # Convert it from pixel length to micrometers
                
                time_step = 1 / 60  # one-minute measurements!
                
                # crop parts of the raw data because of measurement errors
                if self.args['data_origin'] == '20090702_E_coli_MG1655_(CGSC_6300)_Wang2010':
                    if (filename == 'xy08_ch2_cell0'):
                        lineage = lineage.drop([1175, 1176, 1177, 1178], axis=0)
                    elif (filename == 'xy04_ch0_cell0'):
                        lineage = lineage.drop([995], axis=0)
                    elif (filename == 'xy10_ch6_cell0'):
                        lineage = lineage.drop([1034, 1035], axis=0)
                    elif (filename == 'xy09_ch7_cell0'):
                        lineage = lineage.iloc[:2628]
                    elif (filename == 'xy03_ch16_cell0'):
                        lineage = lineage.drop([701, 702], axis=0)
                        lineage = lineage.iloc[:2165]
                    elif (filename == 'xy05_ch10_cell0'):
                        lineage = lineage.drop([2326, 2327, 2328], axis=0)
                    elif (filename == 'xy10_ch10_cell0'):
                        lineage = lineage.iloc[:2451]
                        lineage = lineage.drop([1603, 1604], axis=0)
                        lineage = lineage.drop([1588, 1589, 1590], axis=0)
                        lineage = lineage.drop([1002, 1003], axis=0)
                        lineage = lineage.drop(np.arange(2015, 2111), axis=0)
                    elif (filename == 'xy06_ch3_cell0'):
                        lineage = lineage.drop([1467, 1468, 1547, 1548], axis=0)
                    elif (filename == 'xy09_ch10_cell0'):
                        lineage = lineage.drop([1427, 1428], axis=0)
                    elif (filename == 'xy01_ch4_cell0'):
                        lineage = lineage.drop(np.arange(1565, 1571), axis=0)
                    elif (filename == 'xy01_ch2_cell0'):
                        lineage = lineage.drop(np.arange(1039, 1045), axis=0)
                        lineage = lineage.drop([544, 545, 546], axis=0)
                    elif (filename == 'xy02_ch9_cell0'):
                        lineage = lineage.drop([336, 337, 338], axis=0)
                    elif (filename == 'xy04_ch6_cell0'):
                        lineage = lineage.iloc[:2628]
                    elif (filename == 'xy09_ch17_cell0'):
                        lineage = lineage.iloc[:2367]
                    elif (filename == 'xy06_ch2_cell0'):
                        lineage = lineage.drop([772, 773], axis=0)
                    elif (filename == 'xy06_ch8_cell0'):
                        lineage = lineage.iloc[:2743]
                    elif (filename == 'xy09_ch16_cell0'):
                        lineage = lineage.iloc[:2073]
                    elif (filename == 'xy06_ch4_cell0'):
                        lineage = lineage.drop([2312, 2310], axis=0)
                    elif (filename == 'xy10_ch2_cell0'):
                        lineage = lineage.iloc[:1834]
                    elif (filename == 'xy09_ch8_cell0'):
                        lineage = lineage.iloc[:2042]
                    elif (filename == 'xy09_ch3_cell0'):
                        lineage = lineage.drop([1565, 1566], axis=0)
                        lineage = lineage.drop(np.arange(1481, 1486), axis=0)
                        lineage = lineage.drop([1420, 1421, 1442, 1443], axis=0)
                        lineage = lineage.drop([350, 351, 352, 353], axis=0)
                    elif (filename == 'xy10_ch4_cell0'):
                        lineage = lineage.iloc[:2686]
                    elif (filename == 'xy09_ch13_cell0'):
                        lineage = lineage.drop([2324, 2325], axis=0)
                    elif (filename == 'xy10_ch15_cell0'):
                        lineage = lineage.drop(np.arange(1987, 1995), axis=0)
                    elif (filename == 'xy04_ch15_cell0'):
                        lineage = lineage.drop([962], axis=0)
                    elif (filename == 'xy10_ch9_cell0'):
                        lineage = lineage.drop([2159, 2160], axis=0)
                    if (filename == 'xy05_ch12_cell0'):
                        lineage = lineage.drop(np.arange(611, 616), axis=0)
                    if (filename == 'xy09_ch14_cell0'):
                        lineage = lineage.drop(np.arange(1307, 1311), axis=0)
                    if (filename == 'xy04_ch13_cell0'):
                        lineage = lineage.drop(np.arange(2604, 2625), axis=0)
                        lineage = lineage.drop(np.arange(1487, 1538), axis=0)
                    if (filename == 'xy06_ch6_cell0'):
                        lineage = lineage.drop(np.arange(1121, 1123), axis=0)
                    if (filename == 'xy03_ch3_cell0'):
                        lineage = lineage.drop([1160, 1161], axis=0)
                    if (filename == 'xy08_ch14_cell0'):
                        lineage = lineage.drop(np.arange(710, 719), axis=0)
                if self.args['data_origin'] == '20090131_E_coli_MG1655_(CGSC_6300)_Wang2010':
                    if (filename == 'xy13c1_ch0_cell0'):
                        lineage = lineage.iloc[:2992]
                    elif (filename == 'xy07c1_ch0_cell0'):
                        lineage = lineage.drop(np.arange(2794, 2944), axis=0)
                if self.args['data_origin'] == '20090525_E_coli_MG1655_(CGSC_6300)_Wang2010':
                    if (filename == 'xy13_ch7_cell0_YFP0002'):
                        lineage = lineage.iloc[:319]
                    elif (filename == 'xy09_ch5_cell0_YFP0002'):
                        lineage = lineage.iloc[:592]
                    elif (filename == 'xy11_ch0_cell0_YFP0001'):
                        lineage = lineage.drop(np.arange(1094, 1294), axis=0)
                    elif (filename == 'xy07_ch0_cell0_YFP0002'):
                        lineage = lineage.iloc[:969]
                    elif (filename == 'xy02_ch14_cell0_YFP0001'):
                        lineage = lineage.iloc[:2118]
                    elif (filename == 'xy08_ch5_cell0_YFP0001'):
                        lineage = lineage.iloc[:2993]
                    elif (filename == 'xy02_ch3_cell0_YFP0002'):
                        lineage = lineage.drop(np.arange(884, 891), axis=0)
                        lineage = lineage.drop(np.arange(783, 800), axis=0)
                    elif (filename == 'xy06_ch4_cell0_YFP0002'):
                        lineage = lineage.drop(np.arange(73, 78), axis=0)
                    elif (filename == 'xy14_ch3_cell0_YFP0002'):
                        lineage = lineage.iloc[:277]
                    elif (filename == 'xy06_ch5_cell0_YFP0001'):
                        lineage = lineage.drop(np.arange(295, 297), axis=0)
                if self.args['data_origin'] == '20090512_E_coli_MG1655_(CGSC_6300)_Wang2010':
                    if (filename == 'xy05_ch8_cell0'):
                        lineage = lineage.drop(np.arange(1122, 1126), axis=0)
                    if (filename == 'xy11_ch9_cell0'):
                        lineage = lineage.drop([540, 541], axis=0)
                    if (filename == 'xy04_ch11_cell0'):
                        lineage = lineage.drop([1377, 1378], axis=0)
                    if (filename == 'xy05_ch19_cell0'):
                        lineage = lineage.drop([699, 700], axis=0)
                    if (filename == 'xy09_ch8_cell0'):
                        lineage = lineage.drop(np.arange(1515, 1520), axis=0)
                    if (filename == 'xy05_ch12_cell0'):
                        lineage = lineage.drop(np.arange(1322, 1325), axis=0)
            else:
                raise IOError('This code is not meant to run the data inputted. Please label the data and put it in as an if-statement.')
            
            return [lineage, time_step]
        
        self.args = args
        
        self.file_paths = sorted(glob.glob(self.args['raw_data'] + f'{slash}*'))  # The names of all the data files in an array to loop over and process
        
        if self.args['check']:
            print('type of MM data we are processing:', args['data_origin'])
            print(self.file_paths)
        
        self.raw_data = pd.DataFrame(columns=['time', 'length', 'lineage_ID', 'filename'])  # Dataframe containing the measurements from the machine
        self.raw_indices = pd.DataFrame(columns=['value', 'type', 'lineage_ID'])  # Dataframe containing the division events and mistakes in the measurements
        
        self.cycle_variables = pd.DataFrame(columns=phenotypic_variables + ['lineage_ID', 'generation'])  # The physiological variables for each cell dataframe
        self.with_outliers_cycle_variables = pd.DataFrame(columns=phenotypic_variables + ['lineage_ID', 'generation'])  # The physiological variables for each cell dataframe, including outliers
        
        # The files in lambda_lb file that contain an extra column in the .txt file. It caused a problem before taking it into account...
        extra_column = [
            'pos0-1',
            'pos0-1-daughter',
            'pos1',
            'pos4',
            'pos5-lower cell',
            'pos5-upper cell',
            'pos6-1-1',
            'pos6-1',
            'pos6-2',
            'pos7-1-1',
            'pos7-1-2',
            'pos7-2-1',
            'pos7-2-2',
            'pos7-3',
            'pos8',
            'pos10-1',
            'pos16-1',
            'pos16-2',
            'pos16-3',
            'pos17-1',
            'pos17-2',
            'pos17-3',
            'pos18-2',
            'pos18-3',
            'pos19-2',
            'pos19-3',
            'pos20'
        ]
        
        # Some of the experiments have a wrong time step for the lambda_lb, we rectify it here
        correct_timestamp = ['pos1-1-daughter', 'pos17-1', 'pos17-3', 'pos20', 'pos17-2', 'pos0-1',
                             'pos16-3', 'pos16-2', 'pos16-1', 'pos18-3', 'pos18-2', 'Pos9-1', 'Pos10',
                             'pos19-3', 'Pos9-2', 'pos19-2', 'pos15']
        
        offset = 0  # In case we can't use some files we want the lineage IDs to be in integer order
        
        jump = 0  # In case we want to start from some lineage that is not the first data file in the folder
        
        for count, file in enumerate(self.file_paths[jump:]):  # load first sheet of each Excel-File, fill internal data structure
            
            filename = file.split('/')[-1].split('.')[0]  # Filename of the data file
            extension = file.split('/')[-1].split('.')[1]  # Extension of the type of file
            
            if self.args['check']:
                print(count + 1 + jump, '/', str(len(self.file_paths)), ':', filename)  # Tells us the trap ID and the source (filename)
            
            lineage, time_step = get_raw_lineage_and_time_step()  # Get the raw data and the time step
            
            lineage['filename'] = filename  # for reference
            lineage['lineage_ID'] = count + 1 - offset  # Add the lineage ID
            lineage['step_size'] = time_step  # Specify what is the time step
            
            # Check if time is going backwards and discard if true
            if not all(x < y for x, y in zip(lineage['time'].values[:-1], lineage['time'].values[1:])):
                print(filename, ': Time is going backwards. We cannot use this data.')
                
                # reset the lineage_ID
                offset += 1
                continue
            
            self.raw_data = self.raw_data.append(lineage, ignore_index=True)  # add it to the raw data
            
            self.remove_faulty_lineages()  # If there are parts of the measurements that are botched and we can take out, then we do that here
            
            physiological_variables, physiological_variables_with_outliers, self.raw_indices = deal_with_indices(lineage, count + 1 - offset, self.raw_indices, self.args['check'])
            
            if self.args['check']:
                check_the_division(args, lineages=[count + 1 - offset], raw_lineages=lineage, raw_indices=self.raw_indices, pu=physiological_variables)
            # check_the_division(args, lineages=[count + 1 - offset], raw_lineages=lineage, raw_indices=self.raw_indices, pu=physiological_variables)

            # append the physiological variables of this lineage to the bigger dataframe for all lineages in an experiment
            self.cycle_variables = self.cycle_variables.append(physiological_variables, ignore_index=True)
            self.with_outliers_cycle_variables = self.with_outliers_cycle_variables.append(physiological_variables_with_outliers, ignore_index=True)
        
        if self.args['check']:
            print('processed data:\n', self.cycle_variables)
            print('cleaned raw data:\n', self.raw_data)
        
        create_folder(self.args['without_outliers'])  # create the necessary folders, just in case it is not created already
        
        # save_dataframes(self)  # Save the dataframe to .csv file
    
    def remove_faulty_lineages(self):
        if self.args['data_origin'] == '20090529_E_coli_Br_SJ119_Wang2010':  # Take out the trajectories that cannot be used
            
            wang_br_sj119_200090529_reject = np.array([
                29, 33, 38, 50, 68, 76, 83, 96, 132, 138, 141, 145, 155, 158, 181, 198, 208, 220, 228, 233, 237, 240, 254, 268, 270, 276, 277, 281, 296, 299
            ]) - 1  # 172, 21, 104, 213
            
            # wang_br_sj119_200090529_cut = {104: 190}
            
            self.file_paths = np.array(self.file_paths)[[r for r in np.arange(len(self.file_paths), dtype=int) if (r not in wang_br_sj119_200090529_reject)]]
        
        if self.args['data_origin'] == '20090930_E_coli_MG1655_lexA3_Wang2010':
            wang_MG1655_lexA3_20090930_reject = np.array([
                34, 40, 42, 116
            ]) - 1
            
            self.file_paths = np.array(self.file_paths)[[r for r in np.arange(len(self.file_paths), dtype=int) if (r not in wang_MG1655_lexA3_20090930_reject)]]
        
        if self.args['data_origin'] == '20090923_E_coli_MG1655_lexA3_Wang2010':
            wang_MG1655_lexA3_20090923_reject = np.array([
                40, 133, 138
            ]) - 1
            
            self.file_paths = np.array(self.file_paths)[[r for r in np.arange(len(self.file_paths), dtype=int) if (r not in wang_MG1655_lexA3_20090923_reject)]]
        
        if self.args['data_origin'] == '20090922_E_coli_MG1655_lexA3_Wang2010':
            wang_MG1655_lexA3_20090922_reject = np.array([
                4, 40, 133, 138
            ]) - 1
            
            self.file_paths = np.array(self.file_paths)[[r for r in np.arange(len(self.file_paths), dtype=int) if (r not in wang_MG1655_lexA3_20090922_reject)]]
        
        if self.args['data_origin'] == '20090210_E_coli_MG1655_(CGSC_6300)_Wang2010':
            wang_MG1655_CGSC_6300_20090210_reject = np.array([
                6, 10
            ]) - 1
            
            self.file_paths = np.array(self.file_paths)[[r for r in np.arange(len(self.file_paths), dtype=int) if (r not in wang_MG1655_CGSC_6300_20090210_reject)]]
        
        if self.args['data_origin'] == '20090129_E_coli_MG1655_(CGSC_6300)_Wang2010':
            wang_MG1655_CGSC_6300_20090129_reject = np.array([
                6, 10
            ]) - 1
            
            self.file_paths = np.array(self.file_paths)[[r for r in np.arange(len(self.file_paths), dtype=int) if (r not in wang_MG1655_CGSC_6300_20090129_reject)]]
        
        if self.args['data_origin'] == '20090702_E_coli_MG1655_(CGSC_6300)_Wang2010':
            wang_MG1655_CGSC_6300_20090702_reject = np.array([
                43, 75, 86
            ]) - 1
            
            self.file_paths = np.array(self.file_paths)[[r for r in np.arange(len(self.file_paths), dtype=int) if (r not in wang_MG1655_CGSC_6300_20090702_reject)]]
        
        if self.args['data_origin'] == '20090131_E_coli_MG1655_(CGSC_6300)_Wang2010':
            wang_MG1655_CGSC_6300_20090131_reject = np.array([
                1, 20, 23, 41, 47, 52, 57
            ]) - 1
            
            self.file_paths = np.array(self.file_paths)[[r for r in np.arange(len(self.file_paths), dtype=int) if (r not in wang_MG1655_CGSC_6300_20090131_reject)]]
        
        if self.args['data_origin'] == '20090525_E_coli_MG1655_(CGSC_6300)_Wang2010':
            wang_MG1655_CGSC_6300_20090131_reject = np.array([
                89, 105, 112, 127, 233, 250, 318
            ]) - 1
            
            self.file_paths = np.array(self.file_paths)[[r for r in np.arange(len(self.file_paths), dtype=int) if (r not in wang_MG1655_CGSC_6300_20090131_reject)]]
        
        if self.args['data_origin'] == '20090512_E_coli_MG1655_(CGSC_6300)_Wang2010':
            wang_MG1655_CGSC_6300_20090512_reject = np.array([
                54, 64, 135, 146
            ]) - 1
            
            self.file_paths = np.array(self.file_paths)[[r for r in np.arange(len(self.file_paths), dtype=int) if (r not in wang_MG1655_CGSC_6300_20090512_reject)]]


""" Create the csv files for physical, trace-centered, and trap-centered units for SM data """


class SM:
    def __init__(self, args):
        
        self.args = args
        
        self.raw_data, self.files = self.input_raw_data()  # organize raw data
        
        self.cycle_variables, self.with_outliers_cycle_variables, self.table_of_outliers, self.raw_indices = self.process_lineages()  # attribute physiological states
        
        create_folder(self.args['without_outliers'])  # create the necessary folders, just in case it is not created already
        
        self.save_dataframes()
    
    def save_dataframes(self):
        
        self.raw_data.to_csv(os.path.dirname(os.path.dirname(self.args['raw_data'])) + '/raw_data_all_in_one.csv', index=False)  # Save the raw data
        
        self.cycle_variables.to_csv(self.args['without_outliers'] + '/physical_units_without_outliers.csv', index=False)  # Save the physiological variables
        
        trace_centered = center_the_lineages(self.cycle_variables, phenotypic_variables)
        trace_centered.to_csv(self.args['without_outliers'] + '/trace_centered_without_outliers.csv', index=False)  # Save the trace centered dataframe
        
        time_averages = get_time_averages_df(self.cycle_variables, phenotypic_variables)
        time_averages.to_csv(self.args['without_outliers'] + '/time_averages_without_outliers.csv', index=False)  # Save the time averaged dataframe
        
        # Now do the ones with filamentation and outliers #
        
        self.with_outliers_cycle_variables.to_csv(self.args['processed_data'] + '/physical_units.csv', index=False)  # Save the physiological variables
        
        trace_centered = center_the_lineages(self.with_outliers_cycle_variables, phenotypic_variables).reset_index(drop=True)
        trace_centered.to_csv(self.args['processed_data'] + '/trace_centered.csv', index=False)  # Save the trace centered dataframe
        
        time_averages = get_time_averages_df(self.with_outliers_cycle_variables.sort_values(['lineage_ID', 'generation']).reset_index(drop=True), phenotypic_variables)
        time_averages.to_csv(self.args['processed_data'] + '/time_averages.csv', index=False)  # Save the time averaged dataframe
        
        # Save the raw indices that are supposed to be highlighted
        self.raw_indices.reset_index(drop=True).sort_values(['lineage_ID']).to_csv(self.args['processed_data'] + '/raw_indices_processing.csv', index=False)
    
    def input_raw_data(self):
        # Where we will put the raw data
        raw_data = pd.DataFrame(columns=['time', 'length', 'dataset', 'trap_ID', 'trace', 'lineage_ID'])
        
        # The files that have the RawData
        files = glob.glob(self.args['raw_data'] + '/*.xls')
        
        # load first sheet of each Excel-File, fill rawdata dataframe
        for count, file in enumerate(files):
            if self.args['check']:
                print(count, file.split('/')[-1], sep=': ')
            
            # creates a dataframe from the excel file
            tmpdata = pd.read_excel(file)
            
            # Make sure there are no NaNs in the data
            assert ~tmpdata.isna().values.any()
            
            # Determine the relationship between the two lineage in a file depending on the name of the file
            if ('sis' in file.split('/')[-1]) or ('SIS' in file.split('/')[-1]):
                dataset = 'SL'
            else:
                dataset = 'NL'
            
            # Separate and categorize the traces in the trap
            if dataset == 'SL':
                a_trace = tmpdata[['timeA', 'lengthA']].rename(columns={'timeA': 'time', 'lengthA': 'length'})
                b_trace = tmpdata[['timeB', 'lengthB']].rename(columns={'timeB': 'time', 'lengthB': 'length'})
            else:
                a_trace = tmpdata[['timeA', 'L1']].rename(columns={'timeA': 'time', 'L1': 'length'})
                b_trace = tmpdata[['timeB', 'L2']].rename(columns={'timeB': 'time', 'L2': 'length'})
            
            # Are they SL or NL?
            a_trace['dataset'] = dataset
            b_trace['dataset'] = dataset
            
            # What trap are they in from the pooled ensemble?
            a_trace['trap_ID'] = (count + 1)
            b_trace['trap_ID'] = (count + 1)
            
            # Arbitrarily name the traces
            a_trace['trace'] = 'A'
            b_trace['trace'] = 'B'
            
            # Give each lineage a unique ID
            a_trace['lineage_ID'] = (count + 1) * 2 - 1
            b_trace['lineage_ID'] = (count + 1) * 2
            
            # Set the floats to be accurate to the 2nd decimal point because of the timesteps in hours
            a_trace['time'] = a_trace['time'].round(2)
            b_trace['time'] = b_trace['time'].round(2)
            
            # Check that they round to a normal 0.05 time-step
            if any([int(np.round(l * 100, 0) % 5) for l in a_trace['time'].values]) or \
                    any([int(np.round(l * 100, 0) % 5) for l in a_trace['time'].values]):
                print('Rounded Wrong to not .05 multiples')
                exit()
            
            # Check if time is going forward for the "A" trace
            time_monotony_a = all(x < y for x, y in zip(a_trace['time'].values[:-1], a_trace['time'].values[1:]))
            # Check if time is going forward for the "B" trace
            time_monotony_b = all(x < y for x, y in zip(b_trace['time'].values[:-1], b_trace['time'].values[1:]))
            
            if (not time_monotony_a) or (not time_monotony_b):
                print(file, ': Time is going backwards. We cannot use this data.')
                print("False is bad! --", "A:", time_monotony_a, "B:", time_monotony_b)
                
                continue
            
            # the data contains all dataframes from the excel files in the directory _infiles
            raw_data = raw_data.append(a_trace, ignore_index=True)
            raw_data = raw_data.append(b_trace, ignore_index=True)
        
        # There must be some data
        assert len(raw_data) > 0
        # There can't be any NaNs
        assert ~raw_data.isna().values.any()
        
        return [raw_data, files]
    
    def process_lineages(self):  # step_size = .05
        # The dataframe for our variables
        cycle_variables = pd.DataFrame()  # columns=order
        with_outliers_cycle_variables = pd.DataFrame()
        table_of_outliers = pd.DataFrame()
        raw_indices = pd.DataFrame()
        
        for lineage_id in self.raw_data.lineage_ID.unique():
            if self.args['check']:
                print('Lineage ID:', lineage_id)
            
            lineage = self.raw_data[(self.raw_data['lineage_ID'] == lineage_id)].copy()  # Identify the lineage
            
            cycle_variables_lineage, with_outliers, raw_indices = deal_with_indices(lineage, lineage_id, raw_indices, self.args['check'])  # Do the physiological variable processing
            
            # Add the SM categorical variables
            cycle_variables_lineage['trap_ID'] = lineage['trap_ID'].unique()[0]
            cycle_variables_lineage['trace'] = lineage['trace'].unique()[0]
            cycle_variables_lineage['dataset'] = lineage['dataset'].unique()[0]
            
            with_outliers['trap_ID'] = lineage['trap_ID'].unique()[0]
            with_outliers['trace'] = lineage['trace'].unique()[0]
            with_outliers['dataset'] = lineage['dataset'].unique()[0]
            
            # Append the cycle variables to the processed dataframe
            cycle_variables = cycle_variables.append(cycle_variables_lineage, ignore_index=True)
            with_outliers_cycle_variables = with_outliers_cycle_variables.append(with_outliers, ignore_index=True)
        
        if self.args['check']:
            print('processed data:\n', cycle_variables)
            print('cleaned raw data:\n', self.raw_data)
        
        # Check how much NaNs were introduced because of the zscore < 3 condition on one of the dataframes (no outliers)
        for variable in phenotypic_variables:
            if variable in ['division_ratio', 'div_and_fold']:  # , 'fold_then_div'
                """
                This is because, by definition, the first two variables have 1 NaN value at the first generation, 
                while the third variable has 1 NaN at the end of each lineage.
                """
                table_of_outliers = table_of_outliers.append({
                    'variable': variable,
                    'percentage': cycle_variables[variable].count() / (len(cycle_variables[variable]) - (1 * len(cycle_variables['lineage_ID'].unique())))
                }, ignore_index=True)
                
                if self.args['check']:
                    print(variable, cycle_variables[variable].count() / (len(cycle_variables[variable]) - (1 * len(cycle_variables['lineage_ID'].unique()))))
            else:
                table_of_outliers = table_of_outliers.append({
                    'variable': variable,
                    'percentage': cycle_variables[variable].count() / (len(cycle_variables[variable]) - (1 * len(cycle_variables['lineage_ID'].unique())))
                }, ignore_index=True)
                
                if self.args['check']:
                    print(variable, cycle_variables[variable].count() / len(cycle_variables[variable]))
        
        # reset the index for good practice
        self.raw_data = self.raw_data.reset_index(drop=True).sort_values(['lineage_ID'])
        cycle_variables = cycle_variables.reset_index(drop=True).sort_values(['lineage_ID', 'generation'])
        with_outliers_cycle_variables = with_outliers_cycle_variables.reset_index(drop=True).sort_values(['lineage_ID', 'generation'])
        
        return [cycle_variables, with_outliers_cycle_variables, table_of_outliers, raw_indices]


""" Create the csv files for physical, trace-centered, and trap-centered units for MM and SM data """
for data_origin in ['Maryam_LongTraces']:#dataset_names:  # For all the experiments process the raw data
    print(data_origin)
    
    """
    check ==> Print on the command line progress on the analysis
    data_origin ==> Name of the dataset we are analysing
    raw_data ==> Where the folder containing the raw data for this dataset is
    processed_data ==> The folder we will put the processed data in
    without_outliers ==> The path to save all the dataframes with (pooled ensemble) outliers taken out 
    """
    arguments = {
        'check': True,
        'data_origin': data_origin,
        'raw_data': os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + f'{slash}Datasets{slash}' + data_origin + f'{slash}RawData{slash}',
        'processed_data': os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + f'{slash}Datasets{slash}' + data_origin + f'{slash}ProcessedData{slash}',
        'without_outliers': os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + f'{slash}Datasets{slash}' + data_origin + f'{slash}ProcessedData{slash}z_score_under_3'
    }
    
    # Make sure the folders where we place the data are created already
    create_folder(arguments['processed_data'])
    
    if data_origin in sm_datasets:  # Get SM data
        SM(arguments)
    else:  # Get MM data
        MM(arguments)
    
    # check_the_division(arguments)  # Optional to see the fits over the raw data.
    
    if arguments['check']:
        print('*' * 200)
