#!/usr/bin/env bash


import pandas as pd
import seaborn as sns
import numpy as np
import os

""" create a folder with the name of the string inputted """


def create_folder(filename):
    # create the folder if not created already
    try:
        # Create target Directory
        os.mkdir(filename)
        print("Directory", filename, "Created ")
    except FileExistsError:
        pass


""" Returns dataframe with same size containing the time-averages of each phenotypic variable instead of the local value """


def get_time_averages_df(info, phenotypic_variables):  # MM
    
    # We keep the trap means here
    means_df = pd.DataFrame(columns=['lineage_ID', 'max_gen', 'generation'] + phenotypic_variables)
    
    # specify a lineage
    for lin_id in info['lineage_ID'].unique():
        
        # the values of the lineage we get from physical units
        lineage = info[(info['lineage_ID'] == lin_id)].copy()
        
        # add its time-average
        to_add = {
            'lineage_ID': [lin_id for _ in np.arange(len(lineage))],
            'max_gen': [len(lineage) for _ in np.arange(len(lineage))], 'generation': np.arange(len(lineage))
        }
        to_add.update({param: [np.mean(lineage[param]) for _ in np.arange(len(lineage))] for param in phenotypic_variables})
        to_add = pd.DataFrame(to_add)
        means_df = means_df.append(to_add, ignore_index=True).reset_index(drop=True)
    
    assert len(info) == len(means_df)
    
    return means_df


""" shuffle the generations inside a lineage to destroy inter-generational correlations """


def shuffle_lineage_generations(df, mm):
    if mm:
        
        # This is where we will keep all the data and later save as a csv
        no_epi_df = pd.DataFrame(columns=phenotypic_variables + ['lineage_ID', 'generation'])
        
        for lin_id in df.lineage_ID.unique():
            lineage = df[(df['lineage_ID'] == lin_id)]
            
            new_lineage = lineage[phenotypic_variables].sample(frac=1, replace=False)  # .reset_index(drop=True)
            new_lineage['lineage_ID'] = lineage.lineage_ID.unique()[0]
            new_lineage[phenotypic_variables] = lineage[phenotypic_variables].sample(frac=1, replace=False)
            new_lineage['generation'] = np.arange(len(lineage))
            
            # adds it to the dataframe
            no_epi_df = no_epi_df.append(new_lineage, ignore_index=True)
    else:
        
        # This is where we will keep all the data and later save as a csv
        no_epi_df = pd.DataFrame(columns=phenotypic_variables + ['lineage_ID', 'dataset', 'trap_ID', 'trace', 'generation'])
        
        # loop for every dataset and its traps and traces
        for dataset in np.unique(df['dataset']):
            
            for trap_id in np.unique(df[(df['dataset'] == dataset)]['trap_ID']):
                
                for trace in ['A', 'B']:
                    
                    lineage = df[(df['dataset'] == dataset) & (df['trap_ID'] == trap_id) & (df['trace'] == trace)]
                    
                    new_lineage = lineage[phenotypic_variables].sample(frac=1, replace=False)  # .reset_index(drop=True)
                    new_lineage['lineage_ID'] = lineage.lineage_ID.unique()[0]
                    new_lineage['dataset'] = dataset
                    new_lineage['trap_ID'] = trap_id
                    new_lineage['trace'] = trace
                    new_lineage['generation'] = np.arange(len(lineage))
                    
                    # adds it to the dataframe
                    no_epi_df = no_epi_df.append(new_lineage, ignore_index=True)
                
                # exit()
                #
                # # With these we mask the dataframes to get the values we want, mainly a lineage's empirical time-average and standard deviation
                # trace_mask = (df['dataset'] == dataset) & (df['trap_ID'] == trap_id) & (df['trace'] == trace)
                #
                # # how many generations we have in this lineage
                # max_gen = len(df[trace_mask])
                #
                # # What we are going to add to the
                # to_add = {
                #     'dataset': [dataset for _ in range(max_gen)],
                #     'trap_ID': [trap_id for _ in range(max_gen)],
                #     'trace': [trace for _ in range(max_gen)],
                #     'generation': np.arange(max_gen)
                # }
                #
                # # shuffle the lineage generations so epigenetic MD memory is lessened and or destroyed
                # shuffled = df[trace_mask].sample(frac=1, replace=False).reset_index(drop=True)
                #
                # # creates the iid array of samples from every cycle variableeter lineage distribution
                # for variable in phenotypic_variables:
                #     to_add.update({variable: shuffled[variable]})
                #
                # # adds it to the dataframe
                # no_epi_df = no_epi_df.append(pd.DataFrame(to_add), ignore_index=True)
    
    # Reset the index as a matter of cleanliness since we are not gonna work with it anyways
    return no_epi_df.reset_index(drop=True)


""" truncate all lineages in a dataframe to the min_gens parameter """


def limit_lineage_length(df, min_gens):
    new_df = pd.DataFrame(columns=df.columns)
    for dataset in df['dataset'].unique():
        for trap_id in df[(df['dataset'] == dataset)]['trap_ID'].unique():
            for trace in ['A', 'B']:
                lineage = df[(df['dataset'] == dataset) & (df['trap_ID'] == trap_id) & (df['trace'] == trace)].copy()
                
                if len(lineage) >= min_gens:  # It has at least these amount of generations and we will truncate them there
                    # (min_gens - 1) is because the first generation is equal to zero, not 1
                    new_lineage = lineage[lineage['generation'] <= (min_gens - 1)][phenotypic_variables].sample(frac=1, replace=False)  # .reset_index(drop=True)
                    new_lineage['dataset'] = dataset
                    new_lineage['trap_ID'] = trap_id
                    new_lineage['trace'] = trace
                    new_lineage['generation'] = np.arange(min_gens)
                    
                    # adds it to the dataframe
                    new_df = new_df.append(new_lineage, ignore_index=True)
                else:
                    pass
    
    return new_df


""" Feed in a dataframe of our format and subtract the time-averages of each lineage """


def trace_center_a_dataframe(df, mm):
    # Trace-centered dataframe
    tc_df = pd.DataFrame(columns=df.columns)
    
    if mm:
        for lin_id in df.lineage_ID.unique():
            lineage = df[(df['lineage_ID'] == lin_id)].copy()
            
            new_lineage = lineage[phenotypic_variables] - lineage[phenotypic_variables].mean()
            new_lineage['lineage_ID'] = lin_id
            new_lineage['generation'] = np.arange(len(lineage))
            
            tc_df = tc_df.append(new_lineage, ignore_index=True)
    else:
        for dataset in df['dataset'].unique():
            for trap_id in df[df['dataset'] == dataset]['trap_ID'].unique():
                for trace in ['A', 'B']:
                    
                    lineage = df[(df['dataset'] == dataset) & (df['trap_ID'] == trap_id) & (df['trace'] == trace)]
                    
                    new_lineage = lineage[phenotypic_variables] - lineage[phenotypic_variables].mean()
                    new_lineage['dataset'] = dataset
                    new_lineage['trap_ID'] = trap_id
                    new_lineage['trace'] = trace
                    new_lineage['generation'] = np.arange(len(lineage))
                    
                    tc_df = tc_df.append(new_lineage, ignore_index=True)
    
    return tc_df


""" Create all information dataframe where the lineage lengths are kept constant but the cells in the trace itself are randomly sampled from the population without replacement """


def shuffle_info_OLD(info):
    # Give it a name, contains S, NL
    new_info = pd.DataFrame(columns=info.columns)
    
    # what is the trace length of each trace? This is the only thing that stays the same
    sizes = {'{} {} {}'.format(dataset, trap_ID, trace): len(info[(info['trap_ID'] == trap_ID) & (info['trace'] == trace) & (info['dataset'] == dataset)]) for dataset in np.unique(info['dataset']) for
             trap_ID in np.unique(info[info['dataset'] == dataset]['trap_ID']) for trace in ['A', 'B']}
    
    for dataset in np.unique(info['dataset']):
        for trap_ID in np.unique(info[info['dataset'] == dataset]['trap_ID']):
            for trace in ['A', 'B']:
                # trace length
                size = sizes['{} {} {}'.format(dataset, trap_ID, trace)]
                
                # sample from the old dataframe
                samples = info.sample(replace=False, n=size)
                
                # drop what we sampled
                info = info.drop(index=samples.index)
                
                # add some correct labels even though they don't matter so that the dataframe structure is still intact
                samples['trap_ID'] = trap_ID
                samples['trace'] = trace
                samples['generation'] = np.arange(size)
                samples['dataset'] = dataset
                
                # add them to the new, shuffled dataframe
                new_info = new_info.append(samples, ignore_index=True)
    
    return new_info


""" Create all information dataframe where the lineage lengths are kept constant but the cells in the trace itself are randomly sampled from the population without replacement """


def shuffle_info(info, mm):
    # Give it a name, contains S, NL
    new_info = pd.DataFrame(columns=info.columns)
    
    # what is the trace length of each trace? This is the only thing that stays the same
    if mm:
        sizes = {'{}'.format(lineage_ID): len(info[(info['lineage_ID'] == lineage_ID)]) for lineage_ID in info['lineage_ID'].unique()}
        
        for lineage_id in np.unique(info['lineage_ID']):
            # trace length
            size = sizes['{}'.format(lineage_id)]
            
            # sample from the old dataframe
            samples = info.sample(replace=False, n=size)
            
            # drop what we sampled
            info = info.drop(index=samples.index)
            
            # add some correct labels even though they don't matter so that the dataframe structure is still intact
            samples['lineage_ID'] = lineage_id
            samples['generation'] = np.arange(size)
            
            # add them to the new, shuffled dataframe
            new_info = new_info.append(samples, ignore_index=True)
    else:
        sizes = {'{} {} {}'.format(dataset, trap_ID, trace): len(info[(info['trap_ID'] == trap_ID) & (info['trace'] == trace) & (info['dataset'] == dataset)]) for dataset in np.unique(info['dataset'])
                 for
                 trap_ID in np.unique(info[info['dataset'] == dataset]['trap_ID']) for trace in ['A', 'B']}
        
        lineage_id = 0
        for dataset in np.unique(info['dataset']):
            for trap_ID in np.unique(info[info['dataset'] == dataset]['trap_ID']):
                for trace in ['A', 'B']:
                    # trace length
                    size = sizes['{} {} {}'.format(dataset, trap_ID, trace)]
                    
                    # sample from the old dataframe
                    samples = info.sample(replace=False, n=size)
                    
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


""" What we use for all main-text figures """


def seaborn_preamble():
    # stylistic reasons
    sns.set_context('paper')
    sns.set_style("ticks", {'axes.grid': True})


""" set of bootstrapped pearson correlations """


def boot_pearson(x, y, number_of_straps):
    from random import choices
    from scipy.stats import pearsonr
    x = np.array(x)
    y = np.array(y)
    r_array = []
    for n in range(number_of_straps):
        indices = choices(np.arange(len(x)), k=len(x))
        r = pearsonr(x[indices], y[indices])[0]
        r_array.append(r)
    
    return np.array(r_array)


""" We use this to cut any lineage-processed dataframe, ie. '.../phyiscal_units.csv' and '.../trace_centered.csv' so that the A and B pairs have the same lineage length """


def cut_uneven_pairs(info):
    # Correct for traces that aren't the same size
    old_info = info.copy()
    new_info = pd.DataFrame(columns=info.columns)
    
    for dataset in np.unique(old_info['dataset']):
        for trap_id in np.unique(old_info[(old_info['dataset'] == dataset)]['trap_ID']):
    
            min_gen = min(len(old_info[(old_info['trap_ID'] == trap_id) & (old_info['trace'] == 'A') & (old_info['dataset'] == dataset)]['generation'].values),
                          len(old_info[(old_info['trap_ID'] == trap_id) & (old_info['trace'] == 'B') & (old_info['dataset'] == dataset)]['generation'].values))

            # min_gen = min(max(old_info[(old_info['trap_ID'] == trap_id) & (old_info['trace'] == 'A') & (old_info['dataset'] == dataset)]['generation'].values),
            #               max(old_info[(old_info['trap_ID'] == trap_id) & (old_info['trace'] == 'B') & (old_info['dataset'] == dataset)]['generation'].values))

            # # What we will add
            # a_to_add = old_info[(old_info['trap_ID'] == trap_id) & (old_info['trace'] == 'A') & (old_info['dataset'] == dataset) & (old_info['generation'] <= min_gen)].copy()
            # b_to_add = old_info[(old_info['trap_ID'] == trap_id) & (old_info['trace'] == 'B') & (old_info['dataset'] == dataset) & (old_info['generation'] <= min_gen)].copy()

            # What we will add
            a_to_add = old_info[(old_info['trap_ID'] == trap_id) & (old_info['trace'] == 'A') & (old_info['dataset'] == dataset)].copy().sort_values('generation').iloc[:min_gen]
            b_to_add = old_info[(old_info['trap_ID'] == trap_id) & (old_info['trace'] == 'B') & (old_info['dataset'] == dataset)].copy().sort_values('generation').iloc[:min_gen]

            # Check that they are the same size
            if len(a_to_add) != len(b_to_add):
                print(a_to_add)
                print(b_to_add)
                raise IOError()
            
            # save them to a new dataframe
            new_info = new_info.append(a_to_add, ignore_index=True)
            new_info = new_info.append(b_to_add, ignore_index=True)
        
        # good practice
        new_info = new_info.reset_index(drop=True)
    
    return new_info


""" Create and add the Control dataset """


def add_control(old_info):
    from random import sample, uniform
    from itertools import combinations
    
    # Choose randomly combinations of trap IDs
    possible_combos = sample(list(combinations(np.unique(old_info['trap_ID']), 2)), 85)
    info = old_info.copy()
    
    new_id = max(np.unique(old_info['trap_ID'])) + 1
    for id_A, id_B in possible_combos:
        
        # Decide whether to use the A or B trace
        s1 = 'A' if (uniform(0, 1) > .5) else 'B'
        s2 = 'A' if (uniform(0, 1) > .5) else 'B'
        
        # define the trace
        A_trace = old_info[(old_info['trap_ID'] == id_A) & (old_info['trace'] == s1)].copy()
        # change the trap id to a new id number even though it is a copy of the other ID number
        A_trace['trap_ID'] = new_id
        # Change the dataset to Control
        A_trace['dataset'] = 'CTRL'
        # SO that it is in line with all the other analysis for the SM data
        A_trace['trace'] = 'A'
        # Add it to the information dataframe
        info = pd.concat([info, A_trace], axis=0)
        
        # Do the same
        B_trace = old_info[(old_info['trap_ID'] == id_B) & (old_info['trace'] == s2)].copy()
        B_trace['trap_ID'] = new_id
        # SO that it is in line with all the other analysis for the SM data
        B_trace['trace'] = 'B'
        B_trace['dataset'] = 'CTRL'
        info = pd.concat([info, B_trace], axis=0)
        
        # print(A_trace, B_trace, sep='\n'*2, end='*' * 100)
        # input()
        
        # create a new ID number to add in the next loop
        new_id += 1
    
    # Just for my preference even though we don't rely on indices
    info = info.reset_index(drop=True)
    
    return info


def add_control_and_cut_extra_intervals(info):
    from random import sample, uniform
    from itertools import combinations
    
    """
    Here we will create the Control dataset and equal the length of pair lineages.
    """
    
    # For the Control
    traces_to_choose_from = info['dataset'] + ', ' + info['trap_ID'].astype(str)
    
    # Choose randomly combinations of trap IDs for the A and B traces
    possible_combos = sample(list(combinations(np.unique(traces_to_choose_from), 2)), 85)
    
    # start them as new IDs
    new_id = max(np.unique(info['trap_ID'])) + 1
    
    # loop through all traces paired together for Control
    for combo in possible_combos:
        # define the trace ID
        dataset_a, id_a = combo[0].split(', ')
        dataset_b, id_b = combo[1].split(', ')
        
        # define the trace
        a_trace = info[(info['trap_ID'] == int(id_a)) & (info['trace'] == 'A') & (info['dataset'] == dataset_a)].copy()
        # change the trap id to a new id number even though it is a copy of the other ID number
        a_trace['trap_ID'] = new_id
        # Change the dataset to Control
        a_trace['dataset'] = 'C'
        # Add it to the information dataframe
        info = pd.concat([info, a_trace], axis=0)
        
        # Do the same
        b_trace = info[(info['trap_ID'] == int(id_b)) & (info['trace'] == 'B') & (info['dataset'] == dataset_b)].copy()
        b_trace['trap_ID'] = new_id
        b_trace['dataset'] = 'C'
        info = pd.concat([info, b_trace], axis=0)
        
        # create a new ID number to add in the next loop
        new_id += 1
    
    # Just for my preference even though we don't rely on indices
    info = info.reset_index(drop=True)
    
    # Correct for traces that aren't the same size
    old_info = info.copy()
    new_info = pd.DataFrame(columns=info.columns)
    
    for dataset in np.unique(old_info['dataset']):
        for trap_id in np.unique(old_info[(old_info['dataset'] == dataset)]['trap_ID']):
            # print(trap_id)
            min_gen = min(max(old_info[(old_info['trap_ID'] == trap_id) & (old_info['trace'] == 'A') & (old_info['dataset'] == dataset)]['interval']),
                          max(old_info[(old_info['trap_ID'] == trap_id) & (old_info['trace'] == 'B') & (old_info['dataset'] == dataset)]['interval']))
            
            # What we will add
            a_to_add = old_info[(old_info['trap_ID'] == trap_id) & (old_info['trace'] == 'A') & (old_info['dataset'] == dataset) & (old_info['interval'] <= min_gen)]
            b_to_add = old_info[(old_info['trap_ID'] == trap_id) & (old_info['trace'] == 'B') & (old_info['dataset'] == dataset) & (old_info['interval'] <= min_gen)]
            
            # Check that they are the same size
            if len(a_to_add) != len(b_to_add):
                print(a_to_add)
                print(b_to_add)
                raise IOError()
            
            # save them to a new dataframe
            new_info = new_info.append(a_to_add, ignore_index=True)
            new_info = new_info.append(b_to_add, ignore_index=True)
        
        # good practice
        new_info = new_info.reset_index(drop=True)
    
    return new_info


def pool_experiments(group, name, outliers=True, dimensions='pu'):
    pooled = pd.DataFrame()
    
    reference_id = {ds: count for count, ds in enumerate(dataset_names) if ds != 'Pooled_SM'}
    
    for exp in group:
        if dimensions == 'pu':
            df_to_get = 'physical_units'
        elif dimensions == 'tc':
            df_to_get = 'trace_centered'
        elif dimensions == 'ta':
            df_to_get = 'time_averages'
        else:
            raise IOError('Incorrect dimensions. Options are pu, tc or ta.')
        
        if outliers:
            dir_name = f'/Users/alestawsky/PycharmProjects/Thesis/Datasets/{exp}/ProcessedData/{df_to_get}.csv'
        else:
            dir_name = f'/Users/alestawsky/PycharmProjects/Thesis/Datasets/{exp}/ProcessedData/z_score_under_3/{df_to_get}_without_outliers.csv'
            
        pu_exp = pd.read_csv(dir_name)
        
        if len(pooled) == 0:
            pu_exp['lineage_ID'] = pu_exp['lineage_ID']
        else:
            pu_exp['lineage_ID'] = pu_exp['lineage_ID'] + pooled['lineage_ID'].max() + 1  # Re-index the lineage_IDs
        pu_exp['experiment_group'] = name
        pu_exp['experiment_name'] = '0-4'  # reference_id[exp]
        
        pooled = pooled.append(pu_exp, ignore_index=True)
    
    return pooled


""" check the precision of the division events and growth fits """


def check_the_division(args, lineages=[], raw_lineages=[], raw_indices=[], pu=[], ax=[]):
    import matplotlib.pyplot as plt
    
    # lineage, cycle_variables_lineage, start_indices, end_indices, raw_lineage, non_positives, singularities, rises, raw_start, raw_end, step_size, total_nans
    
    # For each experiment we have their time step (step size)
    if args['data_origin'] == 'MG1655_inLB_LongTraces':
        step_size = 5 / 60
    elif args['data_origin'] == 'Maryam_LongTraces':
        step_size = 3 / 60
    elif args['data_origin'] in tanouchi_datasets:
        step_size = 1 / 60
    elif args['data_origin'] in wang_datasets:
        step_size = 1 / 60
    elif args['data_origin'] in sm_datasets:
        step_size = 3 / 60
    elif args['data_origin'] == 'Lambda_LB':
        step_size = 6 / 60
    else:
        raise IOError('This code is not meant to run the data inputted. Please label the data and put it in as an if-statement.')
    
    if len(raw_lineages) == 0:
        raw_lineages = pd.read_csv(os.path.dirname(os.path.dirname(args['processed_data'])) + '/raw_data_all_in_one.csv')
    if len(raw_indices) == 0:
        raw_indices = pd.read_csv(args['processed_data'] + 'raw_indices_processing.csv')
    if len(pu) == 0:
        pu = pd.read_csv(args['processed_data'] + 'z_score_under_3/physical_units_without_outliers.csv')
    if len(lineages) == 0:
        lineages = raw_lineages.lineage_ID.unique()
    
    for lin_id in lineages:
        if ax == []:
            seaborn_preamble()
            
            fig, ax = plt.subplots(tight_layout=True, figsize=[13, 6])
            
            plot_it = True
        else:
            plot_it = False
        
        rl = raw_lineages[raw_lineages['lineage_ID'] == lin_id].copy().sort_values('time').reset_index(drop=True)
        ri = raw_indices[raw_indices['lineage_ID'] == lin_id].copy().sort_values('value').reset_index(drop=True)
        cycle_variables_lineage = pu[pu['lineage_ID'] == lin_id].copy().sort_values('generation').reset_index(drop=True)
        
        non_positives = ri[ri['type'] == 'non_positives']['value'].values.copy()
        singularities = ri[ri['type'] == 'singularities']['value'].values.copy()
        rises = ri[ri['type'] == 'rises']['value'].values.copy()
        start_indices = ri[ri['type'] == 'start']['value'].values.copy()
        end_indices = ri[ri['type'] == 'end']['value'].values.copy()
        
        ax.plot(rl['time'].values, rl['length'].values, marker='o')
        if len(non_positives) > 0:
            ax.scatter(rl['time'].iloc[non_positives].values, rl['length'].iloc[non_positives].values, label='non_positives', marker='o')
        if len(singularities) > 0:
            ax.scatter(rl['time'].iloc[singularities].values, rl['length'].iloc[singularities].values, label='singularities', marker='v')
        if len(rises) > 0:
            ax.scatter(rl['time'].iloc[rises].values, rl['length'].iloc[rises].values, label='failures', marker='^')
        
        for start, end, gt, gr, lb in zip(start_indices, end_indices, cycle_variables_lineage.generationtime, cycle_variables_lineage.growth_rate, cycle_variables_lineage.length_birth):
            
            if np.isnan([gt, lb, gr]).any():
                continue
            
            line = lb * np.exp(np.arange(0, int(np.round(gt / step_size + 1, 0))) * step_size * gr)  # The fit line based on the phenotypic variables
            
            ax.plot(rl['time'].iloc[int(start):int(start) + len(line)], line, color='orange', ls='--')
        
        # ax.scatter(rl['time'].iloc[start_indices].values, rl['length'].iloc[start_indices].values, color='green', label='start indices')
        # ax.scatter(rl['time'].iloc[end_indices].values, rl['length'].iloc[end_indices].values, color='red', label='end indices')
        
        if plot_it:
            ax.set_yscale('log')
            ax.legend()
            plt.tight_layout()
            plt.show()
            plt.close()
            
            ax = []


phenotypic_variables = ['div_then_fold', 'div_and_fold', 'fold_then_div', 'fold_growth', 'division_ratio', 'added_length', 'generationtime', 'length_birth', 'length_final', 'growth_rate']
symbols = {
    'physical_units': dict(
        zip(phenotypic_variables, [r'$f_n e^{\phi_{n+1}}$', r'$r$', r'$f_{n+1} e^{\phi_n}$', r'$\phi$', r'$f$', r'$\Delta$', r'$\tau$', r'$x_0$', r'$x_\tau$', r'$\alpha$'])),
    'trace_centered': dict(zip(phenotypic_variables,
                               [r'$\delta (f_n e^{\phi_{n+1}})$', r'$\delta r$', r'$\delta (f_{n+1} e^{\phi_n})$', r'$\delta \phi$', r'$\delta f$', r'$\delta \Delta$',
                                r'$\delta \tau$', r'$\delta x_0$', r'$\delta x_\tau$', r'$\delta \alpha$'])),
    'time_averages': dict(
        zip(phenotypic_variables,
            [r'$\overline{f_n e^{\phi_{n+1}}}$', r'$\overline{r}$', r'$\overline{f_{n+1} e^{\phi_n}}$', r'$\overline{\phi}$', r'$\overline{f}$', r'$\overline{\Delta}$',
             r'$\overline{\tau}$', r'$\overline{x_0}$', r'$\overline{x_\tau}$', r'$\overline{\alpha}$'])),
    'model': dict(zip(phenotypic_variables, [r'$\phi$', r'$f$', r'$\Delta$', r'$\tau$', r'$x_0$', r'$x_\tau$', r'$\alpha$'])),
    'old_var': dict(zip(phenotypic_variables, [r'$\phi_{old}$', r'$f_{old}$', r'$\Delta_{old}$', r'$\tau_{old}$', r'$x_{0, old}$', r'$x_{\tau, old}$', r'$\alpha_{old}$'])),
    'young_var': dict(zip(phenotypic_variables, [r'$\phi_{young}$', r'$f_{young}$', r'$\Delta_{young}$', r'$\tau_{young}$', r'$x_{0, young}$', r'$x_{\tau, young}$', r'$\alpha_{young}$'])),
    'with_n_ta': dict(zip(phenotypic_variables, [r'$\overline{\phi_{n}}$', r'$\overline{f_{n}}$', r'$\overline{\Delta_{n}}$', r'$\overline{\tau_{n}}$', r'$\overline{x_{0, n}}$',
                                                 r'$\overline{x_{\tau, n}}$', r'$\overline{\alpha_{n}}$'])),
    'unique_ta': dict(
        zip(phenotypic_variables,
            [r'$\overline{f_n e^{\phi_{n+1}}}$', r'$\overline{r}}$', r'$\overline{f_{n+1} e^{\phi_n}}$', r'$\overline{\phi}$', r'$\overline{f}$', r'$\overline{\Delta}$',
             r'$\overline{\tau}$', r'$\overline{x_0}$', r'$\overline{x_\tau}$', r'$\overline{\alpha}$'])),
    # 'new_pu': dict(zip(phenotypic_variables,
    #                    [r'$\overline{\phi}$', r'$\overline{f}$', r'$\overline{\Delta}$', r'$\overline{\tau}$', r'$\overline{x_0}$', r'$\overline{x_\tau}$', r'$\overline{\alpha}$',
    #                     r'$\overline{\frac{x_{0, n+1}}{x_{0, n}}}$']))
}
datasets = ['SL', 'NL', 'CTRL']
units = dict(zip(phenotypic_variables, [r'', r'', r'', r'', r'', r'$(\mu m)$', r'$(hr)$', r'$(\mu m)$', r'$(\mu m)$', r'$(\frac{1}{hr})$']))
hierarchy = [r'$\phi$', r'$f$', r'$\Delta$', r'$\tau$', r'$x_0$', r'$x_\tau$', r'$\alpha$']
# This is a manual thing to make the distribution look better without the outliers
bounds = {
    'generationtime': [0.01, 1], 'fold_growth': [.27, 1], 'growth_rate': [.8, 1.8], 'length_birth': [.7, 4.5], 'length_final': [2.3, 8.3], 'division_ratio': [.35, .65], 'added_length': [.4, 5]
}
symbols_bounds = {symbols['physical_units'][key]: val for key, val in bounds.items()}
wang_datasets = ['20090529_E_coli_Br_SJ119_Wang2010', '20090930_E_coli_MG1655_lexA3_Wang2010', '20090923_E_coli_MG1655_lexA3_Wang2010', '20090922_E_coli_MG1655_lexA3_Wang2010',
                 '20090210_E_coli_MG1655_(CGSC_6300)_Wang2010', '20090129_E_coli_MG1655_(CGSC_6300)_Wang2010', '20090702_E_coli_MG1655_(CGSC_6300)_Wang2010',
                 '20090131_E_coli_MG1655_(CGSC_6300)_Wang2010', '20090525_E_coli_MG1655_(CGSC_6300)_Wang2010', '20090512_E_coli_MG1655_(CGSC_6300)_Wang2010', '20090412_E_coli_Br_SJ108_Wang2010']
br_wang_exps = ['20090529_E_coli_Br_SJ119_Wang2010', '20090412_E_coli_Br_SJ108_Wang2010']
cgsc_6300_wang_exps = ['20090210_E_coli_MG1655_(CGSC_6300)_Wang2010', '20090129_E_coli_MG1655_(CGSC_6300)_Wang2010', '20090702_E_coli_MG1655_(CGSC_6300)_Wang2010',
                       '20090131_E_coli_MG1655_(CGSC_6300)_Wang2010', '20090525_E_coli_MG1655_(CGSC_6300)_Wang2010', '20090512_E_coli_MG1655_(CGSC_6300)_Wang2010']
lexA3_wang_exps = ['20090930_E_coli_MG1655_lexA3_Wang2010', '20090923_E_coli_MG1655_lexA3_Wang2010', '20090922_E_coli_MG1655_lexA3_Wang2010']
tanouchi_datasets = ['MC4100_25C (Tanouchi 2015)', 'MC4100_27C (Tanouchi 2015)', 'MC4100_37C (Tanouchi 2015)']
sm_datasets = ['1015_NL', '062718_SL', '071318_SL', '072818_SL_NL', '101218_SL_NL', 'Pooled_SM']
mm_datasets = ['Lambda_LB', 'Maryam_LongTraces', 'MG1655_inLB_LongTraces']  # 8-31-16 Continue
dataset_names = sm_datasets + mm_datasets + tanouchi_datasets + cgsc_6300_wang_exps + lexA3_wang_exps
cmap = sns.color_palette('tab10')

phenotypic_variables = ['div_and_fold', 'fold_growth', 'division_ratio', 'added_length', 'generationtime', 'length_birth', 'length_final', 'growth_rate']

# '20090529_E_coli_Br_SJ119_Wang2010' '20090930_E_coli_MG1655_lexA3_Wang2010' '20090923_E_coli_MG1655_lexA3_Wang2010' '20090922_E_coli_MG1655_lexA3_Wang2010' '20090412_E_coli_Br_SJ108_Wang2010' '20090210_E_coli_MG1655_(CGSC_6300)_Wang2010' '20090129_E_coli_MG1655_(CGSC_6300)_Wang2010' '20090702_E_coli_MG1655_(CGSC_6300)_Wang2010' '20090131_E_coli_MG1655_(CGSC_6300)_Wang2010' '20090525_E_coli_MG1655_(CGSC_6300)_Wang2010' '20090512_E_coli_MG1655_(CGSC_6300)_Wang2010'
