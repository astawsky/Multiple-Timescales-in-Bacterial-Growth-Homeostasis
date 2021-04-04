#!/usr/bin/env bash

from AnalysisCode.global_variables import (
    symbols, phenotypic_variables, cgsc_6300_wang_exps, lexA3_wang_exps, mm_datasets, dataset_names
)
import pandas as pd
import numpy as np


ds_names = [ds for ds in dataset_names if ds != 'Pooled_SM']

# Different strains of all the
strain = [
    'MG1655', 'MG1655', 'MG1655', 'MG1655', 'MG1655', 'MG1655', 'MG1655', 'MG1655', 'K-12 substr. MC4100', 'K-12 substr. MC4100', 'K-12 substr. MC4100', 'MG1655 (CGSC 6300)', 'MG1655 (CGSC 6300)',
    'MG1655 (CGSC 6300)', 'MG1655 (CGSC 6300)', 'MG1655 (CGSC 6300)', 'MG1655 (CGSC 6300)', 'MG1655 lexA3', 'MG1655 lexA3', 'MG1655 lexA3'
]

dates = ['15/10/18', '27/06/18', '13/07/18', '28/07/18', '12/10/18', 'Pooled exps.', 'Pooled exps.', 'Pooled exps.', '', '', '', '10/02/09', '29/01/09', '02/07/09', '31/01/09', '25/05/09',
         '12/05/09', '30/09/09', '23/09/09', '22/09/09']
# dates = ['15/10/18', '27/06/18', '13/07/18', '28/07/18', '12/10/18', 'Pooled exps.', 'Pooled exps.', 'Pooled exps.', '', '', '', '29/5/09', '30/9/09', '23/9/09', '22/9/09', '10/2/09',
#          '29/1/09', '2/7/09', '31/1/09', '25/5/09', '12/5/09']

temp = ['32', '32', '32', '32', '32', '32', '32', '32', '25', '27', '37', '37', '37', '37', '37', '37', '37', '37', '37', '37']
# temp = ['32', '32', '32', '32', '32', '32', '32', '32', '25', '27', '37', '37', '37', '37', '37', '37', '37', '37', '37', '37']

references = [
    'Vashistha 2021', 'Vashistha 2021', 'Vashistha 2021', 'Vashistha 2021', 'Vashistha 2021',
    'Susman 2018', 'Korham 2020', 'Susman 2018', 'Tanouchi 2015', 'Tanouchi 2015', 'Tanouchi 2015', 'Wang 2010', 'Wang 2010',
    'Wang 2010', 'Wang 2010', 'Wang 2010', 'Wang 2010', 'Wang 2010', 'Wang 2010', 'Wang 2010'
]

num_of_lineages = []

generation_per_lineage = []

for ds in ds_names:
    print(ds)
    # import the data
    pu = pd.read_csv(f'/Users/alestawsky/PycharmProjects/Thesis/Datasets/{ds}/ProcessedData/z_score_under_3/physical_units_without_outliers.csv')
    # number of lineages
    num_of_lineages.append(len(pu.lineage_ID.unique()))
    # mean and std of gen per lineage
    m = np.mean([len(pu[pu['lineage_ID'] == lin_id]['generation'].values) for lin_id in pu.lineage_ID.unique()])
    s = np.std([len(pu[pu['lineage_ID'] == lin_id]['generation'].values) for lin_id in pu.lineage_ID.unique()])
    generation_per_lineage.append(f'${int(np.round(m, 0))}\pm{int(np.round(s, 0))}$')  # Add it as a string

assert len(ds_names) == len(dates) == len(strain) == len(references) == len(temp) == len(generation_per_lineage)

gc = {count: {
    'Exp.': f'${count}$', 'Ref.': ref, '# of lins.':  f'${nl}$', '# of gens.': gpl, 'Temp. (C)':  f'${t}$', 'Strain':  stra,
    'Date':  f'${da}$'
} for ds, ref, count, nl, gpl, t, stra, da in zip(ds_names, references, range(len(references)), num_of_lineages, generation_per_lineage, temp, strain, dates)}

growth_conditions = pd.DataFrame.from_dict(gc, "index")

print(growth_conditions.to_latex(float_format="{:0.2f}".format, index=False, escape=False))


cv_latex = pd.DataFrame(columns=['Experiment Label'] + [symbols['physical_units'][p] for p in phenotypic_variables])

count = 0
for ds in ds_names:
    print(count, ds)
    
    pu = pd.read_csv(f'/Users/alestawsky/PycharmProjects/Thesis/Datasets/{ds}/ProcessedData/z_score_under_3/physical_units_without_outliers.csv')
    
    to_add = {'Experiment Label': count}
    
    to_add.update({symbols['physical_units'][p]: pu[p].std() / pu[p].mean() for p in phenotypic_variables})
    
    cv_latex = cv_latex.append(to_add, ignore_index=True)
    
    count += 1
 
to_add = {'Experiment Label': 'CV'}
# print('999')
#
# print(np.round(cv_latex.mean(), 2))
# print(np.round(cv_latex.std(), 2))
# print(np.round(cv_latex.std() / cv_latex.mean(), 2))
#
# exit()

# for m, s, p in zip(np.round(cv_latex.mean(), 2).values[1:], np.round(cv_latex.std(), 2).values[1:], np.array(cv_latex.columns)[1:]):
#     print(p, s / m)
to_add.update({p: f'{m} {s}' for m, s, p in zip(np.round(cv_latex.mean(), 2).values[1:], np.round(cv_latex.std(), 2).values[1:], np.array(cv_latex.columns)[1:])})

cv_latex = cv_latex.append(to_add, ignore_index=True)

print(cv_latex.to_latex(float_format="{:0.2f}".format, index=False))
