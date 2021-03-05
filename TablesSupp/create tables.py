#!/usr/bin/env bash

from AnalysisCode.global_variables import (
    symbols, phenotypic_variables, cgsc_6300_wang_exps, lexA3_wang_exps, mm_datasets, dataset_names
)
import pandas as pd
import numpy as np

strain = [
    'MG1655', 'MG1655', 'MG1655', 'MG1655', 'MG1655', 'MG1655', 'MG1655', 'MG1655', 'K-12 substr. MC4100', 'K-12 substr. MC4100', 'K-12 substr. MC4100', 'B/r SJ119', 'MG1655 lexA3', 'MG1655 lexA3',
    'MG1655 lexA3', 'MG1655 (CGSC 6300)', 'MG1655 (CGSC 6300)', 'MG1655 (CGSC 6300)', 'MG1655 (CGSC 6300)', 'MG1655 (CGSC 6300)', 'MG1655 (CGSC 6300)'
]

dates = ['15/10/18', '27/6/18', '13/7/18', '28/7/18', '12/10/18', 'Pooled exps.', 'Pooled exps.', 'Pooled exps.', '', '', '', '29/5/9', '30/9/9', '23/9/9', '22/9/9', '10/2/9',
         '29/1/9', '2/7/9', '31/1/9', '25/5/9', '12/5/9']

temp = ['32', '32', '32', '32', '32', '32', '32', '32', '25', '27', '37', '37', '37', '37', '37', '37', '37', '37', '37', '37']

references = [
    'Vashistha 2021', 'Vashistha 2021', 'Vashistha 2021', 'Vashistha 2021', 'Vashistha 2021',
    'Susman 2018', 'Korham 2020', 'Susman 2018', 'Tanouchi 2015 (25C)', 'Tanouchi 2015 (27C)', 'Tanouchi 2015 (37C)', 'Wang 2010 (10/2/9)', 'Wang 2010 (29/1/9)',
    'Wang 2010 (2/7/9)', 'Wang 2010 (31/1/9)', 'Wang 2010 (25/5/9)', 'Wang 2010 (12/5/9)', 'Wang 2010 (30/9/9)', 'Wang 2010 (23/9/9)', 'Wang 2010 (22/9/9)'
]

num_of_lineages = []

generation_per_lineage = []

ds_names = [ds for ds in dataset_names if ds != 'Pooled_SM']

for ds in ds_names:
    print(ds)
    pu = pd.read_csv(f'/Users/alestawsky/PycharmProjects/Thesis/Datasets/{ds}/ProcessedData/z_score_under_3/physical_units_without_outliers.csv')
    num_of_lineages.append(len(pu.lineage_ID.unique()))
    m = np.mean([len(pu[pu['lineage_ID'] == lin_id]['generation'].values) for lin_id in pu.lineage_ID.unique()])
    s = np.std([len(pu[pu['lineage_ID'] == lin_id]['generation'].values) for lin_id in pu.lineage_ID.unique()])
    generation_per_lineage.append(f'${int(np.round(m, 0))}\pm{int(np.round(s, 0))}$')

for name, group in zip(['Wang 2010, CGSC 6300', 'Wang 2010, lexA3', 'Susman and Kohram', 'Vashistha 2021', 'Tanouchi 25C', 'Tanouchi 27C', 'Tanouchi 37C'],
                       [cgsc_6300_wang_exps, lexA3_wang_exps, mm_datasets, ['Pooled_SM'], ['MC4100_25C (Tanouchi 2015)'], ['MC4100_27C (Tanouchi 2015)'], ['MC4100_37C (Tanouchi 2015)']]):
    print(name, group)
    
    for ds in group:
        print(ds)
        pu = pd.read_csv(f'/Users/alestawsky/PycharmProjects/Thesis/Datasets/{ds}/ProcessedData/z_score_under_3/physical_units_without_outliers.csv')
        num_of_lineages.append(len(pu.lineage_ID.unique()))
        m = np.mean([len(pu[pu['lineage_ID'] == lin_id]['generation'].values) for lin_id in pu.lineage_ID.unique()])
        s = np.std([len(pu[pu['lineage_ID'] == lin_id]['generation'].values) for lin_id in pu.lineage_ID.unique()])
        generation_per_lineage.append(f'${m}\pm{s}$')

print(len(ds_names), len(references))

gc = {count: {
    'Exp.': f'${count}$', 'Ref.': ref, '# of lins.':  f'${nl}$', '# of gens.': gpl, 'Temp. (C)':  f'${t}$', 'Strain':  stra,
    'Date':  f'${da}$'
} for ds, ref, count, nl, gpl, t, stra, da in zip(dataset_names, references, range(len(references)), num_of_lineages, generation_per_lineage, temp, strain, dates) if ds != 'Pooled_SM'}

growth_conditions = pd.DataFrame.from_dict(gc, "index")

print(growth_conditions.to_latex(float_format="{:0.2f}".format, index=False, escape=False))

length_latex = pd.DataFrame(columns=['Exp.',
                                     '# of lineages',
                                     '# of cycles per lineage',
                                     '# lineages with at least 30 cycles'])
cv_latex = pd.DataFrame(columns=['Experiment Label'] + [symbols['physical_units'][p] for p in phenotypic_variables])

count = 0
for ds in dataset_names:
    if ds == 'Pooled_SM':
        continue
    print(count, ds)
    
    pu = pd.read_csv(f'/Users/alestawsky/PycharmProjects/Thesis/Datasets/{ds}/ProcessedData/z_score_under_3/physical_units_without_outliers.csv')
    
    to_add = {'Experiment Label': count}
    
    to_add.update({symbols['physical_units'][p]: pu[p].std() / pu[p].mean() for p in phenotypic_variables})
    
    cv_latex = cv_latex.append(to_add, ignore_index=True)
    
    count += 1

print(cv_latex.mean())

print(cv_latex.std())

print(cv_latex.to_latex(float_format="{:0.2f}".format, index=False))
