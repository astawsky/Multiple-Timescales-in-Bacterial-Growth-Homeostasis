#!/usr/bin/env bash

from AnalysisCode.global_variables import phenotypic_variables, create_folder, symbols, units, dataset_names
import numpy as np
import pandas as pd
import os
import ipywidgets as widgets
    
    
pooled_checkbox = widgets.Checkbox(
    value=False,
    description='Pooled',
    disabled=False,
#     indent=False
)  

remove_outliers_checkbox = widgets.Checkbox(
    value=False,
    description='Remove 3 stdvs',
    disabled=False,
#     indent=False
)

widgets.Valid(
    value=False,
    description='Valid!',
#     indent=False
)

# Mapping between the variable names and what to display
inv_map = {
    'size ratio': 'div_and_fold',
    'total growth': 'fold_growth',
    'division ratio': 'division_ratio',
    'added size': 'added_length',
    'inter-division time': 'generationtime',
    'initial size': 'length_birth',
    'division size': 'length_final',
    'growth-rate': 'growth_rate'
}

dataset_dropdown = widgets.Dropdown(
    options=dataset_names,
    value='Pooled_SM',
    description='Datasets:',
    disabled=False
)

variable_multiselect = widgets.SelectMultiple(
    options=inv_map,
    value=['length_birth'],
    description='Variable:',
#     disabled=False,
    continuous_update=False,
    # orientation='horizontal',
    readout=True,
#     readout_format='d'
)


widgets.Dropdown(
    options=inv_map,
    value='length_birth',
    description='Variable:',
    disabled=False
)

variable_1_dropdown = widgets.Dropdown(
    options=inv_map,
    value='length_birth',
    description='Variable 1:',
    disabled=False
)

variable_2_dropdown = widgets.Dropdown(
    options=inv_map,
    value='length_birth',
    description='Variable 2:',
    disabled=False
)

output = widgets.Output()  # To create refresh the widgets

# For the lineage ID
lin_id_multiselect = widgets.SelectMultiple(
    options=np.arange(1, 430, dtype=int),
    value=[1],
    description='Lineage(s):',
#     disabled=False,
    continuous_update=False,
    # orientation='horizontal',
#     readout=True,
#     readout_format='d'
)


# updating the lineage_ID
def update_li(change):  # ds, li, op
    output.clear_output()  # Clears?
    
    # Choose the correct dataframe
    pu = pd.read_csv(os.path.abspath('') + f'/Datasets/{dataset_dropdown.value}/ProcessedData/z_score_under_3/physical_units_without_outliers.csv')
    
    lin_id_multiselect.options = pu.lineage_ID.unique().tolist()  # update the lineage IDs to the current dataset

lin_id_multiselect.observe(update_li)  # Will always be the case b/c it only depends on the dataset  , dataset_dropdown, lin_id_multiselect, output
    
    


