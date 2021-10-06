#!/usr/bin/env bash

from AnalysisCode.global_variables import slash, dataset_names
from AnalysisCode.process_rawdata import main as process_data
from AttractorManifold.manifold import main as manifold_3d
from ConditionedVariance.variance_decomposed import main as variance_decomposed
from ConditionedVarianceSupp.decomposition import main as variance_decomposed_supp
from CovarianceDecomposition.covariance_decomposition_pyramids import main as pyramids
from HistogramGenerationsPerLineage.code_lineage_lengths import main as histogram_of_generations_per_lineage
from IntergenerationalCorrelations.Intergenerational_code import main as intergenerational_corrs
from MicroenvironmentAverageBehavior.alpha_tau_code import main as microenv_averages
from PairTimescaleCorrelations.pair_lineage_correlations import main as pair_lineage_corrs
from PersistenceMainText.dfa_dataframe_figures_and_illustrations import main as dfa_maintext
from PersistenceSupp.supp_dfa_figures_and_illustrations import main as dfa_supp
from PooledAverageBehavior.different_timescaled import main as pooled_average
from SizeVariablesAverageBehavior.size_variables_code import main as size_variables
from TablesSupp.create_tables import main as tables_supp
from TimescaleCorrelations.single_cell_correlations import main as single_cell_corrs
from TrendForDFA.trending_variables import main as dfa_trend
from NewIntroductoryFigure.plot_intro_figure import main as new_intro_figure
from NewIntroductoryFigure.new_fig_1 import main as new_fig_1
import os

# Arguments for type of noise we want added
noise_args = {
    'types_of_noise': [None],  # ['Add', 'Add'],
    'amount_of_noise': [None],  # [.1, .2],
    'check': False,
    'ds_names': dataset_names
# [
#         '20090131_E_coli_MG1655_(CGSC_6300)_Wang2010', '20090525_E_coli_MG1655_(CGSC_6300)_Wang2010',
#         '20090512_E_coli_MG1655_(CGSC_6300)_Wang2010', '20090930_E_coli_MG1655_lexA3_Wang2010',
#         '20090923_E_coli_MG1655_lexA3_Wang2010', '20090922_E_coli_MG1655_lexA3_Wang2010'
#     ]
}

for tof, aof in zip(noise_args["types_of_noise"], noise_args["amount_of_noise"]):
    print(tof, aof)

    # Making the raw, processed and without_outliers arguments for noise or no noise
    if (tof == None) & (aof == None):  # Not using any noise

        noise_args.update({
            # Functions that return path of where data is stored
            'source_raw_data': lambda dummy: os.path.dirname(
                os.path.abspath(__file__)) + f'{slash}Datasets{slash}' + dummy + f'{slash}RawData{slash}',
            'raw_data': lambda dummy: os.path.dirname(
                os.path.abspath(
                    __file__)) + f'{slash}Datasets{slash}' + dummy + f'{slash}',
            'processed_data': lambda dummy: os.path.dirname(
                os.path.abspath(
                    __file__)) + f'{slash}Datasets{slash}' + dummy + f'{slash}ProcessedData{slash}',
            'without_outliers': lambda dummy: os.path.dirname(os.path.abspath(
                __file__)) + f'{slash}Datasets{slash}' + dummy + f'{slash}ProcessedData{slash}z_score_under_3{slash}',
            'noise_index': ''
        })
    else:  # Using noise
        noise_args.update({
            'processed_data': lambda dummy: os.path.dirname(
                os.path.abspath(
                    __file__)) + f'{slash}Datasets{slash}' + dummy + f'{slash}{tof}_{aof}{slash}',
            'without_outliers': lambda dummy: os.path.dirname(os.path.abspath(
                __file__)) + f'{slash}Datasets{slash}' + dummy + f'{slash}{tof}_{aof}{slash}z_score_under_3{slash}',
            'noise_index': f'_{tof}_{aof}'
        })

    # new_intro_figure(**noise_args)
    new_fig_1(**noise_args)
    exit()

    process_data(**noise_args)

    manifold_3d(**noise_args)  # Uses datasets specified in noise arguments, though it should be all...

    variance_decomposed(**noise_args)  # Requires datasets 'Pooled_SM', 'lambda_lb', 'Maryam_LongTraces'

    variance_decomposed_supp(**noise_args)

    pyramids(**noise_args)

    histogram_of_generations_per_lineage(**noise_args)  # Not necessary since it is not published

    intergenerational_corrs(**noise_args)

    microenv_averages(**noise_args)

    pair_lineage_corrs(**noise_args)

    dfa_maintext(**noise_args)

    dfa_supp(**noise_args)

    pooled_average(**noise_args)

    size_variables(**noise_args)

    tables_supp(**noise_args)

    single_cell_corrs(**noise_args)

    dfa_trend(**noise_args)
