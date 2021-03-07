#!/usr/bin/env bash

from AnalysisCode.global_variables import (
    symbols, cmap, shuffle_info, phenotypic_variables, get_time_averages_df
)
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.stats import linregress, pearsonr
import matplotlib.patches as mpatches


def add_scatterplot_of_averages(var1, var2, pooled, time_average, ax, type_of_lineage, marker='x'):
    if pooled:
        for exp, c in zip(time_average.experiment.unique(), cmap):
            # lineages = df[(df['experiment'] == exp)].lineage_ID.unique()
            ax.scatter(time_average.sort_values('lineage_ID')[var1],  # So they don't overlap too much
                       time_average.sort_values('lineage_ID')[var2],  # [df[(df['experiment'] == exp) & (df['lineage_ID'] == lin_id)][var2].mean() for lin_id in lineages[:min(len(lineages), 50)]]
                       marker=marker, zorder=500, label=exp.split('(')[0] if type_of_lineage == 'Trace' else '', alpha=.2 if type_of_lineage == 'Trace' else .1)
    else:
        first = time_average.sort_values('lineage_ID')[var1]  # [df[df['lineage_ID'] == lin_id][var1].mean() for lin_id in df.lineage_ID.unique() if len(df[df['lineage_ID'] == lin_id]) > 6]
        second = time_average.sort_values('lineage_ID')[var2]  # [df[df['lineage_ID'] == lin_id][var2].mean() for lin_id in df.lineage_ID.unique() if len(df[df['lineage_ID'] == lin_id]) > 6]
        ax.scatter(first, second, marker=marker, c=cmap[0] if type_of_lineage == 'Trace' else cmap[1], zorder=500, alpha=.2 if type_of_lineage == 'Trace' else .1,
                   label=type_of_lineage)
        print(f'Trace averages correlation: {pearsonr(first, second)[0]}' if marker == 'x'
              else f'Artificial averages correlation: {pearsonr(first, second)[0]}')


def plot_binned_data(df, var1, var2, num, ax):
    labels = np.arange(num)
    data_binned, bins = pd.qcut(df[var1].values, num, retbins=True, labels=labels)
    
    # bin_inds = {bin: (data_binned == bin) for bin in bins}
    
    x_binned, y_binned = [], []
    
    for label in labels:
        indices = (data_binned == label)
        x_binned.append(df.loc[indices][var1].mean())
        y_binned.append(df.loc[indices][var2].mean())
    
    ax.plot(x_binned, y_binned, marker='s', label='binned', zorder=1000, alpha=.6, c='k')


def kde_scatterplot_variables(df, var1, var2, num, ax, line_func=[], line_label='', pooled=False, artificial=None, sym1=None, sym2=None):  # df is pu of ONE experiment
    # The symbols for each variable
    if sym1 == None:
        sym1 = symbols['physical_units'][var1]  # if var1 != 'division_ratio' else r'$\ln(f)$'
    if sym2 == None:
        sym2 = symbols['physical_units'][var2]

    df = df[df['max_gen'] > 15].copy().reset_index(drop=True)  # Take out extra small lineages
    
    print(len(df))
    
    if isinstance(artificial, pd.DataFrame):  # If artificial lineages is a dataframe, plot those too
        artificial = artificial[artificial['max_gen'] > 7].copy().reset_index(drop=True)  # Take out extra small lineages
        add_scatterplot_of_averages(var1, var2, pooled, artificial, ax, 'Artificial', marker='^')  # How a random average behavior is supposed to act when only keeping per-cycle correlations
        
        no_nans_art = artificial[[var1, var2]].copy().dropna()  # drop the NaNs
        print(f'pooled correlation (Artificial): {pearsonr(no_nans_art[var1].values, no_nans_art[var2].values)[0]}')
    
    # To speed it up we sample randomly 1,000 points
    # sns.kdeplot(data=df.sample(frac=.8, replace=False), x=var1, y=var2, color='gray', ax=ax)  # Do the kernel distribution approxiamtion for variables in their physical dimensions
    add_scatterplot_of_averages(var1, var2, pooled, df, ax, 'Trace', marker='x')  # Put the average behavior
    
    if len(line_func) == 0:  # if we didn't put any
        pass
    elif line_func == 'regression':  # do the regression line!
        x, y = df[[var1, var2]].dropna()[var1].values, df[[var1, var2]].dropna()[var2].values
        slope, intercept = linregress(x, y)[:2]
        fake_x = np.linspace(np.nanmin(x), np.nanmax(x))
        ax.plot(fake_x, intercept + slope * fake_x, color='black', ls='--', label=f'{sym2}={np.round(intercept, 2)}+{np.round(slope, 2)}*{sym1}')
    else:  # Plot using the other functions for the min and max in the x distribution
        for count, func in enumerate(line_func):
            fake_x = np.linspace(np.nanmin(df[var1].values), np.nanmax(df[var1].values))  # x linespace
            ax.plot(fake_x, func(fake_x), color='black' if count == 0 else 'gray', ls='--', label=line_label)  # plot the x distribution
    
    # plt.title(ds)
    # plot_binned_data(df, var1, var2, num, ax)  # plot the binned data
    ax.set_xlabel(sym1)
    ax.set_ylabel(sym2)
    # ax.legend(title='')
    
    no_nans = df[[var1, var2]].copy().dropna()
    
    print('kde scatter plot')
    print(f'pooled correlation (Trace): {pearsonr(no_nans[var1].values, no_nans[var2].values)[0]}')
    print('-' * 200)


def plot_pair_scatterplots(df, var1, var2, ax, sym1=None, sym2=None):
    # x_a = [df[(df['trap_ID'] == trap_id) & (df['trace'] == 'A') & (df['dataset'] == 'NL')][var1].mean() for trap_id in df[(df['dataset'] == 'NL')].trap_ID.unique()]
    # y_b = [df[(df['trap_ID'] == trap_id) & (df['trace'] == 'B') & (df['dataset'] == 'NL')][var2].mean() for trap_id in df[(df['dataset'] == 'NL')].trap_ID.unique()]
    x_a = []
    y_b = []
    
    # diagonal_length = []
    # offdiagonal_length = []
    
    for trap_id in df[(df['dataset'] == 'NL')].trap_ID.unique():
        lin_a = df[(df['trap_ID'] == trap_id) & (df['trace'] == 'A') & (df['dataset'] == 'NL')].copy()
        lin_b = df[(df['trap_ID'] == trap_id) & (df['trace'] == 'B') & (df['dataset'] == 'NL')].copy()
        
        if (len(lin_a) < 7) or (len(lin_b) < 7):
            continue
        x_a.append(lin_a[var1].mean())
        x_a.append(lin_b[var2].mean())
        y_b.append(lin_b[var2].mean())
        y_b.append(lin_a[var1].mean())
    
    x_a = np.array(x_a)
    y_b = np.array(y_b)
    
    ax.grid(True)
    ax.scatter(x_a, y_b)
    
    diag_spead = np.std((x_a + y_b) / 2)
    off_spread = np.std((np.append((x_a - np.sqrt(x_a * y_b)), (y_b - np.sqrt(x_a * y_b)))))
    center = np.mean(np.append(x_a, y_b))
    
    ax.plot(np.linspace(center - diag_spead, center + diag_spead), np.linspace(center - diag_spead, center + diag_spead), ls='-', c='k', linewidth=3)
    ax.plot(np.linspace(center - off_spread, center + off_spread), - np.linspace(center - off_spread, center + off_spread) + 2 * center, ls='-', c='k', linewidth=3)
    
    # slope, intercept = linregress(x_a, y_b)[:2]
    # low_x, low_y = np.mean(x_a) - np.std((np.array(y_b)+np.array(x_a))/2), np.mean(y_b) - np.std(np.array(y_b)-(intercept+slope*np.array(x_a)))
    # high_x, high_y = np.mean(x_a) + np.std((np.array(y_b)+np.array(x_a))/2), np.mean(y_b) + np.std(np.array(y_b)-(intercept+slope*np.array(x_a)))
    #
    # diag_low, diag_high = np.mean(x_a) - np.std((np.array(y_b)+np.array(x_a))/2), np.mean(x_a) + np.std((np.array(y_b)+np.array(x_a))/2)
    #
    # new_int = (np.mean(y_b) + np.mean(x_a) / slope)
    #
    # ax.plot(np.linspace(low_x, high_x), intercept + slope * np.linspace(low_x, high_x), ls='-', c='k', linewidth=3)
    #
    # ax.plot(np.linspace((-low_y + new_int)*slope, (-high_y + new_int)*slope), (np.mean(y_b) + np.mean(x_a) / slope) - np.linspace((-low_y + new_int)*slope, (-high_y + new_int)*slope) / slope, ls='-', c='k', linewidth=3)
    
    x_a, y_b = list(x_a), list(y_b)
    
    minimum, maximum = np.min(x_a + y_b) - (.2 * np.std(x_a + y_b)), np.max(x_a + y_b) + (.2 * np.std(x_a + y_b))
    ax.set_xlim([minimum, maximum])
    ax.set_ylim([minimum, maximum])
    ax.set_xticks(np.round(np.linspace(minimum, maximum, 4), 2))
    ax.set_yticks(np.round(np.linspace(minimum, maximum, 4), 2))
    if sym1 == None:
        sym1 = symbols['time_averages'][var1] + r'$^{\, A}$'
    if sym2 == None:
        sym2 = symbols['time_averages'][var2] + r'$^{\, B}$'
    ax.set_xlabel(sym1)
    ax.set_ylabel(sym2)
    
    print(f'pair scatterplot {var1} {var2}: {pearsonr(x_a, y_b)[0]}')


#########################################
# What appears in the main text #
#########################################


pu = pd.read_csv(f'/Users/alestawsky/PycharmProjects/Thesis/Datasets/Pooled_SM/ProcessedData/z_score_under_3/physical_units_without_outliers.csv')
ta = pd.read_csv(f'/Users/alestawsky/PycharmProjects/Thesis/Datasets/Pooled_SM/ProcessedData/z_score_under_3/time_averages_without_outliers.csv').drop('generation', axis=1).drop_duplicates()
# pu['fold_growth'] = np.log(pu['fold_growth'])
art = shuffle_info(pu, False)
art_ta = get_time_averages_df(shuffle_info(pu, False), phenotypic_variables).drop('generation', axis=1).drop_duplicates().reset_index(drop=True)

num = 10

scale = 1.5

sns.set_context('paper', font_scale=1 * scale)
sns.set_style("ticks", {'axes.grid': False})

fig, axes = plt.subplots(2, 3, tight_layout=True, figsize=[6.5 * scale, 4.2 * scale])

axes[0, 0].set_title('A', x=-.2, fontsize='xx-large')
axes[0, 1].set_title('B', x=-.2, fontsize='xx-large')
axes[0, 2].set_title('C', x=-.2, fontsize='xx-large')
axes[1, 0].set_title('D', x=-.2, fontsize='xx-large')
axes[1, 1].set_title('E', x=-.2, fontsize='xx-large')
axes[1, 2].set_title('F', x=-.2, fontsize='xx-large')

# axes[0, 2].set_xlim([.76, 2])
# axes[0, 2].set_ylim([.18, .95])
# axes[1, 2].set_xlim([.4, .63])
# axes[1, 2].set_ylim([1.06, 2.96])

kde_scatterplot_variables(
    df=ta.copy(),
    var1='growth_rate',
    var2='generationtime',
    num=num,
    ax=axes[0, 2],
    line_func=[lambda x: np.log(2) / x],  # 'regression',  #lambda x: np.log(2) / x, lambda x: -x ;;;;; None, , lambda x: pu['fold_growth'].mean() / x
    pooled=False,
    artificial=art_ta
)

changed = pu.copy()
changed['fold_growth'] = np.exp(changed['fold_growth'])
changed_ta = get_time_averages_df(changed, phenotypic_variables).drop('generation', axis=1).drop_duplicates().reset_index(drop=True)
art_changed_ta = get_time_averages_df(shuffle_info(changed, False), phenotypic_variables).drop('generation', axis=1).drop_duplicates().reset_index(drop=True)

kde_scatterplot_variables(
    df=changed_ta,
    var1='division_ratio',
    var2='fold_growth',
    num=num,
    ax=axes[1, 2],
    line_func=[lambda x: 1 / x],  # 'regression',  #lambda x: np.log(2) / x, lambda x: -x ;;;;; None, , lambda x: np.mean(np.exp(pu['fold_growth']) * np.mean(pu['division_ratio'])) / x
    pooled=False,
    artificial=art_changed_ta,
    sym2=r'$e^{\phi}$'
)

handles = [mpatches.Patch(color=cmap[1], label='Artificial'), mpatches.Patch(color=cmap[0], label='Trace')]
_, labels = axes[0, 2].get_legend_handles_labels()
axes[0, 2].legend(handles, labels, fontsize='xx-small', markerscale=.5)

plot_pair_scatterplots(pu, 'generationtime', 'generationtime', axes[0, 1])
plot_pair_scatterplots(pu, 'growth_rate', 'growth_rate', axes[0, 0])
plot_pair_scatterplots(pu, 'fold_growth', 'fold_growth', axes[1, 0])
plot_pair_scatterplots(pu, 'division_ratio', 'division_ratio', axes[1, 1])  # , sym1=r'$\overline{\ln(f)}^{\, A}$', sym2=r'$\overline{\ln(f)}^{\, B}$')
# plt.legend()
plt.savefig('alpha_tau.png', dpi=300)
plt.show()
plt.close()
