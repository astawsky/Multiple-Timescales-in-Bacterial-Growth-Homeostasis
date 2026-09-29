"""
Bayesian Model Comparison Framework for Bacterial Growth Phenotype Prediction

112-model sweep: 7 target partitions x 16 mother predictor subsets.
Each model: multivariate MvStudentT likelihood, LKJ Cholesky residual covariance,
Mundlak TC+TA encoding (Simpson's-paradox correction), homeostasis potential
with learned sigma_h.

Usage:
    python analysis/bayesian_model_comparison.py --dry-run
    python analysis/bayesian_model_comparison.py --n-models 3
    python analysis/bayesian_model_comparison.py --datasets Maryam_LongTraces MG1655_inLB_LongTraces
"""

import argparse
import logging
import os
import sys
import warnings
from collections import namedtuple
from itertools import combinations
from pathlib import Path

# If the system's C++ compiler cannot find <iostream> (common on macOS when
# Xcode Command Line Tools headers are missing), pytensor's C backend will crash.
# Force the NumPy backend by setting cxx= unless the user has already configured
# a working C++ compiler via PYTENSOR_FLAGS. To enable C compilation for speed,
# run: PYTENSOR_FLAGS=cxx=/path/to/clang++ python analysis/bayesian_model_comparison.py
_existing_pt_flags = os.environ.get('PYTENSOR_FLAGS', '')
if 'cxx' not in _existing_pt_flags:
    os.environ['PYTENSOR_FLAGS'] = ('cxx=,' + _existing_pt_flags).rstrip(',')

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm

try:
    import pytensor.tensor as pt
    from pytensor.tensor.rewriting.blockwise import local_blockwise_alloc as _lba
    # Pytensor 2.19 bug: local_blockwise_alloc AssertionError for MvStudentT+LKJ
    # graphs with batched observations. The rewriter uses warn_inplace which re-raises
    # instead of logging. Returning None signals "no rewrite" and lets compilation continue.
    _lba_orig_fn = _lba.fn
    def _lba_safe_fn(fgraph, node):
        try:
            return _lba_orig_fn(fgraph, node)
        except AssertionError:
            return None
    _lba.fn = _lba_safe_fn
except (ImportError, AttributeError):
    try:
        import theano.tensor as pt  # type: ignore  # PyMC v4 fallback
    except ImportError:
        pass

sys.path.insert(0, str(Path(__file__).parent.parent))
from AnalysisCode.global_variables import retrieve_dataframe_directory

# ── Constants ─────────────────────────────────────────────────────────────────

# The three daughter targets / daughter predictors (non-overlapping primitives)
PRIMITIVES_3 = ['generationtime', 'growth_rate', 'division_ratio']

# All valid mother predictors (includes length_birth as 4th)
MOTHER_VARS = ['generationtime', 'growth_rate', 'division_ratio', 'length_birth']

# Composed variables that must never appear as predictors or targets (leakage)
COMPOSED_VARS = frozenset(['fold_growth', 'added_length', 'length_final', 'div_and_fold'])

NU = 4  # MvStudentT degrees of freedom

ModelConfig = namedtuple('ModelConfig', ['group', 'targets', 'd_preds', 'mother_subset'])


# ── Model enumeration ─────────────────────────────────────────────────────────

def _powerset(iterable):
    s = sorted(iterable)
    for r in range(len(s) + 1):
        for combo in combinations(s, r):
            yield frozenset(combo)


def generate_configs():
    """Yield all 112 ModelConfig namedtuples (7 target groups x 16 mother subsets)."""
    # Build 7 target partitions: all non-empty subsets of PRIMITIVES_3, largest first
    target_partitions = []
    for k in range(len(PRIMITIVES_3), 0, -1):
        for targets_tuple in combinations(PRIMITIVES_3, k):
            targets = frozenset(targets_tuple)
            d_preds = frozenset(PRIMITIVES_3) - targets
            target_partitions.append((targets, d_preds))

    for i, (targets, d_preds) in enumerate(target_partitions):
        group = f'G{i + 1}'
        for mother_subset in _powerset(MOTHER_VARS):  # 16 subsets including empty
            yield ModelConfig(
                group=group,
                targets=targets,
                d_preds=d_preds,
                mother_subset=mother_subset,
            )


# ── Data loading ──────────────────────────────────────────────────────────────

def load_mother_daughter_pairs(dataset_names, include_outliers=True):
    """
    Load and align consecutive-generation mother-daughter pairs from MM datasets.

    Returns a dict with keys:
        mother_pu, daughter_pu, mother_tc, daughter_tc, mother_ta, daughter_ta
    Each value is a DataFrame of aligned pairs (mother row n, daughter row n+1).
    Lineage IDs are re-indexed to avoid collisions when pooling multiple datasets.
    """
    buckets = {k: [] for k in
               ['mother_pu', 'daughter_pu',
                'mother_tc', 'daughter_tc',
                'mother_ta', 'daughter_ta']}

    lineage_offset = 0

    for ds_name in dataset_names:
        pu_df = pd.read_csv(retrieve_dataframe_directory(ds_name, 'pu', include_outliers))
        tc_df = pd.read_csv(retrieve_dataframe_directory(ds_name, 'tc', include_outliers))
        ta_df = pd.read_csv(retrieve_dataframe_directory(ds_name, 'ta', include_outliers))

        # Re-index lineage IDs to avoid collisions across datasets
        max_id = int(max(df['lineage_ID'].max() for df in (pu_df, tc_df, ta_df)))
        for df in (pu_df, tc_df, ta_df):
            df['lineage_ID'] = df['lineage_ID'] + lineage_offset
        lineage_offset += max_id + 1

        for lin_id in pu_df['lineage_ID'].unique():
            m_pu = pu_df[pu_df['lineage_ID'] == lin_id].sort_values('generation').reset_index(drop=True)
            m_tc = tc_df[tc_df['lineage_ID'] == lin_id].sort_values('generation').reset_index(drop=True)
            m_ta = ta_df[ta_df['lineage_ID'] == lin_id].sort_values('generation').reset_index(drop=True)

            if len(m_pu) < 2:
                continue

            # Mother = generation n, daughter = generation n+1
            buckets['mother_pu'].append(m_pu.iloc[:-1])
            buckets['daughter_pu'].append(m_pu.iloc[1:])
            buckets['mother_tc'].append(m_tc.iloc[:-1])
            buckets['daughter_tc'].append(m_tc.iloc[1:])
            buckets['mother_ta'].append(m_ta.iloc[:-1])
            buckets['daughter_ta'].append(m_ta.iloc[1:])

    return {k: pd.concat(v, ignore_index=True) for k, v in buckets.items()}


# ── Design matrix ─────────────────────────────────────────────────────────────

def _zscore_col(arr):
    std = float(np.nanstd(arr))
    if std < 1e-10:
        return np.zeros_like(arr, dtype=float)
    return (arr - np.nanmean(arr)) / std


def build_X(config, data):
    """
    Build the Mundlak-encoded design matrix.

    Each predictor variable contributes two columns: {var}_tc (within-lineage
    deviation) and {var}_ta (lineage mean), each z-scored independently.
    Raises AssertionError if any composed variable appears in the predictor sets.
    """
    bad_m = frozenset(config.mother_subset) & COMPOSED_VARS
    bad_d = frozenset(config.d_preds) & COMPOSED_VARS
    assert not bad_m, f"Composed variable in mother_subset: {bad_m}"
    assert not bad_d, f"Composed variable in d_preds: {bad_d}"

    cols, names = [], []

    for var in sorted(config.mother_subset):
        cols.append(_zscore_col(data['mother_tc'][var].values))
        cols.append(_zscore_col(data['mother_ta'][var].values))
        names += [f'm_{var}_tc', f'm_{var}_ta']

    for var in sorted(config.d_preds):
        cols.append(_zscore_col(data['daughter_tc'][var].values))
        cols.append(_zscore_col(data['daughter_ta'][var].values))
        names += [f'd_{var}_tc', f'd_{var}_ta']

    if not cols:
        return np.empty((len(data['mother_pu']), 0), dtype=float), []

    return np.column_stack(cols).astype(float), names


# ── Homeostasis potential ─────────────────────────────────────────────────────

def add_homeostasis_potential(config, mu, target_list, d_obs_pu, daughter_ta_arr, sigma_h):
    """
    Add pm.Potential encoding f_d * exp(alpha_d * tau_d) ~ Normal(1, sigma_h).

    Works in PU space:
      - Predicted targets: mu_tc[:, idx] + daughter_ta (converts TC prediction to PU)
      - Observed d_preds: taken directly from daughter PU values
    Skips silently if any of the three homeostasis components (tau, alpha, f) is absent.
    """
    def _get(var):
        if var in config.targets:
            idx = target_list.index(var)
            return mu[:, idx] + daughter_ta_arr[:, idx]
        if var in config.d_preds:
            arr = d_obs_pu.get(var)
            return pt.as_tensor_variable(arr.astype(float)) if arr is not None else None
        return None

    tau_v   = _get('generationtime')
    alpha_v = _get('growth_rate')
    f_v     = _get('division_ratio')

    if tau_v is None or alpha_v is None or f_v is None:
        return

    constraint = f_v * pt.exp(alpha_v * tau_v)
    pm.Potential('homeostasis', pm.Normal.logp(constraint, 1.0, sigma_h).sum())


# ── PyMC model ────────────────────────────────────────────────────────────────

def build_model(config, X, Y, d_obs_pu, daughter_ta_arr):
    """
    Build the multivariate Bayesian regression model for one ModelConfig.

    X              : (n_obs, n_preds) design matrix, Mundlak-encoded, z-scored
    Y              : (n_obs, n_targets) daughter TC values
    d_obs_pu       : dict var -> (n_obs,) PU values for daughter predictors
    daughter_ta_arr: (n_obs, n_targets) TA values for target variables (for homeostasis)
    """
    n_obs, _ = X.shape
    target_list = sorted(config.targets)
    n_targets = len(target_list)

    # Augment X with a leading ones column so mu = X_aug @ beta_aug is a pure
    # matrix multiply with no broadcast dims — avoids pytensor local_blockwise_alloc bug.
    X_aug = np.concatenate([np.ones((n_obs, 1), dtype=float), X], axis=1)
    n_aug = X_aug.shape[1]

    with pm.Model() as model:
        # P4: LKJ Cholesky prior on residual covariance
        chol, _, _ = pm.LKJCholeskyCov(
            'chol_cov', n=n_targets, eta=2.0,
            sd_dist=pm.HalfNormal.dist(1.0, shape=n_targets),
        )

        # beta[0, :] acts as the intercept; beta[1:, :] are regression weights.
        # Single parameter matrix keeps mu shape unambiguous: (n_obs, n_targets).
        beta = pm.Normal('beta', 0.0, 1.0, shape=(n_aug, n_targets))
        mu = pm.math.dot(X_aug, beta)

        # P2: homeostasis potential with learned tightness
        sigma_h = pm.HalfNormal('sigma_h', 1.0)
        add_homeostasis_potential(
            config, mu, target_list, d_obs_pu, daughter_ta_arr, sigma_h)

        # Likelihood
        pm.MvStudentT('obs', nu=NU, mu=mu, chol=chol, observed=Y)

    return model


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_metrics(idata, model, Y, target_list):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        loo_result  = az.loo(idata, var_name='obs')
        waic_result = az.waic(idata, var_name='obs')

    with model:
        ppc = pm.sample_posterior_predictive(
            idata, var_names=['obs'], progressbar=True)

    # ppc.posterior_predictive['obs']: (chain, draw, n_obs, n_targets)
    ppc_mean = np.array(ppc.posterior_predictive['obs']).mean(axis=(0, 1))

    # Mean R² across targets
    r2_vals = []
    for i in range(len(target_list)):
        y_t = Y[:, i]
        ss_res = np.sum((y_t - ppc_mean[:, i]) ** 2)
        ss_tot = np.sum((y_t - y_t.mean()) ** 2)
        r2_vals.append(1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan)

    # Correlation fidelity (Frobenius distance, only meaningful for >=2 targets)
    if len(target_list) > 1:
        corr_fid = float(np.linalg.norm(
            np.corrcoef(Y.T) - np.corrcoef(ppc_mean.T), 'fro'))
    else:
        corr_fid = float('nan')

    return {
        'loo_elpd':      float(loo_result.elpd_loo),
        'waic':          float(waic_result.elpd_waic),
        'r2':            float(np.nanmean(r2_vals)),
        'corr_fidelity': corr_fid,
        'sigma_h_mean':  float(np.array(idata.posterior['sigma_h']).mean()),
    }


# ── Sweep ─────────────────────────────────────────────────────────────────────

def run_sweep(dataset_names=None, n_models=None, results_dir='analysis/results',
              include_outliers=False, draws=500, tune=500, chains=2,
              nuts_sampler='numpyro'):
    """
    Fit all 112 models and collect comparison metrics.

    dataset_names : list of MM dataset names to pool (default: ['Maryam_LongTraces'])
    n_models      : fit only the first N models (None = all 112)
    results_dir   : directory for CSV and InferenceData output
    draws/tune    : NUTS draws and tuning steps per chain (default 500/500)
    chains        : number of MCMC chains (default 2)
    nuts_sampler  : 'numpyro' (fast JAX-JIT) or 'pymc' (pytensor fallback)
    """
    if dataset_names is None:
        dataset_names = ['Maryam_LongTraces']

    os.makedirs(results_dir, exist_ok=True)

    print(f'Loading mother-daughter pairs from: {dataset_names}')
    data = load_mother_daughter_pairs(dataset_names, include_outliers=include_outliers)
    n_pairs = len(data['mother_pu'])
    print(f'  {n_pairs} mother-daughter pairs loaded')

    all_configs = list(generate_configs())
    all_configs = all_configs[1:]
    if n_models is not None:
        all_configs = all_configs[:n_models]

    results = []

    for i, config in enumerate(all_configs):
        target_list  = sorted(config.targets)
        mother_str   = '+'.join(sorted(config.mother_subset)) if config.mother_subset else 'null'
        model_id     = f'{config.group}_{mother_str}'
        print(f'\n[{i + 1}/{len(all_configs)}] {model_id}'
              f'  targets={target_list}'
              f'  d_preds={sorted(config.d_preds)}')

        # Warn when all three non-lb mother primitives are present (homeostasis constraint
        # also operates on the mother side, so the prior may be doubly-applied)
        _degenerate_threshold = frozenset(['generationtime', 'growth_rate', 'division_ratio'])
        if _degenerate_threshold.issubset(config.mother_subset):
            logging.warning(
                '%s: all 3 non-lb primitives in mother_subset — homeostasis constraint '
                'operates on both mother and daughter sides', model_id)

        X, _ = build_X(config, data)
        Y    = data['daughter_tc'][target_list].values.astype(float)

        # Build valid-row mask (drop NaN in X, Y, mother cols, d_obs, daughter_ta)
        nan_mask = np.isnan(X).any(axis=1) | np.isnan(Y).any(axis=1)
        for var in list(config.mother_subset):
            nan_mask |= (np.isnan(data['mother_tc'][var].values) |
                         np.isnan(data['mother_ta'][var].values))
        for var in list(config.d_preds) + target_list:
            for rep in ('pu', 'tc', 'ta'):
                col_df = data[f'daughter_{rep}']
                if var in col_df.columns:
                    nan_mask |= np.isnan(col_df[var].values)
        valid = ~nan_mask
        print(f'  {valid.sum()} valid pairs (dropped {(~valid).sum()} NaN rows)')

        X_fit = X[valid]
        Y_fit = Y[valid]

        d_obs_pu = {v: data['daughter_pu'][v].values[valid].astype(float)
                    for v in config.d_preds}

        daughter_ta_arr = data['daughter_ta'][target_list].values[valid].astype(float)

        model = build_model(config, X_fit, Y_fit, d_obs_pu, daughter_ta_arr)

        with model:
            idata = pm.sample(
                draws=draws, tune=tune, chains=chains,
                nuts_sampler=nuts_sampler,
                target_accept=0.9,
                progressbar=True,
                idata_kwargs={'log_likelihood': True},
            )

        metrics = compute_metrics(idata, model, Y_fit, target_list)

        row = {
            'group':          config.group,
            'mother_subset':  mother_str,
            'targets':        '+'.join(target_list),
            'd_preds':        '+'.join(sorted(config.d_preds)),
            **metrics,
        }
        results.append(row)
        print(f'  loo_elpd={metrics["loo_elpd"]:.2f}  r2={metrics["r2"]:.3f}'
              f'  corr_fid={metrics["corr_fidelity"]:.3f}  sigma_h={metrics["sigma_h_mean"]:.3f}')

        # Persist InferenceData; incremental CSV saved without delta_loo_elpd
        idata.to_netcdf(os.path.join(results_dir, f'idata_{model_id}.nc'))
        pd.DataFrame(results).to_csv(
            os.path.join(results_dir, 'comparison_results_partial.csv'), index=False)

    df = pd.DataFrame(results)

    # Compute delta_loo_elpd per group (null M0 = 0 reference; null has mother_subset='null')
    null_elpd = (
        df[df['mother_subset'] == 'null']
        .set_index('group')['loo_elpd']
        .to_dict()
    )
    df['delta_loo_elpd'] = df.apply(
        lambda r: r['loo_elpd'] - null_elpd.get(r['group'], float('nan')), axis=1)
    # Null model rows get exactly 0
    df.loc[df['mother_subset'] == 'null', 'delta_loo_elpd'] = 0.0

    df.to_csv(os.path.join(results_dir, 'comparison_results.csv'), index=False)
    print(f'\nDone. {len(df)} models saved to {results_dir}/comparison_results.csv')
    return df


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Bayesian model comparison sweep for bacterial growth phenotypes')
    parser.add_argument('--datasets', nargs='+', default=['Maryam_LongTraces'],
                        metavar='DS', help='MM dataset name(s) to pool')
    parser.add_argument('--n-models', type=int, default=None, metavar='N',
                        help='Fit only first N models (for testing)')
    parser.add_argument('--dry-run', action='store_true',
                        help='Print all 112 configs without sampling')
    parser.add_argument('--results-dir', default='analysis/results', metavar='DIR',
                        help='Output directory for results (default: analysis/results)')
    parser.add_argument('--include-outliers', action='store_true',
                        help='Use data with outliers included (default: outliers removed)')
    parser.add_argument('--draws', type=int, default=500, metavar='N',
                        help='NUTS draws per chain (default: 500)')
    parser.add_argument('--tune', type=int, default=500, metavar='N',
                        help='NUTS tuning steps per chain (default: 500)')
    parser.add_argument('--chains', type=int, default=2, metavar='N',
                        help='Number of MCMC chains (default: 2)')
    parser.add_argument('--nuts-sampler', default='numpyro',
                        choices=['numpyro', 'blackjax', 'pymc'],
                        help='NUTS backend (default: numpyro)')
    args = parser.parse_args()

    if args.dry_run:
        configs = list(generate_configs())
        print(f'Total configs: {len(configs)}')
        for j, c in enumerate(configs):
            print(f'  [{j + 1:3d}] {c.group}'
                  f'  targets={sorted(c.targets)}'
                  f'  d_preds={sorted(c.d_preds)}'
                  f'  mother={sorted(c.mother_subset)}')
        return

    run_sweep(
        dataset_names=args.datasets,
        n_models=args.n_models,
        results_dir=args.results_dir,
        include_outliers=args.include_outliers,
        draws=args.draws,
        tune=args.tune,
        chains=args.chains,
        nuts_sampler=args.nuts_sampler,
    )


if __name__ == '__main__':
    main()
