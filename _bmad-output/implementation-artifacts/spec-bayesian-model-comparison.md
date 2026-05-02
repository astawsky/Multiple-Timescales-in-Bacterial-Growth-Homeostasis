---
title: 'Bayesian Model Comparison Framework for Bacterial Growth Phenotype Prediction'
type: 'feature'
created: '2026-05-02'
status: 'done'
baseline_commit: '0c421400263b256978d6313ef7369ebf5d6d4ebc'
context:
  - 'AnalysisCode/global_variables.py'
---

<frozen-after-approval reason="human-owned intent — do not modify unless human renegotiates">

## Intent

**Problem:** No systematic framework exists to compare which subsets of mother/daughter phenotypic variables best predict daughter phenotypes in E. coli growth data, nor whether modeling correlations jointly (multivariate) outperforms independent predictions.

**Approach:** Implement a 112-model PyMC Bayesian sweep (7 target partitions × 16 mother predictor subsets) using multivariate Student-T likelihoods, Mundlak TC+TA encoding for Simpson's-paradox correction, a learned homeostasis potential, and full LOO-CV/WAIC/correlation-fidelity model comparison.

## Boundaries & Constraints

**Always:**
- Primitives only: `generationtime`, `growth_rate`, `division_ratio`, `length_birth` — no composed variables (`fold_growth`, `added_length`, `length_final`, `div_and_fold`) as predictors or targets to avoid leakage
- All predictors enter as Mundlak pairs: TC column + TA column, each z-scored independently
- Mother-daughter pairs constructed from MM data: mother = generation n, daughter = generation n+1 within same lineage
- Homeostasis potential `f_d × exp(α_d × τ_d) ~ Normal(1, σ_h)` with `σ_h ~ HalfNormal(1)` sampled per model, applied to all 7 target groups using observed daughter predictors for conditioning
- LKJ Cholesky prior on residual covariance matrix (P4)
- Results serialized to `analysis/results/` as a single pandas DataFrame CSV plus per-model ArviZ InferenceData objects

**Always (dataset):**
- Default dataset: `Maryam_LongTraces`; `run_sweep(dataset_names=['Maryam_LongTraces'])` accepts a list to support pooling additional MM datasets in future runs without code changes
- `length_birth` is the fourth mother predictor — 2⁴=16 mother subsets, 112 models total

**Never:**
- Hierarchical random effects — Mundlak handles Simpson's paradox without them
- Composed variables in predictor or target sets
- Univariate models — multivariate MvStudentT only
- P3 horseshoe prior — deferred to follow-up analysis

## I/O & Edge-Case Matrix

| Scenario | Input / State | Expected Output / Behavior | Error Handling |
|----------|--------------|---------------------------|----------------|
| Full sweep | MM dataset loaded, all 112 configs | `results/comparison_results.csv` with 112 rows, one per model | Abort with message if dataset missing |
| Null model (M0, G1) | No predictors, all 3 targets | Intercept-only baseline; ELPD reference point for ΔELPD | N/A |
| G5–G7 homeostasis | Single target, 2 daughter predictors observed | σ_h posterior constrains target via point formula | Skip potential if any conditioning variable is NaN |
| Degenerate subset | Mother subset where all 3 primitives present | P2 prior active; log warning that homeostasis constraint operates on mother side too | N/A |

</frozen-after-approval>

## Code Map

- `AnalysisCode/global_variables.py` — `retrieve_dataframe_directory()`, `phenotypic_variables`, dataset name constants
- `analysis/bayesian_model_comparison.py` — new: all model logic, enumeration, fitting, results
- `analysis/results/` — new: output directory for CSV summary and InferenceData files

## Tasks & Acceptance

**Execution:**
- [x] `analysis/` -- create directory -- houses new standalone analysis scripts
- [x] `analysis/bayesian_model_comparison.py` -- implement full script -- 7 sections below
  1. **Imports & constants**: pymc, arviz, numpy, pandas, itertools; define `PRIMITIVES = ['generationtime', 'growth_rate', 'division_ratio', 'length_birth']`
  2. **Data loading**: `load_mother_daughter_pairs(dataset_names: list)` — accepts list of MM dataset names; loads PU, TC, TA CSVs for each via `retrieve_dataframe_directory`; aligns consecutive generations n→n+1 per lineage; re-indexes `lineage_ID` to avoid collisions when pooling; returns dict `{'pu': df_m/df_d pairs, 'tc': ..., 'ta': ...}`
  3. **Design matrix**: `build_X(mother_subset, d_preds, data)` — for each var in mother_subset and d_preds, append `{var}_tc` and `{var}_ta` columns; z-score each column independently; return standardized numpy array + column names
  4. **Model enumeration**: `generate_configs()` — yields all 112 `ModelConfig(targets, d_preds, mother_subset)` namedtuples via `itertools.product(target_partitions, powerset(PRIMITIVES))`
  5. **PyMC model**: `build_model(config, X, Y_tc, Y_ta, d_obs)` — LKJ Cholesky covariance; `beta ~ Normal(0,1)`; `mu = X @ beta`; homeostasis potential via `add_homeostasis_potential(config, mu_pred, d_obs, sigma_h)`; `obs ~ MvStudentT(nu=4, mu=mu, chol=chol)`
  6. **Homeostasis potential**: `add_homeostasis_potential(config, mu_pred, d_obs, sigma_h)` — resolves τ, α, f components (predicted tensor if target, observed column if daughter predictor); adds `pm.Potential('homeostasis', Normal.logp(f*exp(α*τ), 1, sigma_h))`
  7. **Sweep & results**: `run_sweep(dataset_names=['Maryam_LongTraces'])` — loads data, iterates configs, fits each model, collects LOO, WAIC, R², Frobenius distance of predicted vs. observed correlation matrix, `sigma_h` posterior mean; saves `analysis/results/comparison_results.csv`

**Acceptance Criteria:**
- Given a valid MM dataset, when `run_sweep(dataset_name)` executes, then `comparison_results.csv` contains exactly 112 rows with columns: `group`, `mother_subset`, `targets`, `d_preds`, `loo_elpd`, `waic`, `r2`, `corr_fidelity`, `sigma_h_mean`
- Given G1+M0 (null model), when compared against any G1 model with predictors, then null model ΔELPD = 0 (reference point)
- Given G5 config with observed α_d and f_d, when homeostasis potential is added, then `pm.Potential` term encodes `τ_d ≈ -log(f_d_obs)/α_d_obs` with learned σ_h
- Given any config with composed variable in predictor set, when `build_X` is called, then `AssertionError` is raised naming the offending variable

## Design Notes

**Mundlak encoding**: each predictor `v` for variable `x` yields two columns `x_tc` (deviation from lineage mean) and `x_ta` (lineage mean). TC coefficient = within-lineage effect; TA coefficient = between-lineage effect. Z-score each independently so priors are on the same scale.

**Homeostasis potential routing**:
```python
# G5 example: target=tau_d, observed alpha_d and f_d in d_obs
tau_pred = mu_pred[:, target_idx['generationtime']]
alpha_obs = d_obs['growth_rate']
f_obs     = d_obs['division_ratio']
constraint = f_obs * pt.exp(alpha_obs * tau_pred)
pm.Potential('homeostasis', pm.Normal.logp(constraint, 1.0, sigma_h))
```

**Target Y**: fit on TC values of daughter variables (within-lineage deviations), since PU has cross-experiment scale issues for the multivariate likelihood.

## Spec Change Log

## Verification

**Commands:**
- `python analysis/bayesian_model_comparison.py --dry-run` -- expected: prints 112 configs, no sampling
- `python analysis/bayesian_model_comparison.py --n-models 3` -- expected: fits first 3 configs, writes partial CSV to `analysis/results/`
- `python -c "import sys; sys.path.insert(0,'.'); from analysis.bayesian_model_comparison import generate_configs; assert len(list(generate_configs())) == 112"` -- expected: passes silently

## Suggested Review Order

**Entry point — design intent**

- Full 112-model sweep: entry point showing how configs drive data loading, model building, and results
  [`bayesian_model_comparison.py:286`](../../analysis/bayesian_model_comparison.py#L286)

**Model enumeration**

- `generate_configs()`: 7 target partitions × 16 mother subsets, verifiable in one line
  [`bayesian_model_comparison.py:62`](../../analysis/bayesian_model_comparison.py#L62)

**Data pipeline**

- Mother-daughter pair alignment: n→n+1 within lineage; re-indexed lineage IDs for multi-dataset pooling
  [`bayesian_model_comparison.py:84`](../../analysis/bayesian_model_comparison.py#L84)

- Lineage offset uses max across all three representations (PU/TC/TA) to prevent ID collisions
  [`bayesian_model_comparison.py:107`](../../analysis/bayesian_model_comparison.py#L107)

**Design matrix & z-scoring**

- Mundlak TC+TA pairs, each column z-scored independently; `nanstd`/`nanmean` prevents NaN propagation
  [`bayesian_model_comparison.py:133`](../../analysis/bayesian_model_comparison.py#L133)

- `build_X`: assembled column-by-column with composed-variable leakage guard (`AssertionError`)
  [`bayesian_model_comparison.py:140`](../../analysis/bayesian_model_comparison.py#L140)

**Homeostasis potential**

- Universal G1–G7 routing: `_get()` yields predicted tensor (if target) or observed PU column (if d_pred)
  [`bayesian_model_comparison.py:182`](../../analysis/bayesian_model_comparison.py#L182)

- Constraint `f × exp(α × τ) ~ Normal(1, σ_h)` added as `pm.Potential`; skips if any component absent
  [`bayesian_model_comparison.py:198`](../../analysis/bayesian_model_comparison.py#L198)

**PyMC model**

- `build_model`: LKJ Cholesky covariance + MvStudentT(ν=4); null model broadcasts intercept without `beta`
  [`bayesian_model_comparison.py:204`](../../analysis/bayesian_model_comparison.py#L204)

**Sweep loop & metrics**

- NaN mask: guards X, Y, mother TC/TA columns, and all daughter representations before subsetting
  [`bayesian_model_comparison.py:331`](../../analysis/bayesian_model_comparison.py#L331)

- Degenerate subset warning: emitted when all three non-lb primitives are in `mother_subset`
  [`bayesian_model_comparison.py:320`](../../analysis/bayesian_model_comparison.py#L320)

- `delta_loo_elpd` post-processing: null model (M0) is 0 reference per group; computed after loop
  [`bayesian_model_comparison.py:380`](../../analysis/bayesian_model_comparison.py#L380)

**Metrics**

- LOO, WAIC, R², Frobenius correlation fidelity, σ_h posterior mean collected per model
  [`bayesian_model_comparison.py:247`](../../analysis/bayesian_model_comparison.py#L247)

**Package initializer**

- Empty `__init__.py` makes `analysis` importable as a Python package
  [`__init__.py:1`](../../analysis/__init__.py#L1)
