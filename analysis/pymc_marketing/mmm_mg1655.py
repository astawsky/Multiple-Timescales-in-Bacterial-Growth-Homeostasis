#!/usr/bin/env python3
"""
PyMC Marketing MMM — MG1655_inLB_LongTraces

Response:  div_and_fold (trace-centered, shifted +1 generation)
Channels:  growth_rate, length_birth, division_ratio, generationtime
           (all trace-centered values at generation n)

Stage A:  Single-lineage MMM (lineage 12, ~258 usable rows) via pymc-marketing.
Stage B:  Hierarchical PyMC model with partial pooling across all 20 lineages.
          Adstock is preprocessed using Stage A's posterior median decay per channel.
          Channel betas are partially pooled across lineages.

Reproducibility
---------------
- RANDOM_SEED is fixed globally.
- Input data is saved as mg1655_model_input.csv.
- idata_single_lineage.nc and idata_hierarchical.nc store the full posteriors.
- results_single_lineage.csv and results_hierarchical.csv merge observed values
  with predictions using explicit _obs / _pred_mean / _pred_hdi_lo / _pred_hdi_hi
  column names so data and predictions are never ambiguous.

Run with:
    /opt/homebrew/Caskroom/miniforge/base/envs/pymc-marketing/bin/python \
        analysis/pymc_marketing/mmm_mg1655.py
"""

import os

# macOS 26 moves libc++ headers out of the SDK, so clang++ can't find <vector>
# regardless of -isysroot. Disable C compilation entirely; PyTensor falls back
# to NumPy for graph bookkeeping. The MCMC itself runs in JAX (nuts_sampler=
# "numpyro"), so sampling speed is unaffected.
os.environ["PYTENSOR_FLAGS"] = "cxx="

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import pymc as pm
import arviz as az
from pymc_marketing.mmm import MMM, GeometricAdstock, LogisticSaturation
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Configuration ──────────────────────────────────────────────────────────────
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TC_PATH = os.path.join(
    ROOT, "Datasets", "MG1655_inLB_LongTraces", "ProcessedData", "trace_centered.csv"
)
OUT_DIR = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(OUT_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

CHANNEL_COLS = ["growth_rate", "length_birth", "division_ratio", "generationtime"]
RESPONSE_COL = "div_and_fold"
L_MAX = 8            # adstock lag window in generations
SINGLE_LID = 12.0    # longest lineage (261 generations)

# ── Data loading and lagged design ────────────────────────────────────────────

def build_lagged_dataset(tc: pd.DataFrame) -> pd.DataFrame:
    """
    Pair channels at generation n with div_and_fold at generation n+1.
    Lineage boundaries are respected: no pairing across lineages.
    """
    records = []
    for lid in sorted(tc["lineage_ID"].unique()):
        lin = (
            tc[tc["lineage_ID"] == lid]
            .sort_values("generation")
            .reset_index(drop=True)
        )
        valid = lin.dropna(subset=CHANNEL_COLS + [RESPONSE_COL]).reset_index(drop=True)
        if len(valid) < L_MAX + 5:
            continue
        for i in range(len(valid) - 1):
            row = {
                "lineage_ID": lid,
                "generation": int(valid["generation"].iloc[i]),
            }
            for col in CHANNEL_COLS:
                row[col] = valid[col].iloc[i]          # channels at n (TC)
            row[RESPONSE_COL] = valid[RESPONSE_COL].iloc[i + 1]  # response at n+1 (TC)
            records.append(row)
    return pd.DataFrame(records)


tc = pd.read_csv(TC_PATH)
model_input = build_lagged_dataset(tc)
model_input.to_csv(os.path.join(OUT_DIR, "mg1655_model_input.csv"), index=False)
print(f"Model input: {len(model_input)} rows, {model_input['lineage_ID'].nunique()} lineages")
print(f"  Columns: {list(model_input.columns)}")

# ── Stage A: Single-lineage pymc-marketing MMM ────────────────────────────────
print("\n── Stage A: Single-lineage MMM (lineage 12) ──")

lin_df = (
    model_input[model_input["lineage_ID"] == SINGLE_LID]
    .copy()
    .reset_index(drop=True)
)
# Re-index generation to 0, 1, 2, ... for the MMM date column
lin_df["generation"] = np.arange(len(lin_df))

X_single = lin_df[["generation"] + CHANNEL_COLS]
y_single = lin_df[RESPONSE_COL]

mmm = MMM(
    date_column="generation",
    channel_columns=CHANNEL_COLS,
    adstock=GeometricAdstock(l_max=L_MAX),
    saturation=LogisticSaturation(),
    yearly_seasonality=None,
)

mmm.fit(
    X=X_single,
    y=y_single,
    nuts_sampler="numpyro",
    chains=2,
    draws=1000,
    tune=1000,
    target_accept=0.9,
    random_seed=RANDOM_SEED,
    progressbar=True,
)

# Sample posterior predictive so idata contains both observed and predicted y
mmm.sample_posterior_predictive(X_single, extend_idata=True, random_seed=RANDOM_SEED)

az.to_netcdf(mmm.idata, os.path.join(OUT_DIR, "idata_single_lineage.nc"))
print("  Saved idata_single_lineage.nc")

# Stage A results ─────────────────────────────────────────────────────────────
idata_s = mmm.idata
post_s = idata_s.posterior

# Posterior predictive predictions
y_pp = idata_s.posterior_predictive["y"]           # (chain, draw, y_dim_0)
pred_mean = y_pp.mean(dim=["chain", "draw"]).values
hdi_pp = az.hdi(idata_s.posterior_predictive, hdi_prob=0.89)["y"].values  # (n, 2)

# Per-channel contributions (original scale = TC units, since input was TC)
ch_contrib = post_s["channel_contribution_original_scale"]  # (chain, draw, date, channel)
ch_contrib_mean = ch_contrib.mean(dim=["chain", "draw"]).values  # (n_time, n_channels)

results_a = lin_df.copy()
results_a[f"{RESPONSE_COL}_obs"] = y_single.values          # what was observed (TC)
results_a[f"{RESPONSE_COL}_pred_mean"] = pred_mean           # posterior mean prediction
results_a[f"{RESPONSE_COL}_pred_hdi_lo"] = hdi_pp[:, 0]     # 89% HDI lower
results_a[f"{RESPONSE_COL}_pred_hdi_hi"] = hdi_pp[:, 1]     # 89% HDI upper
for j, ch in enumerate(CHANNEL_COLS):
    results_a[f"{ch}_contribution_mean"] = ch_contrib_mean[:, j]

results_a.to_csv(os.path.join(OUT_DIR, "results_single_lineage.csv"), index=False)
print("  Saved results_single_lineage.csv")

# Stage A summary stats
alpha_post = post_s["adstock_alpha"].values  # (chain, draw, channel)
print("\n  Posterior median adstock alpha per channel:")
for j, ch in enumerate(CHANNEL_COLS):
    median = np.median(alpha_post[:, :, j])
    q25, q75 = np.percentile(alpha_post[:, :, j], [25, 75])
    print(f"    {ch:20s}: {median:.3f}  [{q25:.3f}, {q75:.3f}]")

# Stage A plots ───────────────────────────────────────────────────────────────
gens = results_a["generation"].values
fig, axes = plt.subplots(2, 2, figsize=(14, 8), tight_layout=True)

ax = axes[0, 0]
ax.plot(gens, results_a[f"{RESPONSE_COL}_obs"], lw=1, alpha=0.7, label="observed")
ax.plot(gens, results_a[f"{RESPONSE_COL}_pred_mean"], lw=1.5, color="C1", label="predicted mean")
ax.fill_between(
    gens,
    results_a[f"{RESPONSE_COL}_pred_hdi_lo"],
    results_a[f"{RESPONSE_COL}_pred_hdi_hi"],
    alpha=0.25, color="C1", label="89% HDI",
)
ax.axhline(0, color="k", lw=0.5, ls="--")
ax.set_xlabel("Generation")
ax.set_ylabel("div_and_fold (TC)")
ax.set_title(f"Lineage {int(SINGLE_LID)} — Observed vs Predicted")
ax.legend(fontsize=8)

ax = axes[0, 1]
for j, ch in enumerate(CHANNEL_COLS):
    ax.plot(gens, results_a[f"{ch}_contribution_mean"], label=ch, lw=1, alpha=0.8)
ax.axhline(0, color="k", lw=0.5, ls="--")
ax.set_xlabel("Generation")
ax.set_ylabel("Contribution (TC)")
ax.set_title("Per-Channel Contributions over Time")
ax.legend(fontsize=8)

ax = axes[1, 0]
for j, ch in enumerate(CHANNEL_COLS):
    samples = alpha_post[:, :, j].flatten()
    az.plot_kde(samples, label=ch, ax=ax)
ax.set_xlabel("Adstock decay α")
ax.set_title("Adstock Decay Posteriors")
ax.legend(fontsize=8)

ax = axes[1, 1]
mean_abs_contrib = [np.abs(results_a[f"{ch}_contribution_mean"]).mean() for ch in CHANNEL_COLS]
ax.bar(CHANNEL_COLS, mean_abs_contrib, color=[f"C{i}" for i in range(len(CHANNEL_COLS))])
ax.set_ylabel("Mean |contribution| (TC)")
ax.set_title("Mean Absolute Channel Contribution")
plt.setp(ax.get_xticklabels(), rotation=15, ha="right")

fig.suptitle("Stage A — Single Lineage MMM (MG1655_inLB)", fontsize=12, fontweight="bold")
fig.savefig(os.path.join(FIG_DIR, "stage_a_summary.png"), dpi=150)
plt.close()
print("  Saved figures/stage_a_summary.png")

# ── Stage B: Hierarchical model with partial pooling ──────────────────────────
print("\n── Stage B: Hierarchical MMM (partial pooling, all lineages) ──")

# Use Stage A posterior median adstock alpha per channel for preprocessing.
# This fixes the adstock transform and lets Stage B focus on partial pooling
# of channel weights (betas) across lineages.
alpha_medians = np.median(alpha_post, axis=(0, 1))  # (n_channels,)
print(f"  Using fixed adstock alphas: { {ch: f'{a:.3f}' for ch, a in zip(CHANNEL_COLS, alpha_medians)} }")


def geometric_adstock_np(x: np.ndarray, alpha: float, l_max: int) -> np.ndarray:
    """Causal geometric adstock: x̃[t] = Σ_{lag=0}^{l_max} alpha^lag * x[t-lag]."""
    weights = alpha ** np.arange(l_max + 1)
    result = np.zeros_like(x, dtype=float)
    for t in range(len(x)):
        for lag in range(min(l_max + 1, t + 1)):
            result[t] += weights[lag] * x[t - lag]
    return result


def logistic_saturation_np(x: np.ndarray, lam: float) -> np.ndarray:
    """Logistic saturation: x / (1 + |lam * x|) — matches pymc-marketing default."""
    return x / (1.0 + np.abs(lam * x))


# Use Stage A posterior median lam per channel for saturation preprocessing
lam_post = post_s["saturation_lam"].values  # (chain, draw, channel)
lam_medians = np.median(lam_post, axis=(0, 1))
print(f"  Using fixed saturation lam:   { {ch: f'{l:.3f}' for ch, l in zip(CHANNEL_COLS, lam_medians)} }")

# Build preprocessed feature array for all lineages
lineage_ids = sorted(model_input["lineage_ID"].unique())
n_lineages = len(lineage_ids)
n_channels = len(CHANNEL_COLS)
lid_to_idx = {lid: i for i, lid in enumerate(lineage_ids)}

# Collect per-lineage arrays (different lengths)
lineage_X_list = []  # list of (n_t, n_channels) arrays — adstocked + saturated
lineage_y_list = []  # list of (n_t,) arrays
lineage_idx_list = []  # list of int — which lineage index

for lid in lineage_ids:
    sub = model_input[model_input["lineage_ID"] == lid].sort_values("generation").reset_index(drop=True)
    X_raw = sub[CHANNEL_COLS].values  # (n_t, n_channels)

    X_transformed = np.zeros_like(X_raw)
    for c in range(n_channels):
        adstocked = geometric_adstock_np(X_raw[:, c], alpha_medians[c], L_MAX)
        X_transformed[:, c] = logistic_saturation_np(adstocked, lam_medians[c])

    lineage_X_list.append(X_transformed)
    lineage_y_list.append(sub[RESPONSE_COL].values)
    lineage_idx_list.append(lid_to_idx[lid])

# Stack into flat arrays with a lineage index vector
X_flat = np.vstack(lineage_X_list)           # (N_total, n_channels)
y_flat = np.concatenate(lineage_y_list)       # (N_total,)
lin_idx_flat = np.concatenate([
    np.full(len(lineage_X_list[i]), lineage_idx_list[i], dtype=int)
    for i in range(n_lineages)
])

print(f"  Hierarchical design: {X_flat.shape[0]} observations, {n_lineages} lineages")

# Hierarchical PyMC model ─────────────────────────────────────────────────────
# Partial pooling structure:
#   intercept[l]  ~ Normal(mu_intercept, sigma_intercept)   per-lineage baseline
#   beta[l, c]    ~ Normal(mu_beta[c], sigma_beta[c])       per-lineage channel weights
# Both use non-centered parameterization for better geometry.

coords = {
    "lineage": lineage_ids,
    "channel": CHANNEL_COLS,
    "obs": np.arange(len(y_flat)),
}

with pm.Model(coords=coords) as hier_model:
    # ── Hyperpriors ──────────────────────────────────────────────────────────
    mu_intercept = pm.Normal("mu_intercept", mu=0.0, sigma=0.2)
    sigma_intercept = pm.HalfNormal("sigma_intercept", sigma=0.1)

    mu_beta = pm.Normal("mu_beta", mu=0.0, sigma=0.5, dims="channel")
    sigma_beta = pm.HalfNormal("sigma_beta", sigma=0.3, dims="channel")

    sigma_obs = pm.HalfNormal("sigma_obs", sigma=0.3)

    # ── Per-lineage parameters (non-centered) ────────────────────────────────
    intercept_offset = pm.Normal("intercept_offset", mu=0.0, sigma=1.0, dims="lineage")
    intercept = pm.Deterministic(
        "intercept", mu_intercept + sigma_intercept * intercept_offset, dims="lineage"
    )

    beta_offset = pm.Normal("beta_offset", mu=0.0, sigma=1.0, dims=("lineage", "channel"))
    beta = pm.Deterministic(
        "beta", mu_beta[None, :] + sigma_beta[None, :] * beta_offset, dims=("lineage", "channel")
    )

    # ── Likelihood ───────────────────────────────────────────────────────────
    # Data are indexed by lin_idx_flat (which lineage each observation belongs to)
    X_data = pm.Data("X_data", X_flat, dims=("obs", "channel"))
    y_data = pm.Data("y_data", y_flat, dims="obs")
    lin_idx = pm.Data("lin_idx", lin_idx_flat, dims="obs")

    mu_obs = (
        intercept[lin_idx]                                    # (N,)
        + (X_data * beta[lin_idx, :]).sum(axis=-1)           # (N,)
    )

    y_obs = pm.Normal("y_obs", mu=mu_obs, sigma=sigma_obs, observed=y_data, dims="obs")

# Sample
with hier_model:
    idata_hier = pm.sample(
        draws=1000,
        tune=1000,
        chains=2,
        target_accept=0.9,
        random_seed=RANDOM_SEED,
        progressbar=True,
    )
    pm.sample_posterior_predictive(
        idata_hier,
        extend_inferencedata=True,
        random_seed=RANDOM_SEED,
    )

az.to_netcdf(idata_hier, os.path.join(OUT_DIR, "idata_hierarchical.nc"))
print("  Saved idata_hierarchical.nc")

# Stage B results DataFrame ───────────────────────────────────────────────────
post_h = idata_hier.posterior

y_pp_h = idata_hier.posterior_predictive["y_obs"]  # (chain, draw, obs)
pred_mean_h = y_pp_h.mean(dim=["chain", "draw"]).values
hdi_h = az.hdi(idata_hier.posterior_predictive, hdi_prob=0.89)["y_obs"].values  # (N, 2)

beta_mean = post_h["beta"].mean(dim=["chain", "draw"]).values  # (n_lineages, n_channels)

results_b = model_input.copy().sort_values(["lineage_ID", "generation"]).reset_index(drop=True)
results_b[f"{RESPONSE_COL}_obs"] = y_flat          # observed (TC)
results_b[f"{RESPONSE_COL}_pred_mean"] = pred_mean_h
results_b[f"{RESPONSE_COL}_pred_hdi_lo"] = hdi_h[:, 0]
results_b[f"{RESPONSE_COL}_pred_hdi_hi"] = hdi_h[:, 1]

# Per-lineage, per-channel contribution: beta[l,c] * X_transformed[obs, c]
for c, ch in enumerate(CHANNEL_COLS):
    contribs = np.array([
        beta_mean[lin_idx_flat[i], c] * X_flat[i, c]
        for i in range(len(y_flat))
    ])
    results_b[f"{ch}_contribution_mean"] = contribs

results_b.to_csv(os.path.join(OUT_DIR, "results_hierarchical.csv"), index=False)
print("  Saved results_hierarchical.csv")

# Stage B plots ───────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 8), tight_layout=True)

# 1. Observed vs predicted (all lineages, scatter)
ax = axes[0, 0]
ax.scatter(
    results_b[f"{RESPONSE_COL}_obs"],
    results_b[f"{RESPONSE_COL}_pred_mean"],
    s=2, alpha=0.2, color="C0",
)
lims = [results_b[f"{RESPONSE_COL}_obs"].min(), results_b[f"{RESPONSE_COL}_obs"].max()]
ax.plot(lims, lims, "k--", lw=1)
ax.set_xlabel("Observed div_and_fold (TC)")
ax.set_ylabel("Predicted mean")
ax.set_title("All Lineages — Observed vs Predicted")

# 2. Posterior mean beta per lineage for each channel
ax = axes[0, 1]
x_pos = np.arange(n_lineages)
width = 0.2
for c, ch in enumerate(CHANNEL_COLS):
    ax.bar(x_pos + c * width, beta_mean[:, c], width=width, label=ch, alpha=0.8)
ax.axhline(0, color="k", lw=0.5, ls="--")
ax.set_xticks(x_pos + width * 1.5)
ax.set_xticklabels([str(int(lid)) for lid in lineage_ids], rotation=45, ha="right", fontsize=7)
ax.set_xlabel("Lineage ID")
ax.set_ylabel("Posterior mean beta")
ax.set_title("Per-Lineage Channel Weights (partial pooling)")
ax.legend(fontsize=7)

# 3. Hyperprior mu_beta posteriors
ax = axes[1, 0]
mu_beta_post = post_h["mu_beta"].values  # (chain, draw, channel)
for c, ch in enumerate(CHANNEL_COLS):
    az.plot_kde(mu_beta_post[:, :, c].flatten(), label=ch, ax=ax)
ax.axvline(0, color="k", lw=0.5, ls="--")
ax.set_xlabel("mu_beta (population-level channel weight)")
ax.set_title("Population-Level Channel Weight Posteriors")
ax.legend(fontsize=8)

# 4. sigma_beta: between-lineage variability per channel
ax = axes[1, 1]
sigma_beta_post = post_h["sigma_beta"].values  # (chain, draw, channel)
for c, ch in enumerate(CHANNEL_COLS):
    az.plot_kde(sigma_beta_post[:, :, c].flatten(), label=ch, ax=ax)
ax.set_xlabel("sigma_beta (between-lineage variability)")
ax.set_title("Between-Lineage Variability in Channel Weights")
ax.legend(fontsize=8)

fig.suptitle("Stage B — Hierarchical MMM (MG1655_inLB)", fontsize=12, fontweight="bold")
fig.savefig(os.path.join(FIG_DIR, "stage_b_summary.png"), dpi=150)
plt.close()
print("  Saved figures/stage_b_summary.png")

# ── Final summary ─────────────────────────────────────────────────────────────
print("\n── Output files ──")
for fname in [
    "mg1655_model_input.csv",
    "idata_single_lineage.nc",
    "idata_hierarchical.nc",
    "results_single_lineage.csv",
    "results_hierarchical.csv",
    "figures/stage_a_summary.png",
    "figures/stage_b_summary.png",
]:
    path = os.path.join(OUT_DIR, fname)
    exists = "✓" if os.path.exists(path) else "✗"
    print(f"  {exists}  {fname}")

print("\nDone.")
