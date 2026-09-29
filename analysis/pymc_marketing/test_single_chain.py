#!/usr/bin/env python3
"""
Single-chain smoke test for Stage A.
Run from the repo root:
    /opt/homebrew/Caskroom/miniforge/base/envs/pymc-marketing/bin/python \
        analysis/pymc_marketing/test_single_chain.py
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
import arviz as az
from pymc_marketing.mmm import MMM, GeometricAdstock, LogisticSaturation

# ── Config ────────────────────────────────────────────────────────────────────
RANDOM_SEED  = 42
CHANNEL_COLS = ["growth_rate", "length_birth", "division_ratio", "generationtime"]
RESPONSE_COL = "div_and_fold"
L_MAX        = 2
SINGLE_LID   = 12.0   # longest lineage (261 generations)

np.random.seed(RANDOM_SEED)

# ── Data ──────────────────────────────────────────────────────────────────────
tc = pd.read_csv(
    "Datasets/MG1655_inLB_LongTraces/ProcessedData/physical_units.csv"
)

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
        row = {"lineage_ID": lid, "generation": int(valid["generation"].iloc[i])}
        for col in CHANNEL_COLS:
            row[col] = valid[col].iloc[i]
        row[RESPONSE_COL] = valid[RESPONSE_COL].iloc[i + 1]
        records.append(row)

model_input = pd.DataFrame(records)

lin_df = (
    model_input[model_input["lineage_ID"] == SINGLE_LID]
    .copy()
    .reset_index(drop=True)
)
lin_df["generation"] = np.arange(len(lin_df))

X = lin_df[["generation"] + CHANNEL_COLS]
y = lin_df[RESPONSE_COL]

print(f"Rows: {len(lin_df)}  |  X shape: {X.shape}  |  y range: [{y.min():.3f}, {y.max():.3f}]")

# ── Model + fit ───────────────────────────────────────────────────────────────
mmm = MMM(
    date_column="generation",
    channel_columns=CHANNEL_COLS,
    adstock=GeometricAdstock(l_max=L_MAX),
    saturation=LogisticSaturation(),
    yearly_seasonality=None,
)

mmm.fit(
    X=X,
    y=y,
    nuts_sampler="numpyro",   # JAX backend — avoids C compilation, faster on Apple Silicon
    chains=4,
    draws=1000,
    tune=1000,
    target_accept=1.0,
    random_seed=RANDOM_SEED,
    progressbar=True,
)

# ── Diagnostics ───────────────────────────────────────────────────────────────
idata = mmm.idata
print()
print("=== Posterior summary ===")
print(
    az.summary(
        idata,
        var_names=["adstock_alpha", "saturation_lam", "saturation_beta", "intercept", "y_sigma"],
        round_to=3,
    ).to_string()
)

print()
print("=== Posterior variables ===")
print(list(idata.posterior.data_vars))

print()
