#!/usr/bin/env python3
"""
Accuracy evaluation and length-trace reconstruction for the single-lineage MMM.

Loads results_single_lineage.csv (predictions in TC units) and physical_units.csv
(observed values in physical units), then:

  1. Accuracy metrics on div_and_fold predictions (R², MAE, RMSE, 89% HDI coverage)
  2. Converts TC predictions back to physical units by adding the lineage mean
  3. Reconstructs length_final trace via cumulative product of predicted div_and_fold
  4. Plots predicted vs observed for both div_and_fold (TC) and length_final (µm)

Run from repo root after mmm_mg1655.py or test_single_chain.py has completed:
    /opt/homebrew/Caskroom/miniforge/base/envs/pymc-marketing/bin/python \
        analysis/pymc_marketing/evaluate.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT    = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT_DIR = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(OUT_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

RESULTS_PATH = os.path.join(OUT_DIR, "results_single_lineage.csv")
PU_PATH      = os.path.join(ROOT, "Datasets", "MG1655_inLB_LongTraces", "ProcessedData", "physical_units.csv")
SINGLE_LID   = 12.0

# ── Load data ─────────────────────────────────────────────────────────────────
results = pd.read_csv(RESULTS_PATH)   # predictions in TC units
pu      = pd.read_csv(PU_PATH)

lin_pu = (
    pu[pu["lineage_ID"] == SINGLE_LID]
    .sort_values("generation")
    .reset_index(drop=True)
)

# Lineage mean of div_and_fold in physical units (used to un-center predictions)
# TC = PU - mean(PU), so PU = TC + mean(PU)
daf_mean_pu = lin_pu["div_and_fold"].mean()
print(f"Lineage {int(SINGLE_LID)}  |  div_and_fold PU mean: {daf_mean_pu:.4f}  (≈1 = perfect homeostasis)")
print(f"Prediction rows: {len(results)}")

# ── 1. Accuracy metrics on div_and_fold (TC) ──────────────────────────────────
obs  = results["div_and_fold_obs"].values
pred = results["div_and_fold_pred_mean"].values
hdi_lo = results["div_and_fold_pred_hdi_lo"].values
hdi_hi = results["div_and_fold_pred_hdi_hi"].values

ss_res = np.sum((obs - pred) ** 2)
ss_tot = np.sum((obs - obs.mean()) ** 2)
r2     = 1 - ss_res / ss_tot
mae    = np.mean(np.abs(obs - pred))
rmse   = np.sqrt(np.mean((obs - pred) ** 2))
coverage = np.mean((obs >= hdi_lo) & (obs <= hdi_hi))

print(f"\n── Accuracy on div_and_fold (TC) ──")
print(f"  R²       : {r2:.4f}")
print(f"  MAE      : {mae:.4f}")
print(f"  RMSE     : {rmse:.4f}")
print(f"  89% HDI coverage: {coverage:.2%}  (should be ≈89%)")

# ── 2. Convert predictions to physical units ───────────────────────────────────
# The model used trace-centered channels to predict trace-centered div_and_fold.
# Adding back the lineage mean recovers the physical-unit prediction.
pred_pu    = pred    + daf_mean_pu
hdi_lo_pu  = hdi_lo + daf_mean_pu
hdi_hi_pu  = hdi_hi + daf_mean_pu

# Observed div_and_fold in physical units (from PU data, aligned to model rows)
# The model pairs channels at gen n with div_and_fold at gen n+1, so we need to
# match by generation index carefully.
obs_pu = lin_pu["div_and_fold"].dropna().reset_index(drop=True)

# ── 3. Reconstruct length_final trace ─────────────────────────────────────────
# div_and_fold_n = length_final_n / length_final_{n-1}
# => length_final_n = length_final_0 × ∏_{k=1}^{n} div_and_fold_k
#
# We use the known length_final at the first valid generation as the anchor.
# The lagged design means results["generation"] is the channel generation (n),
# and pred_pu corresponds to div_and_fold at generation n+1.
# Align: anchor at the first div_and_fold generation available in PU.

valid_pu = lin_pu.dropna(subset=["div_and_fold", "length_final"]).reset_index(drop=True)
anchor_length_final = valid_pu["length_final"].iloc[0]   # known starting point

# Observed length_final from physical units (aligned to the same generations)
obs_length_final = valid_pu["length_final"].values        # (N,) in µm

# Reconstruct using predicted div_and_fold (physical units)
# We have pred_pu[i] = predicted div_and_fold at generation (results.generation[i] + 1)
# Build cumulative product starting from anchor
n_pred = len(pred_pu)
reconstructed_length_final = np.zeros(n_pred + 1)
reconstructed_length_final[0] = anchor_length_final
for i in range(n_pred):
    reconstructed_length_final[i + 1] = reconstructed_length_final[i] * pred_pu[i]

# HDI bands on reconstructed length (propagate HDI multiplicatively)
recon_lo = np.zeros(n_pred + 1)
recon_hi = np.zeros(n_pred + 1)
recon_lo[0] = recon_hi[0] = anchor_length_final
for i in range(n_pred):
    recon_lo[i + 1] = recon_lo[i] * hdi_lo_pu[i]
    recon_hi[i + 1] = recon_hi[i] * hdi_hi_pu[i]

# Accuracy on reconstructed length_final (trim to overlap length)
n_overlap = min(len(obs_length_final), len(reconstructed_length_final))
obs_lf_trim   = obs_length_final[:n_overlap]
pred_lf_trim  = reconstructed_length_final[:n_overlap]

ss_res_lf = np.sum((obs_lf_trim - pred_lf_trim) ** 2)
ss_tot_lf = np.sum((obs_lf_trim - obs_lf_trim.mean()) ** 2)
r2_lf  = 1 - ss_res_lf / ss_tot_lf
mae_lf = np.mean(np.abs(obs_lf_trim - pred_lf_trim))

print(f"\n── Accuracy on reconstructed length_final (µm) ──")
print(f"  R²   : {r2_lf:.4f}  (degrades over time due to error compounding)")
print(f"  MAE  : {mae_lf:.4f} µm")

# ── 4. Plots ───────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 8), tight_layout=True)

# (a) div_and_fold TC: observed vs predicted over time
ax = axes[0, 0]
gens = results["generation"].values
ax.plot(gens, obs,  lw=1,   alpha=0.7, label="observed (TC)")
ax.plot(gens, pred, lw=1.5, color="C1", label="predicted mean (TC)")
ax.fill_between(gens, hdi_lo, hdi_hi, alpha=0.25, color="C1", label="89% HDI")
ax.axhline(0, color="k", lw=0.5, ls="--")
ax.set_xlabel("Generation")
ax.set_ylabel("div_and_fold (TC)")
ax.set_title(f"div_and_fold — Observed vs Predicted  (R²={r2:.3f})")
ax.legend(fontsize=8)

# (b) div_and_fold: scatter of obs vs pred (TC)
ax = axes[0, 1]
ax.scatter(obs, pred, s=6, alpha=0.4, color="C0")
lims = [min(obs.min(), pred.min()), max(obs.max(), pred.max())]
ax.plot(lims, lims, "k--", lw=1, label="y = x")
ax.set_xlabel("Observed div_and_fold (TC)")
ax.set_ylabel("Predicted mean (TC)")
ax.set_title(f"Predicted vs Observed Scatter  (MAE={mae:.4f})")
ax.legend(fontsize=8)

# (c) Reconstructed length_final over generations
ax = axes[1, 0]
gen_anchor = valid_pu["generation"].iloc[0]
gen_recon  = np.arange(n_pred + 1) + gen_anchor
ax.plot(valid_pu["generation"].values[:n_overlap], obs_lf_trim,
        lw=1, alpha=0.8, label="observed (µm)")
ax.plot(gen_recon[:n_overlap], pred_lf_trim,
        lw=1.5, color="C1", label="reconstructed mean (µm)")
ax.fill_between(gen_recon[:n_overlap],
                recon_lo[:n_overlap], recon_hi[:n_overlap],
                alpha=0.2, color="C1", label="89% HDI (propagated)")
ax.set_xlabel("Generation")
ax.set_ylabel("length_final (µm)")
ax.set_title(f"Reconstructed length_final  (R²={r2_lf:.3f}, MAE={mae_lf:.3f} µm)")
ax.legend(fontsize=8)

# (d) Cumulative drift: |reconstructed - observed| over generation
ax = axes[1, 1]
drift = np.abs(pred_lf_trim - obs_lf_trim)
ax.plot(valid_pu["generation"].values[:n_overlap], drift, lw=1, color="C3")
ax.set_xlabel("Generation")
ax.set_ylabel("|reconstructed − observed|  (µm)")
ax.set_title("Cumulative Error Drift in Reconstructed Length")

fig.suptitle(
    f"Evaluation — Lineage {int(SINGLE_LID)} (MG1655_inLB)",
    fontsize=12, fontweight="bold",
)
fig.savefig(os.path.join(FIG_DIR, "evaluation.png"), dpi=150)
plt.close()
print(f"\nSaved figures/evaluation.png")
