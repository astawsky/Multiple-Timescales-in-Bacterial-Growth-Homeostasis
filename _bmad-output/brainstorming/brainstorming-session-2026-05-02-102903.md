---
stepsCompleted: [1]
inputDocuments: []
session_topic: 'Bayesian model comparison framework for predicting phenotypic variables using subsets of mother/daughter cell variables'
session_goals: 'Code/analysis plan — a framework of PyMC Bayesian linear regression models with different subsets of correlated mother/daughter predictors, enabling model comparison to determine which variable subsets improve prediction vs. simpler models'
selected_approach: 'ai-recommended'
techniques_used: ['First Principles Thinking', 'Morphological Analysis', 'Decision Tree Mapping']
ideas_generated: []
context_file: ''
---

# Brainstorming Session Results

**Facilitator:** Alejandro
**Date:** 2026-05-02

## Session Overview

**Topic:** Bayesian model comparison framework for predicting phenotypic variables using subsets of mother/daughter cell variables in E. coli growth homeostasis data

**Goals:** Build a code/analysis plan for a PyMC Bayesian linear regression framework where:
- All predictor variables are fully correlated (correlated design matrix)
- Models differ by which subset of "mother" and/or "daughter" variables are included as predictors
- Model comparison determines whether adding more variables meaningfully improves prediction or adds unnecessary assumptions

### Session Setup

_Fresh session started 2026-05-02. Topic centers on statistical model selection for bacterial growth phenotype prediction using sister/mother machine data._

---

## Phase 2: Morphological Analysis — Fixed Decisions

| Axis | Decision | Value |
|------|----------|-------|
| 1 — Targets | Minimum non-overlapping primitives, swept | {τ_d, α_d, f_d} — partition into targets vs. daughter predictors |
| 2 — Mother predictors | All subsets swept | 2^4 = 16 combinations of {τ_m, α_m, f_m, Sb_m_TC} |
| 3 — Daughter predictors | D4 | All non-target daughter primitives (complement of target set) |
| 4 — Representation | Mundlak | TC + TA for all mother predictors {τ_m, α_m, f_m, Sb_m} uniformly |
| 5 — Architecture | Multivariate | Joint PyMC model with LKJ residual covariance |
| 6 — Priors | P2 + P4 | Homeostasis prior universally applicable G1–G7 with learned σ_h ~ HalfNormal (per-model, sampled); LKJ on residual covariance |
| 7 — Metrics | Full suite | LOO-CV, WAIC, R², correlation fidelity (Frobenius), ELPD diff |

**Total models: 7 target partitions × 16 mother predictor subsets = 112 models**

**Comparison grouping: models are LOO-comparable only within the same target set → 7 groups of 16**

**Output location: new standalone script — `analysis/bayesian_model_comparison.py`**

## Phase 3: Decision Tree Mapping — Implementation Logic

### Node 1 — Model enumeration
Iterate over 7 target partitions × 16 mother predictor subsets. For each config: targets = subset of {τ_d, α_d, f_d}; daughter predictors = complement; null model (M0) in G2–G7 still includes daughter predictors.

### Node 2 — Design matrix construction
All predictors enter as Mundlak TC+TA pairs. Standardize TC and TA columns separately after split.

### Node 3 — PyMC model
MvStudentT likelihood (robust to outlier cells, ν as config flag). LKJ Cholesky prior on residual covariance (P4). Beta coefficients Normal(0,1) per predictor per target.

### Node 4 — Homeostasis potential (universal G1–G7)
`get_component(var, targets, mu_pred, d_preds)` returns predicted tensor if target, observed column if daughter predictor. Constraint `f × exp(α × τ) ~ Normal(1, σ_h)` with `σ_h ~ HalfNormal(1)` sampled per model.

### Node 5 — Results collection
Per model: LOO-CV, WAIC, R², correlation fidelity (Frobenius distance predicted vs. observed correlation matrix), σ_h posterior mean.

### Node 6 — Comparison logic
Within-group: rank by ELPD difference from null. Cross-group: extract per-observation log-likelihood for shared target variable, compare marginal LOO across groups.

---

## Phase 1: First Principles Thinking

**[FP #1]**: Practical Utility as the Loss Function
_Concept_: The winning model is not the one with smallest marginal residuals per variable, but the one that minimizes prediction error on daughter variables AND replicates the joint correlation structure (daughter-daughter and daughter-mother correlations).
_Novelty_: Evaluation criterion is a two-component score — not a single scalar — making correlation fidelity a first-class metric alongside predictive accuracy.

**[FP #2]**: Architecture is a Comparison Axis
_Concept_: Both multivariate (joint) and univariate (independent) model families are included in the framework. This turns the architecture choice into a finding: does modeling the joint distribution matter?
_Novelty_: Multivariate models have correlation structure in the likelihood (rewarded for getting it right); univariate models have it evaluated post-hoc. The asymmetry is itself a scientific result.

**[FP #3]**: The Variable Space Has Algebraic Dependency Structure
_Concept_: Phenotypic variables divide into primitives {τ, α, f, Sb_TC} and composed variables {fold_growth = exp(α×τ), added_size, division_size}. Valid predictor subsets must not mix a composed variable with its algebraic components.
_Novelty_: Valid subset space is much smaller than 2^n — it is constrained by algebraic independence. Composed variables are leakage-flagged, not simply excluded.

**[FP #4]**: Birth Size Valid Only in TC Form
_Concept_: Raw birth size (PU) is experiment-specific and not comparable across datasets. Only trace-centered birth size (Sb_TC) is a valid predictor for cross-experiment analysis.
_Novelty_: Data representation is a validity constraint, not just a modeling choice — using PU birth size is a category error for multi-experiment frameworks.

**[FP #5]**: Mundlak Device Replaces Hierarchical Model
_Concept_: Instead of sampling lineage-level random effects (expensive), include TC and TA representations of each predictor simultaneously. β_within is estimated from TC variation; β_between from TA variation. Simpson's paradox is resolved without hierarchical overhead.
_Novelty_: The existing PU/TC/TA data representations map directly onto the Mundlak decomposition — the infrastructure is already built. No new computation needed; just a design choice in how predictors enter the model.

**[FP #6]**: Homeostasis Constraint as a Prior
_Concept_: f × exp(α × τ) ≈ 1 is a biological law that constrains the joint distribution of {f, α, τ}. Rather than excluding these variables when used together, encode the constraint as a tight log-normal prior on their product.
_Novelty_: Turns a potential collinearity problem into a regularizing prior that embeds domain knowledge into the model structure.
