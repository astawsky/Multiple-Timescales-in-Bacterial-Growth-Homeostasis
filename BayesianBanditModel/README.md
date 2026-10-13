# Bayesian bandit model of growth homeostasis

Analysis added on 2026-10-13. A hierarchical Thompson-sampling adder model fitted to the 16 experiments in `Datasets/`.

- `growth_homeostasis_bandit.ipynb` — the full analysis, executed, with outputs (statistics, sisters vs neighbours, the two e^phi laws, the simulator, fit and validation).
- `growth_homeostasis_bandit_clean.ipynb` — same notebook without outputs.
- `fit_bandit_model.py` — fits the model to every experiment (`python fit_bandit_model.py` from this folder; writes `fit_results.json`).
- `growth-homeostasis-bandit*.html`, `how-the-model-was-built.html` — self-contained write-ups (v3 is the fitted/validated version).

Uses the repository's processed `z_score_under_3` files and the variable definitions in `AnalysisCode/global_variables.py`.
