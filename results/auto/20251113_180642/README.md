# Auto baseline run 2025-11-13 18:06 (2nd execution)

**Context:** Repeat of the baseline to capture deterministic artifacts after smoothing driver/vehicle draws. Same `python -m src.experiments.auto.baseline` config, with the debug histograms written to `debug_auto_distributions/`.

**Key notes:**

- GLM/NN/ADV_NN ROC AUCs improved (≈0.76/0.79/0.81) compared to the first run, thanks to the more stable sampling seeds and refreshed feature normalization.
- ADV-NN continues to trade a small accuracy drop for lower EO/DP gaps; `dp_ratio_fixed_r ≈ 2.93` vs GLM’s ~5.8.
- Fixed-rate fairness (2% target) shows a gap of ~0.07 in DP difference for GLM versus ~0.06 for NN and ~0.10 for ADV-NN, so the adversary still reduces disparity while maintaining an ROCAUC ≈ 0.81.

**Artifacts:**

- `baseline_results.csv` – tabular metrics for all three models.
- `debug_auto_distributions/` – histograms like `auto_annual_mileage.png`, `auto_premium_charged.png` documenting the simulated feature distributions.
- `results/auto/fairness_accuracy_frontier.png` – the frontier plot produced by `python -m src.experiments.auto.fairness_frontier` using this run’s metrics.

**Next steps:** Use this run as the canonical auto baseline and iterate on `bias_strength` / pricing coefficients before re-running to compare DP/EO improvements.
