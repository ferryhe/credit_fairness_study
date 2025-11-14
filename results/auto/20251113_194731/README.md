# Auto bias sweep run — 20251113_194731

Tracked the latest `python -m src.experiments.auto.bias_sweep` execution after tuning the fairness metrics helper so the logged EO/DP gaps now match the diagnostics.

## Why this run
- Validate the bias sweep after refactoring fairness metric computation.
- Confirm that the NN still suffers large EO/DP gaps at `bias_strength=2.0`, even though the aggregated CSV no longer reports zeros.

## Settings
- `bias_strength` sweep: 0.0, 0.25, 0.5, 1.0, 2.0 (same as previous runs).
- Models: `GLM`, `NN`, and `ADV_NN` with the same training/eval configs; fairness stats now come from `compute_fairness_metrics`.
- Global DP target: 2% acceptance for fixed-rate fairness metrics.

## Results
- The shared `metrics/auto_bias_sweep_metrics.csv` lists ROC AUC and fairness gaps per model; the NN at `bias_strength=2.0` now shows `eo_gap_tpr≈0.087` and `eo_gap_fpr≈0.302` with `dp_ratio_fixed_2pct≈1.57` (matching the diagnostic plots that explored the same threshold).
- The helpers log warnings when a ratio cannot be computed (e.g., no group-0 positives), which means the diagnostics will no longer silently drop to 0 or 1.

## Artifacts

- `metrics/auto_bias_sweep_metrics.csv`: aggregated metrics from this sweep (includes updated fairness numbers).
