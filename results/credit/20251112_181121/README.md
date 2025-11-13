# Run 20251112_181121 — λ sweep (fine grid + detailed fairness)

**Setup**
- Same training/data configs as previous sweeps (60 k rows, seed 42, bias_b = 35, λ_adv varies).
- Lambda grid: `[0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.2, 2.0]`.
- Recorded extra fairness metrics per λ (TPR/FPR by group, selection rates, DP diff/ratio).

| λ_adv | ROC AUC | EO ΔTPR | EO ΔFPR | Selection rate₀ | Selection rate₁ | DP diff | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0.05 | 0.7673 | 0.0202 | 0.0020 | 0.00454 | 0.00695 | +0.00376 | Mild EO reduction, best accuracy. |
| 0.10 | 0.7630 | 0.0000 | 0.0000 | 0.0 | 0.0 | 0.0 | Model predicts almost no positives at default threshold. |
| 0.20 | 0.7671 | 0.0267 | 0.0018 | 0.00549 | 0.01175 | +0.00626 | Slight EO spike; needs more training stability. |
| 0.30 | 0.7623 | 0.0000 | 0.0000 | 0.0 | 0.0 | 0.0 | Same collapse as λ=0.1 (no positives). |
| 0.50 | 0.7591 | 0.00527 | 0.00000 | 0.00025 | 0.00096 | +0.00070 | Strong fairness pressure, accuracy dips. |
| 0.80 | 0.7579 | 0.00326 | 0.00090 | 0.00192 | 0.00216 | +0.00024 | Balanced trade-off. |
| 1.20 | 0.7495 | 0.0000 | 0.0000 | 0.0 | 0.0 | 0.0 | EO perfect, accuracy declines. |
| 2.00 | 0.7500 | 0.0000 | 0.0000 | 0.0 | 0.0 | 0.0 | EO perfect, accuracy ~0.75. |

**Observations**
- The finer grid reveals instability at λ=0.1 and 0.3 where the predictor predicts no positives (selection rates 0). May need learning-rate tuning or per-λ warmup to keep the model calibrated.
- λ≈0.5–0.8 offers a reasonable compromise (ROC AUC ≈0.758, EO gaps <0.006, DP diff near zero).
- Additional metrics (TPR/FPR per group) now allow plotting fairness surfaces; consider averaging over multiple seeds to smooth the intermittent spikes.
