# Run 20251112_180036 — Baseline with fixed acceptance rate (2%)

**Settings**
- Simulator / training configs identical to earlier baseline (60 k rows, seed 42, bias_b = 35, λ_adv = 0.1).
- Added threshold adjustment so that each model’s *overall* selection rate equals 2% on the test set (target_acceptance_rate = 0.02).

## Default threshold (0.5)

| Model | ROC AUC | EO ΔTPR | EO ΔFPR | DP diff | DP ratio |
| --- | --- | --- | --- | --- | --- |
| GLM | 0.772 | 0.066 | 0.0149 | +0.0203 | 3.41 |
| NN  | 0.766 | 0.0287 | 0.0054 | +0.0079 | 2.82 |
| ADV_NN | 0.766 | 0.0359 | 0.0044 | +0.0079 | 4.64 |

## Fixed acceptance rate (r = 2%)

| Model | Threshold | EO ΔTPR | EO ΔFPR | DP diff | DP ratio |
| --- | --- | --- | --- | --- | --- |
| GLM | 0.4712 | 0.0723 | 0.0160 | +0.0219 | 2.77 |
| NN  | 0.4610 | 0.0703 | 0.0109 | +0.0171 | 2.22 |
| ADV_NN | 0.4288 | 0.0661 | 0.0152 | +0.0204 | 2.58 |

**Observations**
- Aligning acceptance rates reduces the extreme DP ratios (>3) seen at threshold 0.5, but EO gaps remain sizable (≈0.07) because fairness is still computed at a single threshold.
- ADV_NN and NN converge to similar EO/DP gaps once the target rate is enforced—useful for comparing policies at a fixed approval budget.

**Next steps**
1. Explore other target rates (e.g., 1%, 5%) to understand sensitivity.
2. Consider per-group thresholds if equal selection rates *and* EO are required simultaneously.
