# Run 20251112_183201 — Fixed-rate comparison (2% approval)

**Setup**
- `target_rate = 0.02` (approve top 2% globally).
- GLM on `[S,D,L,Z]`, PlainNN with validation split (seed 42), ADV_NN with `lambda_adv = 0.8`, 5 warm-up epochs.
- Dataset: default simulator (60 k rows, bias_b = 35, seed 42).

| Model | ROC AUC | PR AUC | EO ΔTPR | EO ΔFPR | DP diff | DP ratio | Threshold | Actual rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GLM | 0.772 | 0.337 | 0.072 | 0.016 | +0.022 | 2.77 | 0.471 | 0.020 |
| NN | 0.757 | 0.309 | 0.053 | 0.007 | +0.015 | 1.96 | 0.447 | 0.020 |
| ADV_NN | 0.759 | 0.325 | 0.050 | 0.006 | +0.014 | 2.05 | 0.424 | 0.020 |

**Observations**
- Enforcing a common approval rate significantly compresses DP ratios (all near 2) while EO gaps remain in the 0.05–0.07 range.
- ADV_NN trades a tiny bit of accuracy vs GLM but slightly improves EO/DP metrics under the fixed-rate constraint.
- Plain NN lags in accuracy but achieves comparable fairness once thresholded to 2%.

Artifacts: `fixed_rate/metrics.csv`.***
