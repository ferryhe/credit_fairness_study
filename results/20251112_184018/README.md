# Run 20251112_184018 — Fixed-rate comparison (1%, 2%, 5%)

**Setup**
- Approval rates: 1%, 2%, 5% (global top-scoring applicants).
- Models: GLM, Plain NN, ADV_NN (`λ = 0.8`, 5 warm-up epochs).
- Dataset: default simulation (60 k rows, bias_b = 35, seed 42).

| Target rate | Model | ROC AUC | EO ΔTPR | EO ΔFPR | DP diff | DP ratio |
| --- | --- | --- | --- | --- | --- | --- |
| 1% | GLM | 0.772 | 0.083 | 0.019 | +0.016 | 3.13 |
| 1% | NN | 0.751 | 0.060 | 0.008 | +0.012 | 3.62 |
| 1% | ADV_NN | 0.752 | 0.038 | 0.004 | +0.0016 | 1.01 |
| 2% | GLM | 0.772 | 0.072 | 0.016 | +0.022 | 2.77 |
| 2% | NN | 0.751 | 0.053 | 0.007 | +0.015 | 2.67 |
| 2% | ADV_NN | 0.752 | 0.050 | 0.006 | +0.013 | 1.27 |
| 5% | GLM | 0.772 | 0.046 | 0.010 | +0.031 | 2.22 |
| 5% | NN | 0.751 | 0.040 | 0.006 | +0.017 | 1.77 |
| 5% | ADV_NN | 0.752 | 0.034 | 0.004 | +0.009 | 1.10 |

**Observations**
- Lower approval rates amplify disparities for GLM/NN (DP ratio >3 at 1%), while ADV_NN keeps DP ratio ≈1 and EO gaps ~0.04.
- As approval rate increases to 5%, fairness gaps shrink for all models, and the advantage of ADV_NN becomes smaller but still present.
- Accuracy differences remain minor across target rates (GLM highest, NN≈ADV_NN).

Artifacts: `fixed_rate/metrics.csv`.
