# Run 20251112_182507 — λ sweep with 5-epoch warm-up

**Warm-up schedule**: predictor-only for epochs 0–4, adversarial training from epoch 5 onward.

| λ_adv | ROC AUC | EO ΔTPR | EO ΔFPR | Selection rate₀ | Selection rate₁ | DP diff | DP ratio | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.05 | 0.7692 | 0.0242 | 0.0035 | 0.01023 | 0.00575 | +0.00448 | 4.50 | Accuracy best; EO gap still noticeable. |
| 0.10 | 0.7666 | 0.0000 | 0.0000 | 0.0 | 0.0 | 0.0 | — | Model predicts no positives at threshold 0.5. |
| 0.20 | 0.7666 | 0.0275 | 0.0023 | 0.01271 | 0.00695 | +0.00427 | 2.59 | Similar to λ=0.05 but slightly lower accuracy. |
| 0.30 | 0.7577 | 0.0307 | 0.0025 | 0.01735 | 0.01031 | +0.00737 | 3.51 | EO gap grows; accuracy dips. |
| 0.50 | 0.7593 | 0.0032 | 0.0011 | 0.00120 | 0.00048 | −0.00016 | 0.75 | Strong EO suppression with low DP gap. |
| 0.80 | 0.7696 | 0.0075 | 0.0004 | 0.00409 | 0.00384 | +0.00167 | 1.77 | Best fairness/accuracy balance in this run. |
| 1.20 | 0.7308 | 0.0011 | 0.0002 | 0.00163 | 0.00048 | +0.00035 | 3.75 | Fairness strong, accuracy drops substantially. |
| 2.00 | 0.7554 | 0.0074 | 0.0010 | 0.00233 | 0.00168 | +0.00066 | 1.64 | Accuracy rebounds slightly; EO gap small. |

**Observations**
- Warm-up of 5 epochs keeps training stable, but λ=0.1 still collapses to zero positives (suggests thresholding/calibration issue rather than training failure).
- λ≈0.5–0.8 gives the cleanest accuracy–fairness trade-offs (AUC ≥ 0.759 with EO gaps ≤ 0.01). Larger λ reduces accuracy markedly without further fairness gains.

Artifacts: `lambda_sweep/metrics.csv` (plus plot if generated).***
