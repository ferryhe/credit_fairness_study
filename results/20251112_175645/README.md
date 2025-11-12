# Run 20251112_175645 — Lambda sweep after adversarial loss tweak

**Change**: Gradient reversal now uses a fixed scale (λ applied only in loss weight) to avoid “double λ” pressure on the predictor. Expect a smoother accuracy/fairness trade-off as `lambda_adv` increases.

**Setup**
- Same data/configs as previous run (60 k samples, bias_b = 35, seed 42).
- Lambdas tested: `[0.1, 0.3, 0.5, 1.0, 3.0]`.

| λ_adv | ROC AUC | EO ΔTPR | EO ΔFPR | Observation |
| --- | --- | --- | --- | --- |
| 0.1 | 0.763 | 0.00000 | 0.00000 | Virtually no EO gap, accuracy ≈ GLM. |
| 0.3 | 0.764 | 0.00846 | 0.00092 | Mild EO suppression, accuracy stays high. |
| 0.5 | 0.758 | 0.00320 | 0.00002 | EO gap shrinks further, small AUC drop. |
| 1.0 | 0.749 | 0.00317 | 0.00054 | Accuracy falls toward 0.75, EO gap low but non-zero. |
| 3.0 | 0.752 | 0.00085 | 0.00050 | Accuracy stabilizes around 0.75 while EO gaps approach 0. |

**Takeaway**: The adjusted loss gives a smoother transition—AUC no longer collapses to 0.5 when EO gaps hit zero. For stronger fairness pressure (ΔTPR ≈ 0.02), consider exploring λ in `[1, 2]` or adding more epochs for the adversary.***
