# Run 20251112_175008

**Context**
- Simulator: default `SimulationConfig` (60 k rows, seed 42, `bias_b = 35`)
- Training: default `TrainingConfig` with updated `lambda_adv = 0.1`, 10 epochs, batch 1024
- Test split: 20 % stratified on `Y`
- Fairness threshold: 0.5

---

## Baseline comparison (`baseline/metrics.csv`)

| Model   | ROC AUC | PR AUC | EO ΔTPR | EO ΔFPR | DP diff | DP ratio |
|---------|---------|--------|---------|---------|---------|----------|
| GLM     | 0.772   | 0.337  | 0.066   | 0.015   | +0.0203 | 3.41 |
| NN      | 0.764   | 0.318  | 0.005   | 0.0003  | +0.0008 | 7.51 |
| ADV_NN  | 0.769   | 0.332  | 0.022   | 0.0045  | +0.0063 | 2.41 |

- ADV_NN matches GLM accuracy while shrinking EO gaps ≈3×.
- Plain NN needs calibration/threshold tuning; it nearly never flags group 0, causing DP ratio ≫ 1.

**Next steps:** tune thresholds (e.g., to fixed approval rate) and log calibration metrics.

---

## Lambda sweep (`lambda_sweep/metrics.csv`, optional `lambda_sweep_plot.png`)

| λ_adv | ROC AUC | EO ΔTPR | EO ΔFPR | Note |
|-------|---------|---------|---------|------|
| 0.1   | 0.761   | 0.00000 | 0.00000 | Strict EO, small accuracy tax. |
| 0.3   | 0.766   | 0.03069 | 0.00383 | Best AUC; slight EO gap. |
| 0.5   | 0.758   | 0.00006 | 0.00017 | Near‑perfect EO, modest AUC. |
| 1.0   | 0.754   | 0.00000 | 0.00000 | More fairness pressure, tiny extra loss. |
| 3.0   | 0.738   | 0.00107 | 0.00000 | Accuracy drops, no EO benefit. |

**Recommendation:** focus on λ ∈ [0.1, 0.5]. Use λ ≈ 0.3 if a small EO gap is tolerable for better AUC.

---

## Measurement bias sanity test (`sanity_bias_vs_nobias/metrics.csv`)

| Scenario  | Model  | ROC AUC | EO ΔTPR | EO ΔFPR | DP diff | DP ratio |
|-----------|--------|---------|---------|---------|---------|----------|
| Biased    | GLM    | 0.772   | 0.066   | 0.0149  | +0.0203 | 3.41 |
| Biased    | ADV_NN | 0.769   | 0.0287  | 0.0034  | +0.0061 | 2.65 |
| Unbiased  | GLM    | 0.767   | 0.0123  | 0.0014  | −0.00094| 0.95 |
| Unbiased  | ADV_NN | 0.765   | 0.0033  | 0.0004  | −0.00055| 0.96 |

Bias removal collapses fairness gaps for both models, confirming that the mitigation is targeting the intended bias mechanism.

---

## Proxy ablation sanity test (`sanity_with_vs_without_proxy/metrics.csv`)

| Scenario        | Model  | ROC AUC | EO ΔTPR | EO ΔFPR | DP diff | DP ratio |
|-----------------|--------|---------|---------|---------|---------|----------|
| With proxy      | GLM    | 0.772   | 0.066   | 0.0149  | +0.0203 | 3.41 |
| With proxy      | ADV_NN | 0.769   | 0.0000  | 0.0000  | 0.0000  | — |
| Without proxy   | GLM    | 0.767   | 0.0689  | 0.0162  | +0.0218 | 3.75 |
| Without proxy   | ADV_NN | 0.766   | 0.0074  | 0.0005  | +0.0013 | 3.00 |

- Removing `Z` alone doesn’t fix fairness for GLM; bias in `S` keeps disparities high.
- ADV_NN neutralizes EO when proxies exist and still outperforms GLM without them.

**Next steps:** explore DP-focused mitigations (e.g., group-specific thresholds) if equal selection rates are required.
