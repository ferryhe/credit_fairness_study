<div align="center">

# Credit Insurance Fairness Study (Prototype)

*Comparing GLM, plain NN, and Equalized-Odds adversarial NN under biased credit signals.*

</div>

---

## 1. Why this project exists

Modern credit underwriting systems rarely observe “true” risk; proxies such as bureau scores or regional indicators often encode historical bias. This prototype simulates such a scenario:

- The *true* default risk is race-neutral and depends on a latent credit factor `S*`.
- Lenders only see a biased score `S`, which is systematically lower for the protected group.
- Even without using race directly, models trained on `S` violate Equalized Odds (EO) and Demographic Parity (DP).

We study three models—GLM, plain NN, and an EO-driven adversarial NN—to quantify the accuracy–fairness trade-off and explore mitigation strategies.

---

## 2. Quick start

```bash
python -m venv .venv
source .venv/bin/activate               # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pytest                                   # run smoke tests
python -m src.experiments.run_baseline   # compare GLM / NN / ADV_NN
```

All experiment scripts write CSV outputs to `results/`.

---

## 3. Code structure at a glance

````text
credit_fairness_study/
├─ README.md
├─ pyproject.toml / requirements.txt
├─ notebooks/
│   └─ exploration.ipynb      # use for ad-hoc EDA or plots
├─ src/
│   ├─ config.py              # dataclasses for sim/train/eval configs
│   ├─ data_generation.py     # biased credit simulator + stratified split
│   ├─ models/
│   │   ├─ glm_model.py       # sklearn logistic regression wrapper
│   │   ├─ nn_model.py        # plain PyTorch MLP + train/predict helpers
│   │   └─ adv_nn_model.py    # EO adversarial network & gradient reversal
│   ├─ training/
│   │   ├─ train_glm.py       # preprocessing + eval pipeline for GLM
│   │   ├─ train_nn.py        # same for plain NN
│   │   └─ train_adv_nn.py    # same for adversarial NN
│   ├─ evaluation/
│   │   ├─ metrics.py         # accuracy metrics (ROC/PR/Brier/LogLoss)
│   │   ├─ fairness.py        # EO & DP computations per group
│   │   └─ reporting.py       # format/save/print metric tables
│   └─ experiments/
│       ├─ run_baseline.py    # main comparison of three models
│       ├─ run_lambda_sweep.py# λ sweep for adversarial strength
│       └─ run_sanity_checks.py# bias/no-bias + proxy ablation studies
└─ tests/
    ├─ test_data_generation.py
    ├─ test_metrics.py
    └─ test_models.py
````

Each submodule has a narrow responsibility so you can swap components (e.g., plug in different models or fairness diagnostics) without touching the rest of the stack.

---

## 4. Problem setup & data-generating process

| Variable | Description | Used for |
| --- | --- | --- |
| `A` | Protected attribute (race), audit-only | fairness auditing |
| `S*` | Latent unbiased credit score | drives true default risk |
| `S` | Observed biased score (`S* - bias_b*A + noise`) | model feature |
| `D` | Debt-to-income ratio (log-normal, clipped) | model feature |
| `L` | Past delinquencies (Poisson) | model feature |
| `Z` | Proxy (ZIP), correlated with `A` | feature / adversary target |
| `Y` | Default outcome (Bernoulli on sigmoid of `S*`, `D`, `L`) | target |

- Bias mechanism: only `S` is biased; `Y` is independent of race conditional on `S*`.  
- Intercept in the outcome model is calibrated via bisection so that `E[Y] ≈ 12%`.  
- Result: fairness gaps emerge purely because we train on biased and proxy-laden features.

---

## 5. Models compared

| Model | Features | Notes |
| --- | --- | --- |
| **GLM** | `[S, D, L, Z]` (numeric scaled, `Z` passthrough) | Strong baseline; fast to train but inherits bias in inputs. |
| **Plain NN** | same features as GLM | Two-layer MLP (16→8 units, dropout 0.1). Can capture nonlinearities but may latch onto proxies even more strongly. |
| **Adversarial NN** | Predictor uses `[S, D, L]`; two adversaries try to predict `Z` within `Y=0` and `Y=1` subsets | Gradient reversal with weight `λ = train_cfg.lambda_adv` pushes the predictor to remove `Z` info per label slice, targeting Equalized Odds. |

---

## 6. Metrics

- **Accuracy**: ROC AUC, PR AUC, Brier score, Log Loss (from `src/evaluation/metrics.py`).
- **Fairness** (at threshold `eval_cfg.threshold`, default 0.5):
  - EO gaps: `|TPR_A - TPR_B|`, `|FPR_A - FPR_B|`.
  - Demographic Parity: selection-rate difference and ratio.
  - Group ROC AUCs to inspect ranking disparities.

Expect the adversarial NN to shrink EO gaps by ~30–60% relative to GLM/NN with only a small accuracy cost when `lambda_adv` is tuned between 0.3 and 1.0.

---

## 7. Built-in experiments

| Script | What it does |
| --- | --- |
| `python -m src.experiments.run_baseline` | Full sim → GLM, NN, ADV_NN comparison. |
| `python -m src.experiments.run_lambda_sweep` | Varies `lambda_adv` ∈ {0.1, 0.3, 0.5, 1.0, 3.0}, logs ROC AUC and EO gaps, optionally plots trade-off curve. |
| `python -m src.experiments.run_sanity_checks` | a) measurement bias on vs off, b) models with/without proxy `Z`; checks that fairness gaps behave as expected. |

All metrics are funneled through `format_metrics_table` / `save_metrics`, so CSV outputs live under `results/`.

---

## 8. Typical observations (default config: 60k rows, 80/20 split)

| Model | ROC AUC | PR AUC | Brier | EO ΔTPR | EO ΔFPR | DP diff | DP ratio |
| --- | --- | --- | --- | --- | --- | --- | --- |
| GLM | ~0.77–0.81 | 0.24–0.32 | 0.09–0.11 | 0.10–0.18 | 0.05–0.12 | 0.08–0.20 | 1.15–1.45 |
| NN | similar ±0.02 AUC vs GLM | similar | similar | EO gaps similar or slightly worse than GLM | — | — | — |
| ADV_NN (`λ ≈ 0.5`) | within 0–1 AUC pts of best model | similar | slightly higher Brier if λ too big | EO gaps down to 0.04–0.09 / 0.02–0.06 | DP diff closer to 0 | DP ratio closer to 1 |

Sanity checks: set `bias_b = 0` and fairness gaps collapse across models; increase `lambda_adv` and EO improves until accuracy drops sharply.

---

## 9. Future directions

- Add calibration diagnostics (ECE, reliability plots).
- Extend experiments with bootstrap confidence intervals.
- Try threshold optimization per policy target (choose TPR/FPR trade-off post-hoc).
- Plug in alternative mitigation methods (reweighting, post-processing) to compare against adversarial training.

---

## 10. Troubleshooting tips

| Symptom | Possible fix |
| --- | --- |
| ROC AUC ≈ 0.5 for all models | Verify simulator parameters (e.g., `alpha_s`, `alpha_d`, `alpha_l` not near zero). |
| EO gaps unchanged when sweeping λ | Ensure adversaries receive **true** labels for slicing and that gradient reversal is applied (see `train_adv_nn`). |
| Training slow | Reduce `n_samples`, epochs, or increase batch size. |
| NaNs in losses | Probabilities are already clipped in metrics; double-check data preprocessing for NaNs. |

---

## 11. Credits & licenses

MIT-style research prototype inspired by fairness-in-ML literature (Hardt et al. 2016, Ganin et al. 2015). Feel free to fork, extend, or plug pieces into your own fairness benchmarking pipeline.
