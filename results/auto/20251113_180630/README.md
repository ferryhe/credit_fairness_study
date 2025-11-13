# Auto baseline run 2025-11-13 18:06

**Context:** First auto pricing baseline with the new generator. `python -m src.experiments.auto.baseline` produced the GLM/NN/ADV_NN metrics below plus the histogram snapshots under `debug_auto_distributions/auto_<feature>.png`.

**Key outcomes:**

- `baseline_results.csv` contains ROC AUC values ~0.67 (GLM), ~0.74 (NN/ADV-NN), with ADV-NN keeping both EO and DP gaps lower than the other models.
- Fixed-rate fairness at 2% shows DP ratios above 3 for GLM and NN because the biasing proxies (violations_obs, past_claims_obs, territory, credit_score) still leak group information.
- Debug histograms (age, mileage, premium, etc.) live in `debug_auto_distributions/`; pay special attention to `auto_annual_mileage.png`, which shows most drivers between 10k–50k miles with a small tail near 60k.

**Next steps:** Use this run as the reference for tuning `bias_strength`, measurement bias parameters, or the pricing GLM coefficients before conducting a new experiment.
