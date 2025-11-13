from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


RESULTS_DIR = Path("results/auto")
FIG_PATH = RESULTS_DIR / "fairness_accuracy_frontier.png"


def _latest_run_dir() -> Path | None:
    candidates = sorted(
        [
            p
            for p in RESULTS_DIR.iterdir()
            if p.is_dir()
            and p.name.replace("_", "").isdigit()
            and (p / "baseline_results.csv").exists()
        ],
        key=lambda d: d.name,
    )
    if not candidates:
        return None
    return candidates[-1]


def main() -> None:
    run_dir = _latest_run_dir()
    if run_dir is None:
        raise FileNotFoundError("No auto run directories under results/auto/")

    metrics_path = run_dir / "baseline_results.csv"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing baseline CSV in {run_dir}")

    df = pd.read_csv(metrics_path)
    plt.figure(figsize=(7, 5))
    plt.grid(True, linestyle="--", alpha=0.5)

    colors = {"GLM": "tab:orange", "NN": "tab:green", "ADV_NN": "tab:blue"}
    for model_name, group in df.groupby("model_name"):
        row = group.iloc[0]
        plt.scatter(
            row["eo_gap_tpr"],
            row["roc_auc"],
            label=model_name,
            color=colors.get(model_name, "tab:purple"),
            s=80,
            marker="o" if model_name != "ADV_NN" else "D",
        )
        plt.annotate(
            model_name,
            (row["eo_gap_tpr"], row["roc_auc"]),
            textcoords="offset points",
            xytext=(5, 5),
            fontsize=9,
            fontweight="bold",
        )

    plt.title("Auto Fairness–Accuracy Frontier (Equalized Odds)")
    plt.xlabel("EO gap (TPR)")
    plt.ylabel("ROC AUC")
    plt.legend()

    FIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(FIG_PATH, dpi=200)
    run_plot_path = run_dir / "fairness_accuracy_frontier.png"
    plt.savefig(run_plot_path, dpi=200)
    plt.close()
    print(
        f"Saved plot to {FIG_PATH} and {run_plot_path} using data from {metrics_path}"
    )


if __name__ == "__main__":
    main()
