"""Compare Mamba with the repository's classical classifiers on one NPZ dataset."""

import argparse
import os
from pathlib import Path

plot_cache = Path(os.environ.get("TMPDIR", "/tmp")) / "3dbrain_matplotlib_cache"
plot_cache.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(plot_cache))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report

from brain_3d.Mamba import CompareMambaToClassical


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", help="NPZ containing X, y and optionally groups")
    parser.add_argument("output", help="Directory for CSV results")
    parser.add_argument("--splits", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--trees", type=int, default=300)
    parser.add_argument("--backend", choices=["reference", "official"], default="reference")
    args = parser.parse_args()

    data = np.load(args.dataset, allow_pickle=False)
    groups = data["groups"] if "groups" in data.files else None
    result = CompareMambaToClassical(
        data["X"], data["y"], groups=groups, n_splits=args.splits,
        test_size=args.test_size, epochs=args.epochs, random_state=args.seed,
        backend=args.backend, n_estimators=args.trees,
    )
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    result["metrics"].to_csv(output / "comparison_folds.csv", index=False)
    result["summary"].to_csv(output / "comparison_summary.csv", index=False)
    result["predictions"].to_csv(output / "comparison_predictions.csv", index=False)

    for model, matrix in result["confusion_matrices"].items():
        table = pd.DataFrame(matrix, index=result["classes"], columns=result["classes"])
        table.to_csv(output / f"confusion_matrix_{model}.csv")
        plt.figure(figsize=(7, 6))
        sns.heatmap(table, annot=True, fmt="d", cmap="Blues")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title(model)
        plt.tight_layout()
        plt.savefig(output / f"confusion_matrix_{model}.png", dpi=180)
        plt.close()

        pred = result["predictions"][result["predictions"]["model"] == model]
        report = classification_report(
            pred["true_label"], pred["predicted_label"], labels=result["classes"],
            output_dict=True, zero_division=0,
        )
        pd.DataFrame(report).T.to_csv(output / f"classification_report_{model}.csv")

    plt.figure(figsize=(11, 6))
    order = result["summary"]["model"]
    sns.boxplot(data=result["metrics"], x="model", y="balanced_accuracy", order=order)
    plt.xticks(rotation=35, ha="right")
    plt.xlabel("")
    plt.ylabel("Test balanced accuracy")
    plt.tight_layout()
    plt.savefig(output / "algorithm_comparison_balanced_accuracy.png", dpi=180)
    plt.close()
    print(result["summary"].to_string(index=False))


if __name__ == "__main__":
    main()
