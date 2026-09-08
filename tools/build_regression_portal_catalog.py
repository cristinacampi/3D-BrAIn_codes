#!/usr/bin/env python3
"""Build portal catalogue records for the trained DIV regression models."""

from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PORTAL_CATALOG_DIR = Path(
    "/Users/cristina/Documents/application/progetti/EU/EIC_pathfinder/3dbrain-portal/app/data/model-catalogs"
)


def title(model_name: str) -> str:
    return model_name.replace("_pca90", "").replace("_", " ").title()


def family(regressor: str) -> str:
    if "extra_trees" in regressor or "random_forest" in regressor:
        return "Tree ensemble"
    if "svr" in regressor:
        return "Support vector regression"
    return "Linear regression"


def build(experiment_id: int) -> list[dict]:
    campaign = ROOT / "outputs" / "analisi_dati_Mark" / f"experiment_{experiment_id}" / "recording_regression_negative_spikes_div_ml_campaign"
    final_models = campaign / "final_models"
    with (final_models / "model_manifest.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    records = []
    for row in rows:
        model_name = row["model_name"]
        feature_set = row["feature_set"]
        records.append(
            {
                "access": "Open",
                "classificationFamily": family(row["regressor"]),
                "downloadUrl": None,
                "framework": "scikit-learn / joblib",
                "id": model_name,
                "input": f"{feature_set} features extracted from processed plausible negative MEA spikes",
                "keywords": ["MEA", "DIV", "regression", feature_set, row["regressor"]],
                "license": "MIT",
                "limitations": "Research use only; performance is based on internal grouped cross-validation and has not been externally validated on an independent dataset.",
                "metrics": [
                    f"MAE: {float(row['cv_mean_test_mae_days']):.2f} days",
                    f"RMSE: {float(row['cv_mean_test_rmse_days']):.2f} days",
                    f"R2: {float(row['cv_mean_test_r2']):.3f}",
                    f"Within 10 days: {100 * float(row['cv_mean_test_within_10_days']):.1f}%",
                    f"CV folds: {int(row['cv_n_folds'])}",
                ],
                "modelType": title(row["regressor"]),
                "name": title(model_name),
                "output": "Predicted culture age as a continuous DIV value",
                "owner": "University of Genoa",
                "purpose": "Recording-level prediction of days in vitro from plausible negative MEA spikes.",
                "repository": "https://github.com/cristinacampi/3D-BrAIn_codes",
                "status": "Research",
                "validation": "Repeated group-disjoint cross-validation by biological_unit_id",
                "version": f"Experiment {experiment_id} DIV regression research model",
                "workPackage": "WP3",
            }
        )
    destination = final_models / "portal_model_catalog.json"
    destination.write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")
    PORTAL_CATALOG_DIR.mkdir(parents=True, exist_ok=True)
    (PORTAL_CATALOG_DIR / f"experiment-{experiment_id}-div-regression.json").write_text(
        json.dumps(records, indent=2) + "\n", encoding="utf-8"
    )
    return records


if __name__ == "__main__":
    for experiment in (4, 8):
        print(f"Experiment {experiment}: {len(build(experiment))} models")
