"""Build a labeled feature table for stratification and classification.

The script joins the experiment labels generated from
``lista_esperimenti_aggiornata.xlsx`` with the well-level spike/burst metrics
created by ``main_compute_spike_burst_metrics.py``.
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_LABELS = Path("outputs/campagna_analisi_etichette/labels_3dbrain.csv")
DEFAULT_METRICS_ROOTS = [
    Path("/data/metrics"),
    Path("/data/26-05-28_3Brain_Ivan_PartnershipUniGe_Pasca120Days/metrics"),
    Path("/data/26-06-04-EXP/metrics"),
]
DEFAULT_OUTPUT_DIR = Path("outputs/campagna_analisi_etichette/analysis_ready")

NON_FEATURE_COLUMNS = {
    "record_id",
    "experiment_id",
    "folder",
    "filename",
    "recording_name",
    "recording_name_key",
    "metrics_path",
    "label_original",
    "label_normalized",
    "label_code",
    "recording_index",
    "div",
    "analysis_group",
    "Comment",
    "ActiveElectrodesID",
}


def recording_key(value):
    """Normalize recording identifiers from filenames and metric CSVs."""
    if pd.isna(value):
        return ""
    text = str(value).strip()
    text = re.sub(r"\.brw$", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", "", text)
    return text.lower()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument(
        "--metrics-root",
        type=Path,
        action="append",
        default=None,
        help=(
            "Directory containing recording metric folders. Can be passed more than once, "
            "for example one root for 28 May and one for 4 June."
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--channel-metrics",
        action="store_true",
        help="Also export a labeled channel-level table when channel metrics are present.",
    )
    return parser.parse_args()


def safe_name(value):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("_")


def find_well_metric_files(metrics_root):
    return sorted(metrics_root.glob("**/spike_burst/well_spike_burst_metrics.csv"))


def find_channel_metric_file(well_metric_path):
    channel_path = well_metric_path.with_name("channel_spike_burst_metrics.csv")
    return channel_path if channel_path.exists() else None


def load_well_metrics(metrics_roots):
    rows = []
    checked_roots = []
    for metrics_root in metrics_roots:
        checked_roots.append(str(metrics_root))
        if not metrics_root.exists():
            continue
        metric_files = find_well_metric_files(metrics_root)
        for metric_path in metric_files:
            metric = pd.read_csv(metric_path)
            if metric.empty:
                continue
            first = metric.iloc[0].copy()
            recording_name = first.get("RecordingName", metric_path.parents[1].name)
            first["recording_name"] = recording_name
            first["recording_name_key"] = recording_key(recording_name)
            first["metrics_path"] = str(metric_path)
            rows.append(first)

    if not rows:
        raise FileNotFoundError(
            "No well_spike_burst_metrics.csv files found.\n"
            "Checked metrics roots:\n  - "
            + "\n  - ".join(checked_roots)
            + "\nRun main/main_compute_spike_burst_metrics.py first, or pass the real metrics folder with "
            "--metrics-root /path/to/metrics. You can pass --metrics-root more than once."
        )

    return pd.DataFrame(rows)


def load_channel_metrics(labeled_well):
    rows = []
    for _, row in labeled_well.iterrows():
        channel_path = find_channel_metric_file(Path(row["metrics_path"]))
        if channel_path is None:
            continue
        channels = pd.read_csv(channel_path)
        channels.insert(0, "record_id", row["record_id"])
        channels.insert(1, "recording_name", row["recording_name"])
        channels.insert(2, "label_normalized", row["label_normalized"])
        channels.insert(3, "label_code", row["label_code"])
        rows.append(channels)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def feature_columns(frame):
    numeric = frame.select_dtypes(include=[np.number, "bool"]).columns.tolist()
    return [
        col
        for col in numeric
        if col not in NON_FEATURE_COLUMNS
        and not col.endswith("_present")
        and col not in {"combination_present", "spontaneous_control"}
    ]


def main():
    args = parse_args()
    metrics_roots = args.metrics_root or DEFAULT_METRICS_ROOTS
    labels = pd.read_csv(args.labels)
    labels["recording_name"] = labels["filename"].map(lambda value: re.sub(r"\.brw$", "", str(value), flags=re.IGNORECASE))
    labels["recording_name_key"] = labels["filename"].map(recording_key)

    metrics = load_well_metrics(metrics_roots)
    labeled = labels.merge(metrics, on="recording_name_key", how="left", suffixes=("", "_metric"))
    if "recording_name_metric" in labeled.columns:
        labeled["recording_name"] = labeled["recording_name_metric"].fillna(labeled["recording_name"])
        labeled = labeled.drop(columns=["recording_name_metric"])

    args.output_dir.mkdir(parents=True, exist_ok=True)

    missing = labeled[labeled["metrics_path"].isna()].copy()
    matched = labeled[labeled["metrics_path"].notna()].copy()

    labeled_path = args.output_dir / "labeled_well_metrics.csv"
    missing_path = args.output_dir / "missing_well_metrics.csv"
    features_path = args.output_dir / "feature_matrix_well.csv"

    labeled.to_csv(labeled_path, index=False)
    missing.to_csv(missing_path, index=False)

    if matched.empty:
        raise FileNotFoundError(
            f"No labels matched metrics under {args.metrics_root}. "
            f"See missing list: {missing_path}"
        )

    cols = [
        "record_id",
        "experiment_id",
        "recording_name",
        "folder",
        "filename",
        "label_original",
        "label_normalized",
        "label_code",
        "excitation_present",
        "inhibition_present",
        "combination_present",
        "spontaneous_control",
        "ready_processed",
        "ready_strict",
        "analysis_group",
    ] + feature_columns(matched)
    features = matched[cols].copy()
    features.to_csv(features_path, index=False)

    if "experiment_id" in features.columns:
        for experiment_id, group in features.groupby("experiment_id", dropna=False):
            experiment_path = args.output_dir / f"feature_matrix_well__{safe_name(experiment_id)}.csv"
            group.to_csv(experiment_path, index=False)

    print("Saved labeled metrics:", labeled_path)
    print("Saved feature matrix:", features_path)
    if "experiment_id" in matched.columns:
        print("Saved per-experiment feature matrices in:", args.output_dir)
    print("Saved missing metrics report:", missing_path)
    print(f"Matched metrics: {len(matched)} / {len(labeled)}")
    if not missing.empty:
        print("Missing metrics by label:")
        print(missing["label_normalized"].value_counts(dropna=False).to_string())

    if args.channel_metrics:
        channel = load_channel_metrics(matched)
        if not channel.empty:
            channel_path = args.output_dir / "labeled_channel_metrics.csv"
            channel.to_csv(channel_path, index=False)
            print("Saved labeled channel metrics:", channel_path)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
