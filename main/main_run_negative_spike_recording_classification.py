"""Recording-level classification for one Mark experiment using plausible negative spikes.

This main mirrors the recording-level classification analysis, but all activity
and waveform features are computed from the same event universe used by the
negative-polarity stratification:

- detector label 0;
- dominant negative trough;
- extremum near the waveform center;
- active channels with firing rate > 0.25 Hz after polarity filtering.

Each recording is one sample, and cross-validation is split by biological unit.
Edit the parameter block below to select the experiment and target.
"""

import os
import json
import re
import sys
import time
from pathlib import Path

plot_cache = Path(os.environ.get("TMPDIR", "/tmp")) / "3dbrain_matplotlib_cache"
plot_cache.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(plot_cache))
os.environ.setdefault("XDG_CACHE_HOME", str(plot_cache))
os.environ.setdefault("NUMBA_CACHE_DIR", str(Path(os.environ.get("TMPDIR", "/tmp")) / "numba_cache"))

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import joblib
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifierCV
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

for parent in Path(__file__).resolve().parents:
    src_path = parent / "src"
    if (src_path / "brain_3d").is_dir():
        sys.path.insert(0, str(src_path))
        break

from brain_3d import Classification, Mamba


# =========================
# PARAMETERS TO EDIT
# =========================

EXPERIMENT_ID = 8
EXPERIMENT_CONFIG = {
    4: {
        "name": "4_human_laminin_series",
        "spike_root": Path("/data/shared/3D_BRAIN/Mark/Single well MEA/4. Human laminin series 1/spikes/"),
    },
    8: {
        "name": "8_human_laminin_series_2",
        "spike_root": Path("/data/shared/3D_BRAIN/Mark/Single well MEA/8. Human laminin series 2/spikes/"),
    },
}
EXPERIMENT_NAME = EXPERIMENT_CONFIG[EXPERIMENT_ID]["name"]
SPIKE_ROOT = EXPERIMENT_CONFIG[EXPERIMENT_ID]["spike_root"]
LABELS_CSV = Path("outputs/analisi_dati_Mark/labels_experiments_4_8.csv")

TARGET_NAME = "culture_phase"
TARGET_CONFIG = {
    "culture_phase": {
        "column": "culture_phase_code",
        "label_column": "culture_phase",
        "classes": [0, 1],
        "min_groups_per_class": 1,
    },
    "div": {
        "column": "div_code",
        "label_column": "div_label",
        "classes": None,
        "min_groups_per_class": 2,
    },
}
OUTPUT_DIR = Path(
    f"outputs/analisi_dati_Mark/experiment_{EXPERIMENT_ID}/"
    f"recording_classification_negative_spikes_{TARGET_NAME}_ml_campaign"
)

TARGET_COLUMN = TARGET_CONFIG[TARGET_NAME]["column"]
TARGET_LABEL_COLUMN = TARGET_CONFIG[TARGET_NAME]["label_column"]
GROUP_COLUMN = "biological_unit_id"
CLASSES_TO_USE = TARGET_CONFIG[TARGET_NAME]["classes"]
MIN_GROUPS_PER_CLASS = TARGET_CONFIG[TARGET_NAME]["min_groups_per_class"]

SAMPLING_RATE_HZ = 19753
RECORDING_DURATION_SECONDS = 180
MIN_ACTIVE_CHANNEL_FIRING_RATE_HZ = 0.25
EXCLUDE_REFERENCE_CHANNEL = True
REFERENCE_CHANNEL = 0

NEGATIVE_SPIKE_LABEL = 0
USE_DETECTOR_LABEL_FILTER = True
USE_WAVEFORM_POLARITY_FILTER = True
POLARITY_DOMINANCE_RATIO = 1.10
REQUIRE_EXTREMUM_NEAR_CENTER = True
CENTER_WINDOW_SAMPLES = 10

MAX_SPIKES_PER_RECORDING = 20000
MAX_SPIKES_FOR_ROCKET_FIT = 12000
NUM_KERNELS = 10000
PCA_EXPLAINED_VARIANCE = 0.90

N_SPLITS = 4
N_REPEATS = 3
RANDOM_STATE = 42
FORCE_REPROCESS = False

RUN_MAMBA = True
MAMBA_EPOCHS = 30
MAMBA_BATCH_SIZE = 16
MAMBA_D_MODEL = 64
MAMBA_N_LAYERS = 2
MAMBA_D_STATE = 16
MAMBA_BACKEND = "reference"
MAMBA_VERBOSE = True

SAVE_FINAL_MODELS = True
SAVE_CV_FOLD_MODELS = False
SAVE_ALL_CLASSICAL_FINAL_MODELS = True
SAVE_TOP_N_CLASSICAL_FINAL_MODELS = 3
SAVE_DUMMY_FINAL_MODEL = False
SAVE_MAMBA_FINAL_MODEL = RUN_MAMBA


# =========================
# SETUP
# =========================

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
done_file = OUTPUT_DIR / "summary_metrics.csv"
if done_file.exists() and not FORCE_REPROCESS:
    print("Negative recording-level classification already processed:", done_file)
    print("Set FORCE_REPROCESS = True in the parameter block to recompute it.")
    raise SystemExit(0)

rng = np.random.default_rng(RANDOM_STATE)

AnalysisStarted = time.perf_counter()
print("\n=== Negative-spike recording classification ===", flush=True)
print(f"Experiment: {EXPERIMENT_ID} - {EXPERIMENT_NAME}", flush=True)
print(f"Target: {TARGET_NAME}", flush=True)
print(f"Spike folder: {SPIKE_ROOT}", flush=True)
print(f"Output folder: {OUTPUT_DIR}", flush=True)
print(f"Mamba enabled: {RUN_MAMBA} ({MAMBA_BACKEND}, {MAMBA_EPOCHS} epochs)", flush=True)


def protocol_from_filename(filename):
    name = str(filename).lower()
    if "100um4ap_50umap5" in name:
        return "4ap_then_ap5"
    if "50umap5_100um4ap" in name:
        return "ap5_then_4ap"
    if "100um4ap" in name:
        return "4ap"
    if "50umap5" in name:
        return "ap5"
    return "spontaneous"


def negative_waveform_mask(spikes, labels):
    keep = np.ones(len(spikes), dtype=bool)
    if USE_DETECTOR_LABEL_FILTER and np.isfinite(labels).any():
        keep &= labels == NEGATIVE_SPIKE_LABEL

    if USE_WAVEFORM_POLARITY_FILTER:
        waveform_min = np.min(spikes, axis=1)
        waveform_max = np.max(spikes, axis=1)
        trough_index = np.argmin(spikes, axis=1)
        center = (spikes.shape[1] - 1) / 2
        waveform_keep = np.abs(waveform_min) >= POLARITY_DOMINANCE_RATIO * np.abs(waveform_max)
        if REQUIRE_EXTREMUM_NEAR_CENTER:
            waveform_keep &= np.abs(trough_index - center) <= CENTER_WINDOW_SAMPLES
        keep &= waveform_keep

    return keep


def parse_model_name(model_name):
    for feature_set_name in ("activity", "waveform", "combined"):
        prefix = f"{feature_set_name}_"
        if model_name.startswith(prefix):
            return feature_set_name, model_name[len(prefix):]
    return None, model_name


def build_recording_feature_matrix(recordings_table, names, waveforms_by_name, extractor, feature_columns):
    rows = []
    for _, rec in recordings_table.set_index("recording_name").loc[names].reset_index().iterrows():
        rec_waves = waveforms_by_name[rec["recording_name"]]
        values = {column: float(rec[column]) for column in activity_columns + waveform_summary_columns}
        if extractor is not None:
            rec_features = np.asarray(extractor.transform(rec_waves[:, np.newaxis, :]))
            rec_mean = rec_features.mean(axis=0)
            rec_std = rec_features.std(axis=0)
            values.update({f"rocket_mean_{idx}": value for idx, value in enumerate(rec_mean)})
            values.update({f"rocket_std_{idx}": value for idx, value in enumerate(rec_std)})
        rows.append(pd.Series(values, name=rec["recording_name"]))

    feature_matrix = pd.DataFrame(rows)
    feature_matrix.index.name = "recording_name"
    return feature_matrix.reindex(columns=feature_columns).replace([np.inf, -np.inf], np.nan).fillna(0)


def to_jsonable(value):
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return to_jsonable(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def save_json(path, payload):
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(to_jsonable(payload), handle, indent=2, sort_keys=True)


def training_dataset_metadata(recordings_table, selected_model_names):
    class_counts = (
        recordings_table.groupby(["label_code", "label_normalized"], as_index=False)
        .agg(
            recordings=("recording_name", "count"),
            biological_units=(GROUP_COLUMN, "nunique"),
            sampled_spikes=("sampled_spikes", "sum"),
            active_channels=("n_active_channels", "sum"),
        )
        .sort_values("label_code")
    )
    return {
        "experiment_id": EXPERIMENT_ID,
        "experiment_name": EXPERIMENT_NAME,
        "target_name": TARGET_NAME,
        "target_column": TARGET_COLUMN,
        "target_label_column": TARGET_LABEL_COLUMN,
        "group_column": GROUP_COLUMN,
        "sample_unit": "recording",
        "split_unit": GROUP_COLUMN,
        "polarity_scope": "negative_plausible",
        "n_recordings": int(len(recordings_table)),
        "n_biological_units": int(recordings_table[GROUP_COLUMN].nunique()),
        "n_classes": int(recordings_table["label_code"].nunique()),
        "total_input_spikes": int(recordings_table["n_input_spikes"].sum()),
        "total_detector_negative_spikes": int(recordings_table["n_after_label_filter"].sum()),
        "total_plausible_negative_spikes": int(recordings_table["n_after_waveform_filter"].sum()),
        "total_training_sampled_spikes": int(recordings_table["sampled_spikes"].sum()),
        "class_mapping": {str(code): str(label) for code, label in name_by_code.items()},
        "class_distribution": class_counts.to_dict("records"),
        "selected_final_models": selected_model_names,
        "feature_sets": {
            "activity": activity_columns,
            "waveform_summary": waveform_summary_columns,
            "waveform_sequence_mamba": [
                "waveform_mean_profile",
                "waveform_std_profile",
                "waveform_median_profile",
                "waveform_percentile_25_profile",
                "waveform_percentile_75_profile",
            ],
        },
        "training_parameters": {
            "sampling_rate_hz": SAMPLING_RATE_HZ,
            "recording_duration_seconds": RECORDING_DURATION_SECONDS,
            "min_active_channel_firing_rate_hz": MIN_ACTIVE_CHANNEL_FIRING_RATE_HZ,
            "exclude_reference_channel": EXCLUDE_REFERENCE_CHANNEL,
            "reference_channel": REFERENCE_CHANNEL,
            "max_spikes_per_recording": MAX_SPIKES_PER_RECORDING,
            "max_spikes_for_rocket_fit": MAX_SPIKES_FOR_ROCKET_FIT,
            "num_kernels": NUM_KERNELS,
            "pca_explained_variance": PCA_EXPLAINED_VARIANCE,
            "n_splits": N_SPLITS,
            "n_repeats": N_REPEATS,
            "random_state": RANDOM_STATE,
        },
        "negative_spike_filter": {
            "negative_spike_label": NEGATIVE_SPIKE_LABEL,
            "use_detector_label_filter": USE_DETECTOR_LABEL_FILTER,
            "use_waveform_polarity_filter": USE_WAVEFORM_POLARITY_FILTER,
            "polarity_dominance_ratio": POLARITY_DOMINANCE_RATIO,
            "require_extremum_near_center": REQUIRE_EXTREMUM_NEAR_CENTER,
            "center_window_samples": CENTER_WINDOW_SAMPLES,
        },
        "mamba_parameters": {
            "run_mamba": RUN_MAMBA,
            "epochs": MAMBA_EPOCHS,
            "batch_size": MAMBA_BATCH_SIZE,
            "d_model": MAMBA_D_MODEL,
            "n_layers": MAMBA_N_LAYERS,
            "d_state": MAMBA_D_STATE,
            "backend": MAMBA_BACKEND,
        },
    }


def model_family(classifier_name):
    if classifier_name == "mamba":
        return "State-space model"
    if "svm" in classifier_name:
        return "SVM"
    if "forest" in classifier_name or "trees" in classifier_name:
        return "Ensemble"
    if "logistic" in classifier_name or "ridge" in classifier_name:
        return "Linear model"
    if "dummy" in classifier_name:
        return "Baseline"
    return "Machine learning"


def model_type_label(classifier_name):
    labels_by_classifier = {
        "mamba": "Selective state-space sequence model",
        "ridge_pca90": "Ridge classifier with PCA",
        "logistic_balanced_pca90": "Balanced logistic regression with PCA",
        "linear_svm_balanced_pca90": "Balanced linear support vector machine with PCA",
        "rbf_svm_balanced_pca90": "Balanced RBF support vector machine with PCA",
        "random_forest_balanced": "Balanced random forest classifier",
        "extra_trees_balanced": "Balanced Extra Trees classifier",
        "dummy_most_frequent": "Most-frequent-class dummy baseline",
    }
    return labels_by_classifier.get(classifier_name, classifier_name.replace("_", " "))


def metrics_for_model(model_name):
    cv_rows = metrics[metrics["model"] == model_name].copy()
    summary_row = algorithm_comparison.set_index("model").loc[model_name].to_dict()
    return {
        "rank": int(summary_row.get("rank", 0)),
        "cv_summary": summary_row,
        "cv_folds": cv_rows.to_dict("records"),
        "training_metrics": {
            "mean_cv_train_accuracy": summary_row.get("mean_train_accuracy"),
            "mean_cv_train_balanced_accuracy": summary_row.get("mean_train_balanced_accuracy"),
        },
        "test_metrics": {
            "mean_cv_test_accuracy": summary_row.get("mean_test_accuracy"),
            "std_cv_test_accuracy": summary_row.get("std_test_accuracy"),
            "mean_cv_test_balanced_accuracy": summary_row.get("mean_test_balanced_accuracy"),
            "std_cv_test_balanced_accuracy": summary_row.get("std_test_balanced_accuracy"),
            "n_cv_folds": summary_row.get("n_folds"),
        },
    }


def recording_manifest():
    columns = [
        "recording_name",
        GROUP_COLUMN,
        TARGET_LABEL_COLUMN,
        "label_code",
        "label_normalized",
        "protocol",
        "div",
        "sampled_spikes",
        "n_active_channels",
        "firing_rate_hz",
    ]
    available_columns = [column for column in columns if column in recordings.columns]
    return recordings[available_columns].to_dict("records")


def model_metadata_record(
    model_name,
    model_type,
    model_path,
    feature_set_name,
    classifier_name,
    feature_columns,
    estimator_parameters,
):
    model_metrics = metrics_for_model(model_name)
    return {
        "model_id": model_name,
        "model_name": model_name,
        "model_type": model_type,
        "classifier": classifier_name,
        "classifier_family": model_family(classifier_name),
        "classifier_description": model_type_label(classifier_name),
        "framework": "PyTorch" if model_type == "torch_state_dict" else "scikit-learn",
        "model_path": str(model_path),
        "created_by_main": Path(__file__).name,
        "dataset": training_dataset_metadata(recordings, [model_name]),
        "training_recordings": recording_manifest(),
        "feature_set": feature_set_name,
        "feature_columns": feature_columns,
        "n_features": len(feature_columns),
        "estimator_parameters": estimator_parameters,
        "preprocessing": {
            "polarity_scope": "negative_plausible",
            "sampling_rate_hz": SAMPLING_RATE_HZ,
            "recording_duration_seconds": RECORDING_DURATION_SECONDS,
            "min_active_channel_firing_rate_hz": MIN_ACTIVE_CHANNEL_FIRING_RATE_HZ,
            "exclude_reference_channel": EXCLUDE_REFERENCE_CHANNEL,
            "reference_channel": REFERENCE_CHANNEL,
            "max_spikes_per_recording": MAX_SPIKES_PER_RECORDING,
            "max_spikes_for_rocket_fit": MAX_SPIKES_FOR_ROCKET_FIT,
            "num_kernels": NUM_KERNELS if feature_set_name in {"waveform", "combined"} else 0,
            "pca_explained_variance": PCA_EXPLAINED_VARIANCE if "pca90" in classifier_name else None,
            "negative_spike_filter": {
                "negative_spike_label": NEGATIVE_SPIKE_LABEL,
                "use_detector_label_filter": USE_DETECTOR_LABEL_FILTER,
                "use_waveform_polarity_filter": USE_WAVEFORM_POLARITY_FILTER,
                "polarity_dominance_ratio": POLARITY_DOMINANCE_RATIO,
                "require_extremum_near_center": REQUIRE_EXTREMUM_NEAR_CENTER,
                "center_window_samples": CENTER_WINDOW_SAMPLES,
            },
        },
        "validation": {
            "strategy": "Repeated StratifiedGroupKFold",
            "split_unit": GROUP_COLUMN,
            "n_splits": N_SPLITS,
            "n_repeats": N_REPEATS,
            "random_state": RANDOM_STATE,
            "independent_holdout_test": False,
            "test_metrics_are_cross_validated": True,
        },
        "metrics": model_metrics,
        "reuse_notes": (
            "Final reusable model fitted on all available labelled recordings after cross-validation. "
            "Reported test metrics are cross-validated held-out-fold metrics, not a separate external test set."
        ),
    }


def portal_catalog_record(metadata):
    return {
        "id": metadata["model_id"],
        "name": metadata["classifier_description"],
        "version": f"Experiment {EXPERIMENT_ID} research model",
        "status": "Research",
        "classificationFamily": metadata["classifier_family"],
        "modelType": metadata["classifier_description"],
        "purpose": f"Recording-level classification of {TARGET_NAME.replace('_', ' ')} from plausible negative MEA spikes.",
        "framework": metadata["framework"],
        "input": f"{metadata['feature_set']} features extracted from processed spikes/<recording> folders",
        "output": "Predicted experimental class with class probabilities when supported by the estimator",
        "validation": f"Repeated group-disjoint cross-validation by {GROUP_COLUMN}",
        "metrics": [
            f"Balanced accuracy: {metadata['metrics']['test_metrics']['mean_cv_test_balanced_accuracy']:.3f}",
            f"Accuracy: {metadata['metrics']['test_metrics']['mean_cv_test_accuracy']:.3f}",
            f"CV folds: {int(metadata['metrics']['test_metrics']['n_cv_folds'])}",
        ],
        "owner": "University of Genoa",
        "workPackage": "WP3",
        "license": "MIT",
        "access": "Restricted",
        "repository": "https://github.com/cristinacampi/3D-BrAIn_codes",
        "downloadUrl": None,
        "keywords": ["MEA", "classification", TARGET_NAME, metadata["feature_set"], metadata["classifier_family"]],
        "limitations": "Research use only; final model has not been externally validated on an independent dataset.",
    }


# =========================
# READ LABELS
# =========================

labels = pd.read_csv(LABELS_CSV)
labels["recording_key"] = (
    labels["filename"]
    .astype(str)
    .str.replace(r"\.brw$", "", regex=True)
    .str.replace(r"\s+", "", regex=True)
    .str.lower()
)
target_mask = labels[TARGET_COLUMN].notna()
if CLASSES_TO_USE is not None:
    target_mask &= labels[TARGET_COLUMN].isin(CLASSES_TO_USE)
labels = labels[(labels["experiment_id"] == EXPERIMENT_ID) & target_mask].copy()

if MIN_GROUPS_PER_CLASS > 1:
    groups_per_class = labels.groupby(TARGET_COLUMN)[GROUP_COLUMN].nunique()
    usable_classes = groups_per_class[groups_per_class >= MIN_GROUPS_PER_CLASS].index
    excluded_labels = labels[~labels[TARGET_COLUMN].isin(usable_classes)].copy()
    labels = labels[labels[TARGET_COLUMN].isin(usable_classes)].copy()
    if not excluded_labels.empty:
        excluded_labels.to_csv(OUTPUT_DIR / "target_classes_excluded_too_few_groups.csv", index=False)
        print(
            f"Excluded {len(excluded_labels)} recordings whose {TARGET_COLUMN} occurs in fewer than "
            f"{MIN_GROUPS_PER_CLASS} biological units."
        )

if labels.empty:
    raise ValueError(f"No labels found for experiment {EXPERIMENT_ID}.")

if not SPIKE_ROOT.exists():
    raise FileNotFoundError(f"Spike root not found for experiment {EXPERIMENT_ID}: {SPIKE_ROOT}")

recording_dirs = {
    re.sub(r"\s+", "", path.name).lower(): path
    for path in SPIKE_ROOT.iterdir()
    if path.is_dir()
}


# =========================
# LOAD NEGATIVE RECORDING FEATURES
# =========================

recording_rows = []
waveforms_by_recording = {}
missing_recordings = []

SortedLabels = labels.sort_values(["DIV", "chip_ID", "filename"])
print(f"\nLoading {len(SortedLabels)} labeled recordings...", flush=True)
for RecordingIndex, (_, label_row) in enumerate(SortedLabels.iterrows(), start=1):
    RecordingStarted = time.perf_counter()
    recording_key = label_row["recording_key"]
    print(
        f"[recording {RecordingIndex}/{len(SortedLabels)}] {label_row['filename']}",
        flush=True,
    )
    if recording_key not in recording_dirs:
        missing_recordings.append(label_row["filename"])
        print("  not found: skipped", flush=True)
        continue

    recording_dir = recording_dirs[recording_key]
    recording_duration_seconds = float(RECORDING_DURATION_SECONDS)
    active_waveforms = []
    active_channel_counts = []
    active_frames = []
    n_input_spikes = 0
    n_after_label_filter = 0
    n_after_waveform_filter = 0
    n_channels_with_negative_spikes = 0

    for spike_path in sorted(recording_dir.glob("spikes_ch*.npy")):
        channel_match = re.search(r"spikes_ch(\d+)\.npy$", spike_path.name)
        if channel_match is None:
            continue

        channel = int(channel_match.group(1))
        if EXCLUDE_REFERENCE_CHANNEL and channel == REFERENCE_CHANNEL:
            continue

        spikes = np.load(spike_path)
        if spikes.size == 0:
            continue
        spikes = np.asarray(spikes).reshape(spikes.shape[0], -1)

        frame_path = spike_path.with_name(f"frames_ch{channel}.npy")
        label_path = spike_path.with_name(f"label_ch{channel}.npy")
        frames = np.load(frame_path).reshape(-1) if frame_path.exists() else np.full(spikes.shape[0], np.nan)
        spike_labels = np.load(label_path).reshape(-1) if label_path.exists() else np.full(spikes.shape[0], np.nan)

        n_spikes = min(spikes.shape[0], len(frames), len(spike_labels))
        spikes = spikes[:n_spikes]
        frames = frames[:n_spikes]
        spike_labels = spike_labels[:n_spikes]
        n_input_spikes += n_spikes

        label_keep = spike_labels == NEGATIVE_SPIKE_LABEL if np.isfinite(spike_labels).any() else np.ones(n_spikes, bool)
        n_after_label_filter += int(label_keep.sum())
        keep = negative_waveform_mask(spikes, spike_labels)
        n_after_waveform_filter += int(keep.sum())

        spikes = spikes[keep]
        frames = frames[keep]
        if len(spikes) == 0:
            continue

        n_channels_with_negative_spikes += 1
        channel_firing_rate_hz = len(spikes) / recording_duration_seconds
        if channel_firing_rate_hz <= MIN_ACTIVE_CHANNEL_FIRING_RATE_HZ:
            continue

        active_waveforms.append(spikes)
        active_channel_counts.append(len(spikes))
        active_frames.append(frames)

    if not active_waveforms:
        print("  no active negative-waveform channels: skipped", flush=True)
        continue

    waves_all = np.vstack(active_waveforms)
    sampled_waves = waves_all
    if len(sampled_waves) > MAX_SPIKES_PER_RECORDING:
        selected = rng.choice(len(sampled_waves), size=MAX_SPIKES_PER_RECORDING, replace=False)
        sampled_waves = sampled_waves[selected]

    channel_counts = np.asarray(active_channel_counts, dtype=float)
    total_active_negative_spikes = float(channel_counts.sum())
    firing_rate_hz = total_active_negative_spikes / recording_duration_seconds
    probabilities = channel_counts / total_active_negative_spikes if total_active_negative_spikes else np.zeros_like(channel_counts)
    channel_entropy = float(-(probabilities * np.log2(probabilities + 1e-12)).sum())

    recording_name = recording_dir.name
    biological_unit_id = label_row[GROUP_COLUMN]
    waveforms_by_recording[recording_name] = sampled_waves
    recording_rows.append(
        {
            "recording_name": recording_name,
            "filename": label_row["filename"],
            "experiment_id": EXPERIMENT_ID,
            "biological_unit_id": biological_unit_id,
            "chip_ID": int(label_row["chip_ID"]),
            "DIV": int(label_row["DIV"]),
            "Days_in_BF": int(label_row["Days_in_BF"]),
            "label_code": int(label_row[TARGET_COLUMN]),
            "label_normalized": label_row[TARGET_LABEL_COLUMN],
            "protocol": label_row[TARGET_LABEL_COLUMN],
            "n_input_spikes": n_input_spikes,
            "n_after_label_filter": n_after_label_filter,
            "n_after_waveform_filter": n_after_waveform_filter,
            "total_spikes": total_active_negative_spikes,
            "sampled_spikes": len(sampled_waves),
            "duration_seconds": recording_duration_seconds,
            "firing_rate_hz": firing_rate_hz,
            "n_channels_with_spikes": int(n_channels_with_negative_spikes),
            "n_active_channels": int(len(channel_counts)),
            "mean_spikes_per_channel": float(np.mean(channel_counts)),
            "median_spikes_per_channel": float(np.median(channel_counts)),
            "max_spikes_per_channel": float(np.max(channel_counts)),
            "channel_entropy": channel_entropy,
            "waveform_mean": float(np.mean(sampled_waves)),
            "waveform_std": float(np.std(sampled_waves)),
            "waveform_min": float(np.min(sampled_waves)),
            "waveform_max": float(np.max(sampled_waves)),
            "waveform_ptp": float(np.ptp(sampled_waves)),
            "waveform_energy": float(np.mean(sampled_waves**2)),
            "polarity_filter": "negative_plausible",
        }
    )
    print(
        f"  loaded {n_input_spikes:,} spikes; kept {len(sampled_waves):,} "
        f"from {len(channel_counts)} active channels "
        f"({time.perf_counter() - RecordingStarted:.1f} s)",
        flush=True,
    )

if missing_recordings:
    pd.Series(missing_recordings, name="filename").to_csv(OUTPUT_DIR / "missing_recordings.csv", index=False)

recordings = pd.DataFrame(recording_rows)
if recordings.empty:
    raise FileNotFoundError(f"No usable negative-waveform recordings found under {SPIKE_ROOT}.")

waveforms_by_recording = {
    row["recording_name"]: waveforms_by_recording[row["recording_name"]]
    for _, row in recordings.iterrows()
}

# Preserve the waveform time axis for Mamba before MiniRocket or scalar
# aggregation. Each recording becomes a five-channel sequence containing the
# mean, standard deviation, median, and interquartile waveform profiles.
WaveformLengths = {
    np.asarray(Waveforms).shape[1] for Waveforms in waveforms_by_recording.values()
}
if len(WaveformLengths) != 1:
    raise ValueError("All recordings must use the same waveform length for Mamba classification.")

MambaSequences = {}
for RecordingName, Waveforms in waveforms_by_recording.items():
    Waveforms = np.asarray(Waveforms, dtype=np.float32)
    MambaSequences[RecordingName] = np.stack(
        [
            np.mean(Waveforms, axis=0),
            np.std(Waveforms, axis=0),
            np.median(Waveforms, axis=0),
            np.percentile(Waveforms, 25, axis=0),
            np.percentile(Waveforms, 75, axis=0),
        ],
        axis=0,
    ).astype(np.float32)

recordings.to_csv(OUTPUT_DIR / "recording_metadata_and_activity_features.csv", index=False)

print("Experiment:", EXPERIMENT_ID, "-", EXPERIMENT_NAME)
print("Classification target:", TARGET_NAME, f"({TARGET_COLUMN})")
print("Polarity filter: negative plausible")
print("Recordings:", len(recordings))
print("Biological units:", recordings[GROUP_COLUMN].nunique())
print("Classes:")
print(recordings.groupby(["label_code", "label_normalized"])["recording_name"].count().to_string())


# =========================
# CROSS-VALIDATED RECORDING-LEVEL CLASSIFICATION
# =========================

activity_columns = [
    "n_input_spikes",
    "n_after_label_filter",
    "n_after_waveform_filter",
    "total_spikes",
    "sampled_spikes",
    "duration_seconds",
    "firing_rate_hz",
    "n_channels_with_spikes",
    "n_active_channels",
    "mean_spikes_per_channel",
    "median_spikes_per_channel",
    "max_spikes_per_channel",
    "channel_entropy",
]

waveform_summary_columns = [
    "waveform_mean",
    "waveform_std",
    "waveform_min",
    "waveform_max",
    "waveform_ptp",
    "waveform_energy",
]

class_table = recordings[["label_code", "label_normalized"]].drop_duplicates().sort_values("label_code")
class_codes = class_table["label_code"].to_numpy()
class_labels = class_table["label_normalized"].to_numpy()
name_by_code = dict(zip(class_codes, class_labels))

def build_classifiers(random_state=RANDOM_STATE):
    return {
        "dummy_most_frequent": make_pipeline(
            DummyClassifier(strategy="most_frequent"),
        ),
        "ridge_pca90": make_pipeline(
            StandardScaler(),
            PCA(n_components=PCA_EXPLAINED_VARIANCE, svd_solver="full"),
            RidgeClassifierCV(alphas=np.logspace(-3, 3, 10)),
        ),
        "logistic_balanced_pca90": make_pipeline(
            StandardScaler(),
            PCA(n_components=PCA_EXPLAINED_VARIANCE, svd_solver="full"),
            LogisticRegression(class_weight="balanced", max_iter=5000, solver="lbfgs"),
        ),
        "linear_svm_balanced_pca90": make_pipeline(
            StandardScaler(),
            PCA(n_components=PCA_EXPLAINED_VARIANCE, svd_solver="full"),
            SVC(kernel="linear", class_weight="balanced", C=1.0),
        ),
        "rbf_svm_balanced_pca90": make_pipeline(
            StandardScaler(),
            PCA(n_components=PCA_EXPLAINED_VARIANCE, svd_solver="full"),
            SVC(kernel="rbf", class_weight="balanced", C=1.0, gamma="scale"),
        ),
        "random_forest_balanced": make_pipeline(
            RandomForestClassifier(
                n_estimators=300,
                class_weight="balanced",
                max_features="sqrt",
                random_state=random_state,
                n_jobs=-1,
            ),
        ),
        "extra_trees_balanced": make_pipeline(
            ExtraTreesClassifier(
                n_estimators=300,
                class_weight="balanced",
                max_features="sqrt",
                random_state=random_state,
                n_jobs=-1,
            ),
        ),
    }


classifiers = build_classifiers()

metric_rows = []
prediction_rows = []
confusion_by_model = {}

for repeat_index in range(N_REPEATS):
    n_splits = min(N_SPLITS, recordings[GROUP_COLUMN].nunique())
    if n_splits < 2:
        raise ValueError(f"At least two {GROUP_COLUMN} values are required for grouped cross-validation.")
    splitter = StratifiedGroupKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=RANDOM_STATE + repeat_index,
    )
    print(f"\n=== Repeat {repeat_index + 1}/{N_REPEATS} ===", flush=True)

    for fold_index, (train_index, test_index) in enumerate(
        splitter.split(recordings, recordings["label_code"], groups=recordings[GROUP_COLUMN])
    ):
        FoldStarted = time.perf_counter()
        train_recordings = recordings.iloc[train_index].reset_index(drop=True)
        test_recordings = recordings.iloc[test_index].reset_index(drop=True)
        print(
            f"\n[fold {fold_index + 1}/{n_splits}] "
            f"train={len(train_recordings)} recordings, test={len(test_recordings)} recordings",
            flush=True,
        )

        train_waves = np.vstack([waveforms_by_recording[name] for name in train_recordings["recording_name"]])
        fit_waves = train_waves
        if len(fit_waves) > MAX_SPIKES_FOR_ROCKET_FIT:
            selected = rng.choice(len(fit_waves), size=MAX_SPIKES_FOR_ROCKET_FIT, replace=False)
            fit_waves = fit_waves[selected]

        print(
            f"  MiniRocket: fitting {NUM_KERNELS:,} kernels on {len(fit_waves):,} waveforms...",
            flush=True,
        )
        RocketStarted = time.perf_counter()
        rocket = Classification.ExtractMiniRocketFeatures(
            fit_waves,
            fit_waves[:1],
            num_kernels=NUM_KERNELS,
            multivariate=False,
        )
        extractor = rocket["extractors"][0]
        print(f"  MiniRocket fitted in {time.perf_counter() - RocketStarted:.1f} s", flush=True)

        feature_rows = []
        selected_recordings = recordings.iloc[np.r_[train_index, test_index]]
        for FeatureIndex, (_, rec) in enumerate(selected_recordings.iterrows(), start=1):
            print(
                f"  MiniRocket transform {FeatureIndex}/{len(selected_recordings)}: "
                f"{rec['recording_name']}",
                flush=True,
            )
            rec_waves = waveforms_by_recording[rec["recording_name"]]
            rec_features = np.asarray(extractor.transform(rec_waves[:, np.newaxis, :]))
            rec_mean = rec_features.mean(axis=0)
            rec_std = rec_features.std(axis=0)
            values = {column: float(rec[column]) for column in activity_columns + waveform_summary_columns}
            values.update({f"rocket_mean_{idx}": value for idx, value in enumerate(rec_mean)})
            values.update({f"rocket_std_{idx}": value for idx, value in enumerate(rec_std)})
            feature_rows.append(pd.Series(values, name=rec["recording_name"]))

        feature_matrix = pd.DataFrame(feature_rows)
        feature_matrix.index.name = "recording_name"
        rocket_columns = [column for column in feature_matrix.columns if column.startswith("rocket_")]
        feature_sets = {
            "activity": activity_columns,
            "waveform": waveform_summary_columns + rocket_columns,
            "combined": activity_columns + waveform_summary_columns + rocket_columns,
        }

        train_names = train_recordings["recording_name"].to_list()
        test_names = test_recordings["recording_name"].to_list()
        y_train = train_recordings["label_code"].to_numpy()
        y_test = test_recordings["label_code"].to_numpy()

        if RUN_MAMBA:
            X_train_mamba = np.stack([MambaSequences[Name] for Name in train_names])
            X_test_mamba = np.stack([MambaSequences[Name] for Name in test_names])
            print(
                f"  Mamba: training {MAMBA_EPOCHS} epochs on "
                f"{X_train_mamba.shape[0]} sequences...",
                flush=True,
            )
            MambaResult = Mamba.MambaClassifierSplit(
                X_train_mamba,
                y_train,
                X_test_mamba,
                y_test=y_test,
                epochs=MAMBA_EPOCHS,
                batch_size=MAMBA_BATCH_SIZE,
                d_model=MAMBA_D_MODEL,
                n_layers=MAMBA_N_LAYERS,
                d_state=MAMBA_D_STATE,
                backend=MAMBA_BACKEND,
                random_state=RANDOM_STATE + repeat_index * N_SPLITS + fold_index,
                verbose=MAMBA_VERBOSE,
            )
            MambaModelName = "waveform_sequence_mamba"
            if SAVE_CV_FOLD_MODELS:
                fold_model_dir = OUTPUT_DIR / "cv_fold_models" / MambaModelName
                fold_model_dir.mkdir(parents=True, exist_ok=True)
                Mamba.torch.save(
                    {
                        "state_dict": MambaResult["model"].state_dict(),
                        "normalizer": MambaResult["normalizer"],
                        "classes": MambaResult["classes"],
                        "class_labels": class_labels,
                        "name_by_code": name_by_code,
                        "input_shape": tuple(X_train_mamba.shape[1:]),
                        "repeat_index": repeat_index,
                        "fold_index": fold_index,
                        "train_recordings": train_names,
                        "test_recordings": test_names,
                    },
                    fold_model_dir / f"repeat_{repeat_index:02d}_fold_{fold_index:02d}.pt",
                )
            MambaTrainPred = MambaResult["train_predictions"]
            MambaTestPred = MambaResult["test_predictions"]
            print(
                f"  Mamba completed: test balanced accuracy="
                f"{balanced_accuracy_score(y_test, MambaTestPred):.3f}, "
                f"training={MambaResult['training_seconds']:.1f} s",
                flush=True,
            )
            if MambaModelName not in confusion_by_model:
                confusion_by_model[MambaModelName] = np.zeros(
                    (len(class_codes), len(class_codes)), dtype=int
                )
            confusion_by_model[MambaModelName] += confusion_matrix(
                y_test, MambaTestPred, labels=class_codes
            )
            metric_rows.append(
                {
                    "model": MambaModelName,
                    "feature_set": "waveform_sequence",
                    "classifier": "mamba",
                    "repeat_index": repeat_index,
                    "fold_index": fold_index,
                    "train_accuracy": accuracy_score(y_train, MambaTrainPred),
                    "test_accuracy": accuracy_score(y_test, MambaTestPred),
                    "train_balanced_accuracy": balanced_accuracy_score(y_train, MambaTrainPred),
                    "test_balanced_accuracy": balanced_accuracy_score(y_test, MambaTestPred),
                    "n_train_recordings": len(y_train),
                    "n_test_recordings": len(y_test),
                    "n_train_biological_units": train_recordings[GROUP_COLUMN].nunique(),
                    "n_test_biological_units": test_recordings[GROUP_COLUMN].nunique(),
                    "n_features": int(np.prod(X_train_mamba.shape[1:])),
                    "n_pca_components": 0,
                    "num_kernels": 0,
                    "training_seconds": MambaResult["training_seconds"],
                    "inference_seconds": MambaResult["inference_seconds"],
                }
            )
            for Rec, TrueLabel, PredictedLabel in zip(
                test_recordings.to_dict("records"), y_test, MambaTestPred
            ):
                prediction_rows.append(
                    {
                        "model": MambaModelName,
                        "feature_set": "waveform_sequence",
                        "classifier": "mamba",
                        "repeat_index": repeat_index,
                        "fold_index": fold_index,
                        "recording_name": Rec["recording_name"],
                        "biological_unit_id": Rec[GROUP_COLUMN],
                        "protocol": Rec["protocol"],
                        "label_code": TrueLabel,
                        "label_normalized": name_by_code.get(TrueLabel, str(TrueLabel)),
                        "prediction_code": PredictedLabel,
                        "prediction_label": name_by_code.get(PredictedLabel, str(PredictedLabel)),
                        "correct": TrueLabel == PredictedLabel,
                    }
                )

        for feature_set_name, feature_columns in feature_sets.items():
            X_train = feature_matrix.loc[train_names, feature_columns].replace([np.inf, -np.inf], np.nan).fillna(0).to_numpy()
            X_test = feature_matrix.loc[test_names, feature_columns].replace([np.inf, -np.inf], np.nan).fillna(0).to_numpy()

            for classifier_name, model in classifiers.items():
                model_name = f"{feature_set_name}_{classifier_name}"
                ModelStarted = time.perf_counter()
                print(f"  Classical model: {model_name}...", flush=True)
                model.fit(X_train, y_train)
                pred_train = model.predict(X_train)
                pred_test = model.predict(X_test)
                pca_step = model.named_steps.get("pca")
                n_pca_components = pca_step.n_components_ if pca_step is not None else 0
                if SAVE_CV_FOLD_MODELS:
                    fold_model_dir = OUTPUT_DIR / "cv_fold_models" / model_name
                    fold_model_dir.mkdir(parents=True, exist_ok=True)
                    joblib.dump(
                        {
                            "model": model,
                            "model_name": model_name,
                            "feature_set": feature_set_name,
                            "classifier": classifier_name,
                            "feature_columns": feature_columns,
                            "rocket_extractor": extractor if feature_set_name in {"waveform", "combined"} else None,
                            "class_codes": class_codes,
                            "class_labels": class_labels,
                            "name_by_code": name_by_code,
                            "repeat_index": repeat_index,
                            "fold_index": fold_index,
                            "train_recordings": train_names,
                            "test_recordings": test_names,
                        },
                        fold_model_dir / f"repeat_{repeat_index:02d}_fold_{fold_index:02d}.joblib",
                    )
                print(
                    f"    completed in {time.perf_counter() - ModelStarted:.1f} s; "
                    f"test balanced accuracy={balanced_accuracy_score(y_test, pred_test):.3f}",
                    flush=True,
                )

                if model_name not in confusion_by_model:
                    confusion_by_model[model_name] = np.zeros((len(class_codes), len(class_codes)), dtype=int)

                confusion_by_model[model_name] += confusion_matrix(y_test, pred_test, labels=class_codes)
                metric_rows.append(
                    {
                        "model": model_name,
                        "feature_set": feature_set_name,
                        "classifier": classifier_name,
                        "repeat_index": repeat_index,
                        "fold_index": fold_index,
                        "train_accuracy": accuracy_score(y_train, pred_train),
                        "test_accuracy": accuracy_score(y_test, pred_test),
                        "train_balanced_accuracy": balanced_accuracy_score(y_train, pred_train),
                        "test_balanced_accuracy": balanced_accuracy_score(y_test, pred_test),
                        "n_train_recordings": len(y_train),
                        "n_test_recordings": len(y_test),
                        "n_train_biological_units": train_recordings[GROUP_COLUMN].nunique(),
                        "n_test_biological_units": test_recordings[GROUP_COLUMN].nunique(),
                        "n_features": X_train.shape[1],
                        "n_pca_components": n_pca_components,
                        "num_kernels": NUM_KERNELS,
                    }
                )

                for rec, true_label, predicted_label in zip(test_recordings.to_dict("records"), y_test, pred_test):
                    prediction_rows.append(
                        {
                            "model": model_name,
                            "feature_set": feature_set_name,
                            "classifier": classifier_name,
                            "repeat_index": repeat_index,
                            "fold_index": fold_index,
                            "recording_name": rec["recording_name"],
                            "biological_unit_id": rec[GROUP_COLUMN],
                            "protocol": rec["protocol"],
                            "label_code": true_label,
                            "label_normalized": name_by_code.get(true_label, str(true_label)),
                            "prediction_code": predicted_label,
                            "prediction_label": name_by_code.get(predicted_label, str(predicted_label)),
                            "correct": true_label == predicted_label,
                        }
                    )

        print(
            f"[fold {fold_index + 1}/{n_splits}] completed in "
            f"{time.perf_counter() - FoldStarted:.1f} s",
            flush=True,
        )


# =========================
# SAVE OUTPUTS
# =========================

print("\nSaving metrics, predictions, confusion matrices, and reports...", flush=True)
metrics = pd.DataFrame(metric_rows)
predictions = pd.DataFrame(prediction_rows)
metrics.to_csv(OUTPUT_DIR / "cv_metrics.csv", index=False)
predictions.to_csv(OUTPUT_DIR / "cv_recording_predictions.csv", index=False)

summary = (
    metrics.groupby("model")
    .agg(
        mean_test_accuracy=("test_accuracy", "mean"),
        std_test_accuracy=("test_accuracy", "std"),
        mean_test_balanced_accuracy=("test_balanced_accuracy", "mean"),
        std_test_balanced_accuracy=("test_balanced_accuracy", "std"),
        mean_train_accuracy=("train_accuracy", "mean"),
        mean_train_balanced_accuracy=("train_balanced_accuracy", "mean"),
        mean_pca_components=("n_pca_components", "mean"),
        n_folds=("test_accuracy", "count"),
    )
    .reset_index()
)
summary.insert(0, "experiment_id", EXPERIMENT_ID)
summary.insert(1, "target_name", TARGET_NAME)
summary.insert(2, "target_column", TARGET_COLUMN)
summary.insert(3, "split_unit", GROUP_COLUMN)
summary.insert(4, "sample_unit", "recording")
summary.insert(5, "polarity_scope", "negative_plausible")
summary.to_csv(OUTPUT_DIR / "summary_metrics.csv", index=False)
algorithm_comparison = summary.sort_values(
    ["mean_test_balanced_accuracy", "mean_test_accuracy"],
    ascending=False,
).reset_index(drop=True)
algorithm_comparison.insert(0, "rank", np.arange(1, len(algorithm_comparison) + 1))
algorithm_comparison.to_csv(OUTPUT_DIR / "algorithm_comparison.csv", index=False)


if SAVE_FINAL_MODELS:
    final_models_dir = OUTPUT_DIR / "final_models"
    final_models_dir.mkdir(parents=True, exist_ok=True)

    print("\nTraining final reusable models on all available recordings...", flush=True)
    model_manifest = []
    portal_catalog_records = []
    all_names = recordings["recording_name"].to_list()
    y_all = recordings["label_code"].to_numpy()
    classical_ranking = algorithm_comparison[
        algorithm_comparison["model"] != "waveform_sequence_mamba"
    ].copy()
    if not SAVE_DUMMY_FINAL_MODEL:
        classical_ranking = classical_ranking[~classical_ranking["model"].str.contains("dummy", regex=False)]
    if SAVE_ALL_CLASSICAL_FINAL_MODELS:
        selected_classical_models = classical_ranking["model"].to_list()
    else:
        selected_classical_models = classical_ranking["model"].head(SAVE_TOP_N_CLASSICAL_FINAL_MODELS).to_list()
    selected_final_model_names = list(selected_classical_models)
    if SAVE_MAMBA_FINAL_MODEL and RUN_MAMBA and "waveform_sequence_mamba" in set(algorithm_comparison["model"]):
        selected_final_model_names.append("waveform_sequence_mamba")

    recordings.to_csv(final_models_dir / "training_recordings.csv", index=False)
    save_json(
        final_models_dir / "training_dataset_metadata.json",
        training_dataset_metadata(recordings, selected_final_model_names),
    )

    selected_feature_sets = {
        parse_model_name(model_name)[0]
        for model_name in selected_classical_models
        if parse_model_name(model_name)[0] is not None
    }
    needs_rocket = bool(selected_feature_sets.intersection({"waveform", "combined"}))
    final_extractor = None
    final_rocket_columns = []
    if needs_rocket:
        all_waves = np.vstack([waveforms_by_recording[name] for name in all_names])
        fit_waves = all_waves
        if len(fit_waves) > MAX_SPIKES_FOR_ROCKET_FIT:
            selected = rng.choice(len(fit_waves), size=MAX_SPIKES_FOR_ROCKET_FIT, replace=False)
            fit_waves = fit_waves[selected]

        print(f"  Final MiniRocket: fitting on {len(fit_waves):,} waveforms...", flush=True)
        rocket = Classification.ExtractMiniRocketFeatures(
            fit_waves,
            fit_waves[:1],
            num_kernels=NUM_KERNELS,
            multivariate=False,
        )
        final_extractor = rocket["extractors"][0]
        first_features = np.asarray(final_extractor.transform(waveforms_by_recording[all_names[0]][:, np.newaxis, :]))
        final_rocket_columns = (
            [f"rocket_mean_{idx}" for idx in range(first_features.shape[1])]
            + [f"rocket_std_{idx}" for idx in range(first_features.shape[1])]
        )

    final_feature_sets = {
        "activity": activity_columns,
        "waveform": waveform_summary_columns + final_rocket_columns,
        "combined": activity_columns + waveform_summary_columns + final_rocket_columns,
    }
    final_feature_matrix = build_recording_feature_matrix(
        recordings,
        all_names,
        waveforms_by_recording,
        final_extractor,
        sorted(set().union(*[set(cols) for cols in final_feature_sets.values()])),
    )

    for model_name in selected_classical_models:
        feature_set_name, classifier_name = parse_model_name(model_name)
        if feature_set_name is None or classifier_name not in build_classifiers():
            continue
        feature_columns = final_feature_sets[feature_set_name]
        X_all = final_feature_matrix[feature_columns].to_numpy()
        model = build_classifiers(random_state=RANDOM_STATE)[classifier_name]
        print(f"  Final classical model: {model_name}...", flush=True)
        model.fit(X_all, y_all)
        bundle = {
            "model": model,
            "model_name": model_name,
            "feature_set": feature_set_name,
            "classifier": classifier_name,
            "feature_columns": feature_columns,
            "rocket_extractor": final_extractor if feature_set_name in {"waveform", "combined"} else None,
            "class_codes": class_codes,
            "class_labels": class_labels,
            "name_by_code": name_by_code,
            "experiment_id": EXPERIMENT_ID,
            "experiment_name": EXPERIMENT_NAME,
            "target_name": TARGET_NAME,
            "target_column": TARGET_COLUMN,
            "target_label_column": TARGET_LABEL_COLUMN,
            "group_column": GROUP_COLUMN,
            "sampling_rate_hz": SAMPLING_RATE_HZ,
            "recording_duration_seconds": RECORDING_DURATION_SECONDS,
            "negative_spike_filter": {
                "negative_spike_label": NEGATIVE_SPIKE_LABEL,
                "use_detector_label_filter": USE_DETECTOR_LABEL_FILTER,
                "use_waveform_polarity_filter": USE_WAVEFORM_POLARITY_FILTER,
                "polarity_dominance_ratio": POLARITY_DOMINANCE_RATIO,
                "require_extremum_near_center": REQUIRE_EXTREMUM_NEAR_CENTER,
                "center_window_samples": CENTER_WINDOW_SAMPLES,
                "min_active_channel_firing_rate_hz": MIN_ACTIVE_CHANNEL_FIRING_RATE_HZ,
                "exclude_reference_channel": EXCLUDE_REFERENCE_CHANNEL,
                "reference_channel": REFERENCE_CHANNEL,
            },
        }
        model_path = final_models_dir / f"{model_name}.joblib"
        joblib.dump(bundle, model_path)
        cv_row = algorithm_comparison.set_index("model").loc[model_name].to_dict()
        manifest_row = {
            "model_name": model_name,
            "metadata_path": str(final_models_dir / f"{model_name}.metadata.json"),
            "model_type": "sklearn_joblib",
            "path": str(model_path),
            "feature_set": feature_set_name,
            "classifier": classifier_name,
            "n_features": len(feature_columns),
            "n_training_recordings": len(all_names),
            "trained_on_all_available_recordings": True,
            "cv_rank": int(cv_row.get("rank", 0)),
            "cv_mean_test_accuracy": float(cv_row.get("mean_test_accuracy", np.nan)),
            "cv_mean_test_balanced_accuracy": float(cv_row.get("mean_test_balanced_accuracy", np.nan)),
            "cv_std_test_balanced_accuracy": float(cv_row.get("std_test_balanced_accuracy", np.nan)),
            "cv_mean_train_balanced_accuracy": float(cv_row.get("mean_train_balanced_accuracy", np.nan)),
            "cv_n_folds": int(cv_row.get("n_folds", 0)),
        }
        model_manifest.append(manifest_row)
        metadata = model_metadata_record(
            model_name=model_name,
            model_type="sklearn_joblib",
            model_path=model_path,
            feature_set_name=feature_set_name,
            classifier_name=classifier_name,
            feature_columns=feature_columns,
            estimator_parameters=model.get_params(deep=True),
        )
        save_json(final_models_dir / f"{model_name}.metadata.json", metadata)
        portal_catalog_records.append(portal_catalog_record(metadata))

    if SAVE_MAMBA_FINAL_MODEL and RUN_MAMBA and "waveform_sequence_mamba" in set(algorithm_comparison["model"]):
        print(f"  Final Mamba model: waveform_sequence_mamba ({MAMBA_EPOCHS} epochs)...", flush=True)
        X_all_mamba = np.stack([MambaSequences[name] for name in all_names])
        mamba_result = Mamba.MambaClassifierSplit(
            X_all_mamba,
            y_all,
            X_all_mamba,
            y_test=y_all,
            epochs=MAMBA_EPOCHS,
            batch_size=MAMBA_BATCH_SIZE,
            d_model=MAMBA_D_MODEL,
            n_layers=MAMBA_N_LAYERS,
            d_state=MAMBA_D_STATE,
            backend=MAMBA_BACKEND,
            random_state=RANDOM_STATE,
            verbose=MAMBA_VERBOSE,
        )
        mamba_path = final_models_dir / "waveform_sequence_mamba.pt"
        Mamba.torch.save(
            {
                "state_dict": mamba_result["model"].state_dict(),
                "normalizer": mamba_result["normalizer"],
                "classes": mamba_result["classes"],
                "class_labels": class_labels,
                "name_by_code": name_by_code,
                "input_shape": tuple(X_all_mamba.shape[1:]),
                "model_config": {
                    "n_channels": int(X_all_mamba.shape[1]),
                    "n_classes": int(len(mamba_result["classes"])),
                    "d_model": MAMBA_D_MODEL,
                    "n_layers": MAMBA_N_LAYERS,
                    "d_state": MAMBA_D_STATE,
                    "backend": MAMBA_BACKEND,
                },
                "experiment_id": EXPERIMENT_ID,
                "experiment_name": EXPERIMENT_NAME,
                "target_name": TARGET_NAME,
                "target_column": TARGET_COLUMN,
                "target_label_column": TARGET_LABEL_COLUMN,
                "training_recordings": all_names,
            },
            mamba_path,
        )
        cv_row = algorithm_comparison.set_index("model").loc["waveform_sequence_mamba"].to_dict()
        manifest_row = {
            "model_name": "waveform_sequence_mamba",
            "metadata_path": str(final_models_dir / "waveform_sequence_mamba.metadata.json"),
            "model_type": "torch_state_dict",
            "path": str(mamba_path),
            "feature_set": "waveform_sequence",
            "classifier": "mamba",
            "n_features": int(np.prod(X_all_mamba.shape[1:])),
            "n_training_recordings": len(all_names),
            "trained_on_all_available_recordings": True,
            "cv_rank": int(cv_row.get("rank", 0)),
            "cv_mean_test_accuracy": float(cv_row.get("mean_test_accuracy", np.nan)),
            "cv_mean_test_balanced_accuracy": float(cv_row.get("mean_test_balanced_accuracy", np.nan)),
            "cv_std_test_balanced_accuracy": float(cv_row.get("std_test_balanced_accuracy", np.nan)),
            "cv_mean_train_balanced_accuracy": float(cv_row.get("mean_train_balanced_accuracy", np.nan)),
            "cv_n_folds": int(cv_row.get("n_folds", 0)),
        }
        model_manifest.append(manifest_row)
        metadata = model_metadata_record(
            model_name="waveform_sequence_mamba",
            model_type="torch_state_dict",
            model_path=mamba_path,
            feature_set_name="waveform_sequence",
            classifier_name="mamba",
            feature_columns=[
                "waveform_mean_profile",
                "waveform_std_profile",
                "waveform_median_profile",
                "waveform_percentile_25_profile",
                "waveform_percentile_75_profile",
            ],
            estimator_parameters={
                "epochs": MAMBA_EPOCHS,
                "batch_size": MAMBA_BATCH_SIZE,
                "d_model": MAMBA_D_MODEL,
                "n_layers": MAMBA_N_LAYERS,
                "d_state": MAMBA_D_STATE,
                "backend": MAMBA_BACKEND,
            },
        )
        save_json(final_models_dir / "waveform_sequence_mamba.metadata.json", metadata)
        portal_catalog_records.append(portal_catalog_record(metadata))

    pd.DataFrame(model_manifest).to_csv(final_models_dir / "model_manifest.csv", index=False)
    save_json(final_models_dir / "model_manifest.json", model_manifest)
    save_json(final_models_dir / "portal_model_catalog.json", portal_catalog_records)
    save_json(
        final_models_dir / "training_config.json",
        {
            "experiment_id": EXPERIMENT_ID,
            "experiment_name": EXPERIMENT_NAME,
            "target_name": TARGET_NAME,
            "target_column": TARGET_COLUMN,
            "target_label_column": TARGET_LABEL_COLUMN,
            "group_column": GROUP_COLUMN,
            "class_mapping": {str(code): str(label) for code, label in name_by_code.items()},
            "save_all_classical_final_models": SAVE_ALL_CLASSICAL_FINAL_MODELS,
            "save_top_n_classical_final_models": SAVE_TOP_N_CLASSICAL_FINAL_MODELS,
            "save_dummy_final_model": SAVE_DUMMY_FINAL_MODEL,
            "save_mamba_final_model": SAVE_MAMBA_FINAL_MODEL,
            "num_kernels": NUM_KERNELS,
            "pca_explained_variance": PCA_EXPLAINED_VARIANCE,
            "mamba": {
                "epochs": MAMBA_EPOCHS,
                "batch_size": MAMBA_BATCH_SIZE,
                "d_model": MAMBA_D_MODEL,
                "n_layers": MAMBA_N_LAYERS,
                "d_state": MAMBA_D_STATE,
                "backend": MAMBA_BACKEND,
            },
        },
    )

for model_name, cm in confusion_by_model.items():
    cm_table = pd.DataFrame(cm, index=class_labels, columns=class_labels)
    cm_table.to_csv(OUTPUT_DIR / f"confusion_matrix_{model_name}.csv")

    plt.figure(figsize=(7, 6))
    sns.heatmap(cm_table, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"Negative recording-level classification - {model_name}")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"confusion_matrix_{model_name}.png", dpi=180)
    plt.close()

    model_predictions = predictions[predictions["model"] == model_name]
    report = classification_report(
        model_predictions["label_code"],
        model_predictions["prediction_code"],
        labels=class_codes,
        target_names=class_labels,
        output_dict=True,
        zero_division=0,
    )
    pd.DataFrame(report).T.to_csv(OUTPUT_DIR / f"classification_report_{model_name}.csv")

print("\nSaved negative recording-level classification outputs in:", OUTPUT_DIR, flush=True)
print(f"Total elapsed time: {time.perf_counter() - AnalysisStarted:.1f} s", flush=True)
print("\nAlgorithm ranking by mean test balanced accuracy:")
print(
    algorithm_comparison[
        ["rank", "model", "mean_test_balanced_accuracy", "std_test_balanced_accuracy", "mean_test_accuracy"]
    ].to_string(index=False)
)
