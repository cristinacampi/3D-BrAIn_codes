"""Reusable recording-level DIV regression utilities for MEA experiments.

This main is the regression counterpart of the DIV classification campaign.
Each sample is one recording, the target is the numeric ``DIV`` value, and
cross-validation is group-disjoint by ``biological_unit_id`` so that recordings
from the same biological unit do not appear in both train and test folds.

Features are extracted from the spike polarity selected by the caller:

- detector label 0 when labels are available;
- dominant negative trough;
- waveform extremum close to the waveform centre;
- active channels after filtering, with firing rate > 0.25 Hz by default.

The script evaluates activity-only, waveform-only and combined feature sets.
Waveform MiniRocket features are fitted inside each training fold only, then
summarised per recording with mean and standard deviation across waveforms.
"""

import json
import os
import re
import sys
import time
from pathlib import Path

plot_cache = Path(os.environ.get("TMPDIR", "/tmp")) / "3dbrain_matplotlib_cache"
plot_cache.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(plot_cache))
os.environ.setdefault("XDG_CACHE_HOME", str(plot_cache))
os.environ.setdefault("NUMBA_CACHE_DIR", str(Path(os.environ.get("TMPDIR", "/tmp")) / "numba_cache"))

import joblib
import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNet, RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

for parent in Path(__file__).resolve().parents:
    src_path = parent / "src"
    if (src_path / "brain_3d").is_dir():
        sys.path.insert(0, str(src_path))
        break

from brain_3d import Classification, SpikeSorting


def RunDIVRegression(
    EXPERIMENT_ID,
    EXPERIMENT_NAME,
    SPIKE_ROOT,
    LABELS_CSV,
    OUTPUT_DIR,
    GROUP_COLUMN,
    TARGET_COLUMN,
    TARGET_NAME,
    SAMPLING_RATE_HZ,
    RECORDING_DURATION_SECONDS,
    MIN_ACTIVE_CHANNEL_FIRING_RATE_HZ,
    EXCLUDE_REFERENCE_CHANNEL,
    REFERENCE_CHANNEL,
    POLARITY_FILTER,
    NEGATIVE_SPIKE_LABEL,
    POSITIVE_SPIKE_LABEL,
    USE_DETECTOR_LABEL_FILTER,
    USE_WAVEFORM_POLARITY_FILTER,
    POLARITY_DOMINANCE_RATIO,
    REQUIRE_EXTREMUM_NEAR_CENTER,
    CENTER_WINDOW_SAMPLES,
    MAX_SPIKES_PER_RECORDING,
    MAX_SPIKES_FOR_ROCKET_FIT,
    NUM_KERNELS,
    PCA_EXPLAINED_VARIANCE,
    N_SPLITS,
    N_REPEATS,
    N_DIV_STRATIFICATION_BINS,
    RANDOM_STATE,
    FORCE_REPROCESS,
    SAVE_FINAL_MODELS,
    SAVE_ALL_FINAL_MODELS,
    SAVE_TOP_N_FINAL_MODELS,
    SAVE_DUMMY_FINAL_MODEL,
):
    """Run DIV regression using only explicitly supplied configuration."""
    # SETUP
    # =========================

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    done_file = OUTPUT_DIR / "summary_metrics.csv"
    if done_file.exists() and not FORCE_REPROCESS:
        print(f"{POLARITY_FILTER.capitalize()}-spike DIV regression already processed:", done_file)
        print("Set FORCE_REPROCESS = True in the parameter block to recompute it.")
        raise SystemExit(0)

    rng = np.random.default_rng(RANDOM_STATE)
    analysis_started = time.perf_counter()

    print(f"\n=== {POLARITY_FILTER.capitalize()}-spike recording-level DIV regression ===", flush=True)
    print(f"Experiment: {EXPERIMENT_ID} - {EXPERIMENT_NAME}", flush=True)
    print(f"Spike folder: {SPIKE_ROOT}", flush=True)
    print(f"Output folder: {OUTPUT_DIR}", flush=True)


    # =========================
    # HELPERS
    # =========================

    def ProtocolFromFilename(filename):
        """Infer the pharmacological protocol encoded in a filename."""
        return Classification.ProtocolFromFilename(filename)


    def SelectedWaveformMask(spikes, labels):
        """Select events using the shared polarity-filtering rules."""
        return SpikeSorting.WaveformPolarityMask(
            spikes,
            labels,
            SpikeSorting.SpikePolarityFilterConfig(
                polarity=POLARITY_FILTER,
                negative_spike_label=NEGATIVE_SPIKE_LABEL,
                positive_spike_label=POSITIVE_SPIKE_LABEL,
                use_detector_label_filter=USE_DETECTOR_LABEL_FILTER,
                use_waveform_polarity_filter=USE_WAVEFORM_POLARITY_FILTER,
                polarity_dominance_ratio=POLARITY_DOMINANCE_RATIO,
                require_extremum_near_center=REQUIRE_EXTREMUM_NEAR_CENTER,
                center_window_samples=CENTER_WINDOW_SAMPLES,
            ),
        )


    def ToJsonable(value):
        """Convert NumPy, path, and nested values into JSON-compatible data."""
        if isinstance(value, dict):
            return {str(key): ToJsonable(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [ToJsonable(item) for item in value]
        if isinstance(value, np.ndarray):
            return ToJsonable(value.tolist())
        if isinstance(value, np.integer):
            return int(value)
        if isinstance(value, np.floating):
            return float(value)
        if isinstance(value, np.bool_):
            return bool(value)
        if isinstance(value, Path):
            return str(value)
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        return str(value)


    def SaveJson(path, payload):
        """Serialize a metadata payload as deterministic indented JSON."""
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(ToJsonable(payload), handle, indent=2, sort_keys=True)


    def BuildRegressors(random_state=RANDOM_STATE):
        """Construct fresh baseline, linear, kernel, and tree regressors."""
        return {
            "dummy_mean": make_pipeline(DummyRegressor(strategy="mean")),
            "ridge_pca90": make_pipeline(
                StandardScaler(),
                PCA(n_components=PCA_EXPLAINED_VARIANCE, svd_solver="full"),
                RidgeCV(alphas=np.logspace(-3, 3, 10)),
            ),
            "elasticnet_pca90": make_pipeline(
                StandardScaler(),
                PCA(n_components=PCA_EXPLAINED_VARIANCE, svd_solver="full"),
                ElasticNet(alpha=0.05, l1_ratio=0.2, max_iter=10000, random_state=random_state),
            ),
            "linear_svr_pca90": make_pipeline(
                StandardScaler(),
                PCA(n_components=PCA_EXPLAINED_VARIANCE, svd_solver="full"),
                SVR(kernel="linear", C=1.0, epsilon=5.0),
            ),
            "rbf_svr_pca90": make_pipeline(
                StandardScaler(),
                PCA(n_components=PCA_EXPLAINED_VARIANCE, svd_solver="full"),
                SVR(kernel="rbf", C=1.0, epsilon=5.0, gamma="scale"),
            ),
            "random_forest": make_pipeline(
                RandomForestRegressor(
                    n_estimators=300,
                    max_features="sqrt",
                    random_state=random_state,
                    n_jobs=-1,
                ),
            ),
            "extra_trees": make_pipeline(
                ExtraTreesRegressor(
                    n_estimators=300,
                    max_features="sqrt",
                    random_state=random_state,
                    n_jobs=-1,
                ),
            ),
        }


    def DivBinsForStratification(div_values):
        """Discretize DIV values only for balanced cross-validation splitting."""
        unique_values = pd.Series(div_values).nunique()
        n_bins = min(N_DIV_STRATIFICATION_BINS, unique_values)
        if n_bins < 2:
            return np.zeros(len(div_values), dtype=int)
        try:
            return pd.qcut(div_values, q=n_bins, labels=False, duplicates="drop").astype(int)
        except ValueError:
            return pd.cut(div_values, bins=n_bins, labels=False, include_lowest=True).astype(int)


    def CvSplits(recordings_table, repeat_index):
        """Build repeat-specific, biological-unit-disjoint regression folds."""
        y = recordings_table[TARGET_COLUMN].to_numpy(dtype=float)
        groups = recordings_table[GROUP_COLUMN].to_numpy()
        n_groups = recordings_table[GROUP_COLUMN].nunique()
        n_splits = min(N_SPLITS, n_groups)
        if n_splits < 2:
            raise ValueError(f"At least two {GROUP_COLUMN} values are required for grouped cross-validation.")

        div_bins = DivBinsForStratification(y)
        try:
            splitter = StratifiedGroupKFold(
                n_splits=n_splits,
                shuffle=True,
                random_state=RANDOM_STATE + repeat_index,
            )
            return list(splitter.split(recordings_table, div_bins, groups=groups)), "StratifiedGroupKFold_on_DIV_bins"
        except ValueError:
            splitter = GroupKFold(n_splits=n_splits)
            return list(splitter.split(recordings_table, y, groups=groups)), "GroupKFold_fallback"


    def RegressionScores(y_true, y_pred):
        """Calculate error, fit, correlation, and temporal-tolerance metrics."""
        y_true = np.asarray(y_true, dtype=float)
        y_pred = np.asarray(y_pred, dtype=float)
        abs_error = np.abs(y_true - y_pred)
        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        if len(np.unique(y_true)) > 1 and len(np.unique(y_pred)) > 1:
            spearman = pd.Series(y_true).corr(pd.Series(y_pred), method="spearman")
            pearson = pd.Series(y_true).corr(pd.Series(y_pred), method="pearson")
        else:
            spearman = np.nan
            pearson = np.nan
        return {
            "mae_days": mean_absolute_error(y_true, y_pred),
            "median_absolute_error_days": float(np.median(abs_error)),
            "rmse_days": rmse,
            "r2": r2_score(y_true, y_pred) if len(y_true) > 1 else np.nan,
            "spearman": spearman,
            "pearson": pearson,
            "within_5_days": float(np.mean(abs_error <= 5)),
            "within_10_days": float(np.mean(abs_error <= 10)),
            "within_15_days": float(np.mean(abs_error <= 15)),
        }


    def BuildRecordingFeatureMatrix(recordings_table, names, waveforms_by_name, extractor, feature_columns):
        """Aggregate activity, waveform, and MiniRocket features by recording."""
        rows = []
        selected_recordings = recordings_table.set_index("recording_name").loc[names].reset_index()
        for _, rec in selected_recordings.iterrows():
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


    def DatasetMetadata(selected_model_names):
        """Describe the recordings, DIV range, groups, and selected models."""
        div_summary = (
            recordings.groupby("DIV", as_index=False)
            .agg(
                recordings=("recording_name", "count"),
                biological_units=(GROUP_COLUMN, "nunique"),
                sampled_spikes=("sampled_spikes", "sum"),
                active_channels=("n_active_channels", "sum"),
            )
            .sort_values("DIV")
        )
        return {
            "experiment_id": EXPERIMENT_ID,
            "experiment_name": EXPERIMENT_NAME,
            "target_name": TARGET_NAME,
            "target_column": TARGET_COLUMN,
            "target_type": "continuous_regression",
            "group_column": GROUP_COLUMN,
            "sample_unit": "recording",
            "split_unit": GROUP_COLUMN,
            "polarity_scope": POLARITY_FILTER,
            "n_recordings": int(len(recordings)),
            "n_biological_units": int(recordings[GROUP_COLUMN].nunique()),
            "div_min": float(recordings["DIV"].min()),
            "div_max": float(recordings["DIV"].max()),
            "total_input_spikes": int(recordings["n_input_spikes"].sum()),
            "total_detector_selected_spikes": int(recordings["n_after_label_filter"].sum()),
            "total_plausible_selected_spikes": int(recordings["n_after_waveform_filter"].sum()),
            "total_training_sampled_spikes": int(recordings["sampled_spikes"].sum()),
            "div_distribution": div_summary.to_dict("records"),
            "selected_final_models": selected_model_names,
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
                "n_div_stratification_bins": N_DIV_STRATIFICATION_BINS,
                "random_state": RANDOM_STATE,
            },
            "spike_polarity_filter": {
                "polarity": POLARITY_FILTER,
                "negative_spike_label": NEGATIVE_SPIKE_LABEL,
                "positive_spike_label": POSITIVE_SPIKE_LABEL,
                "use_detector_label_filter": USE_DETECTOR_LABEL_FILTER,
                "use_waveform_polarity_filter": USE_WAVEFORM_POLARITY_FILTER,
                "polarity_dominance_ratio": POLARITY_DOMINANCE_RATIO,
                "require_extremum_near_center": REQUIRE_EXTREMUM_NEAR_CENTER,
                "center_window_samples": CENTER_WINDOW_SAMPLES,
            },
        }


    def ModelMetadataRecord(model_name, model_path, feature_set_name, regressor_name, feature_columns, estimator_parameters):
        """Build the reusable-model metadata record for one fitted regressor."""
        cv_rows = metrics[metrics["model"] == model_name].copy()
        summary_row = algorithm_comparison.set_index("model").loc[model_name].to_dict()
        return {
            "model_id": model_name,
            "model_name": model_name,
            "model_type": "sklearn_joblib",
            "task": "DIV regression",
            "regressor": regressor_name,
            "framework": "scikit-learn",
            "model_path": str(model_path),
            "created_by_main": Path(__file__).name,
            "dataset": DatasetMetadata([model_name]),
            "training_recordings": recordings[
                [
                    "recording_name",
                    GROUP_COLUMN,
                    "DIV",
                    "protocol",
                    "sampled_spikes",
                    "n_active_channels",
                    "firing_rate_hz",
                ]
            ].to_dict("records"),
            "feature_set": feature_set_name,
            "feature_columns": feature_columns,
            "n_features": len(feature_columns),
            "estimator_parameters": estimator_parameters,
            "preprocessing": {
                "polarity_scope": POLARITY_FILTER,
                "sampling_rate_hz": SAMPLING_RATE_HZ,
                "recording_duration_seconds": RECORDING_DURATION_SECONDS,
                "min_active_channel_firing_rate_hz": MIN_ACTIVE_CHANNEL_FIRING_RATE_HZ,
                "exclude_reference_channel": EXCLUDE_REFERENCE_CHANNEL,
                "reference_channel": REFERENCE_CHANNEL,
                "max_spikes_per_recording": MAX_SPIKES_PER_RECORDING,
                "max_spikes_for_rocket_fit": MAX_SPIKES_FOR_ROCKET_FIT,
                "num_kernels": NUM_KERNELS if feature_set_name in {"waveform", "combined"} else 0,
                "pca_explained_variance": PCA_EXPLAINED_VARIANCE if "pca90" in regressor_name else None,
            },
            "validation": {
                "strategy": "Repeated group-disjoint CV, stratified on DIV quantile bins when possible",
                "split_unit": GROUP_COLUMN,
                "n_splits": N_SPLITS,
                "n_repeats": N_REPEATS,
                "random_state": RANDOM_STATE,
                "independent_holdout_test": False,
                "test_metrics_are_cross_validated": True,
            },
            "metrics": {
                "rank": int(summary_row.get("rank", 0)),
                "cv_summary": summary_row,
                "cv_folds": cv_rows.to_dict("records"),
                "training_metrics": {
                    "mean_cv_train_mae_days": summary_row.get("mean_train_mae_days"),
                    "mean_cv_train_rmse_days": summary_row.get("mean_train_rmse_days"),
                    "mean_cv_train_r2": summary_row.get("mean_train_r2"),
                },
                "test_metrics": {
                    "mean_cv_test_mae_days": summary_row.get("mean_test_mae_days"),
                    "std_cv_test_mae_days": summary_row.get("std_test_mae_days"),
                    "mean_cv_test_rmse_days": summary_row.get("mean_test_rmse_days"),
                    "mean_cv_test_r2": summary_row.get("mean_test_r2"),
                    "mean_cv_test_spearman": summary_row.get("mean_test_spearman"),
                    "mean_cv_within_5_days": summary_row.get("mean_test_within_5_days"),
                    "mean_cv_within_10_days": summary_row.get("mean_test_within_10_days"),
                    "mean_cv_within_15_days": summary_row.get("mean_test_within_15_days"),
                    "n_cv_folds": summary_row.get("n_folds"),
                },
            },
            "reuse_notes": (
                "Final reusable model fitted on all available labelled recordings after cross-validation. "
                "Reported test metrics are cross-validated held-out-fold metrics, not a separate external test set."
            ),
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
    if EXPERIMENT_ID is None:
        labels = labels[target_mask].copy()
    else:
        labels = labels[(labels["experiment_id"] == EXPERIMENT_ID) & target_mask].copy()

    if labels.empty:
        raise ValueError(f"No DIV labels found for experiment {EXPERIMENT_ID}.")

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

    sorted_labels = labels.sort_values(["DIV", "chip_ID", "filename"])
    print(f"\nLoading {len(sorted_labels)} labeled recordings...", flush=True)
    for recording_index, (_, label_row) in enumerate(sorted_labels.iterrows(), start=1):
        recording_started = time.perf_counter()
        recording_key = label_row["recording_key"]
        print(f"[recording {recording_index}/{len(sorted_labels)}] {label_row['filename']}", flush=True)
        if recording_key not in recording_dirs:
            missing_recordings.append(label_row["filename"])
            print("  not found: skipped", flush=True)
            continue

        recording_dir = recording_dirs[recording_key]
        recording_duration_seconds = float(RECORDING_DURATION_SECONDS)
        active_waveforms = []
        active_channel_counts = []
        n_input_spikes = 0
        n_after_label_filter = 0
        n_after_waveform_filter = 0
        n_channels_with_negative_spikes = 0

        for channel, arrays in SpikeSorting.LoadSpikeRecording(recording_dir).items():
            if EXCLUDE_REFERENCE_CHANNEL and channel == REFERENCE_CHANNEL:
                continue

            spikes = arrays["Spikes"]
            if spikes.size == 0:
                continue
            spikes = np.asarray(spikes).reshape(spikes.shape[0], -1)

            spike_labels = arrays["Labels"]

            n_spikes = min(spikes.shape[0], len(spike_labels))
            spikes = spikes[:n_spikes]
            spike_labels = spike_labels[:n_spikes]
            n_input_spikes += n_spikes

            label_keep = SpikeSorting.WaveformPolarityMask(
                spikes,
                spike_labels,
                SpikeSorting.SpikePolarityFilterConfig(
                    polarity=POLARITY_FILTER,
                    negative_spike_label=NEGATIVE_SPIKE_LABEL,
                    positive_spike_label=POSITIVE_SPIKE_LABEL,
                    use_detector_label_filter=USE_DETECTOR_LABEL_FILTER,
                    use_waveform_polarity_filter=False,
                ),
            )
            n_after_label_filter += int(label_keep.sum())
            keep = SelectedWaveformMask(spikes, spike_labels)
            n_after_waveform_filter += int(keep.sum())

            spikes = spikes[keep]
            if len(spikes) == 0:
                continue

            n_channels_with_negative_spikes += 1
            channel_firing_rate_hz = len(spikes) / recording_duration_seconds
            if channel_firing_rate_hz <= MIN_ACTIVE_CHANNEL_FIRING_RATE_HZ:
                continue

            active_waveforms.append(spikes)
            active_channel_counts.append(len(spikes))

        if not active_waveforms:
            print(f"  no active {POLARITY_FILTER}-waveform channels: skipped", flush=True)
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
        waveforms_by_recording[recording_name] = sampled_waves
        recording_rows.append(
            {
                "recording_name": recording_name,
                "filename": label_row["filename"],
                "experiment_id": EXPERIMENT_ID,
                "biological_unit_id": label_row[GROUP_COLUMN],
                "chip_ID": int(label_row["chip_ID"]),
                "DIV": float(label_row["DIV"]),
                "div_label": label_row.get("div_label", f"DIV_{int(label_row['DIV']):03d}"),
                "div_code": int(label_row.get("div_code", int(label_row["DIV"]))),
                "Days_in_BF": int(label_row["Days_in_BF"]),
                "protocol": ProtocolFromFilename(label_row["filename"]),
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
                "polarity_filter": POLARITY_FILTER,
            }
        )
        print(
            f"  loaded {n_input_spikes:,} spikes; kept {len(sampled_waves):,} "
            f"from {len(channel_counts)} active channels ({time.perf_counter() - recording_started:.1f} s)",
            flush=True,
        )

    if missing_recordings:
        pd.Series(missing_recordings, name="filename").to_csv(OUTPUT_DIR / "missing_recordings.csv", index=False)

    recordings = pd.DataFrame(recording_rows)
    if recordings.empty:
        raise FileNotFoundError(f"No usable {POLARITY_FILTER}-waveform recordings found under {SPIKE_ROOT}.")

    waveforms_by_recording = {
        row["recording_name"]: waveforms_by_recording[row["recording_name"]]
        for _, row in recordings.iterrows()
    }

    recordings.to_csv(OUTPUT_DIR / "recording_metadata_and_activity_features.csv", index=False)

    print("Experiment:", EXPERIMENT_ID, "-", EXPERIMENT_NAME)
    print("Regression target:", TARGET_NAME, f"({TARGET_COLUMN})")
    print("Polarity filter: negative plausible")
    print("Recordings:", len(recordings))
    print("Biological units:", recordings[GROUP_COLUMN].nunique())
    print("DIV range:", int(recordings["DIV"].min()), "-", int(recordings["DIV"].max()))


    # =========================
    # CROSS-VALIDATED REGRESSION
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

    regressors = BuildRegressors()
    metric_rows = []
    prediction_rows = []

    for repeat_index in range(N_REPEATS):
        splits, split_strategy = CvSplits(recordings, repeat_index)
        print(f"\n=== Repeat {repeat_index + 1}/{N_REPEATS} ({split_strategy}) ===", flush=True)

        for fold_index, (train_index, test_index) in enumerate(splits):
            fold_started = time.perf_counter()
            train_recordings = recordings.iloc[train_index].reset_index(drop=True)
            test_recordings = recordings.iloc[test_index].reset_index(drop=True)
            print(
                f"\n[fold {fold_index + 1}/{len(splits)}] "
                f"train={len(train_recordings)} recordings, test={len(test_recordings)} recordings",
                flush=True,
            )

            train_waves = np.vstack([waveforms_by_recording[name] for name in train_recordings["recording_name"]])
            fit_waves = train_waves
            if len(fit_waves) > MAX_SPIKES_FOR_ROCKET_FIT:
                selected = rng.choice(len(fit_waves), size=MAX_SPIKES_FOR_ROCKET_FIT, replace=False)
                fit_waves = fit_waves[selected]

            print(f"  MiniRocket: fitting {NUM_KERNELS:,} kernels on {len(fit_waves):,} waveforms...", flush=True)
            rocket_started = time.perf_counter()
            rocket = Classification.ExtractMiniRocketFeatures(
                fit_waves,
                fit_waves[:1],
                num_kernels=NUM_KERNELS,
                multivariate=False,
            )
            extractor = rocket["extractors"][0]
            print(f"  MiniRocket fitted in {time.perf_counter() - rocket_started:.1f} s", flush=True)

            feature_rows = []
            selected_recordings = recordings.iloc[np.r_[train_index, test_index]]
            for feature_index, (_, rec) in enumerate(selected_recordings.iterrows(), start=1):
                print(
                    f"  MiniRocket transform {feature_index}/{len(selected_recordings)}: "
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
            y_train = train_recordings[TARGET_COLUMN].to_numpy(dtype=float)
            y_test = test_recordings[TARGET_COLUMN].to_numpy(dtype=float)

            for feature_set_name, feature_columns in feature_sets.items():
                X_train = feature_matrix.loc[train_names, feature_columns].replace([np.inf, -np.inf], np.nan).fillna(0).to_numpy()
                X_test = feature_matrix.loc[test_names, feature_columns].replace([np.inf, -np.inf], np.nan).fillna(0).to_numpy()

                for regressor_name, model in BuildRegressors(random_state=RANDOM_STATE + repeat_index * N_SPLITS + fold_index).items():
                    model_name = f"{feature_set_name}_{regressor_name}"
                    model_started = time.perf_counter()
                    print(f"  Regressor: {model_name}...", flush=True)
                    model.fit(X_train, y_train)
                    pred_train = model.predict(X_train)
                    pred_test = model.predict(X_test)
                    pca_step = model.named_steps.get("pca")
                    n_pca_components = pca_step.n_components_ if pca_step is not None else 0
                    train_scores = RegressionScores(y_train, pred_train)
                    test_scores = RegressionScores(y_test, pred_test)
                    elapsed = time.perf_counter() - model_started
                    print(
                        f"    completed in {elapsed:.1f} s; "
                        f"test MAE={test_scores['mae_days']:.2f} days, "
                        f"within 10 days={test_scores['within_10_days']:.2f}",
                        flush=True,
                    )

                    metric_rows.append(
                        {
                            "model": model_name,
                            "feature_set": feature_set_name,
                            "regressor": regressor_name,
                            "repeat_index": repeat_index,
                            "fold_index": fold_index,
                            "split_strategy": split_strategy,
                            "n_train_recordings": len(y_train),
                            "n_test_recordings": len(y_test),
                            "n_train_biological_units": train_recordings[GROUP_COLUMN].nunique(),
                            "n_test_biological_units": test_recordings[GROUP_COLUMN].nunique(),
                            "n_features": X_train.shape[1],
                            "n_pca_components": n_pca_components,
                            "num_kernels": NUM_KERNELS if feature_set_name in {"waveform", "combined"} else 0,
                            "training_seconds": elapsed,
                            **{f"train_{key}": value for key, value in train_scores.items()},
                            **{f"test_{key}": value for key, value in test_scores.items()},
                        }
                    )

                    for rec, true_div, predicted_div in zip(test_recordings.to_dict("records"), y_test, pred_test):
                        prediction_rows.append(
                            {
                                "model": model_name,
                                "feature_set": feature_set_name,
                                "regressor": regressor_name,
                                "repeat_index": repeat_index,
                                "fold_index": fold_index,
                                "recording_name": rec["recording_name"],
                                "biological_unit_id": rec[GROUP_COLUMN],
                                "protocol": rec["protocol"],
                                "true_div": true_div,
                                "predicted_div": predicted_div,
                                "absolute_error_days": abs(true_div - predicted_div),
                            }
                        )

            print(f"[fold {fold_index + 1}/{len(splits)}] completed in {time.perf_counter() - fold_started:.1f} s", flush=True)


    # =========================
    # SAVE OUTPUTS
    # =========================

    print("\nSaving DIV regression outputs...", flush=True)
    metrics = pd.DataFrame(metric_rows)
    predictions = pd.DataFrame(prediction_rows)
    metrics.to_csv(OUTPUT_DIR / "cv_metrics.csv", index=False)
    predictions.to_csv(OUTPUT_DIR / "cv_recording_predictions.csv", index=False)

    summary = (
        metrics.groupby("model")
        .agg(
            mean_test_mae_days=("test_mae_days", "mean"),
            std_test_mae_days=("test_mae_days", "std"),
            mean_test_rmse_days=("test_rmse_days", "mean"),
            mean_test_r2=("test_r2", "mean"),
            mean_test_spearman=("test_spearman", "mean"),
            mean_test_pearson=("test_pearson", "mean"),
            mean_test_within_5_days=("test_within_5_days", "mean"),
            mean_test_within_10_days=("test_within_10_days", "mean"),
            mean_test_within_15_days=("test_within_15_days", "mean"),
            mean_train_mae_days=("train_mae_days", "mean"),
            mean_train_rmse_days=("train_rmse_days", "mean"),
            mean_train_r2=("train_r2", "mean"),
            mean_pca_components=("n_pca_components", "mean"),
            n_folds=("test_mae_days", "count"),
        )
        .reset_index()
    )
    summary.insert(0, "experiment_id", EXPERIMENT_ID)
    summary.insert(1, "target_name", TARGET_NAME)
    summary.insert(2, "target_column", TARGET_COLUMN)
    summary.insert(3, "split_unit", GROUP_COLUMN)
    summary.insert(4, "sample_unit", "recording")
    summary.insert(5, "polarity_scope", POLARITY_FILTER)
    summary.to_csv(OUTPUT_DIR / "summary_metrics.csv", index=False)

    algorithm_comparison = summary.sort_values(
        ["mean_test_mae_days", "mean_test_rmse_days"],
        ascending=True,
    ).reset_index(drop=True)
    algorithm_comparison.insert(0, "rank", np.arange(1, len(algorithm_comparison) + 1))
    algorithm_comparison.to_csv(OUTPUT_DIR / "algorithm_comparison.csv", index=False)

    best_model_name = algorithm_comparison.loc[0, "model"]
    best_predictions = predictions[predictions["model"] == best_model_name]

    plt.figure(figsize=(7, 6))
    sns.scatterplot(data=best_predictions, x="true_div", y="predicted_div", hue="feature_set", s=70)
    min_div = min(best_predictions["true_div"].min(), best_predictions["predicted_div"].min())
    max_div = max(best_predictions["true_div"].max(), best_predictions["predicted_div"].max())
    plt.plot([min_div, max_div], [min_div, max_div], color="black", linewidth=1, linestyle="--")
    plt.xlabel("True DIV")
    plt.ylabel("Predicted DIV")
    plt.title(f"Best DIV regressor: {best_model_name}")
    plt.legend(loc="upper left", bbox_to_anchor=(1.02, 1), frameon=False)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "best_model_true_vs_predicted_div.png", dpi=180, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(8, 4.8))
    plot_df = algorithm_comparison.head(15).copy()
    sns.barplot(data=plot_df, y="model", x="mean_test_mae_days", color="#4c78a8")
    plt.xlabel("Mean CV test MAE (days)")
    plt.ylabel("")
    plt.title("DIV regression model ranking")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "model_ranking_mae.png", dpi=180, bbox_inches="tight")
    plt.close()

    SaveJson(OUTPUT_DIR / "dataset_metadata.json", DatasetMetadata(algorithm_comparison["model"].to_list()))


    # =========================
    # SAVE FINAL REUSABLE MODELS
    # =========================

    if SAVE_FINAL_MODELS:
        final_models_dir = OUTPUT_DIR / "final_models"
        final_models_dir.mkdir(parents=True, exist_ok=True)
        print("\nTraining final reusable regressors on all available recordings...", flush=True)

        all_names = recordings["recording_name"].to_list()
        y_all = recordings[TARGET_COLUMN].to_numpy(dtype=float)
        model_ranking = algorithm_comparison.copy()
        if not SAVE_DUMMY_FINAL_MODEL:
            model_ranking = model_ranking[~model_ranking["model"].str.contains("dummy", regex=False)]
        if SAVE_ALL_FINAL_MODELS:
            selected_models = model_ranking["model"].to_list()
        else:
            selected_models = model_ranking["model"].head(SAVE_TOP_N_FINAL_MODELS).to_list()

        final_models_dir.mkdir(parents=True, exist_ok=True)
        recordings.to_csv(final_models_dir / "training_recordings.csv", index=False)
        SaveJson(final_models_dir / "training_dataset_metadata.json", DatasetMetadata(selected_models))

        selected_feature_sets = {model_name.split("_", 1)[0] for model_name in selected_models}
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
        final_feature_matrix = BuildRecordingFeatureMatrix(
            recordings,
            all_names,
            waveforms_by_recording,
            final_extractor,
            sorted(set().union(*[set(cols) for cols in final_feature_sets.values()])),
        )

        model_manifest = []
        for model_name in selected_models:
            feature_set_name, regressor_name = model_name.split("_", 1)
            if feature_set_name not in final_feature_sets:
                continue
            regressors_for_model = BuildRegressors(random_state=RANDOM_STATE)
            if regressor_name not in regressors_for_model:
                continue

            feature_columns = final_feature_sets[feature_set_name]
            X_all = final_feature_matrix[feature_columns].to_numpy()
            model = regressors_for_model[regressor_name]
            print(f"  Final regressor: {model_name}...", flush=True)
            model.fit(X_all, y_all)

            model_path = final_models_dir / f"{model_name}.joblib"
            bundle = {
                "model": model,
                "model_name": model_name,
                "task": "div_regression",
                "feature_set": feature_set_name,
                "regressor": regressor_name,
                "feature_columns": feature_columns,
                "rocket_extractor": final_extractor if feature_set_name in {"waveform", "combined"} else None,
                "experiment_id": EXPERIMENT_ID,
                "experiment_name": EXPERIMENT_NAME,
                "target_name": TARGET_NAME,
                "target_column": TARGET_COLUMN,
                "group_column": GROUP_COLUMN,
                "sampling_rate_hz": SAMPLING_RATE_HZ,
                "recording_duration_seconds": RECORDING_DURATION_SECONDS,
                "spike_polarity_filter": {
                    "polarity": POLARITY_FILTER,
                    "negative_spike_label": NEGATIVE_SPIKE_LABEL,
                    "positive_spike_label": POSITIVE_SPIKE_LABEL,
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
            joblib.dump(bundle, model_path)

            cv_row = algorithm_comparison.set_index("model").loc[model_name].to_dict()
            metadata_path = final_models_dir / f"{model_name}.metadata.json"
            metadata = ModelMetadataRecord(
                model_name=model_name,
                model_path=model_path,
                feature_set_name=feature_set_name,
                regressor_name=regressor_name,
                feature_columns=feature_columns,
                estimator_parameters=model.get_params(deep=True),
            )
            SaveJson(metadata_path, metadata)
            model_manifest.append(
                {
                    "model_name": model_name,
                    "metadata_path": str(metadata_path),
                    "model_type": "sklearn_joblib",
                    "path": str(model_path),
                    "feature_set": feature_set_name,
                    "regressor": regressor_name,
                    "n_features": len(feature_columns),
                    "n_training_recordings": len(all_names),
                    "trained_on_all_available_recordings": True,
                    "cv_rank": int(cv_row.get("rank", 0)),
                    "cv_mean_test_mae_days": float(cv_row.get("mean_test_mae_days", np.nan)),
                    "cv_mean_test_rmse_days": float(cv_row.get("mean_test_rmse_days", np.nan)),
                    "cv_mean_test_r2": float(cv_row.get("mean_test_r2", np.nan)),
                    "cv_mean_test_spearman": float(cv_row.get("mean_test_spearman", np.nan)),
                    "cv_mean_test_within_10_days": float(cv_row.get("mean_test_within_10_days", np.nan)),
                    "cv_n_folds": int(cv_row.get("n_folds", 0)),
                }
            )

        pd.DataFrame(model_manifest).to_csv(final_models_dir / "model_manifest.csv", index=False)
        SaveJson(final_models_dir / "model_manifest.json", model_manifest)
        SaveJson(
            final_models_dir / "training_config.json",
            {
                "experiment_id": EXPERIMENT_ID,
                "experiment_name": EXPERIMENT_NAME,
                "target_name": TARGET_NAME,
                "target_column": TARGET_COLUMN,
                "group_column": GROUP_COLUMN,
                "save_all_final_models": SAVE_ALL_FINAL_MODELS,
                "save_top_n_final_models": SAVE_TOP_N_FINAL_MODELS,
                "save_dummy_final_model": SAVE_DUMMY_FINAL_MODEL,
                "num_kernels": NUM_KERNELS,
                "pca_explained_variance": PCA_EXPLAINED_VARIANCE,
            },
        )

    print(f"\nSaved {POLARITY_FILTER}-spike DIV regression outputs in:", OUTPUT_DIR, flush=True)
    print(f"Total elapsed time: {time.perf_counter() - analysis_started:.1f} s", flush=True)
    print("\nAlgorithm ranking by mean CV test MAE:")
    print(
        algorithm_comparison[
            [
                "rank",
                "model",
                "mean_test_mae_days",
                "mean_test_rmse_days",
                "mean_test_r2",
                "mean_test_within_10_days",
            ]
        ].to_string(index=False)
    )
