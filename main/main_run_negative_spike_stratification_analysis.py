import os
import re
import sys
import time
from pathlib import Path

plot_cache = Path(os.environ.get("TMPDIR", "/tmp")) / "3dbrain_matplotlib_cache"
plot_cache.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(plot_cache))
os.environ.setdefault("XDG_CACHE_HOME", str(plot_cache))

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

for parent in Path(__file__).resolve().parents:
    src_path = parent / "src"
    if (src_path / "brain_3d").is_dir():
        sys.path.insert(0, str(src_path))
        break

from brain_3d import Stratification


# =========================
# PARAMETERS TO EDIT
# =========================


SPIKE_ROOTS = [
     Path("/data/shared/3D_BRAIN/Mark/Single well MEA/8. Human laminin series 2/spikes/"),
     Path("/data/shared/3D_BRAIN/Mark/Single well MEA/4. Human laminin series 1/spikes"),
]
#SPIKE_ROOTS = [
#     Path("Organoidi/26-06-04-EXP/spikes)
#]

OUTPUT_DIR = Path("outputs/analisi_dati_Mark/spike_stratification_neg")
#OUTPUT_DIR = Path("outputs/analisi_dati_3Brain/spike_stratification_neg")

THRESHOLD_VARIANCE = 0.90
MAX_CLASSES = [2, 3, 4, 5, 6]
MAX_SPIKES_PER_RECORDING = 15000
MAX_SPIKES_GLOBAL = 150000
MIN_SPIKES = 20
SAMPLING_RATE_HZ = 19753
RECORDING_DURATION_SECONDS = 180
RANDOM_STATE = 42
SKIP_ALREADY_PROCESSED = True
RUN_PER_RECORDING_CLUSTERING = True
FORCE_TRANSVERSAL_RECLUSTER = True

USE_ONLY_ACTIVE_CHANNELS = True
MIN_ACTIVE_CHANNEL_FIRING_RATE_HZ = 0.25
ACTIVE_RATE_AFTER_POLARITY_FILTER = True

EXCLUDE_REFERENCE_CHANNEL = True
REFERENCE_CHANNEL = 0
GRID_ROWS = 64
GRID_COLUMNS = 64

# The spike detector writes label 0 for negative spikes and label 1 for positive spikes.
POLARITY_FILTER = "negative"  # choices: "negative", "positive", "none"
USE_DETECTOR_LABEL_FILTER = True
NEGATIVE_SPIKE_LABEL = 0
POSITIVE_SPIKE_LABEL = 1

# Extra waveform-based sanity check. For negative spikes, the negative trough must
# dominate the positive peak. This removes polarity-driven trivial clustering.
USE_WAVEFORM_POLARITY_FILTER = True
POLARITY_DOMINANCE_RATIO = 1.10
REQUIRE_EXTREMUM_NEAR_CENTER = True
CENTER_WINDOW_SAMPLES = 10

PLOT_CHANNEL_MAPS = True
PLOT_CLUSTER_PERCENTAGE_CHANNEL_MAPS = True


# =========================
# SETUP
# =========================

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(RANDOM_STATE)

recording_dirs = []
for spike_root in SPIKE_ROOTS:
    if not spike_root.exists():
        print("Spike folder not found:", spike_root)
        continue

    experiment_match = re.search(r"(\d{2}-\d{2})", str(spike_root))
    experiment_name = experiment_match.group(1) if experiment_match else "unknown_experiment"

    for recording_dir in sorted(spike_root.iterdir()):
        if recording_dir.is_dir() and any(recording_dir.glob("spikes_ch*.npy")):
            recording_dirs.append((experiment_name, recording_dir))

if not recording_dirs:
    raise FileNotFoundError("No folders with spikes_ch*.npy files were found.")


# =========================
# PER-RECORDING CLUSTERING
# =========================

summary_rows = []
experiments_done = {}

for experiment_name, recording_dir in recording_dirs:
    print("\nRecording:", recording_dir.name)

    safe_recording_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", recording_dir.name).strip("_")
    experiment_output_dir = OUTPUT_DIR / experiment_name
    out_dir = experiment_output_dir / "per_recording" / safe_recording_name
    done_files = [
        out_dir / "run_info.csv",
        out_dir / "spike_cluster_assignments.csv",
        out_dir / "PCA_KM_m_label_predette.csv",
    ]
    already_processed = SKIP_ALREADY_PROCESSED and all(path.exists() for path in done_files)

    max_frame = 0
    for frame_path in sorted(recording_dir.glob("frames_ch*.npy")):
        channel_match = re.search(r"frames_ch(\d+)\.npy$", frame_path.name)
        if channel_match is None:
            continue
        channel = int(channel_match.group(1))
        if EXCLUDE_REFERENCE_CHANNEL and channel == REFERENCE_CHANNEL:
            continue
        frames_for_duration = np.load(frame_path).reshape(-1)
        if len(frames_for_duration):
            max_frame = max(max_frame, int(np.nanmax(frames_for_duration)))

    if RECORDING_DURATION_SECONDS is None:
        recording_duration_seconds = max_frame / SAMPLING_RATE_HZ if max_frame > 0 else np.nan
    else:
        recording_duration_seconds = float(RECORDING_DURATION_SECONDS)

    spike_tables = []
    n_input_spikes = 0
    n_after_label_filter = 0
    n_after_waveform_filter = 0
    n_channels_with_spikes = 0
    n_active_channels = 0
    skipped_inactive_channels = 0
    channels_with_spike_file = set()
    expected_channels = set(range(GRID_ROWS * GRID_COLUMNS))
    if EXCLUDE_REFERENCE_CHANNEL:
        expected_channels.discard(REFERENCE_CHANNEL)

    for spike_path in sorted(recording_dir.glob("spikes_ch*.npy")):
        channel_match = re.search(r"spikes_ch(\d+)\.npy$", spike_path.name)
        if channel_match is None:
            continue

        channel = int(channel_match.group(1))
        if EXCLUDE_REFERENCE_CHANNEL and channel == REFERENCE_CHANNEL:
            continue

        channels_with_spike_file.add(channel)

        spikes = np.load(spike_path)
        if spikes.size == 0:
            skipped_inactive_channels += 1
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
        n_channels_with_spikes += 1

        keep = np.ones(n_spikes, dtype=bool)

        if USE_DETECTOR_LABEL_FILTER and POLARITY_FILTER != "none" and np.isfinite(spike_labels).any():
            wanted_label = NEGATIVE_SPIKE_LABEL if POLARITY_FILTER == "negative" else POSITIVE_SPIKE_LABEL
            keep &= spike_labels == wanted_label

        n_after_label_filter += int(np.sum(keep))

        if USE_WAVEFORM_POLARITY_FILTER and POLARITY_FILTER != "none":
            waveform_min = np.min(spikes, axis=1)
            waveform_max = np.max(spikes, axis=1)
            trough_index = np.argmin(spikes, axis=1)
            peak_index = np.argmax(spikes, axis=1)
            center = (spikes.shape[1] - 1) / 2

            if POLARITY_FILTER == "negative":
                waveform_keep = np.abs(waveform_min) >= POLARITY_DOMINANCE_RATIO * np.abs(waveform_max)
                extremum_index = trough_index
            else:
                waveform_keep = np.abs(waveform_max) >= POLARITY_DOMINANCE_RATIO * np.abs(waveform_min)
                extremum_index = peak_index

            if REQUIRE_EXTREMUM_NEAR_CENTER:
                waveform_keep &= np.abs(extremum_index - center) <= CENTER_WINDOW_SAMPLES

            keep &= waveform_keep

        n_after_waveform_filter += int(np.sum(keep))

        spikes = spikes[keep]
        frames = frames[keep]
        spike_labels = spike_labels[keep]
        if len(spikes) == 0:
            skipped_inactive_channels += 1
            continue

        if ACTIVE_RATE_AFTER_POLARITY_FILTER:
            spikes_for_rate = len(spikes)
        else:
            spikes_for_rate = n_spikes

        if np.isfinite(recording_duration_seconds) and recording_duration_seconds > 0:
            channel_firing_rate_hz = spikes_for_rate / recording_duration_seconds
        else:
            channel_firing_rate_hz = np.nan

        if USE_ONLY_ACTIVE_CHANNELS and not (
            np.isfinite(channel_firing_rate_hz) and channel_firing_rate_hz > MIN_ACTIVE_CHANNEL_FIRING_RATE_HZ
        ):
            skipped_inactive_channels += 1
            continue

        n_active_channels += 1
        metadata = pd.DataFrame(
            {
                "recording_name": recording_dir.name,
                "channel": channel,
                "spike_index": np.arange(len(spikes)),
                "frame": frames,
                "spike_label": spike_labels,
                "channel_firing_rate_hz": channel_firing_rate_hz,
                "recording_duration_seconds": recording_duration_seconds,
                "polarity_filter": POLARITY_FILTER,
            }
        )
        waveform = pd.DataFrame(spikes, columns=[f"w{i}" for i in range(spikes.shape[1])])
        spike_tables.append(pd.concat([metadata, waveform], axis=1))

    missing_spike_file_channels = expected_channels - channels_with_spike_file
    print(
        f"  polarity filter: input {n_input_spikes}, "
        f"after label {n_after_label_filter}, after waveform {n_after_waveform_filter}"
    )
    if USE_ONLY_ACTIVE_CHANNELS:
        print(
            f"  active-channel filter: expected {len(expected_channels)} channels, "
            f"found {len(channels_with_spike_file)} spike files, "
            f"kept {n_active_channels} active channels, "
            f"skipped {skipped_inactive_channels + len(missing_spike_file_channels)} channels "
            f"({skipped_inactive_channels} inactive/empty after polarity filter, "
            f"{len(missing_spike_file_channels)} without spike file)"
        )

    if not spike_tables:
        print("  skipped: no biologically plausible active spikes loaded")
        summary_rows.append(
            {
                "experiment": experiment_name,
                "recording_name": recording_dir.name,
                "n_input_spikes": n_input_spikes,
                "n_after_label_filter": n_after_label_filter,
                "n_after_waveform_filter": n_after_waveform_filter,
                "n_spikes": 0,
                "n_active_channels": 0,
                "n_clusters": None,
                "output_dir": str(out_dir),
                "status": "skipped_no_spikes_after_filters",
            }
        )
        continue

    spike_df = pd.concat(spike_tables, ignore_index=True)
    n_after_active_filter = len(spike_df)
    if len(spike_df) > MAX_SPIKES_PER_RECORDING:
        spike_df = spike_df.sample(n=MAX_SPIKES_PER_RECORDING, random_state=RANDOM_STATE).reset_index(drop=True)

    if len(spike_df) < MIN_SPIKES:
        print(f"  skipped: only {len(spike_df)} spikes after filters")
        summary_rows.append(
            {
                "experiment": experiment_name,
                "recording_name": recording_dir.name,
                "n_input_spikes": n_input_spikes,
                "n_after_label_filter": n_after_label_filter,
                "n_after_waveform_filter": n_after_waveform_filter,
                "n_after_active_filter": n_after_active_filter,
                "n_spikes": len(spike_df),
                "n_active_channels": n_active_channels,
                "n_clusters": None,
                "output_dir": str(out_dir),
                "status": "skipped_too_few_spikes_after_filters",
            }
        )
        continue

    waveform_cols = [col for col in spike_df.columns if col.startswith("w")]
    if experiment_name not in experiments_done:
        experiments_done[experiment_name] = []
    experiments_done[experiment_name].append(spike_df)

    if already_processed:
        run_info = pd.read_csv(out_dir / "run_info.csv")
        n_clusters = int(run_info["n_clusters"].iloc[0]) if "n_clusters" in run_info.columns else None
        if PLOT_CHANNEL_MAPS and PLOT_CLUSTER_PERCENTAGE_CHANNEL_MAPS:
            assignments = pd.read_csv(out_dir / "spike_cluster_assignments.csv")
            channel_map_dir = out_dir / "channel_maps"
            channel_map_dir.mkdir(exist_ok=True)
            waveform_counts = np.zeros((GRID_ROWS, GRID_COLUMNS))
            for channel in assignments["channel"].astype(int):
                row = channel // GRID_COLUMNS
                col = channel % GRID_COLUMNS
                waveform_counts[row, col] += 1
            np.savetxt(channel_map_dir / "waveforms_number.csv", waveform_counts, delimiter=",", fmt="%.6g")
            plt.figure(figsize=(7, 7))
            image = plt.imshow(np.ma.masked_where(waveform_counts == 0, waveform_counts), cmap="viridis")
            plt.colorbar(image)
            plt.xticks([])
            plt.yticks([])
            plt.title("Number of negative waveforms")
            plt.tight_layout()
            plt.savefig(channel_map_dir / "waveforms_number.png", dpi=300, bbox_inches="tight")
            plt.close()
            cluster_ids = sorted(int(cluster) for cluster in assignments["cluster"].dropna().unique() if int(cluster) >= 0)
            for cluster_id in cluster_ids:
                cluster_counts = np.zeros((GRID_ROWS, GRID_COLUMNS))
                cluster_assignments = assignments[assignments["cluster"] == cluster_id]
                for channel in cluster_assignments["channel"].astype(int):
                    row = channel // GRID_COLUMNS
                    col = channel % GRID_COLUMNS
                    cluster_counts[row, col] += 1
                cluster_percentage = np.divide(
                    100 * cluster_counts,
                    waveform_counts,
                    out=np.zeros_like(cluster_counts),
                    where=waveform_counts > 0,
                )
                np.savetxt(
                    channel_map_dir / f"cluster_{cluster_id}_waveform_percentage_by_channel.csv",
                    cluster_percentage,
                    delimiter=",",
                    fmt="%.6g",
                )
                plt.figure(figsize=(7, 7))
                image = plt.imshow(np.ma.masked_where(waveform_counts == 0, cluster_percentage), cmap="viridis", vmin=0, vmax=100)
                plt.colorbar(image, label="% of waveforms in this channel")
                plt.xticks([])
                plt.yticks([])
                plt.title(f"Cluster {cluster_id}: local waveform percentage")
                plt.tight_layout()
                plt.savefig(
                    channel_map_dir / f"cluster_{cluster_id}_waveform_percentage_by_channel.png",
                    dpi=300,
                    bbox_inches="tight",
                )
                plt.close()
        print("  already processed, keeping existing clustering:", out_dir)
        summary_rows.append(
            {
                "experiment": experiment_name,
                "recording_name": recording_dir.name,
                "n_input_spikes": n_input_spikes,
                "n_after_label_filter": n_after_label_filter,
                "n_after_waveform_filter": n_after_waveform_filter,
                "n_after_active_filter": n_after_active_filter,
                "n_spikes": len(spike_df),
                "n_active_channels": n_active_channels,
                "n_clusters": n_clusters,
                "output_dir": str(out_dir),
                "status": "already_processed",
            }
        )
        continue

    if not RUN_PER_RECORDING_CLUSTERING:
        print("  per-recording clustering disabled; using filtered spikes only for transversal analysis")
        summary_rows.append(
            {
                "experiment": experiment_name,
                "recording_name": recording_dir.name,
                "n_input_spikes": n_input_spikes,
                "n_after_label_filter": n_after_label_filter,
                "n_after_waveform_filter": n_after_waveform_filter,
                "n_after_active_filter": n_after_active_filter,
                "n_spikes": len(spike_df),
                "n_active_channels": n_active_channels,
                "n_clusters": None,
                "output_dir": str(out_dir),
                "status": "per_recording_not_run",
            }
        )
        continue

    data = spike_df[waveform_cols].to_numpy(dtype=float)
    data = Stratification.Whitening(data.copy())

    t0 = time.time()
    n_clusters, clusters = Stratification.Clustering(
        Data=data,
        Algo="PCA&KM",
        DistanceStr="m",
        pMinkowski=2,
        ThresholdVariance=THRESHOLD_VARIANCE,
        MaxClasses=MAX_CLASSES,
        SamplingRate=1,
        Normalization="OFF",
    )
    elapsed = time.time() - t0

    cluster_label = np.full(len(spike_df), -1, dtype=int)
    for cluster_id, row_indexes in enumerate(clusters):
        cluster_label[np.asarray(row_indexes, dtype=int).ravel()] = cluster_id

    pca_plot = PCA(n_components=min(2, data.shape[0], data.shape[1]), random_state=RANDOM_STATE)
    pc = pca_plot.fit_transform(StandardScaler().fit_transform(data))
    if pc.shape[1] == 1:
        pc = np.column_stack([pc[:, 0], np.zeros(len(pc))])

    assignments = spike_df[
        [
            "recording_name",
            "channel",
            "spike_index",
            "frame",
            "spike_label",
            "channel_firing_rate_hz",
            "recording_duration_seconds",
            "polarity_filter",
        ]
    ].copy()
    assignments["PC1"] = pc[:, 0]
    assignments["PC2"] = pc[:, 1]
    assignments["cluster"] = cluster_label

    centroids = (
        pd.concat([assignments[["cluster"]], spike_df[waveform_cols]], axis=1)
        .groupby("cluster", sort=True)[waveform_cols]
        .mean()
        .reset_index()
    )
    centroid_std = (
        pd.concat([assignments[["cluster"]], spike_df[waveform_cols]], axis=1)
        .groupby("cluster", sort=True)[waveform_cols]
        .std()
        .fillna(0)
        .reset_index()
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    assignments.to_csv(out_dir / "spike_cluster_assignments.csv", index=False)
    assignments[["cluster"]].to_csv(out_dir / "PCA_KM_m_label_predette.csv", index=False, header=False)
    centroids.to_csv(out_dir / "cluster_centroid_waveforms.csv", index=False)
    centroid_std.to_csv(out_dir / "cluster_centroid_waveform_std.csv", index=False)
    pd.crosstab(assignments["cluster"], assignments["channel"]).to_csv(out_dir / "cluster_by_channel.csv")
    pd.crosstab(assignments["cluster"], assignments["spike_label"]).to_csv(out_dir / "cluster_by_spike_label.csv")

    pd.DataFrame(
        [
            {
                "recording_name": recording_dir.name,
                "data_used": "negative biologically plausible spikes from spikes_ch*.npy",
                "algo": "PCA&KM",
                "distance": "m",
                "distance_meaning": "euclidean_minkowski_p2",
                "threshold_variance": THRESHOLD_VARIANCE,
                "max_classes": " ".join(map(str, MAX_CLASSES)),
                "polarity_filter": POLARITY_FILTER,
                "use_detector_label_filter": USE_DETECTOR_LABEL_FILTER,
                "use_waveform_polarity_filter": USE_WAVEFORM_POLARITY_FILTER,
                "polarity_dominance_ratio": POLARITY_DOMINANCE_RATIO,
                "require_extremum_near_center": REQUIRE_EXTREMUM_NEAR_CENTER,
                "center_window_samples": CENTER_WINDOW_SAMPLES,
                "n_input_spikes": n_input_spikes,
                "n_after_label_filter": n_after_label_filter,
                "n_after_waveform_filter": n_after_waveform_filter,
                "n_after_active_filter": n_after_active_filter,
                "n_spikes": len(spike_df),
                "n_active_channels": n_active_channels,
                "n_clusters": n_clusters,
                "use_only_active_channels": USE_ONLY_ACTIVE_CHANNELS,
                "active_rate_after_polarity_filter": ACTIVE_RATE_AFTER_POLARITY_FILTER,
                "min_active_channel_firing_rate_hz": MIN_ACTIVE_CHANNEL_FIRING_RATE_HZ,
                "recording_duration_seconds": recording_duration_seconds,
                "sampling_rate_hz": SAMPLING_RATE_HZ,
                "elapsed_seconds": elapsed,
            }
        ]
    ).to_csv(out_dir / "run_info.csv", index=False)

    plt.figure(figsize=(8, 6))
    plt.scatter(assignments["PC1"], assignments["PC2"], c=assignments["cluster"], s=8, cmap="tab10")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title(recording_dir.name)
    plt.tight_layout()
    plt.savefig(out_dir / "pca_by_cluster.png", dpi=180)
    plt.close()

    plt.figure(figsize=(10, 5))
    x = np.arange(len(waveform_cols))
    for _, row in centroids.iterrows():
        cluster_id = int(row["cluster"])
        mean_waveform = row[waveform_cols].to_numpy(dtype=float)
        std_waveform = (
            centroid_std.loc[centroid_std["cluster"] == cluster_id, waveform_cols]
            .iloc[0]
            .to_numpy(dtype=float)
        )
        line = plt.plot(x, mean_waveform, label=f"cluster {cluster_id}")[0]
        plt.fill_between(
            x,
            mean_waveform - std_waveform,
            mean_waveform + std_waveform,
            color=line.get_color(),
            alpha=0.2,
            linewidth=0,
        )
    plt.xlabel("waveform sample")
    plt.ylabel("amplitude")
    plt.title("Cluster centroid waveforms")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "cluster_centroid_waveforms.png", dpi=180)
    plt.close()

    if PLOT_CHANNEL_MAPS:
        channel_map_dir = out_dir / "channel_maps"
        channel_map_dir.mkdir(exist_ok=True)
        waveform_counts = np.zeros((GRID_ROWS, GRID_COLUMNS))
        for channel in assignments["channel"].astype(int):
            row = channel // GRID_COLUMNS
            col = channel % GRID_COLUMNS
            waveform_counts[row, col] += 1
        np.savetxt(channel_map_dir / "waveforms_number.csv", waveform_counts, delimiter=",", fmt="%.6g")
        plt.figure(figsize=(7, 7))
        image = plt.imshow(np.ma.masked_where(waveform_counts == 0, waveform_counts), cmap="viridis")
        plt.colorbar(image)
        plt.xticks([])
        plt.yticks([])
        plt.title("Number of negative waveforms")
        plt.tight_layout()
        plt.savefig(channel_map_dir / "waveforms_number.png", dpi=300, bbox_inches="tight")
        plt.close()
        if PLOT_CLUSTER_PERCENTAGE_CHANNEL_MAPS:
            cluster_ids = sorted(int(cluster) for cluster in assignments["cluster"].dropna().unique() if int(cluster) >= 0)
            for cluster_id in cluster_ids:
                cluster_counts = np.zeros((GRID_ROWS, GRID_COLUMNS))
                cluster_assignments = assignments[assignments["cluster"] == cluster_id]
                for channel in cluster_assignments["channel"].astype(int):
                    row = channel // GRID_COLUMNS
                    col = channel % GRID_COLUMNS
                    cluster_counts[row, col] += 1
                cluster_percentage = np.divide(
                    100 * cluster_counts,
                    waveform_counts,
                    out=np.zeros_like(cluster_counts),
                    where=waveform_counts > 0,
                )
                np.savetxt(
                    channel_map_dir / f"cluster_{cluster_id}_waveform_percentage_by_channel.csv",
                    cluster_percentage,
                    delimiter=",",
                    fmt="%.6g",
                )
                plt.figure(figsize=(7, 7))
                image = plt.imshow(np.ma.masked_where(waveform_counts == 0, cluster_percentage), cmap="viridis", vmin=0, vmax=100)
                plt.colorbar(image, label="% of waveforms in this channel")
                plt.xticks([])
                plt.yticks([])
                plt.title(f"Cluster {cluster_id}: local waveform percentage")
                plt.tight_layout()
                plt.savefig(
                    channel_map_dir / f"cluster_{cluster_id}_waveform_percentage_by_channel.png",
                    dpi=300,
                    bbox_inches="tight",
                )
                plt.close()

    summary_rows.append(
        {
            "experiment": experiment_name,
            "recording_name": recording_dir.name,
            "n_input_spikes": n_input_spikes,
            "n_after_label_filter": n_after_label_filter,
            "n_after_waveform_filter": n_after_waveform_filter,
            "n_after_active_filter": n_after_active_filter,
            "n_spikes": len(spike_df),
            "n_active_channels": n_active_channels,
            "n_clusters": n_clusters,
            "output_dir": str(out_dir),
            "status": "processed",
        }
    )
    print("  saved in:", out_dir)


if not summary_rows:
    raise ValueError("No recording had enough spikes.")

summary = pd.DataFrame(summary_rows)
summary.to_csv(OUTPUT_DIR / "per_recording_summary.csv", index=False)
for experiment_name, group in summary.groupby("experiment", dropna=False):
    experiment_output_dir = OUTPUT_DIR / str(experiment_name)
    experiment_output_dir.mkdir(parents=True, exist_ok=True)
    group.to_csv(experiment_output_dir / "per_recording_summary.csv", index=False)


# =========================
# TRANSVERSAL ANALYSIS PER EXPERIMENT
# =========================

for experiment_name, spike_dfs in experiments_done.items():
    out_dir = OUTPUT_DIR / experiment_name / "transversal_global_spike_clusters"
    done_files = [
        out_dir / "run_info.csv",
        out_dir / "spike_cluster_assignments.csv",
        out_dir / "PCA_KM_m_label_predette.csv",
    ]
    if not FORCE_TRANSVERSAL_RECLUSTER and SKIP_ALREADY_PROCESSED and all(path.exists() for path in done_files):
        print("\nTransversal analysis already processed, recomputing counts/profile only:", out_dir)
        assignments = pd.read_csv(out_dir / "spike_cluster_assignments.csv")
        pd.crosstab(assignments["recording_name"], assignments["cluster"], normalize="index").to_csv(
            OUTPUT_DIR / experiment_name / "recording_global_cluster_profile.csv"
        )
        pd.crosstab(assignments["recording_name"], assignments["cluster"]).to_csv(
            OUTPUT_DIR / experiment_name / "recording_global_cluster_counts.csv"
        )
        if PLOT_CHANNEL_MAPS:
            channel_map_dir = out_dir / "channel_maps"
            channel_map_dir.mkdir(exist_ok=True)
            waveform_counts = np.zeros((GRID_ROWS, GRID_COLUMNS))
            for channel in assignments["channel"].astype(int):
                row = channel // GRID_COLUMNS
                col = channel % GRID_COLUMNS
                waveform_counts[row, col] += 1
            np.savetxt(channel_map_dir / "waveforms_number.csv", waveform_counts, delimiter=",", fmt="%.6g")
            plt.figure(figsize=(7, 7))
            image = plt.imshow(np.ma.masked_where(waveform_counts == 0, waveform_counts), cmap="viridis")
            plt.colorbar(image)
            plt.xticks([])
            plt.yticks([])
            plt.title("Number of negative waveforms")
            plt.tight_layout()
            plt.savefig(channel_map_dir / "waveforms_number.png", dpi=300, bbox_inches="tight")
            plt.close()
            if PLOT_CLUSTER_PERCENTAGE_CHANNEL_MAPS:
                cluster_ids = sorted(int(cluster) for cluster in assignments["cluster"].dropna().unique() if int(cluster) >= 0)
                for cluster_id in cluster_ids:
                    cluster_counts = np.zeros((GRID_ROWS, GRID_COLUMNS))
                    cluster_assignments = assignments[assignments["cluster"] == cluster_id]
                    for channel in cluster_assignments["channel"].astype(int):
                        row = channel // GRID_COLUMNS
                        col = channel % GRID_COLUMNS
                        cluster_counts[row, col] += 1
                    cluster_percentage = np.divide(
                        100 * cluster_counts,
                        waveform_counts,
                        out=np.zeros_like(cluster_counts),
                        where=waveform_counts > 0,
                    )
                    np.savetxt(
                        channel_map_dir / f"cluster_{cluster_id}_waveform_percentage_by_channel.csv",
                        cluster_percentage,
                        delimiter=",",
                        fmt="%.6g",
                    )
                    plt.figure(figsize=(7, 7))
                    image = plt.imshow(np.ma.masked_where(waveform_counts == 0, cluster_percentage), cmap="viridis", vmin=0, vmax=100)
                    plt.colorbar(image, label="% of waveforms in this channel")
                    plt.xticks([])
                    plt.yticks([])
                    plt.title(f"Cluster {cluster_id}: local waveform percentage")
                    plt.tight_layout()
                    plt.savefig(
                        channel_map_dir / f"cluster_{cluster_id}_waveform_percentage_by_channel.png",
                        dpi=300,
                        bbox_inches="tight",
                    )
                    plt.close()
        continue

    pooled_df = pd.concat(spike_dfs, ignore_index=True)
    if len(pooled_df) > MAX_SPIKES_GLOBAL:
        pooled_df = pooled_df.sample(n=MAX_SPIKES_GLOBAL, random_state=RANDOM_STATE).reset_index(drop=True)

    waveform_cols = [col for col in pooled_df.columns if col.startswith("w")]
    data = pooled_df[waveform_cols].to_numpy(dtype=float)
    data = Stratification.Whitening(data.copy())

    t0 = time.time()
    n_clusters, clusters = Stratification.Clustering(
        Data=data,
        Algo="PCA&KM",
        DistanceStr="m",
        pMinkowski=2,
        ThresholdVariance=THRESHOLD_VARIANCE,
        MaxClasses=MAX_CLASSES,
        SamplingRate=1,
        Normalization="OFF",
    )
    elapsed = time.time() - t0

    cluster_label = np.full(len(pooled_df), -1, dtype=int)
    for cluster_id, row_indexes in enumerate(clusters):
        cluster_label[np.asarray(row_indexes, dtype=int).ravel()] = cluster_id

    pca_plot = PCA(n_components=min(2, data.shape[0], data.shape[1]), random_state=RANDOM_STATE)
    pc = pca_plot.fit_transform(StandardScaler().fit_transform(data))
    if pc.shape[1] == 1:
        pc = np.column_stack([pc[:, 0], np.zeros(len(pc))])

    assignments = pooled_df[
        [
            "recording_name",
            "channel",
            "spike_index",
            "frame",
            "spike_label",
            "channel_firing_rate_hz",
            "recording_duration_seconds",
            "polarity_filter",
        ]
    ].copy()
    assignments["PC1"] = pc[:, 0]
    assignments["PC2"] = pc[:, 1]
    assignments["cluster"] = cluster_label

    out_dir.mkdir(parents=True, exist_ok=True)
    assignments.to_csv(out_dir / "spike_cluster_assignments.csv", index=False)
    assignments[["cluster"]].to_csv(out_dir / "PCA_KM_m_label_predette.csv", index=False, header=False)
    pd.crosstab(assignments["recording_name"], assignments["cluster"], normalize="index").to_csv(
        OUTPUT_DIR / experiment_name / "recording_global_cluster_profile.csv"
    )
    pd.crosstab(assignments["recording_name"], assignments["cluster"]).to_csv(
        OUTPUT_DIR / experiment_name / "recording_global_cluster_counts.csv"
    )
    pd.DataFrame(
        [
            {
                "experiment": experiment_name,
                "data_used": "sampled negative biologically plausible spikes from one experiment",
                "algo": "PCA&KM",
                "distance": "m",
                "distance_meaning": "euclidean_minkowski_p2",
                "threshold_variance": THRESHOLD_VARIANCE,
                "max_classes": " ".join(map(str, MAX_CLASSES)),
                "polarity_filter": POLARITY_FILTER,
                "use_detector_label_filter": USE_DETECTOR_LABEL_FILTER,
                "use_waveform_polarity_filter": USE_WAVEFORM_POLARITY_FILTER,
                "polarity_dominance_ratio": POLARITY_DOMINANCE_RATIO,
                "require_extremum_near_center": REQUIRE_EXTREMUM_NEAR_CENTER,
                "center_window_samples": CENTER_WINDOW_SAMPLES,
                "max_spikes_global": MAX_SPIKES_GLOBAL,
                "n_spikes": len(pooled_df),
                "n_clusters": n_clusters,
                "elapsed_seconds": elapsed,
            }
        ]
    ).to_csv(out_dir / "run_info.csv", index=False)

    plt.figure(figsize=(8, 6))
    plt.scatter(assignments["PC1"], assignments["PC2"], c=assignments["cluster"], s=8, cmap="tab10")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title(f"Transversal negative spike clustering {experiment_name}")
    plt.tight_layout()
    plt.savefig(out_dir / "pca_by_cluster.png", dpi=180)
    plt.close()

    if PLOT_CHANNEL_MAPS:
        channel_map_dir = out_dir / "channel_maps"
        channel_map_dir.mkdir(exist_ok=True)
        waveform_counts = np.zeros((GRID_ROWS, GRID_COLUMNS))
        for channel in assignments["channel"].astype(int):
            row = channel // GRID_COLUMNS
            col = channel % GRID_COLUMNS
            waveform_counts[row, col] += 1
        np.savetxt(channel_map_dir / "waveforms_number.csv", waveform_counts, delimiter=",", fmt="%.6g")
        plt.figure(figsize=(7, 7))
        image = plt.imshow(np.ma.masked_where(waveform_counts == 0, waveform_counts), cmap="viridis")
        plt.colorbar(image)
        plt.xticks([])
        plt.yticks([])
        plt.title("Number of negative waveforms")
        plt.tight_layout()
        plt.savefig(channel_map_dir / "waveforms_number.png", dpi=300, bbox_inches="tight")
        plt.close()
        if PLOT_CLUSTER_PERCENTAGE_CHANNEL_MAPS:
            cluster_ids = sorted(int(cluster) for cluster in assignments["cluster"].dropna().unique() if int(cluster) >= 0)
            for cluster_id in cluster_ids:
                cluster_counts = np.zeros((GRID_ROWS, GRID_COLUMNS))
                cluster_assignments = assignments[assignments["cluster"] == cluster_id]
                for channel in cluster_assignments["channel"].astype(int):
                    row = channel // GRID_COLUMNS
                    col = channel % GRID_COLUMNS
                    cluster_counts[row, col] += 1
                cluster_percentage = np.divide(
                    100 * cluster_counts,
                    waveform_counts,
                    out=np.zeros_like(cluster_counts),
                    where=waveform_counts > 0,
                )
                np.savetxt(
                    channel_map_dir / f"cluster_{cluster_id}_waveform_percentage_by_channel.csv",
                    cluster_percentage,
                    delimiter=",",
                    fmt="%.6g",
                )
                plt.figure(figsize=(7, 7))
                image = plt.imshow(np.ma.masked_where(waveform_counts == 0, cluster_percentage), cmap="viridis", vmin=0, vmax=100)
                plt.colorbar(image, label="% of waveforms in this channel")
                plt.xticks([])
                plt.yticks([])
                plt.title(f"Cluster {cluster_id}: local waveform percentage")
                plt.tight_layout()
                plt.savefig(
                    channel_map_dir / f"cluster_{cluster_id}_waveform_percentage_by_channel.png",
                    dpi=300,
                    bbox_inches="tight",
                )
                plt.close()

    print("\nTransversal analysis saved in:", out_dir)

print("\nDone. Summary saved in:", OUTPUT_DIR / "per_recording_summary.csv")
