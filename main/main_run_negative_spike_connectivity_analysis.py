"""Connectivity analysis for MEA recordings using plausible negative spikes.

This main mirrors ``main_run_connectivity_analysis.py`` but applies the same
event filter used in negative-polarity stratification before computing
connectivity:

- detector label 0;
- dominant negative trough;
- extremum near the waveform center;
- reference channel excluded;
- active channels with firing rate > 0.25 Hz.
"""

import argparse
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")

for parent in Path(__file__).resolve().parents:
    src_path = parent / "src"
    if (src_path / "brain_3d").is_dir():
        sys.path.insert(0, str(src_path))
        break

from brain_3d import Connectivity


# =========================
# PARAMETERS TO EDIT
# =========================

#SPIKE_ROOT = Path("/data/privato/Organoidi/26-06-04-EXP/spikes")
#OUTPUT_ROOT = Path("outputs/analisi_dati_3Brain/connectivity_negative_spikes/26-06")

SPIKE_ROOT = Path("/data/shared/3D_BRAIN/Mark/Single well MEA/4. Human laminin series 1/spikes")
OUTPUT_ROOT = Path("outputs/analisi_dati_Mark//4_series_1/connectvity_neg")

SAMPLING_RATE_HZ = 19753
RECORDING_DURATION_SECONDS = 180
BIN_SIZE_SECONDS = 0.05
STTC_DT_SECONDS = 0.02
CORRELATION_THRESHOLD = 0.2
STTC_THRESHOLD = 0.2
MIN_ACTIVE_CHANNEL_FIRING_RATE_HZ = 0.25
EXCLUDED_CHANNELS = [0]

NEGATIVE_SPIKE_LABEL = 0
USE_DETECTOR_LABEL_FILTER = True
USE_WAVEFORM_POLARITY_FILTER = True
POLARITY_DOMINANCE_RATIO = 1.10
REQUIRE_EXTREMUM_NEAR_CENTER = True
CENTER_WINDOW_SAMPLES = 10

VERBOSE = True
SKIP_EXISTING = True
GENERATE_NETWORK_MAPS = True
NETWORK_MAP_TOP_PERCENTILE = 95
NETWORK_MAP_TOP_N = 120


# =========================
# HELPERS
# =========================

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


def load_negative_spike_frames(spike_dir):
    spike_frames = {}
    filter_rows = []

    for spike_path in sorted(spike_dir.glob("spikes_ch*.npy")):
        channel_match = re.search(r"spikes_ch(\d+)\.npy$", spike_path.name)
        if channel_match is None:
            continue

        channel = int(channel_match.group(1))
        if channel in set(EXCLUDED_CHANNELS):
            continue

        spikes = np.load(spike_path)
        if spikes.size == 0:
            continue
        spikes = np.asarray(spikes).reshape(spikes.shape[0], -1)

        frame_path = spike_path.with_name(f"frames_ch{channel}.npy")
        label_path = spike_path.with_name(f"label_ch{channel}.npy")
        frames = np.load(frame_path).reshape(-1) if frame_path.exists() else np.full(spikes.shape[0], np.nan)
        labels = np.load(label_path).reshape(-1) if label_path.exists() else np.full(spikes.shape[0], np.nan)

        n_spikes = min(spikes.shape[0], len(frames), len(labels))
        spikes = spikes[:n_spikes]
        frames = frames[:n_spikes]
        labels = labels[:n_spikes]

        label_keep = labels == NEGATIVE_SPIKE_LABEL if np.isfinite(labels).any() else np.ones(n_spikes, bool)
        keep = negative_waveform_mask(spikes, labels)
        kept_frames = np.sort(frames[keep].astype(int))
        firing_rate_hz = len(kept_frames) / float(RECORDING_DURATION_SECONDS)
        is_active = firing_rate_hz > MIN_ACTIVE_CHANNEL_FIRING_RATE_HZ

        filter_rows.append(
            {
                "channel": channel,
                "input_spikes": n_spikes,
                "detector_negative_spikes": int(label_keep.sum()),
                "plausible_negative_spikes": int(keep.sum()),
                "firing_rate_hz": firing_rate_hz,
                "is_active": is_active,
            }
        )

        if is_active and len(kept_frames):
            spike_frames[channel] = kept_frames

    if not spike_frames:
        raise ValueError("No active channels remain after negative-spike filtering.")

    return spike_frames, pd.DataFrame(filter_rows)


def save_results(result, output_dir, filter_summary):
    output_dir.mkdir(parents=True, exist_ok=True)
    channels = result["channels"]

    pd.DataFrame(result["correlation_matrix"], index=channels, columns=channels).to_csv(
        output_dir / "spike_count_correlation_matrix.csv"
    )
    pd.DataFrame(result["sttc_matrix"], index=channels, columns=channels).to_csv(output_dir / "sttc_matrix.csv")
    result["correlation_strength"].to_csv(output_dir / "correlation_node_strength.csv", index=False)
    result["sttc_strength"].to_csv(output_dir / "sttc_node_strength.csv", index=False)
    result["channel_firing_rates"].to_csv(output_dir / "channel_firing_rates.csv", index=False)
    filter_summary.to_csv(output_dir / "negative_spike_filter_summary_by_channel.csv", index=False)

    if "correlation_edges" in result:
        result["correlation_edges"].to_csv(output_dir / "correlation_edges.csv", index=False)
    if "sttc_edges" in result:
        result["sttc_edges"].to_csv(output_dir / "sttc_edges.csv", index=False)

    np.save(output_dir / "binned_spike_counts.npy", result["counts"])
    np.save(output_dir / "bin_edges_seconds.npy", result["bin_edges"])


def save_network_maps(result, output_dir, map_dir=None):
    if not GENERATE_NETWORK_MAPS:
        return {"shown_top_sttc_edges": 0, "shown_top_correlation_edges": 0}

    map_dir = output_dir / "network_maps" if map_dir is None else Path(map_dir)
    map_dir.mkdir(parents=True, exist_ok=True)

    Connectivity.PlotConnectivityMatrix(
        result["sttc_matrix"],
        SavePath=map_dir / "sttc_connectivity_matrix.png",
        Title="STTC connectivity matrix",
    )
    Connectivity.PlotConnectivityMatrix(
        result["correlation_matrix"],
        SavePath=map_dir / "correlation_connectivity_matrix.png",
        Title="Spike-count correlation matrix",
    )

    Connectivity.PlotNodeStrengthGrid(
        result["sttc_strength"],
        SavePath=map_dir / "sttc_node_strength_map.png",
        Title="STTC node strength",
        Cmap="magma",
    )
    Connectivity.PlotNodeStrengthGrid(
        result["correlation_strength"],
        SavePath=map_dir / "correlation_node_strength_map.png",
        Title="Spike-count correlation node strength",
        Cmap="viridis",
    )

    _, _, selected_sttc_edges = Connectivity.PlotNetworkEdgeGrid(
        result["sttc_edges"],
        SavePath=map_dir / "top_sttc_network_edges_map.png",
        ChannelValues=result["sttc_strength"],
        TopPercentile=NETWORK_MAP_TOP_PERCENTILE,
        TopN=NETWORK_MAP_TOP_N,
        Title="Top STTC network edges",
        Cmap="magma",
    )
    selected_sttc_edges.to_csv(map_dir / "top_sttc_network_edges.csv", index=False)

    _, _, selected_correlation_edges = Connectivity.PlotNetworkEdgeGrid(
        result["correlation_edges"],
        SavePath=map_dir / "top_correlation_network_edges_map.png",
        ChannelValues=result["correlation_strength"],
        TopPercentile=NETWORK_MAP_TOP_PERCENTILE,
        TopN=NETWORK_MAP_TOP_N,
        Title="Top spike-count correlation network edges",
        Cmap="viridis",
    )
    selected_correlation_edges.to_csv(map_dir / "top_correlation_network_edges.csv", index=False)
    return {
        "shown_top_sttc_edges": len(selected_sttc_edges),
        "shown_top_correlation_edges": len(selected_correlation_edges),
    }


def save_network_maps_from_existing_outputs(output_dir, map_dir=None):
    if not GENERATE_NETWORK_MAPS:
        return

    required = [
        output_dir / "sttc_matrix.csv",
        output_dir / "spike_count_correlation_matrix.csv",
        output_dir / "sttc_node_strength.csv",
        output_dir / "correlation_node_strength.csv",
        output_dir / "sttc_edges.csv",
        output_dir / "correlation_edges.csv",
    ]
    if not all(path.exists() for path in required):
        print("Network maps not generated: missing saved connectivity tables.", flush=True)
        return

    result = {
        "sttc_matrix": pd.read_csv(output_dir / "sttc_matrix.csv", index_col=0).to_numpy(),
        "correlation_matrix": pd.read_csv(output_dir / "spike_count_correlation_matrix.csv", index_col=0).to_numpy(),
        "sttc_strength": pd.read_csv(output_dir / "sttc_node_strength.csv"),
        "correlation_strength": pd.read_csv(output_dir / "correlation_node_strength.csv"),
        "sttc_edges": pd.read_csv(output_dir / "sttc_edges.csv"),
        "correlation_edges": pd.read_csv(output_dir / "correlation_edges.csv"),
    }
    return save_network_maps(result, output_dir, map_dir=map_dir)


def network_maps_exist(output_dir):
    map_dir = output_dir / "network_maps"
    return all(
        (map_dir / name).exists()
        for name in [
            "sttc_node_strength_map.png",
            "correlation_node_strength_map.png",
            "top_sttc_network_edges_map.png",
            "top_correlation_network_edges_map.png",
            "sttc_connectivity_matrix.png",
            "correlation_connectivity_matrix.png",
        ]
    )


def generate_maps_for_existing_output_root(output_root, maps_output_root=None):
    recording_dirs = [
        path for path in sorted(output_root.iterdir())
        if path.is_dir() and (path / "sttc_matrix.csv").exists()
    ]
    if not recording_dirs:
        raise FileNotFoundError(f"No saved connectivity recordings found in {output_root}")
    for index, output_dir in enumerate(recording_dirs, start=1):
        print(f"Generating maps {index}/{len(recording_dirs)}: {output_dir.name}", flush=True)
        map_dir = None if maps_output_root is None else maps_output_root / output_dir.name
        save_network_maps_from_existing_outputs(output_dir, map_dir=map_dir)
    print(f"Generated connectivity maps for {len(recording_dirs)} recordings.", flush=True)


# =========================
# ANALYSIS
# =========================

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spike-root", type=Path, default=SPIKE_ROOT)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--maps-only", action="store_true", help="Generate maps from existing connectivity CSV files.")
    parser.add_argument("--maps-output-root", type=Path, help="Optional separate destination used with --maps-only.")
    args = parser.parse_args()
    spike_root = args.spike_root
    output_root = args.output_root

    if args.maps_only:
        generate_maps_for_existing_output_root(output_root, maps_output_root=args.maps_output_root)
        return

    if not spike_root.exists():
        raise FileNotFoundError(f"Spike root not found: {spike_root}")

    recording_dirs = [
        recording_dir
        for recording_dir in sorted(spike_root.iterdir())
        if recording_dir.is_dir() and any(recording_dir.glob("frames_ch*.npy"))
    ]
    if not recording_dirs:
        raise FileNotFoundError(f"No recordings with frames_ch*.npy found in {spike_root}")

    output_root.mkdir(parents=True, exist_ok=True)
    summary_rows = []
    print(f"Found {len(recording_dirs)} recordings in {spike_root}", flush=True)

    for recording_index, recording_dir in enumerate(recording_dirs, start=1):
        recording_name = recording_dir.name
        output_dir = output_root / recording_name
        done_file = output_dir / "sttc_matrix.csv"

        print("\n" + "=" * 80, flush=True)
        print(f"Recording {recording_index}/{len(recording_dirs)}: {recording_name}", flush=True)

        if SKIP_EXISTING and done_file.exists():
            print(f"Already processed, skipping: {done_file}", flush=True)
            if GENERATE_NETWORK_MAPS and not network_maps_exist(output_dir):
                print("Existing connectivity found; generating missing network maps.", flush=True)
                save_network_maps_from_existing_outputs(output_dir)
            summary_rows.append({"recording_name": recording_name, "status": "skipped_existing", "active_channels": None})
            continue

        try:
            t0 = time.time()
            spike_frames, filter_summary = load_negative_spike_frames(recording_dir)
            spike_times = Connectivity.FramesToSeconds(spike_frames, SamplingRate=SAMPLING_RATE_HZ)
            channels = sorted(spike_times)

            correlation_matrix, channels, counts, bin_edges = Connectivity.SpikeCountCorrelation(
                spike_times,
                BinSize=BIN_SIZE_SECONDS,
                Duration=RECORDING_DURATION_SECONDS,
                Channels=channels,
            )
            sttc_matrix, _ = Connectivity.STTCMatrix(
                spike_times,
                Dt=STTC_DT_SECONDS,
                Duration=RECORDING_DURATION_SECONDS,
                Channels=channels,
                Verbose=VERBOSE,
            )

            channel_firing_rates = pd.DataFrame(
                {
                    "channel": list(spike_frames),
                    "firing_rate_hz": [len(frames) / float(RECORDING_DURATION_SECONDS) for frames in spike_frames.values()],
                    "is_active": True,
                }
            )
            result = {
                "channels": channels,
                "spike_frames": spike_frames,
                "spike_times": spike_times,
                "channel_firing_rates": channel_firing_rates,
                "counts": counts,
                "bin_edges": bin_edges,
                "correlation_matrix": correlation_matrix,
                "sttc_matrix": sttc_matrix,
                "correlation_strength": Connectivity.NodeStrength(correlation_matrix, channels),
                "sttc_strength": Connectivity.NodeStrength(sttc_matrix, channels),
                "correlation_edges": Connectivity.MatrixToEdges(
                    correlation_matrix,
                    channels,
                    Threshold=CORRELATION_THRESHOLD,
                ),
                "sttc_edges": Connectivity.MatrixToEdges(sttc_matrix, channels, Threshold=STTC_THRESHOLD),
            }

            save_results(result, output_dir, filter_summary)
            map_summary = save_network_maps(result, output_dir)
            active_channels = len(channels)
            total_active_spikes = int(sum(len(frames) for frames in spike_frames.values()))
            summary_rows.append(
                {
                    "recording_name": recording_name,
                    "status": "completed",
                    "active_channels": active_channels,
                    "active_negative_spikes": total_active_spikes,
                    "sttc_edges": len(result["sttc_edges"]),
                    "correlation_edges": len(result["correlation_edges"]),
                    "shown_top_sttc_edges": map_summary["shown_top_sttc_edges"],
                    "shown_top_correlation_edges": map_summary["shown_top_correlation_edges"],
                    "elapsed_seconds": time.time() - t0,
                }
            )
            print(f"Analysis saved to: {output_dir}", flush=True)
            print(f"Active channels analyzed: {active_channels}", flush=True)
            print(f"Active negative spikes: {total_active_spikes}", flush=True)
            if GENERATE_NETWORK_MAPS:
                print(f"Network maps saved to: {output_dir / 'network_maps'}", flush=True)
        except Exception as error:
            print(f"ERROR in {recording_name}: {error}", flush=True)
            summary_rows.append(
                {"recording_name": recording_name, "status": "error", "active_channels": None, "error": str(error)}
            )

        pd.DataFrame(summary_rows).to_csv(output_root / "connectivity_summary.csv", index=False)

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(output_root / "connectivity_summary.csv", index=False)
    print("\nBatch analysis completed.", flush=True)
    print(summary["status"].value_counts().to_string(), flush=True)
    print(f"Summary: {output_root / 'connectivity_summary.csv'}", flush=True)


if __name__ == "__main__":
    main()
