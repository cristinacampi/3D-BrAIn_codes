"""Analyze functional connectivity from polarity-filtered MEA spikes."""

import sys
import time
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")

for Parent in Path(__file__).resolve().parents:
    SourcePath = Parent / "src"
    if (SourcePath / "brain_3d").is_dir():
        sys.path.insert(0, str(SourcePath))
        break

from brain_3d import Connectivity, SpikeSorting

SPIKE_ROOT = Path("spikes")
POLARITY_FILTER = "negative"  # choices: "negative", "positive", "none"
OUTPUT_ROOT = Path(f"outputs/connectivity_{POLARITY_FILTER}_spikes")
SAMPLING_RATE_HZ = 19753
RECORDING_DURATION_SECONDS = 180
CORRELATION_BIN_SIZE_SECONDS = 0.05
STTC_COINCIDENCE_WINDOW_SECONDS = 0.02
CORRELATION_THRESHOLD = 0.2
STTC_THRESHOLD = 0.2
EDGE_USE_ABSOLUTE = True
MIN_ACTIVE_CHANNEL_FIRING_RATE_HZ = 0.25
EXCLUDED_CHANNELS = [0]
NEGATIVE_SPIKE_LABEL = 0
POSITIVE_SPIKE_LABEL = 1
USE_DETECTOR_LABEL_FILTER = True
USE_WAVEFORM_POLARITY_FILTER = True
POLARITY_DOMINANCE_RATIO = 1.10
REQUIRE_EXTREMUM_NEAR_CENTER = True
CENTER_WINDOW_SAMPLES = 10
VERBOSE = True
SKIP_EXISTING = True
GENERATE_NETWORK_MAPS = True
NETWORK_MAP_TOP_EDGE_PERCENT = 5.0
NETWORK_MAP_TOP_N = 120


if __name__ == "__main__":
    if CORRELATION_BIN_SIZE_SECONDS <= 0 or STTC_COINCIDENCE_WINDOW_SECONDS <= 0:
        raise ValueError("Connectivity time windows must be greater than zero.")
    if not 0 <= CORRELATION_THRESHOLD <= 1 or not 0 <= STTC_THRESHOLD <= 1:
        raise ValueError("Connectivity thresholds must be between zero and one.")
    if not 0 < NETWORK_MAP_TOP_EDGE_PERCENT <= 100:
        raise ValueError("NETWORK_MAP_TOP_EDGE_PERCENT must be in (0, 100].")
    if not SPIKE_ROOT.exists():
        raise FileNotFoundError(f"Spike root not found: {SPIKE_ROOT}")

    RecordingDirectories = [
        Directory for Directory in sorted(SPIKE_ROOT.iterdir())
        if Directory.is_dir() and any(Directory.glob("frames_ch*.npy"))
    ]
    if not RecordingDirectories:
        raise FileNotFoundError(f"No recordings with frames_ch*.npy found in {SPIKE_ROOT}")

    FilterConfig = SpikeSorting.SpikePolarityFilterConfig(
        polarity=POLARITY_FILTER,
        negative_spike_label=NEGATIVE_SPIKE_LABEL,
        positive_spike_label=POSITIVE_SPIKE_LABEL,
        use_detector_label_filter=USE_DETECTOR_LABEL_FILTER,
        use_waveform_polarity_filter=USE_WAVEFORM_POLARITY_FILTER,
        polarity_dominance_ratio=POLARITY_DOMINANCE_RATIO,
        require_extremum_near_center=REQUIRE_EXTREMUM_NEAR_CENTER,
        center_window_samples=CENTER_WINDOW_SAMPLES,
    )
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    SummaryRows = []

    for RecordingIndex, RecordingDirectory in enumerate(RecordingDirectories, start=1):
        StartTime = time.time()
        OutputDirectory = OUTPUT_ROOT / RecordingDirectory.name
        DoneFile = OutputDirectory / "sttc_matrix.csv"
        print(f"Recording {RecordingIndex}/{len(RecordingDirectories)}: {RecordingDirectory.name}")
        if SKIP_EXISTING and DoneFile.exists():
            SummaryRows.append({"recording_name": RecordingDirectory.name, "status": "skipped_existing"})
            continue

        try:
            SpikeFrames, FilterRows = {}, []
            for Channel, Arrays in SpikeSorting.LoadSpikeRecording(RecordingDirectory).items():
                if Channel in EXCLUDED_CHANNELS or Arrays["Spikes"].size == 0:
                    continue
                Spikes = np.asarray(Arrays["Spikes"]).reshape(len(Arrays["Spikes"]), -1)
                Count = min(len(Spikes), len(Arrays["Frames"]), len(Arrays["Labels"]))
                Spikes, Frames, Labels = Spikes[:Count], Arrays["Frames"][:Count], Arrays["Labels"][:Count]
                Keep = SpikeSorting.WaveformPolarityMask(Spikes, Labels, FilterConfig)
                KeptFrames = np.sort(Frames[Keep].astype(int))
                Rate = len(KeptFrames) / float(RECORDING_DURATION_SECONDS)
                FilterRows.append({
                    "channel": Channel, "input_spikes": Count,
                    "detector_selected_spikes": int(SpikeSorting.WaveformPolarityMask(
                        Spikes, Labels, SpikeSorting.SpikePolarityFilterConfig(
                            polarity=POLARITY_FILTER,
                            negative_spike_label=NEGATIVE_SPIKE_LABEL,
                            positive_spike_label=POSITIVE_SPIKE_LABEL,
                            use_detector_label_filter=USE_DETECTOR_LABEL_FILTER,
                            use_waveform_polarity_filter=False,
                        ),
                    ).sum()),
                    "plausible_selected_spikes": int(Keep.sum()),
                    "firing_rate_hz": Rate, "is_active": Rate > MIN_ACTIVE_CHANNEL_FIRING_RATE_HZ,
                })
                if Rate > MIN_ACTIVE_CHANNEL_FIRING_RATE_HZ and len(KeptFrames):
                    SpikeFrames[Channel] = KeptFrames
            if not SpikeFrames:
                raise ValueError(f"No active channels remain after {POLARITY_FILTER}-spike filtering.")

            SpikeTimes = Connectivity.FramesToSeconds(SpikeFrames, SamplingRate=SAMPLING_RATE_HZ)
            Channels = sorted(SpikeTimes)
            Correlation, Channels, Counts, BinEdges = Connectivity.SpikeCountCorrelation(
                SpikeTimes, BinSize=CORRELATION_BIN_SIZE_SECONDS,
                Duration=RECORDING_DURATION_SECONDS, Channels=Channels,
            )
            Sttc, _ = Connectivity.STTCMatrix(
                SpikeTimes, Dt=STTC_COINCIDENCE_WINDOW_SECONDS,
                Duration=RECORDING_DURATION_SECONDS, Channels=Channels, Verbose=VERBOSE,
            )
            CorrelationEdges = Connectivity.MatrixToEdges(
                Correlation, Channels, Threshold=CORRELATION_THRESHOLD, Absolute=EDGE_USE_ABSOLUTE
            )
            SttcEdges = Connectivity.MatrixToEdges(
                Sttc, Channels, Threshold=STTC_THRESHOLD, Absolute=EDGE_USE_ABSOLUTE
            )
            CorrelationStrength = Connectivity.NodeStrength(Correlation, Channels, Absolute=EDGE_USE_ABSOLUTE)
            SttcStrength = Connectivity.NodeStrength(Sttc, Channels, Absolute=EDGE_USE_ABSOLUTE)

            OutputDirectory.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(Correlation, index=Channels, columns=Channels).to_csv(OutputDirectory / "spike_count_correlation_matrix.csv")
            pd.DataFrame(Sttc, index=Channels, columns=Channels).to_csv(OutputDirectory / "sttc_matrix.csv")
            CorrelationEdges.to_csv(OutputDirectory / "correlation_edges.csv", index=False)
            SttcEdges.to_csv(OutputDirectory / "sttc_edges.csv", index=False)
            CorrelationStrength.to_csv(OutputDirectory / "correlation_node_strength.csv", index=False)
            SttcStrength.to_csv(OutputDirectory / "sttc_node_strength.csv", index=False)
            pd.DataFrame(FilterRows).to_csv(OutputDirectory / "spike_filter_summary_by_channel.csv", index=False)
            np.save(OutputDirectory / "binned_spike_counts.npy", Counts)
            np.save(OutputDirectory / "bin_edges_seconds.npy", BinEdges)

            if GENERATE_NETWORK_MAPS:
                MapDirectory = OutputDirectory / "network_maps"
                MapDirectory.mkdir(parents=True, exist_ok=True)
                TopPercentile = 100.0 - NETWORK_MAP_TOP_EDGE_PERCENT
                for Name, Matrix, Strength, Edges, Cmap in (
                    ("sttc", Sttc, SttcStrength, SttcEdges, "magma"),
                    ("correlation", Correlation, CorrelationStrength, CorrelationEdges, "viridis"),
                ):
                    Connectivity.PlotConnectivityMatrix(Matrix, SavePath=MapDirectory / f"{Name}_connectivity_matrix.png", Title=f"{Name.upper()} connectivity matrix")
                    Connectivity.PlotNodeStrengthGrid(Strength, SavePath=MapDirectory / f"{Name}_node_strength_map.png", Title=f"{Name.upper()} node strength", Cmap=Cmap)
                    Selected = Connectivity.SelectStrongEdges(Edges, TopPercentile=TopPercentile, TopN=NETWORK_MAP_TOP_N, Absolute=EDGE_USE_ABSOLUTE)
                    SelectedStrength = Connectivity.EdgeNodeStrength(Selected, Absolute=EDGE_USE_ABSOLUTE)
                    Connectivity.PlotNetworkEdgeGrid(Selected, SavePath=MapDirectory / f"top_{Name}_network_edges_map.png", ChannelValues=SelectedStrength, Absolute=EDGE_USE_ABSOLUTE, Title=f"Top {NETWORK_MAP_TOP_EDGE_PERCENT:g}% {Name} edges", Cmap=Cmap)
                    Selected.to_csv(MapDirectory / f"top_{Name}_network_edges.csv", index=False)

            SummaryRows.append({
                "recording_name": RecordingDirectory.name, "status": "completed",
                "polarity_filter": POLARITY_FILTER,
                "active_channels": len(Channels), "active_selected_spikes": sum(map(len, SpikeFrames.values())),
                "sttc_edges": len(SttcEdges), "correlation_edges": len(CorrelationEdges),
                "elapsed_seconds": time.time() - StartTime,
            })
        except Exception as Error:
            print(f"ERROR in {RecordingDirectory.name}: {Error}")
            SummaryRows.append({"recording_name": RecordingDirectory.name, "status": "error", "error": str(Error)})
        pd.DataFrame(SummaryRows).to_csv(OUTPUT_ROOT / "connectivity_summary.csv", index=False)

    print(pd.DataFrame(SummaryRows)["status"].value_counts().to_string())
