import os
import pickle
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

for Parent in Path(__file__).resolve().parents:
    SrcPath = Parent / "src"
    if (SrcPath / "brain_3d").is_dir():
        sys.path.insert(0, str(SrcPath))
        break


from brain_3d import BrwFunctions as brw_f





#PATH_ROOT = "/data/26-06-04-EXP//"
PATH_ROOT = "/data/shared/3D_BRAIN/Mark/Single well MEA/4. Human laminin series 1/"

#PATH_ROOT = "/data/shared/3D_BRAIN/Mark/Single well MEA/8. Human laminin series 2/"
SPIKE_ROOT = PATH_ROOT + "/spikes/"
TRACE_ROOT = PATH_ROOT + "/traces_h5/"
METRIC_ROOT = PATH_ROOT + "/metrics/"
WELL_ID = "Well_A1"

CHANNEL_START = 0
CHANNEL_END = None
SPIKE_WINDOW = 41
SKIP_EXISTING_METRICS = False

ACTIVE_ELECTRODE_THRESHOLD = 0.0
N_MIN_SPIKES_BURST = 5
ISI_MAX_SECONDS_BURST = 0.1

SPIKE_METRIC_NAMES = [
    "SamplingRate", "NumFrames", "SpikesN", "SpikesP", "SpikesFramesAE",
    "SpikesCount", "SpikesRate", "ISIMap", "ISIVarianceMap",
    "ISIWellAverage", "ISIVarianceWell", "NSpikesWell", "SpikesWellRate",
    "ActiveElectrodesID", "ActiveElectrodesNumber", "PeakToPeakMap", "PeakToPeakStd",
    "PeakToPeakAverageWell", "PeakToPeakStdWell",
]

BURST_METRIC_NAMES = [
    "Bursts", "NBursts", "BurstsRate", "BurstsNSpikes", "BurstsSpikesPercentage",
    "BurstsISI", "BurstsISIAverage", "BurstsNSpikesAverage", "BurstsDuration",
    "BurstsDurationAverage", "IBI", "IBIAverage", "IBIStd", "BurstsWell",
    "NBurstsWell", "BurstsRateWell", "BurstsDurationWellAverage",
    "BurstsDurationWellStd", "IBIWell", "IBIWellAverage", "IBIWellStd",
    "BurstsNSpikesWell", "BurstsNSpikesWellAverage", "BurstsNSpikesWellStd",
    "BurstsISIWell", "BurstsISIWellAverage", "BurstsISIWellStd",
    "BurstsSpikesPercentageWell",
]


def load_recording_metadata(RecordingName):
    H5Path = Path(TRACE_ROOT) / RecordingName / "channel_traces.h5"
    if H5Path.exists():
        with h5py.File(H5Path, "r") as File:
            Well = File[WELL_ID]
            SamplingRate = float(Well.attrs["SamplingRate"])
            NumFrames = int(Well.attrs["NumFrames"])
            NumChannels = int(len(Well["ChannelIndices"]))
        return SamplingRate, NumChannels, NumFrames

    BrwPath = Path(PATH_ROOT) / f"{RecordingName}.brw"
    if not BrwPath.exists():
        raise FileNotFoundError(f"Missing metadata source for {RecordingName}: no H5 and no BRW")

    BRW, SamplingRate, NumChannels, _, NumFrames = brw_f.ReadBRW(str(BrwPath), WELL_ID)
    BRW.close()
    return SamplingRate, NumChannels, NumFrames


def processed_recordings():
    SpikeRoot = Path(SPIKE_ROOT)
    if not SpikeRoot.exists():
        raise FileNotFoundError(f"Spike root not found: {SPIKE_ROOT}")

    for SpikeDir in sorted(SpikeRoot.iterdir()):
        if not SpikeDir.is_dir():
            continue
        if any(SpikeDir.glob("spikes_ch*.npy")):
            yield SpikeDir.name


def split_channel_spikes(SpikeDir, Channel):
    SpikePath = SpikeDir / f"spikes_ch{Channel}.npy"
    FramePath = SpikeDir / f"frames_ch{Channel}.npy"
    LabelPath = SpikeDir / f"label_ch{Channel}.npy"

    if not (SpikePath.exists() and FramePath.exists() and LabelPath.exists()):
        EmptyFrames = np.array([], dtype=int)
        EmptyDataset = np.zeros((0, SPIKE_WINDOW))
        return EmptyFrames, EmptyFrames, EmptyDataset, EmptyDataset

    Dataset = np.load(SpikePath)
    Frames = np.load(FramePath).astype(int)
    Labels = np.load(LabelPath).reshape(-1).astype(int)

    NegativeMask = Labels == 0
    PositiveMask = Labels == 1
    return Frames[NegativeMask], Frames[PositiveMask], Dataset[NegativeMask], Dataset[PositiveMask]


def channel_range(RecordingName, NumChannels):
    H5Path = Path(TRACE_ROOT) / RecordingName / "channel_traces.h5"
    if H5Path.exists():
        with h5py.File(H5Path, "r") as File:
            Channels = np.asarray(File[WELL_ID]["ChannelIndices"], dtype=int)
    else:
        Channels = np.arange(NumChannels, dtype=int)

    Start = CHANNEL_START
    End = len(Channels) if CHANNEL_END is None else min(CHANNEL_END, len(Channels))
    return Channels[Start:End]


def save_metrics(RecordingName, Channels, SamplingRate, NumFrames, SpikesN, SpikesP, DatasetN, DatasetP):
    SaveDir = Path(METRIC_ROOT) / RecordingName / "spike_burst"
    os.makedirs(SaveDir, exist_ok=True)

    SpikeMetrics = brw_f.SpikesMetric(
        SamplingRate=SamplingRate,
        NumFrames=NumFrames,
        SpikesN=SpikesN,
        SpikesP=SpikesP,
        DatasetN=DatasetN,
        DatasetP=DatasetP,
        Threshold=ACTIVE_ELECTRODE_THRESHOLD,
    )
    SpikeMetric = dict(zip(SPIKE_METRIC_NAMES, SpikeMetrics))

    BurstMetrics = brw_f.BurstsMetric(
        SamplingRate=SpikeMetric["SamplingRate"],
        NumFrames=SpikeMetric["NumFrames"],
        SpikesN=SpikeMetric["SpikesN"],
        SpikesP=SpikeMetric["SpikesP"],
        SpikesFramesAE=SpikeMetric["SpikesFramesAE"],
        SpikesCount=SpikeMetric["SpikesCount"],
        NSpikesWell=SpikeMetric["NSpikesWell"],
        ActiveElectrodesID=SpikeMetric["ActiveElectrodesID"],
        NMinSpikes=N_MIN_SPIKES_BURST,
        ISIMaxSeconds=ISI_MAX_SECONDS_BURST,
    )
    BurstMetric = dict(zip(BURST_METRIC_NAMES, BurstMetrics))

    RawSpikesCount = np.asarray([len(set(Neg) | set(Pos)) for Neg, Pos in zip(SpikesN, SpikesP)])
    RawSpikesRate = RawSpikesCount / NumFrames * SamplingRate

    ChannelMetrics = pd.DataFrame(
        {
            "Channel": Channels,
            "RawSpikesCount": RawSpikesCount,
            "RawSpikesRate": RawSpikesRate,
            "IsActiveElectrode": RawSpikesRate >= ACTIVE_ELECTRODE_THRESHOLD,
            "SpikesCount": SpikeMetric["SpikesCount"],
            "SpikesRate": SpikeMetric["SpikesRate"],
            "ISIMap": SpikeMetric["ISIMap"],
            "ISIVarianceMap": SpikeMetric["ISIVarianceMap"],
            "PeakToPeakMap": SpikeMetric["PeakToPeakMap"],
            "PeakToPeakStd": SpikeMetric["PeakToPeakStd"],
            "NBursts": BurstMetric["NBursts"],
            "BurstsRate": BurstMetric["BurstsRate"],
            "BurstsSpikesPercentage": BurstMetric["BurstsSpikesPercentage"],
            "BurstsISIAverage": BurstMetric["BurstsISIAverage"],
            "BurstsNSpikesAverage": BurstMetric["BurstsNSpikesAverage"],
            "BurstsDurationAverage": BurstMetric["BurstsDurationAverage"],
            "IBIAverage": BurstMetric["IBIAverage"],
            "IBIStd": BurstMetric["IBIStd"],
        }
    )

    WellMetrics = pd.DataFrame(
        [
            {
                "RecordingName": RecordingName,
                "SamplingRate": SamplingRate,
                "NumFrames": NumFrames,
                "ActiveElectrodesNumber": SpikeMetric["ActiveElectrodesNumber"],
                "ActiveElectrodesID": " ".join(map(str, SpikeMetric["ActiveElectrodesID"])),
                "NSpikesWell": SpikeMetric["NSpikesWell"],
                "SpikesWellRate": SpikeMetric["SpikesWellRate"],
                "ISIWellAverage": SpikeMetric["ISIWellAverage"],
                "ISIVarianceWell": SpikeMetric["ISIVarianceWell"],
                "PeakToPeakAverageWell": SpikeMetric["PeakToPeakAverageWell"],
                "PeakToPeakStdWell": SpikeMetric["PeakToPeakStdWell"],
                "NBurstsWell": BurstMetric["NBurstsWell"],
                "BurstsRateWell": BurstMetric["BurstsRateWell"],
                "BurstsDurationWellAverage": BurstMetric["BurstsDurationWellAverage"],
                "BurstsDurationWellStd": BurstMetric["BurstsDurationWellStd"],
                "IBIWellAverage": BurstMetric["IBIWellAverage"],
                "IBIWellStd": BurstMetric["IBIWellStd"],
                "BurstsNSpikesWellAverage": BurstMetric["BurstsNSpikesWellAverage"],
                "BurstsNSpikesWellStd": BurstMetric["BurstsNSpikesWellStd"],
                "BurstsISIWellAverage": BurstMetric["BurstsISIWellAverage"],
                "BurstsISIWellStd": BurstMetric["BurstsISIWellStd"],
                "BurstsSpikesPercentageWell": BurstMetric["BurstsSpikesPercentageWell"],
            }
        ]
    )

    ChannelMetrics.to_csv(SaveDir / "channel_spike_burst_metrics.csv", index=False)
    WellMetrics.to_csv(SaveDir / "well_spike_burst_metrics.csv", index=False)
    with open(SaveDir / "full_spike_burst_metrics.pkl", "wb") as File:
        pickle.dump({"spike_metrics": SpikeMetric, "burst_metrics": BurstMetric}, File)

    return SaveDir, WellMetrics, ChannelMetrics


def main() -> None:
    Processed = list(processed_recordings())
    if not Processed:
        raise FileNotFoundError(f"No processed spike folders found in {SPIKE_ROOT}")

    SummaryRows = []
    for RecordingName in Processed:
        SaveDir = Path(METRIC_ROOT) / RecordingName / "spike_burst"
        if SKIP_EXISTING_METRICS and (SaveDir / "channel_spike_burst_metrics.csv").exists():
            print(f"Skipping {RecordingName}: metrics already exist")
            continue

        print("\nRecording:", RecordingName)
        SamplingRate, NumChannels, NumFrames = load_recording_metadata(RecordingName)
        Channels = channel_range(RecordingName, NumChannels)
        SpikeDir = Path(SPIKE_ROOT) / RecordingName

        SpikesN = []
        SpikesP = []
        DatasetN = []
        DatasetP = []
        for Channel in Channels:
            ChannelSpikesN, ChannelSpikesP, ChannelDatasetN, ChannelDatasetP = split_channel_spikes(SpikeDir, int(Channel))
            SpikesN.append(ChannelSpikesN)
            SpikesP.append(ChannelSpikesP)
            DatasetN.append(ChannelDatasetN)
            DatasetP.append(ChannelDatasetP)

        SaveDir, WellMetrics, ChannelMetrics = save_metrics(
            RecordingName,
            Channels,
            SamplingRate,
            NumFrames,
            SpikesN,
            SpikesP,
            DatasetN,
            DatasetP,
        )
        SummaryRows.append(WellMetrics.iloc[0].to_dict())

        print("Channels:", len(Channels))
        print("Channel metrics saved in:", SaveDir / "channel_spike_burst_metrics.csv")
        print("Well metrics saved in:", SaveDir / "well_spike_burst_metrics.csv")
        print(ChannelMetrics.head())

    if SummaryRows:
        Summary = pd.DataFrame(SummaryRows)
        SummaryPath = Path(METRIC_ROOT) / "all_recordings_spike_burst_summary.csv"
        os.makedirs(METRIC_ROOT, exist_ok=True)
        Summary.to_csv(SummaryPath, index=False)
        print("\nSummary saved in:", SummaryPath)


if __name__ == "__main__":
    main()
