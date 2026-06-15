import os
import pickle
import sys
from pathlib import Path
import debugpy
debugpy.breakpoint()

import numpy as np
import pandas as pd

def add_project_src_to_path() -> None:
    for parent in Path(__file__).resolve().parents:
        src_path = parent / "src"
        if (src_path / "brain_3d").is_dir():
            sys.path.insert(0, str(src_path))
            return
    raise ModuleNotFoundError("Cannot find src/brain_3d from this script location.")

add_project_src_to_path()

from brain_3d import BrwFunctions
print("BrwFunctions loaded from:", BrwFunctions.__file__)

PATH_ROOT = '/data/26-05-28_3Brain_Ivan_PartnershipUniGe_Pasca120Days/'
BRW_FILE = "00_1W38-60_Hu_iPSC_Brain_Org_120DIV_Spontaneous_Raw_00"
# Directory containing spikes_ch<Channel>.npy, frames_ch<Channel>.npy and label_ch<Channel>.npy.
SPIKE_DIR = PATH_ROOT +'/spikes/' + BRW_FILE + '/'
SAVE_DIR = PATH_ROOT + '/metrics/' + BRW_FILE + '/spike_burst/'
os.makedirs(SAVE_DIR, exist_ok=True)

BRW_FILE = BRW_FILE + '.brw'
WELL_ID = "Well_A1"

CHANNEL_START = 0
CHANNEL_END = None
SPIKE_WINDOW = 41

ACTIVE_ELECTRODE_THRESHOLD = 0
N_MIN_SPIKES_BURST = 5
ISI_MAX_SECONDS_BURST = 0.1


SPIKE_METRIC_NAMES = [
    "SamplingRate",
    "NumFrames",
    "SpikesN",
    "SpikesP",
    "SpikesFramesAE",
    "SpikesCount",
    "SpikesRate",
    "ISIMap",
    "ISIVarianceMap",
    "ISIWellAverage",
    "ISIVarianceWell",
    "NSpikesWell",
    "SpikesWellRate",
    "ActiveElectrodesID",
    "ActiveElectrodesNumber",
    "PeakToPeakMap",
    "PeakToPeakStd",
    "PeakToPeakAverageWell",
    "PeakToPeakStdWell",
]

BURST_METRIC_NAMES = [
    "Bursts",
    "NBursts",
    "BurstsRate",
    "BurstsNSpikes",
    "BurstsSpikesPercentage",
    "BurstsISI",
    "BurstsISIAverage",
    "BurstsNSpikesAverage",
    "BurstsDuration",
    "BurstsDurationAverage",
    "IBI",
    "IBIAverage",
    "IBIStd",
    "BurstsWell",
    "NBurstsWell",
    "BurstsRateWell",
    "BurstsDurationWellAverage",
    "BurstsDurationWellStd",
    "IBIWell",
    "IBIWellAverage",
    "IBIWellStd",
    "BurstsNSpikesWell",
    "BurstsNSpikesWellAverage",
    "BurstsNSpikesWellStd",
    "BurstsISIWell",
    "BurstsISIWellAverage",
    "BurstsISIWellStd",
    "BurstsSpikesPercentageWell",
]


def load_recording_metadata():
    _, SamplingRate, NumChannels, _, NumFrames = BrwFunctions.ReadBRW(PATH_ROOT + BRW_FILE, WELL_ID)
    return SamplingRate, NumChannels, NumFrames


def split_channel_spikes(Channel):
    SpikePath = f"{SPIKE_DIR}spikes_ch{Channel}.npy"
    FramePath = f"{SPIKE_DIR}frames_ch{Channel}.npy"
    LabelPath = f"{SPIKE_DIR}label_ch{Channel}.npy"

    if not (os.path.exists(SpikePath) and os.path.exists(FramePath) and os.path.exists(LabelPath)):
        EmptyFrames = np.array([], dtype=int)
        EmptyDataset = np.zeros((0, SPIKE_WINDOW))
        return EmptyFrames, EmptyFrames, EmptyDataset, EmptyDataset

    Dataset = np.load(SpikePath)
    Frames = np.load(FramePath).astype(int)
    Labels = np.load(LabelPath).reshape(-1).astype(int)

    NegativeMask = Labels == 0
    PositiveMask = Labels == 1
    SpikesN = Frames[NegativeMask]
    SpikesP = Frames[PositiveMask]
    DatasetN = Dataset[NegativeMask]
    DatasetP = Dataset[PositiveMask]
    return SpikesN, SpikesP, DatasetN, DatasetP


def load_all_spikes(NumChannels):
    LastChannel = NumChannels if CHANNEL_END is None else min(CHANNEL_END, NumChannels)
    SpikesN = []
    SpikesP = []
    DatasetN = []
    DatasetP = []

    for Channel in range(CHANNEL_START, LastChannel):
        ChannelSpikesN, ChannelSpikesP, ChannelDatasetN, ChannelDatasetP = split_channel_spikes(Channel)
        SpikesN.append(ChannelSpikesN)
        SpikesP.append(ChannelSpikesP)
        DatasetN.append(ChannelDatasetN)
        DatasetP.append(ChannelDatasetP)

    return SpikesN, SpikesP, DatasetN, DatasetP


def save_channel_metrics(SpikeMetrics, BurstMetrics):
    SpikeMetric = dict(zip(SPIKE_METRIC_NAMES, SpikeMetrics))
    BurstMetric = dict(zip(BURST_METRIC_NAMES, BurstMetrics))
    Channels = np.arange(len(SpikeMetric["SpikesCount"])) + CHANNEL_START

    ChannelMetrics = pd.DataFrame(
        {
            "Channel": Channels,
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
    ChannelMetrics.to_csv(SAVE_DIR + "channel_spike_burst_metrics.csv", index=False)
    return ChannelMetrics


def save_well_metrics(SpikeMetrics, BurstMetrics):
    SpikeMetric = dict(zip(SPIKE_METRIC_NAMES, SpikeMetrics))
    BurstMetric = dict(zip(BURST_METRIC_NAMES, BurstMetrics))

    WellMetrics = pd.DataFrame(
        [
            {
                "SamplingRate": SpikeMetric["SamplingRate"],
                "NumFrames": SpikeMetric["NumFrames"],
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
    WellMetrics.to_csv(SAVE_DIR + "well_spike_burst_metrics.csv", index=False)
    return WellMetrics


def save_full_metrics(SpikeMetrics, BurstMetrics):
    FullMetrics = {
        "spike_metrics": dict(zip(SPIKE_METRIC_NAMES, SpikeMetrics)),
        "burst_metrics": dict(zip(BURST_METRIC_NAMES, BurstMetrics)),
    }
    with open(SAVE_DIR + "full_spike_burst_metrics.pkl", "wb") as File:
        pickle.dump(FullMetrics, File)


def main() -> None:
    os.makedirs(SAVE_DIR, exist_ok=True)
    SamplingRate, NumChannels, NumFrames = load_recording_metadata()
    SpikesN, SpikesP, DatasetN, DatasetP = load_all_spikes(NumChannels)

    SpikeMetrics = BrwFunctions.SpikesMetric(
        SamplingRate=SamplingRate,
        NumFrames=NumFrames,
        SpikesN=SpikesN,
        SpikesP=SpikesP,
        DatasetN=DatasetN,
        DatasetP=DatasetP,
        Threshold=ACTIVE_ELECTRODE_THRESHOLD,
    )
    SpikeMetric = dict(zip(SPIKE_METRIC_NAMES, SpikeMetrics))

    BurstMetrics = BrwFunctions.BurstsMetric(
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

    ChannelMetrics = save_channel_metrics(SpikeMetrics, BurstMetrics)
    WellMetrics = save_well_metrics(SpikeMetrics, BurstMetrics)
    save_full_metrics(SpikeMetrics, BurstMetrics)

    print("Channel metrics saved in:", SAVE_DIR + "channel_spike_burst_metrics.csv")
    print("Well metrics saved in:", SAVE_DIR + "well_spike_burst_metrics.csv")
    print("Full metrics saved in:", SAVE_DIR + "full_spike_burst_metrics.pkl")
    print(WellMetrics)
    print(ChannelMetrics.head())


if __name__ == "__main__":
    main()
