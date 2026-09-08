import os
import sys
import time
from pathlib import Path

import h5py
import numpy as np
from joblib import Parallel, delayed


for Parent in Path(__file__).resolve().parents:
    SrcPath = Parent / "src"
    if (SrcPath / "brain_3d").is_dir():
        sys.path.insert(0, str(SrcPath))
        break

from brain_3d import BrwFunctions, SpikeSorting



RECORDING_DIR = "/data/shared/3D_BRAIN/Mark/Single well MEA/4. Human laminin series 1"
#RECORDING_DIR = "/data/shared/3D_BRAIN/Mark/Single well MEA/8. Human laminin series 2/" 
TRACE_ROOT = RECORDING_DIR + "/traces_h5/"
SPIKE_ROOT = RECORDING_DIR + "/spikes/"
WELL_ID = "Well_A1"

CHANNEL_START = 0
CHANNEL_END = None
MAX_WORKERS = 100

SKIP_EXISTING_SPIKES = True

NOTCH_CUT = 50
LOW_CUT = 300
HIGH_CUT = 3000
IDX_NUMBER = 41
HALF_IDX_NUMBER = IDX_NUMBER // 2
DETECTION_THRESHOLD = 5


def extract_waveforms(DataFull, Frames, NumFrames):
    Dataset = np.zeros((len(Frames), IDX_NUMBER))
    for Index, PeakFrame in enumerate(Frames):
        PeakFrame = int(PeakFrame)
        if PeakFrame < HALF_IDX_NUMBER:
            Dataset[Index, HALF_IDX_NUMBER - PeakFrame:IDX_NUMBER] = DataFull[0:PeakFrame + HALF_IDX_NUMBER + 1]
        elif PeakFrame >= NumFrames - HALF_IDX_NUMBER:
            Aux = DataFull[PeakFrame - HALF_IDX_NUMBER:NumFrames]
            Dataset[Index, 0:len(Aux)] = Aux
        else:
            Dataset[Index] = DataFull[PeakFrame - HALF_IDX_NUMBER:PeakFrame + HALF_IDX_NUMBER + 1]
    return Dataset


def detect_channel_spikes(Channel, ChannelRow, H5Path, SpikeDir):
    print(Channel)
    with h5py.File(H5Path, "r") as File:
        Traces = File[WELL_ID]["Traces"]
        SamplingRate = float(File[WELL_ID].attrs["SamplingRate"])
        NumFrames = int(File[WELL_ID].attrs["NumFrames"])
        DataFull = np.asarray(Traces[:, ChannelRow], dtype=float)

    DataFull = DataFull - np.median(DataFull)
    DataFull = BrwFunctions.NotchFilter(DataFull, NOTCH_CUT, SamplingRate)
    DataFull = BrwFunctions.BandpassFilter(DataFull, LOW_CUT, HIGH_CUT, SamplingRate)
    DataFull = DataFull - np.mean(DataFull)

    Step = int(SamplingRate * 0.05)
    FramesN = SpikeSorting.SpikesDetection(DataFull, Step, DETECTION_THRESHOLD, "neg")
    FramesP = SpikeSorting.SpikesDetection(DataFull, Step, DETECTION_THRESHOLD, "pos")

    FramesPNew = []
    FramesNNew = set(FramesN)
    for FrameP in FramesP:
        PreviousNeg = set(range(FrameP - HALF_IDX_NUMBER, FrameP)) & set(FramesN)
        FollowingNeg = set(range(FrameP, FrameP + HALF_IDX_NUMBER + 1)) & set(FramesN)

        if PreviousNeg:
            FrameN = sorted(PreviousNeg)[-1]
            if np.abs(DataFull[FrameN]) < np.abs(DataFull[FrameP]):
                FramesPNew.append(FrameP)
                FramesNNew.discard(FrameN)
        elif FollowingNeg:
            FrameN = sorted(FollowingNeg)[0]
            if np.abs(DataFull[FrameN]) < np.abs(DataFull[FrameP]):
                FramesPNew.append(FrameP)
                FramesNNew.discard(FrameN)
        else:
            FramesPNew.append(FrameP)

    FramesN = sorted(FramesNNew)
    FramesP = sorted(FramesPNew)
    DatasetN = extract_waveforms(DataFull, FramesN, NumFrames)
    DatasetP = extract_waveforms(DataFull, FramesP, NumFrames)

    DatasetTot = np.zeros((DatasetN.shape[0] + DatasetP.shape[0], IDX_NUMBER))
    DatasetTot[0:DatasetN.shape[0]] = DatasetN
    DatasetTot[DatasetN.shape[0]:DatasetN.shape[0] + DatasetP.shape[0]] = DatasetP

    np.save(f"{SpikeDir}/spikes_ch{Channel}.npy", DatasetTot)
    np.save(f"{SpikeDir}/frames_ch{Channel}.npy", np.concatenate([FramesN, FramesP]))
    np.save(f"{SpikeDir}/label_ch{Channel}.npy", np.concatenate([np.zeros((DatasetN.shape[0], 1)), np.ones((DatasetP.shape[0], 1))]))


def main() -> None:
    H5Paths = sorted(Path(TRACE_ROOT).glob("*/channel_traces.h5"))
    if not H5Paths:
        raise FileNotFoundError(f"No channel_traces.h5 files found in {TRACE_ROOT}")

    for H5Path in H5Paths:
        RecordingName = H5Path.parent.name
        SpikeDir = SPIKE_ROOT + RecordingName + "/"
        os.makedirs(SpikeDir, exist_ok=True)

        with h5py.File(H5Path, "r") as File:
            Channels = np.asarray(File[WELL_ID]["ChannelIndices"], dtype=int)
            NumFrames = int(File[WELL_ID].attrs["NumFrames"])
            SamplingRate = float(File[WELL_ID].attrs["SamplingRate"])

        ChannelEnd = len(Channels) if CHANNEL_END is None else min(CHANNEL_END, len(Channels))
        SpikeJobs = [(ChannelRow, int(Channels[ChannelRow])) for ChannelRow in range(CHANNEL_START, ChannelEnd)]
        if SKIP_EXISTING_SPIKES:
            SpikeJobs = [
                (ChannelRow, Channel)
                for ChannelRow, Channel in SpikeJobs
                if not os.path.exists(f"{SpikeDir}/spikes_ch{Channel}.npy")
            ]

        print("\nRecording:", RecordingName)
        print("H5:", H5Path)
        print("Output:", SpikeDir)
        print("Frames:", NumFrames)
        print("Sampling rate:", SamplingRate)
        print("Channels to process:", len(SpikeJobs))

        for BatchStart in range(0, len(SpikeJobs), MAX_WORKERS):
            BatchJobs = SpikeJobs[BatchStart:BatchStart + MAX_WORKERS]
            if not BatchJobs:
                continue

            Start = time.time()
            Parallel(n_jobs=len(BatchJobs), backend="loky")(
                delayed(detect_channel_spikes)(Channel, ChannelRow, str(H5Path), SpikeDir)
                for ChannelRow, Channel in BatchJobs
            )
            print(
                f"spike detection channels {BatchJobs[0][1]}-{BatchJobs[-1][1]} "
                f"({len(BatchJobs)} workers): {time.time() - Start:.2f} s"
            )


if __name__ == "__main__":
    main()
