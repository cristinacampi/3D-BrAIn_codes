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

RECORDING_DIR = "/data/shared/3D_BRAIN/Mark/Single well MEA/4. Human laminin series 1"
#RECORDING_DIR = "/data/shared/3D_BRAIN/Mark/Single well MEA/8. Human laminin series 2"

#RECORDING_DIR = "/data/26-06-04-EXP//"
TRACE_ROOT = RECORDING_DIR + "/traces_h5/"
CHANNEL_ROOT = RECORDING_DIR + "/channels/"
WELL_ID = "Well_A1"

CHANNEL_START = 0
CHANNEL_END = None
MAX_WORKERS = 100
SKIP_EXISTING = True
OUTPUT_DTYPE = None  # use None to keep the HDF5 dtype


def export_channel(H5Path, ChannelRow, Channel, OutputDir):
    OutputPath = f"{OutputDir}/ch{Channel}.npy"
    if SKIP_EXISTING and os.path.exists(OutputPath):
        return OutputPath

    with h5py.File(H5Path, "r") as File:
        Trace = File[WELL_ID]["Traces"][:, ChannelRow]
        Trace = np.asarray(Trace if OUTPUT_DTYPE is None else Trace.astype(OUTPUT_DTYPE))

    np.save(OutputPath, Trace)
    return OutputPath


def main() -> None:
    H5Paths = sorted(Path(TRACE_ROOT).glob("*/channel_traces.h5"))
    if not H5Paths:
        raise FileNotFoundError(f"No channel_traces.h5 files found in {TRACE_ROOT}")

    for H5Path in H5Paths:
        RecordingName = H5Path.parent.name
        OutputDir = CHANNEL_ROOT + RecordingName + "/"
        os.makedirs(OutputDir, exist_ok=True)

        with h5py.File(H5Path, "r") as File:
            Channels = np.asarray(File[WELL_ID]["ChannelIndices"], dtype=int)
            NumFrames = int(File[WELL_ID].attrs["NumFrames"])
            TraceDType = File[WELL_ID]["Traces"].dtype

        ChannelEnd = len(Channels) if CHANNEL_END is None else min(CHANNEL_END, len(Channels))
        Jobs = [(ChannelRow, int(Channels[ChannelRow])) for ChannelRow in range(CHANNEL_START, ChannelEnd)]
        if SKIP_EXISTING:
            Jobs = [(Row, Channel) for Row, Channel in Jobs if not os.path.exists(f"{OutputDir}/ch{Channel}.npy")]

        print("\nRecording:", RecordingName)
        print("H5:", H5Path)
        print("Output:", OutputDir)
        print("Frames:", NumFrames)
        print("H5 dtype:", TraceDType)
        print("Output dtype:", OUTPUT_DTYPE or TraceDType)
        print("Channels to export:", len(Jobs))

        for BatchStart in range(0, len(Jobs), MAX_WORKERS):
            BatchJobs = Jobs[BatchStart:BatchStart + MAX_WORKERS]
            if not BatchJobs:
                continue

            Start = time.time()
            Parallel(n_jobs=len(BatchJobs), backend="loky")(
                delayed(export_channel)(str(H5Path), ChannelRow, Channel, OutputDir)
                for ChannelRow, Channel in BatchJobs
            )
            print(
                f"exported channels {BatchJobs[0][1]}-{BatchJobs[-1][1]} "
                f"({len(BatchJobs)} workers): {time.time() - Start:.2f} s"
            )


if __name__ == "__main__":
    main()
