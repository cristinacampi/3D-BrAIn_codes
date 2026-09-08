import os
import sys
import time
from pathlib import Path

import h5py
import numpy as np

for Parent in Path(__file__).resolve().parents:
    SrcPath = Parent / "src"
    if (SrcPath / "brain_3d").is_dir():
        sys.path.insert(0, str(SrcPath))
        break

from brain_3d import BrwFunctions


#RECORDING_DIR = "/data/26-06-04-EXP/"
RECORDING_DIR = "/data/shared/3D_BRAIN/Mark/Single well MEA/4. Human laminin series 1"
#RECORDING_DIR = "/data/shared/3D_BRAIN/Mark/Single well MEA/8. Human laminin series 2"
WELL_ID = "Well_A1"

SKIP_RECORDINGS_STARTING_WITH = "11-6-25 D79 Chip 101"
BUILD_TRACES_H5 = True
REBUILD_H5_IF_EXISTS = False
SKIP_EXPERIMENTS_WITH_EXISTING_H5 = True

START_TIME = 0
READ_CHUNK_DURATION = 10
TRACE_DTYPE = "float16"
H5_COMPRESSION = "lzf"  # fast. Use "gzip" for smaller but slower files.

CHANNEL_START = 0
CHANNEL_END = None


def main() -> None:
    for BrwFilename in sorted(Path(RECORDING_DIR).glob("*.brw")):
        RecordingName = BrwFilename.stem
        if SKIP_RECORDINGS_STARTING_WITH and RecordingName.startswith(SKIP_RECORDINGS_STARTING_WITH):
            continue

        TraceDir = RECORDING_DIR + "/traces_h5/" + RecordingName + "/"
        os.makedirs(TraceDir, exist_ok=True)

        H5Path = TraceDir + "channel_traces.h5"
        if (
            BUILD_TRACES_H5
            and SKIP_EXPERIMENTS_WITH_EXISTING_H5
            and not REBUILD_H5_IF_EXISTS
            and os.path.exists(H5Path)
        ):
            print(f"Skipping {RecordingName}: {H5Path} already exists")
            continue

        BRW, SamplingRate, NumChannels, TotDuration, NumFrames = BrwFunctions.ReadBRW(str(BrwFilename), WELL_ID)
        ChannelEnd = NumChannels if CHANNEL_END is None else min(CHANNEL_END, NumChannels)
        Channels = np.arange(CHANNEL_START, ChannelEnd, dtype=int)

        if BUILD_TRACES_H5 and (REBUILD_H5_IF_EXISTS or not os.path.exists(H5Path)):
            ChunkFrames = max(1, int(SamplingRate * READ_CHUNK_DURATION))
            with h5py.File(H5Path, "w") as OutFile:
                WellGroup = OutFile.create_group(WELL_ID)
                WellGroup.attrs["SourceBRW"] = str(BrwFilename)
                WellGroup.attrs["SamplingRate"] = float(SamplingRate)
                WellGroup.attrs["NumFrames"] = int(NumFrames)
                WellGroup.attrs["TotDuration"] = float(TotDuration)
                WellGroup.attrs["TraceDType"] = TRACE_DTYPE
                WellGroup.attrs["Compression"] = H5_COMPRESSION
                WellGroup.create_dataset("ChannelIndices", data=Channels, compression=H5_COMPRESSION)
                Traces = WellGroup.create_dataset(
                    "Traces",
                    shape=(int(NumFrames), len(Channels)),
                    dtype=TRACE_DTYPE,
                    chunks=(min(ChunkFrames, int(NumFrames)), min(256, len(Channels))),
                    compression=H5_COMPRESSION,
                )

                CurrentStart = START_TIME
                Start = time.time()
                while CurrentStart < TotDuration:
                    CurrentDuration = min(READ_CHUNK_DURATION, TotDuration - CurrentStart)
                    if CurrentDuration <= 0:
                        break

                    DataChunk, Frames = BrwFunctions.ReadingRawData(
                        BRW,
                        WELL_ID,
                        SamplingRate,
                        CurrentStart,
                        CurrentDuration,
                    )
                    Frames = Frames.astype(int)
                    Valid = (Frames >= 0) & (Frames < NumFrames)
                    Frames = Frames[Valid]
                    DataChunk = DataChunk[Valid]
                    Traces[Frames, :] = DataChunk[:, Channels].astype(TRACE_DTYPE, copy=False)
                    OutFile.flush()

                    print(
                        f"read {CurrentStart:.2f}-{CurrentStart + CurrentDuration:.2f} s "
                        f"({len(Frames)} frames): {time.time() - Start:.2f} s total"
                    )
                    CurrentStart += CurrentDuration

        BRW.close()


if __name__ == "__main__":
    main()
