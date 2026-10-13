import struct

import h5py
import numpy as np

from brain_3d import BrwFunctions


def test_decode_event_based_raw_data_populates_requested_channel(tmp_path):
    path = tmp_path / "event_based.brw"
    encoded = b"".join(
        [
            struct.pack("<i", 3),
            struct.pack("<i", 20),
            struct.pack("<q", 0),
            struct.pack("<q", 2),
            struct.pack("<hh", -12, 34),
        ]
    )

    with h5py.File(path, "w") as brw:
        brw.attrs["SamplingRate"] = 10.0
        brw.create_dataset("TOC", data=np.array([[0, 0], [0, 2], [0, 4]]))
        well = brw.create_group("Well_A1")
        well.create_dataset(
            "EventsBasedSparseRawTOC", data=np.array([0, len(encoded), len(encoded)])
        )
        well.create_dataset("EventsBasedSparseRaw", data=np.frombuffer(encoded, dtype=np.uint8))

        data = {3: np.zeros(2, dtype=np.int16)}
        decoded = BrwFunctions.DecodeEventBasedRawData(
            brw, data, "Well_A1", StartTime=0, Duration=0.2
        )

    assert decoded[3].tolist() == [-12, 34]
