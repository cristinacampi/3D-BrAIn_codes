import numpy as np

from brain_3d import Classification, SpikeSorting


def test_negative_waveform_mask_applies_label_polarity_and_center_rules():
    spikes = np.array(
        [
            [0.0, 1.0, -4.0, 1.0, 0.0],
            [0.0, 1.0, -4.0, 1.0, 0.0],
            [-4.0, 1.0, 0.0, 1.0, 0.0],
            [0.0, -1.0, 2.0, -1.0, 0.0],
        ]
    )
    labels = np.array([0, 1, 0, 0])
    config = SpikeSorting.NegativeSpikeFilterConfig(center_window_samples=1)

    mask = SpikeSorting.NegativeWaveformMask(spikes, labels, config)

    assert mask.tolist() == [True, False, False, False]


def test_negative_waveform_mask_works_without_detector_labels():
    spikes = np.array([[0.0, 1.0, -3.0, 1.0, 0.0]])

    mask = SpikeSorting.NegativeWaveformMask(spikes)

    assert mask.tolist() == [True]


def test_waveform_polarity_mask_selects_positive_events():
    spikes = np.array([
        [0.0, -1.0, 5.0, -0.5, 0.0],
        [0.0, 1.0, -5.0, 0.5, 0.0],
    ])
    labels = np.array([1, 0])
    config = SpikeSorting.SpikePolarityFilterConfig(
        polarity="positive", center_window_samples=1
    )

    mask = SpikeSorting.WaveformPolarityMask(spikes, labels, config)

    assert mask.tolist() == [True, False]


def test_waveform_polarity_mask_none_keeps_all_events():
    spikes = np.array([[0.0, -3.0, 0.0], [0.0, 4.0, 0.0]])
    labels = np.array([0, 1])
    config = SpikeSorting.SpikePolarityFilterConfig(polarity="none")

    mask = SpikeSorting.WaveformPolarityMask(spikes, labels, config)

    assert mask.tolist() == [True, True]


def test_protocol_from_filename_uses_specific_combined_protocol_first():
    assert Classification.ProtocolFromFilename("sample_100uM4AP_50uMAP5.brw") == "4ap_then_ap5"
