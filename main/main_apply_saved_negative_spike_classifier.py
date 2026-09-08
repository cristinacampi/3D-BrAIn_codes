"""Apply a saved negative-spike recording classifier to new MEA recordings.

This main loads one reusable model produced by
``main_run_negative_spike_recording_classification.py`` and applies it to one
or more processed ``spikes/<recording>`` folders.

Supported saved models:

- classical scikit-learn bundles saved as ``final_models/<model>.joblib``;
- Mamba sequence models saved as ``final_models/waveform_sequence_mamba.pt``.

The new recordings are filtered with the same plausible negative-spike rules
used during training. Each recording is one prediction sample.
"""

import os
import re
import sys
from pathlib import Path

plot_cache = Path(os.environ.get("TMPDIR", "/tmp")) / "3dbrain_matplotlib_cache"
plot_cache.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(plot_cache))
os.environ.setdefault("XDG_CACHE_HOME", str(plot_cache))
os.environ.setdefault("NUMBA_CACHE_DIR", str(Path(os.environ.get("TMPDIR", "/tmp")) / "numba_cache"))

import joblib
import numpy as np
import pandas as pd

for parent in Path(__file__).resolve().parents:
    src_path = parent / "src"
    if (src_path / "brain_3d").is_dir():
        sys.path.insert(0, str(src_path))
        break

from brain_3d import Mamba


# =========================
# PARAMETERS TO EDIT
# =========================

MODEL_PATH = Path(
    "outputs/analisi_dati_Mark/experiment_8/"
    "recording_classification_negative_spikes_culture_phase_ml_campaign/"
    "final_models/activity_extra_trees_balanced.joblib"
)

# Use either a root containing many recording folders or one single recording.
NEW_SPIKE_ROOT = Path("/data/shared/3D_BRAIN/Mark/new_recordings/spikes")
SINGLE_RECORDING_DIR = None

OUTPUT_DIR = Path("outputs/analisi_dati_Mark/new_predictions")

RECORDING_DURATION_SECONDS = 180
MIN_ACTIVE_CHANNEL_FIRING_RATE_HZ = 0.25
EXCLUDE_REFERENCE_CHANNEL = True
REFERENCE_CHANNEL = 0

NEGATIVE_SPIKE_LABEL = 0
USE_DETECTOR_LABEL_FILTER = True
USE_WAVEFORM_POLARITY_FILTER = True
POLARITY_DOMINANCE_RATIO = 1.10
REQUIRE_EXTREMUM_NEAR_CENTER = True
CENTER_WINDOW_SAMPLES = 10

MAX_SPIKES_PER_RECORDING = 20000
RANDOM_STATE = 42


# =========================
# FEATURE EXTRACTION
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


def load_recording_waveforms(recording_dir, rng):
    active_waveforms = []
    active_channel_counts = []
    n_input_spikes = 0
    n_after_label_filter = 0
    n_after_waveform_filter = 0
    n_channels_with_negative_spikes = 0

    for spike_path in sorted(recording_dir.glob("spikes_ch*.npy")):
        channel_match = re.search(r"spikes_ch(\d+)\.npy$", spike_path.name)
        if channel_match is None:
            continue

        channel = int(channel_match.group(1))
        if EXCLUDE_REFERENCE_CHANNEL and channel == REFERENCE_CHANNEL:
            continue

        spikes = np.load(spike_path)
        if spikes.size == 0:
            continue
        spikes = np.asarray(spikes).reshape(spikes.shape[0], -1)

        label_path = spike_path.with_name(f"label_ch{channel}.npy")
        labels = np.load(label_path).reshape(-1) if label_path.exists() else np.full(spikes.shape[0], np.nan)

        n_spikes = min(spikes.shape[0], len(labels))
        spikes = spikes[:n_spikes]
        labels = labels[:n_spikes]
        n_input_spikes += n_spikes

        label_keep = labels == NEGATIVE_SPIKE_LABEL if np.isfinite(labels).any() else np.ones(n_spikes, bool)
        n_after_label_filter += int(label_keep.sum())
        keep = negative_waveform_mask(spikes, labels)
        n_after_waveform_filter += int(keep.sum())

        spikes = spikes[keep]
        if len(spikes) == 0:
            continue

        n_channels_with_negative_spikes += 1
        channel_firing_rate_hz = len(spikes) / float(RECORDING_DURATION_SECONDS)
        if channel_firing_rate_hz <= MIN_ACTIVE_CHANNEL_FIRING_RATE_HZ:
            continue

        active_waveforms.append(spikes)
        active_channel_counts.append(len(spikes))

    if not active_waveforms:
        raise ValueError("No active plausible negative-spike channels found.")

    waves_all = np.vstack(active_waveforms)
    sampled_waves = waves_all
    if len(sampled_waves) > MAX_SPIKES_PER_RECORDING:
        selected = rng.choice(len(sampled_waves), size=MAX_SPIKES_PER_RECORDING, replace=False)
        sampled_waves = sampled_waves[selected]

    channel_counts = np.asarray(active_channel_counts, dtype=float)
    total_active_negative_spikes = float(channel_counts.sum())
    probabilities = channel_counts / total_active_negative_spikes if total_active_negative_spikes else np.zeros_like(channel_counts)
    channel_entropy = float(-(probabilities * np.log2(probabilities + 1e-12)).sum())

    metadata = {
        "recording_name": recording_dir.name,
        "n_input_spikes": n_input_spikes,
        "n_after_label_filter": n_after_label_filter,
        "n_after_waveform_filter": n_after_waveform_filter,
        "total_spikes": total_active_negative_spikes,
        "sampled_spikes": len(sampled_waves),
        "duration_seconds": float(RECORDING_DURATION_SECONDS),
        "firing_rate_hz": total_active_negative_spikes / float(RECORDING_DURATION_SECONDS),
        "n_channels_with_spikes": int(n_channels_with_negative_spikes),
        "n_active_channels": int(len(channel_counts)),
        "mean_spikes_per_channel": float(np.mean(channel_counts)),
        "median_spikes_per_channel": float(np.median(channel_counts)),
        "max_spikes_per_channel": float(np.max(channel_counts)),
        "channel_entropy": channel_entropy,
        "waveform_mean": float(np.mean(sampled_waves)),
        "waveform_std": float(np.std(sampled_waves)),
        "waveform_min": float(np.min(sampled_waves)),
        "waveform_max": float(np.max(sampled_waves)),
        "waveform_ptp": float(np.ptp(sampled_waves)),
        "waveform_energy": float(np.mean(sampled_waves**2)),
    }
    return sampled_waves.astype(np.float32), metadata


def mamba_sequence(waveforms):
    return np.stack(
        [
            np.mean(waveforms, axis=0),
            np.std(waveforms, axis=0),
            np.median(waveforms, axis=0),
            np.percentile(waveforms, 25, axis=0),
            np.percentile(waveforms, 75, axis=0),
        ],
        axis=0,
    ).astype(np.float32)


def classical_feature_row(metadata, waveforms, bundle):
    values = {key: float(metadata[key]) for key in metadata if isinstance(metadata[key], (int, float))}
    extractor = bundle.get("rocket_extractor")
    if extractor is not None:
        rec_features = np.asarray(extractor.transform(waveforms[:, np.newaxis, :]))
        rec_mean = rec_features.mean(axis=0)
        rec_std = rec_features.std(axis=0)
        values.update({f"rocket_mean_{idx}": value for idx, value in enumerate(rec_mean)})
        values.update({f"rocket_std_{idx}": value for idx, value in enumerate(rec_std)})
    feature_columns = bundle["feature_columns"]
    row = pd.DataFrame([{column: values.get(column, 0.0) for column in feature_columns}])
    return row.replace([np.inf, -np.inf], np.nan).fillna(0)


def label_from_mapping(mapping, code):
    for key in (code, int(code) if isinstance(code, np.integer) else code, str(code)):
        if key in mapping:
            return mapping[key]
    return str(code)


def predict_classical(model_path, recording_dirs):
    bundle = joblib.load(model_path)
    rng = np.random.default_rng(RANDOM_STATE)
    rows = []
    for recording_dir in recording_dirs:
        waveforms, metadata = load_recording_waveforms(recording_dir, rng)
        features = classical_feature_row(metadata, waveforms, bundle)
        prediction_code = bundle["model"].predict(features.to_numpy())[0]
        prediction_label = label_from_mapping(bundle.get("name_by_code", {}), prediction_code)
        row = dict(metadata)
        row.update(
            {
                "model_path": str(model_path),
                "model_name": bundle.get("model_name", model_path.stem),
                "prediction_code": prediction_code,
                "prediction_label": prediction_label,
            }
        )
        if hasattr(bundle["model"], "predict_proba"):
            probabilities = bundle["model"].predict_proba(features.to_numpy())[0]
            for class_code, probability in zip(bundle["class_codes"], probabilities):
                class_label = label_from_mapping(bundle.get("name_by_code", {}), class_code)
                row[f"probability_{class_label}"] = float(probability)
        rows.append(row)
    return pd.DataFrame(rows)


def predict_mamba(model_path, recording_dirs):
    checkpoint = Mamba.torch.load(model_path, map_location="cpu", weights_only=False)
    config = checkpoint["model_config"]
    model = Mamba.MEAMambaClassifier(**config)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    rng = np.random.default_rng(RANDOM_STATE)
    rows = []
    for recording_dir in recording_dirs:
        waveforms, metadata = load_recording_waveforms(recording_dir, rng)
        sequence = mamba_sequence(waveforms)
        expected_shape = tuple(checkpoint["input_shape"])
        if tuple(sequence.shape) != expected_shape:
            raise ValueError(
                f"{recording_dir.name} has Mamba input shape {tuple(sequence.shape)}, "
                f"expected {expected_shape}."
            )

        mean = checkpoint["normalizer"]["mean"]
        std = checkpoint["normalizer"]["std"]
        x = ((sequence[np.newaxis, :, :] - mean) / std).astype(np.float32)
        with Mamba.torch.no_grad():
            logits = model(Mamba.torch.from_numpy(x))
        probabilities = Mamba.torch.softmax(logits, dim=1).numpy()[0]
        best_index = int(np.argmax(probabilities))
        prediction_code = checkpoint["classes"][best_index]
        prediction_label = label_from_mapping(checkpoint.get("name_by_code", {}), prediction_code)

        row = dict(metadata)
        row.update(
            {
                "model_path": str(model_path),
                "model_name": "waveform_sequence_mamba",
                "prediction_code": prediction_code,
                "prediction_label": prediction_label,
            }
        )
        for class_code, probability in zip(checkpoint["classes"], probabilities):
            class_label = label_from_mapping(checkpoint.get("name_by_code", {}), class_code)
            row[f"probability_{class_label}"] = float(probability)
        rows.append(row)
    return pd.DataFrame(rows)


def find_recording_dirs():
    if SINGLE_RECORDING_DIR is not None:
        recording_dir = Path(SINGLE_RECORDING_DIR)
        if not recording_dir.exists():
            raise FileNotFoundError(f"Recording folder not found: {recording_dir}")
        return [recording_dir]

    if not NEW_SPIKE_ROOT.exists():
        raise FileNotFoundError(f"New spike root not found: {NEW_SPIKE_ROOT}")
    recording_dirs = [
        path for path in sorted(NEW_SPIKE_ROOT.iterdir())
        if path.is_dir() and any(path.glob("spikes_ch*.npy"))
    ]
    if not recording_dirs:
        raise FileNotFoundError(f"No recording folders with spikes_ch*.npy found under {NEW_SPIKE_ROOT}")
    return recording_dirs


def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Saved model not found: {MODEL_PATH}")

    recording_dirs = find_recording_dirs()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Applying model: {MODEL_PATH}", flush=True)
    print(f"Recordings: {len(recording_dirs)}", flush=True)

    if MODEL_PATH.suffix == ".joblib":
        predictions = predict_classical(MODEL_PATH, recording_dirs)
    elif MODEL_PATH.suffix == ".pt":
        predictions = predict_mamba(MODEL_PATH, recording_dirs)
    else:
        raise ValueError("MODEL_PATH must point to a .joblib or .pt saved model.")

    out_csv = OUTPUT_DIR / f"predictions_{MODEL_PATH.stem}.csv"
    predictions.to_csv(out_csv, index=False)
    print(f"Predictions saved to: {out_csv}", flush=True)
    print(predictions[["recording_name", "model_name", "prediction_label"]].to_string(index=False))


if __name__ == "__main__":
    main()
