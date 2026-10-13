# Main Script Reference

The eleven programs in `main/` are the executable entry points of 3D-BrAIn.
They deliberately do not use `argparse`: edit the uppercase configuration
values near the beginning of a script and then run it with `python3`.

Reusable calculations, file readers, feature builders, models, and plotting
functions live in `src/brain_3d/`. The main scripts select paths and parameters,
validate run conditions, choose analysis branches, and call those functions.

## Recommended execution order

```text
BRW recordings
  |
  +-- 1. main_extract_traces_h5.py
  |       -> traces_h5/<recording>/channel_traces.h5
  |
  +-- 2. main_detect_spikes_from_h5.py
  |       -> spikes/<recording>/{spikes,frames,label}_ch*.npy
  |
  +-- 3a. main_compute_spike_burst_metrics.py
  |       +-- 4a. main_prepare_labeled_metrics.py
  |
  +-- 3b. main_run_negative_spike_stratification_analysis.py
  +-- 3c. main_run_negative_spike_connectivity_analysis.py
  +-- 3d. main_run_negative_spike_recording_classification.py
  |       +-- 4b. main_apply_saved_negative_spike_classifier.py
  +-- 3e. main_run_negative_spike_div_regression_analysis.py
```

`main_export_h5_to_channel_npy.py` is optional after step 1. Spike detection
reads `channel_traces.h5` directly. `main_run_mamba_algorithm_comparison.py` is
an independent benchmark that starts from a prepared NPZ dataset.

## Before every server run

1. Activate the project environment and work from the repository root.
2. Open the selected main and edit its uppercase paths and parameters.
3. Check sampling rate, duration, well identifier, reference channel, channel
   range, skip/rebuild flags, label columns, and biological grouping.
4. Run the script without command-line options.
5. Archive the edited main, Git revision, environment, labels, and outputs.

Paths are interpreted relative to the current working directory unless an
absolute `Path` is supplied. Do not combine experiments merely because they
have the same directory structure.

## 1. Extract BRW traces to HDF5

```bash
python3 main/main_extract_traces_h5.py
```

**Input:** `*.brw` files directly below `RECORDING_DIR`.

**Output:** `traces_h5/<recording>/channel_traces.h5`, containing
`<WELL_ID>/Traces`, `ChannelIndices`, and recording attributes.

Important settings are `WELL_ID`, `START_TIME`, `READ_CHUNK_DURATION`,
`TRACE_DTYPE`, `H5_COMPRESSION`, `CHANNEL_START`, and `CHANNEL_END`.
`REBUILD_H5_IF_EXISTS` permits replacement; use it carefully. The script calls
`BrwFunctions.ReadBRW` and `BrwFunctions.ReadingRawData`. Both event-based and
legacy raw-data layouts are selected inside the reusable reader.

## 2. Export HDF5 traces by channel (optional)

```bash
python3 main/main_export_h5_to_channel_npy.py
```

**Input:** `TRACE_ROOT/*/channel_traces.h5`.

**Output:** `CHANNEL_ROOT/<recording>/ch<channel>.npy`.

Use this only when another workflow needs one complete trace per file.
`OUTPUT_DTYPE=None` preserves the HDF5 type. `MAX_WORKERS` controls parallel
exports and should be chosen for the server capacity. Each export calls
`BrwFunctions.ExportH5Channel`.

## 3. Detect spikes from HDF5

```bash
python3 main/main_detect_spikes_from_h5.py
```

**Input:** `TRACE_ROOT/<recording>/channel_traces.h5`.

**Output:** for each channel, `spikes_ch*.npy` (waveforms), `frames_ch*.npy`
(frame indexes), and `label_ch*.npy` (`0` negative, `1` positive).

The main configures the notch and band-pass frequencies, waveform length
`IDX_NUMBER`, detection threshold, channel range, worker count, and skip rule.
Each channel is processed and saved by `SpikeSorting.ProcessH5Channel`, which
uses `SpikeSorting.DetectSpikesInTrace`.

## 4. Compute spike and burst metrics

```bash
python3 main/main_compute_spike_burst_metrics.py
```

**Input:** processed spike folders plus HDF5 or BRW metadata.

**Output:** per-recording channel/well CSV files and a complete pickle under
`METRIC_ROOT/<recording>/spike_burst/`; a combined well summary is written to
`METRIC_ROOT/all_recordings_spike_burst_summary.csv`.

`ACTIVE_ELECTRODE_THRESHOLD`, `N_MIN_SPIKES_BURST`, and
`ISI_MAX_SECONDS_BURST` define activity and burst criteria. The script uses
`BrwFunctions.ReadRecordingMetadata`, `SplitChannelSpikes`,
`SpikesMetric`, `BurstsMetric`, and `SaveSpikeBurstMetrics`.

## 5. Join labels and metrics

```bash
python3 main/main_prepare_labeled_metrics.py
```

**Input:** `DEFAULT_LABELS` and one or more `DEFAULT_METRICS_ROOTS`.

**Output:** `labeled_well_metrics.csv`, `feature_matrix_well.csv`,
`missing_well_metrics.csv`, per-experiment feature matrices, and optionally
`labeled_channel_metrics.csv`.

Set `INCLUDE_CHANNEL_METRICS=True` only when the larger channel-level table is
needed. Recording-name normalization and table assembly use generic functions
in `Classification`.

## 6. Stratify spike waveforms

```bash
python3 main/main_run_negative_spike_stratification_analysis.py
```

**Input:** one or more roots containing processed spike recordings.

**Output:** per-recording and transversal clustering tables, PCA plots,
centroid waveforms, run summaries, and optional 64 x 64 channel maps.

The polarity mode may be `negative`, `positive`, or `none`. The default
negative filter combines detector label, waveform dominance, centre-window,
reference-channel, and active-rate criteria. `MAX_CLASSES`, PCA variance,
sampling limits, recomputation flags, and map flags control the analysis.
Spike loading/filtering comes from `SpikeSorting`; clustering functions come
from `Stratification`.

## 7. Compute functional connectivity

```bash
python3 main/main_run_negative_spike_connectivity_analysis.py
```

**Input:** processed spike folders.

**Output:** correlation and STTC matrices, thresholded edge tables, node
strengths, filter summaries, binned counts, network maps, and
`connectivity_summary.csv`.

Check `SAMPLING_RATE_HZ`, `RECORDING_DURATION_SECONDS`, correlation bin size,
STTC coincidence window, thresholds, excluded channels, active-rate threshold,
network-map selection, and `POLARITY_FILTER` (`negative`, `positive`, or
`none`). Waveforms are filtered with `SpikeSorting.WaveformPolarityMask`;
matrices, edges, strengths, and plots are computed by `Connectivity`.

## 8. Train recording classifiers

```bash
python3 main/main_run_negative_spike_recording_classification.py
```

**Input:** `SPIKE_ROOT` and `LABELS_CSV` with the configured target and
`GROUP_COLUMN`.

**Output:** recording features, missing/excluded records, fold predictions,
cross-validation metrics, algorithm rankings, confusion matrices, reports,
and optional reusable models with JSON metadata.

Select the experiment and target only in the main configuration. Repeated
measurements sharing a biological unit must have the same group identifier so
they cannot cross folds. Classical models combine activity, waveform summary,
and optional MiniRocket features. Mamba is controlled by `RUN_MAMBA` and the
`MAMBA_*` settings. `FORCE_REPROCESS=True` recomputes an existing result.
Reusable feature/model/metadata functions live in `Classification`; the
sequence model lives in `Mamba`.

## 9. Apply a saved classifier

```bash
python3 main/main_apply_saved_negative_spike_classifier.py
```

**Input:** a `.joblib` classical bundle or `.pt` Mamba checkpoint and either a
single processed recording or a root containing several recordings.

**Output:** `predictions_<model-name>.csv` with labels, codes, filter counts,
recording summaries, and probabilities when supported.

The filter and activity parameters must match training. The main explicitly
selects the `.joblib` or `.pt` branch; reusable loading, feature construction,
and prediction functions are in `Classification`.

## 10. Train DIV regressors

```bash
python3 main/main_run_negative_spike_div_regression_analysis.py
```

**Input:** processed spikes and labels containing continuous `TARGET_COLUMN`
(normally `DIV`) and `GROUP_COLUMN`.

**Output:** recording features, grouped CV predictions and metrics, regression
plots, rankings, and optional final `.joblib` models with metadata.

Configure experiment selection, filter parameters, MiniRocket/PCA settings,
grouped folds, DIV stratification bins, and model-saving policy in the main.
The reusable workflow is called through `Regression.RunDIVRegression` with all
configuration values passed explicitly.

## 11. Compare Mamba and classical algorithms

```bash
python3 main/main_run_mamba_algorithm_comparison.py
```

**Input:** an NPZ file containing `X`, `y`, and optionally `groups`.

**Output:** fold metrics, summaries, predictions, confusion matrices,
classification reports, and a balanced-accuracy comparison plot.

Set dataset/output paths, folds, epochs, seed, test fraction, tree count, and
Mamba backend. The comparison is performed by `Mamba.CompareMambaToClassical`.

## Failure checks

- `NameError` during BRW decoding: record the complete traceback and BRW layout;
  do not rename variables blindly because old and event-based formats differ.
- No recordings found: verify the current directory and configured roots.
- No active channels: verify duration, sampling rate, polarity labels, reference
  channel, and active-rate threshold.
- Too few groups/classes: correct the labels and biological grouping rather
  than allowing related recordings into different folds.
- Existing outputs skipped: inspect the relevant skip/force/rebuild setting.
- Out-of-memory or excessive processes: reduce channel range, sampling limits,
  `MAX_WORKERS`, MiniRocket kernels, or Mamba batch size.
