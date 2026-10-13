# Data Format

This page defines the data contracts used by the public analysis commands.
Paths are examples; no public command contains project-specific absolute paths.

## Raw Recordings

`main_extract_traces_h5.py` expects a directory containing 3Brain `.brw`
recordings. The well identifier defaults to `Well_A1`. Recording metadata,
including sampling rate, channel count, and frame count, are read from the BRW
file and retained in the extracted HDF5 representation.

## Processed Spike Folders

Each immediate subdirectory of a spike root represents one recording:

```text
spike_root/
├── recording_A/
│   ├── spikes_ch0.npy
│   ├── frames_ch0.npy
│   ├── label_ch0.npy
│   ├── spikes_ch1.npy
│   ├── frames_ch1.npy
│   └── label_ch1.npy
└── recording_B/
    └── ...
```

For channel `N`:

| File | Shape | Meaning |
|---|---:|---|
| `spikes_chN.npy` | events x samples | Extracted waveforms centred on detected events. |
| `frames_chN.npy` | events | Event positions in acquisition frames. |
| `label_chN.npy` | events | Detector polarity code: `0` negative, `1` positive. |

The three arrays are truncated to their shortest common event count if lengths
differ. This prevents index errors but should be treated as a data-quality
warning. All recordings used together in Mamba classification must have the
same waveform length.

Channel `0` is excluded by default in the negative-spike analyses because it is
the reference. A 64 x 64 grid therefore has 4095 candidate recording channels.
The channel-to-grid convention is:

```text
row = channel // 64
column = channel % 64
```

Confirm that this zero-based convention matches the exported data before
interpreting physical maps.

## Labels CSV

Classification and regression expect one row per recording. Recording names
are matched case-insensitively after removing a terminal `.brw` extension and
whitespace.

Required shared columns:

| Column | Type | Purpose |
|---|---|---|
| `filename` | string | Recording filename or folder name. |
| `biological_unit_id` | string/int | Organoid, culture, subject, or other unit that must not cross folds. |
| `DIV` | numeric | Days in vitro; required by DIV regression and current recording metadata. |
| `chip_ID` | integer | Chip identifier used in metadata and sorting. |
| `Days_in_BF` | integer | Days in bioreactor/flask metadata used by current pipelines. |

Classification additionally requires a numeric target column and a readable
label column. Built-in target presets are:

| `--target` | Numeric target | Readable target | Default classes |
|---|---|---|---|
| `culture_phase` | `culture_phase_code` | `culture_phase` | `0 1` |
| `div` | `div_code` | `div_label` | all eligible classes |

Custom targets require both `--target-column` and `--target-label-column`.
`experiment_id` is needed only when the selected main configuration filters
the labels by experiment.

Example:

```csv
filename,experiment_id,biological_unit_id,chip_ID,DIV,Days_in_BF,culture_phase_code,culture_phase
organoid01_DIV033.brw,4,organoid01,123,33,12,0,early
organoid01_DIV047.brw,4,organoid01,123,47,26,1,late
organoid02_DIV033.brw,4,organoid02,124,33,12,0,early
```

Repeated measurements of the same organoid must share the same
`biological_unit_id`. This keeps all conditions or DIV values from that
organoid together during cross-validation.

## Generic Mamba NPZ

`main_run_mamba_algorithm_comparison.py` accepts a NumPy archive with:

| Array | Required | Shape | Meaning |
|---|---|---:|---|
| `X` | yes | samples x channels x time | Sequence input. |
| `y` | yes | samples | Class codes. |
| `groups` | no | samples | Group IDs for leakage-resistant folds. |

Always provide `groups` when multiple samples originate from one recording,
well, organoid, culture, animal, or participant.

## Validation Before Analysis

Check the following before a long run:

```bash
# Count recording folders.
find /data/experiment/spikes -mindepth 1 -maxdepth 1 -type d | wc -l

# Count waveform arrays in one recording.
find /data/experiment/spikes/recording_A -name 'spikes_ch*.npy' | wc -l

# Inspect the parameter block before running the analysis.
sed -n '1,120p' main/main_run_negative_spike_stratification_analysis.py
```

Record the actual sampling rate and duration. Defaults are conveniences, not a
substitute for recording metadata.
