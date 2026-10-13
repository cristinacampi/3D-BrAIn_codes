# Configurable Analysis Scripts

The programs in `main/` contain the analysis orchestration. They do not parse
command-line arguments. Before running a program, open it and edit its
`PARAMETERS TO EDIT` block (paths, sampling rate, duration, thresholds, output
folder, and recomputation flags).

Run a configured program without additional arguments:

```bash
python3 main/main_extract_traces_h5.py
```

## Processing Order

1. `main_extract_traces_h5.py` converts BRW recordings into
   `traces_h5/<recording>/channel_traces.h5`.
2. `main_detect_spikes_from_h5.py` reads those HDF5 traces and writes
   `spikes_ch*.npy`, `frames_ch*.npy`, and `label_ch*.npy`.
3. `main_compute_spike_burst_metrics.py` computes channel- and well-level
   metrics.
4. Run any required downstream analysis: stratification, connectivity,
   recording classification, or DIV regression.

`main_export_h5_to_channel_npy.py` is optional. Spike detection reads the HDF5
files directly and does not require per-channel trace exports.

## Complete reference

The detailed guide documents all eleven main scripts, their inputs, outputs,
editable parameters, package functions, execution relationships, and common
failures: [Main Script Reference](MAIN_SCRIPTS.md).

## Other Scripts

- `main_prepare_labeled_metrics.py` joins label tables to computed metrics.
- `main_apply_saved_negative_spike_classifier.py` applies a trained model to
  new processed spike folders.
- `main_run_mamba_algorithm_comparison.py` compares Mamba and classical models
  on a configured NPZ dataset.

For reproducibility, archive the edited main file, Git revision, environment,
labels, and outputs associated with each analysis run.
