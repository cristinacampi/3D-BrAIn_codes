# Output Reference

Generated outputs are ignored by Git because they may be large, derived from
restricted data, or specific to one run. Archive final outputs separately with
the exact code revision, command, environment, and input manifest.

## Spike and Burst Metrics

For each recording, `<metrics-root>/<recording>/spike_burst/` contains:

| File | Contents |
|---|---|
| `channel_spike_burst_metrics.csv` | One row per channel: activity, spike, ISI, amplitude, burst, and IBI metrics. |
| `well_spike_burst_metrics.csv` | One recording-level row with well summaries. |
| `full_spike_burst_metrics.pkl` | Complete Python objects used to derive the tables. Load only from trusted sources. |

`all_recordings_spike_burst_summary.csv` concatenates well-level rows from the
recordings processed in that invocation.

## Stratification

The output root contains `per_recording_summary.csv` and one subdirectory per
experiment identifier. The identifier is inferred from a `NN-NN` token in the
spike-root path; otherwise `unknown_experiment` is used.

Each `per_recording/<recording>/` directory contains:

| File | Contents |
|---|---|
| `run_info.csv` | Parameters, filter counts, active channels, selected cluster count, and runtime. |
| `spike_cluster_assignments.csv` | Recording, channel, event, frame, PC1/PC2, and assigned cluster. |
| `cluster_centroid_waveforms.csv` | Mean waveform for each cluster. |
| `cluster_centroid_waveform_std.csv` | Pointwise standard deviation for each cluster. |
| `pca_by_cluster.png` | Two-component visualisation coloured by cluster. |
| `cluster_centroid_waveforms.png` | Mean cluster waveforms with one-SD bands. |
| `channel_maps/waveforms_number.*` | Number of analysed waveforms on the 64 x 64 grid. |
| `channel_maps/cluster_*_waveform_percentage_by_channel.*` | Within-channel percentage assigned to each cluster. |

`transversal_global_spike_clusters/` contains the corresponding pooled
analysis. `recording_global_cluster_counts.csv` gives counts and
`recording_global_cluster_profile.csv` gives row-normalised proportions.

## Classification

| File or directory | Contents |
|---|---|
| `recording_metadata_and_activity_features.csv` | Included recordings, labels, filter counts, activity, and waveform summaries. |
| `missing_recordings.csv` | Labelled recordings not found under the spike root, when present. |
| `target_classes_excluded_too_few_groups.csv` | Classes removed for insufficient independent groups, when present. |
| `cv_metrics.csv` | One row per repeat, fold, feature set, and model. |
| `cv_recording_predictions.csv` | Held-out prediction for every recording/fold occurrence. |
| `summary_metrics.csv` | Aggregated cross-validation metrics. |
| `algorithm_comparison.csv` | Models ranked by mean test balanced accuracy and accuracy. |
| `confusion_matrix_<model>.csv/.png` | Aggregated held-out confusion matrix. |
| `classification_report_<model>.csv` | Per-class precision, recall, F1, and support. |
| `cv_fold_models/` | Optional fold-specific model bundles. |
| `final_models/` | Reusable models refitted on all eligible labelled recordings. |

`final_models/` includes `training_recordings.csv`, `model_manifest.csv`, model
bundles (`.joblib` or `.pt`), and JSON metadata with the dataset, feature set,
preprocessing, validation, estimator parameters, and train/test CV metrics.

## DIV Regression

Regression uses the same recording metadata, CV metric, prediction, summary,
comparison, and `final_models/` pattern. Key figures are
`best_model_true_vs_predicted_div.png` and `model_ranking_mae.png`. Interpret
errors in days and compare every estimator with `dummy_mean`.

## Connectivity

Each recording output contains:

| File | Contents |
|---|---|
| `spike_count_correlation_matrix.csv` | Pairwise spike-count correlations. |
| `sttc_matrix.csv` | Pairwise STTC values. |
| `correlation_edges.csv`, `sttc_edges.csv` | Threshold-passing edge lists. |
| `correlation_node_strength.csv`, `sttc_node_strength.csv` | Full-matrix node strength. |
| `spike_filter_summary_by_channel.csv` | Counts through detector, waveform, and activity filters for the selected polarity. |
| `binned_spike_counts.npy`, `bin_edges_seconds.npy` | Correlation input counts and bin edges. |

`network_maps/` stores matrix heatmaps, global node-strength grids,
strongest-edge sensor maps, and selected edge lists. `connectivity_summary.csv`
provides one row per recording at the output root.

## Saved-Model Predictions

Model application writes `predictions_<model-name>.csv`. Depending on the model,
the table includes recording name, predicted code and readable class, input and
retained spike counts, active-channel count, and class probabilities.

## Completion and Reuse Rules

Skip and recomputation behavior is controlled by uppercase values inside each
main, such as `SKIP_EXISTING`, `FORCE_REPROCESS`, or
`REBUILD_H5_IF_EXISTS`. The scripts do not accept `--force` or `--maps-only`
command-line options. Recompute only when inputs, scientific defaults, or code
changed.
