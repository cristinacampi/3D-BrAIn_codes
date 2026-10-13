# Saved Models and New Data

## Training Artifacts

Recording-level classification and DIV regression rank models by grouped
cross-validation and then refit selected models on all eligible labelled
recordings. Reusable bundles are written under the analysis directory's
`final_models/` folder.

For every algorithm, retain together:

- the `.joblib` or `.pt` model file;
- its JSON metadata;
- `model_manifest.csv`;
- `training_recordings.csv`;
- `dataset_metadata.json` when present;
- the cross-validation tables and exact code revision.

The metadata records the training dataset, biological units, label mapping,
feature set, estimator parameters, plausible-negative filter, MiniRocket/PCA
settings, and training/test cross-validation metrics. The final model is fitted
on all available labelled recordings, so it has no internal holdout result of
its own.

## Apply a Classifier

Set `MODEL_PATH`, `NEW_SPIKE_ROOT`, and `OUTPUT_DIR` in
`main_apply_saved_negative_spike_classifier.py`, then run:

```bash
python3 main/main_apply_saved_negative_spike_classifier.py
```

To apply it to one recording, set `SINGLE_RECORDING_DIR` instead of
`NEW_SPIKE_ROOT`, then run the same script:

```bash
python3 main/main_apply_saved_negative_spike_classifier.py
```

The application code loads raw processed spike arrays, repeats the saved
negative-waveform and active-channel preprocessing, extracts the required
recording-level activity, waveform, and MiniRocket features, and applies the
estimator. You do not need to create a feature CSV manually.

## Compatibility Checklist

Before prediction, verify:

- the new detector uses the same label coding (`0` negative);
- waveform sample length and centring match training;
- sampling rate and recording duration are correct;
- channel `0` has the same reference meaning;
- voltage scale and upstream filters are comparable;
- the model's class vocabulary is meaningful for the new experiment;
- the new biological domain is covered by training data.

A technically successful prediction is not evidence of external validity.
Evaluate performance on a labelled external cohort before using predictions as
scientific conclusions.

## Model Catalogue

The metadata and manifest are designed for ingestion into an external model
catalogue. `tools/upload_models_to_portal.py` handles portal upload. Keep portal
credentials in `tools/.env.portal-upload`, which is ignored by Git, and inspect
the generated catalogue records before publishing them.

## Trusted Files

Joblib, pickle, and PyTorch model files can execute code during loading. Load
only artifacts produced by a trusted pipeline and verify their provenance and
checksum after transfer.
