# Reproducibility and Publication Checklist

Use this checklist for each analysis campaign and before a release, manuscript,
or data deposit.

## Dataset Provenance

- Assign a stable dataset and experiment identifier.
- Record culture, organoid, chip, well, condition, DIV, and acquisition date.
- Preserve the raw BRW checksum and the labels-file checksum.
- State inclusion and exclusion criteria for recordings and biological units.
- Keep distinct experiments separate unless pooling is scientifically planned.
- Describe repeated measurements and define `biological_unit_id` accordingly.

## Acquisition and Preprocessing

- Record hardware, plate geometry, sampling rate, duration, and voltage units.
- Report notch, band-pass, detection threshold, and waveform length.
- Report whether channel `0` is the reference and confirm grid indexing.
- Report counts before and after detector-label, waveform, and active-channel
  filters.
- Record the active-channel threshold and whether it is applied after polarity
  filtering.
- Report random seeds and spike-sampling caps.

## Statistical Analysis

- Define the statistical sample: spike, channel, recording, or biological unit.
- Report all candidate clusters/models and model-selection criteria.
- For clustering, report whitening, PCA variance, distance, candidate `k`, and
  per-recording versus transversal scope.
- For classification/regression, confirm group-disjoint folds and report class,
  DIV, recording, and biological-unit distributions.
- Fit PCA, MiniRocket, scalers, and other learned transforms inside training
  folds.
- Compare against a dummy baseline and report variability across folds/repeats.
- Distinguish cross-validated test folds from an independent external test set.
- For connectivity, report bin/window sizes, thresholds, sign rule, and network
  map edge selection.
- Perform sensitivity analyses before interpreting a single connectivity scale
  or threshold as robust.

## Computational Record

Run from a clean, committed revision and save:

```bash
git rev-parse HEAD
python --version
python -m pip freeze
```

Archive the exact command, standard output/error log, labels CSV, parameter CSV
or JSON, and model metadata. Avoid replacing old results in place; use a new
versioned output directory when scientific settings change.

## Quality Control

- Review `missing_recordings.csv` and excluded-class tables.
- Compare expected, present, active, and retained channel counts.
- Inspect representative waveforms and centroid variability.
- Check cluster proportions for sampling artifacts or polarity separation.
- Inspect train-versus-test metrics and confusion matrices for overfitting.
- Confirm that all folds contain valid classes/groups.
- Compare regression errors with the target range and dummy baseline.
- Inspect connectivity matrices for NaN values, saturation, and edge density.
- Verify that legends, labels, and colour bars do not overlap plotted data.

## Publication Record

- Cite the software release and all algorithmic references.
- State the software version and commit hash.
- Provide a machine-readable input manifest and non-sensitive labels schema.
- Deposit generated tables and models only when data governance permits it.
- Label exploratory analyses and unvalidated models explicitly.
- State limitations, missing data, parameter sensitivity, and external
  validation status.

## Release Checklist

- Update `CHANGELOG.md`, package version, Sphinx version, and `CITATION.cff`.
- Run `python3 -m pytest -q`.
- Build Sphinx with warnings treated as errors.
- Build and visually inspect the printable PDF manual.
- Create a signed or annotated Git tag.
- Attach the PDF manual and checksums as release assets.
