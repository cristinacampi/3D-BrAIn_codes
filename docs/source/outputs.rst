Output Reference
================

Metrics
-------

Metrics outputs include channel and well CSV tables plus a complete pickle
artifact for each recording. The metrics root also receives an all-recordings
well summary.

Stratification
--------------

Each recording receives run parameters, assignments, centroid means and
standard deviations, PCA plots, waveform-count maps, and within-channel cluster
percentage maps. The transversal folder also provides per-recording cluster
counts and row-normalised profiles.

Supervised Models
-----------------

Classification and regression save recording metadata, fold metrics,
held-out predictions, ranked summaries, diagnostic figures, and reusable final
models. ``final_models/`` contains model bundles, a training-recording manifest,
a model manifest, and JSON provenance metadata.

Connectivity
------------

Each recording receives correlation and STTC matrices, edge tables, node
strength, firing-rate and filter summaries, binned counts, parameter tables, and
sensor-grid network maps. ``connectivity_summary.csv`` summarises recordings.

Generated outputs are ignored by Git. Archive release outputs separately with
the code revision and exact command. See ``docs/OUTPUT_REFERENCE.md`` for the
file-by-file reference.
