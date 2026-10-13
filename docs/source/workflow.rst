Data and Workflow
=================

Processed Data Layout
---------------------

Each immediate subdirectory of a spike root represents one recording and
contains matched arrays for each channel:

.. code-block:: text

   spikes/recording_A/
   |-- spikes_ch1.npy   # events x waveform samples
   |-- frames_ch1.npy   # event frame indexes
   `-- label_ch1.npy    # 0 negative, 1 positive

Channel ``0`` is excluded as the reference by the negative-spike workflows. A
64 x 64 array therefore has 4095 candidate recording channels before filtering.

Labels Table
------------

Supervised workflows use one labels CSV row per recording. Required columns
include ``filename``, ``biological_unit_id``, ``chip_ID``, ``DIV``, and
``Days_in_BF``. Classification additionally needs numeric class codes and
human-readable labels. ``experiment_id`` is required only when the selected
main configuration filters by experiment.

Repeated measurements from one organoid must share the same
``biological_unit_id`` so they remain in one cross-validation partition.

Public Workflow
---------------

1. Extract BRW traces with ``main_extract_traces_h5.py``.
2. Detect channel-wise events with ``main_detect_spikes_from_h5.py``.
3. Compute metrics with ``main_compute_spike_burst_metrics.py``.
4. Run stratification, connectivity, classification, or DIV regression.
5. Inspect quality-control tables before interpreting figures or metrics.
6. Archive the command, Git revision, environment, labels, and parameter files.

Edit the parameter block in each main before running it. Keep unrelated
experimental campaigns in separate runs unless pooling is part of the analysis
design.

The complete description of all executable scripts is maintained in
``docs/MAIN_SCRIPTS.md``.
