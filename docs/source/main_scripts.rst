Main Scripts
============

All executable analyses are configured by editing uppercase values near the
start of the selected file and running it without command-line options.
Reusable calculations live in ``src/brain_3d``; the scripts in ``main`` retain
paths, run conditions, branching, and orchestration.

Recommended Order
-----------------

1. ``main_extract_traces_h5.py``
2. ``main_detect_spikes_from_h5.py``
3. ``main_compute_spike_burst_metrics.py``
4. One or more downstream analyses: stratification, connectivity,
   classification, or DIV regression.

``main_export_h5_to_channel_npy.py`` is optional. Label/metric joining follows
metric computation. Saved-model application follows classification training.
The Mamba comparison script is an independent NPZ benchmark.

Complete Reference
------------------

The repository guide ``docs/MAIN_SCRIPTS.md`` documents all eleven scripts,
including inputs, outputs, parameters, module calls, execution relationships,
and common server failures.

Run Pattern
-----------

.. code-block:: bash

   # Edit the uppercase configuration values first.
   python3 main/main_extract_traces_h5.py

Do not append CLI options: the current public mains intentionally contain no
argument parser.
