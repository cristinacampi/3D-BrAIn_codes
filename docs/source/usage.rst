Usage Guide
===========

Module Responsibilities
-----------------------

``brw_functions``
   BRW file loading, raw-data decoding, downsampling, filtering, peak detection, data-frame conversion, and plotting helpers.

``bxr_functions``
   BXR file loading and extraction of spikes, bursts, waveforms, false positives, and raster-style visualizations.

``spike_sorting``
   Spike detection, template selection, template matching, cross-correlograms, and per-channel sorting helpers.

``stratification``
   Distance metrics for waveforms, normalization, whitening, PCA/ICA/kernel PCA, hierarchical clustering, k-means, k-shape, Leiden clustering, recursive clustering, and classification against centroids.

``FCM``
   Standalone fuzzy C-means implementation.

``merging_tree``
   Binary tree representation for hierarchical merges, with NetworkX/Matplotlib visualization support.

``gan_functions`` and ``vaegan_functions``
   PyTorch dataset, encoder, decoder, generator, discriminator, and training components for MEA signal synthesis experiments.

Notes
-----

The package contains research-oriented functions with several optional heavy dependencies. For reproducible work, keep the exact Python environment with ``requirements.txt`` or ``environment.yml`` and document the input file formats, well IDs, sampling rate, and clustering parameters used in each analysis.
