Usage Guide
===========

Module Responsibilities
-----------------------

``BrwFunctions``
   BRW file loading, raw-data decoding, downsampling, filtering, peak detection, data-frame conversion, and plotting helpers.

``BxrFunctions``
   BXR file loading and extraction of spikes, bursts, waveforms, false positives, and raster-style visualizations.

``SpikeSorting``
   Spike detection, template selection, template matching, cross-correlograms, and per-channel sorting helpers.

``Stratification``
   Distance metrics for waveforms, normalization, whitening, PCA/ICA/kernel PCA, hierarchical clustering, k-means, k-shape, Leiden clustering, recursive clustering, and Classification against centroids.

``Fcm``
   Standalone fuzzy C-means implementation.

``MergingTree``
   Binary tree representation for hierarchical merges, with NetworkX/Matplotlib visualization support.

``GanFunctions`` and ``VaeganFunctions``
   PyTorch dataset, encoder, decoder, generator, discriminator, and training components for MEA signal synthesis experiments.

Notes
-----

The package contains research-oriented functions with several optional heavy dependencies. For reproducible work, keep the exact Python environment with ``requirements.txt`` or ``environment.yml`` and document the input file formats, well IDs, sampling rate, and clustering parameters used in each analysis.
