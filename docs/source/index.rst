3D-BrAIn: 3D Brain Activity In vitro Network
============================================

Welcome to the **3D-BrAIn** documentation! 

3D-BrAIn is a comprehensive Python package for advanced spike sorting, clustering, and deep learning-based signal synthesis for microelectrode array (MEA) recordings of neural networks.

.. image:: https://img.shields.io/badge/python-3.8+-blue.svg
   :target: https://www.python.org/downloads/
   :alt: Python 3.8+

.. image:: https://img.shields.io/badge/License-MIT-yellow.svg
   :target: https://opensource.org/licenses/MIT
   :alt: License: MIT

Key Features
============

- **Spike Detection & Sorting**: Multiple algorithms for detecting and clustering spike events
- **Advanced Clustering**: K-means, Hierarchical Clustering, Leiden graph-based clustering
- **Distance Metrics**: DTW, WDTW, LCSS, EDR, Minkowski, correlation-based distances
- **Template Matching**: Pearson correlation-based template matching with greedy matching
- **GAN-based Synthesis**: VAE-GAN architecture for generating synthetic neural signals
- **Comprehensive Analysis**: Burst detection, network metrics, cross-correlations
- **Parallel Processing**: Multi-threaded and GPU-accelerated computations
- **File I/O**: Support for BRW (3Brain), BXR formats, HDF5, and CSV

Contents
========

.. toctree::
   :maxdepth: 2
   :caption: Getting Started:

   installation
   getting_started

.. toctree::
   :maxdepth: 2
   :caption: User Guide:

   usage

.. toctree::
   :maxdepth: 2
   :caption: API Reference:

   modules

.. toctree::
   :maxdepth: 1
   :caption: Additional Resources:

   GitHub Repository <https://github.com/cristinacampi/3D-BrAIn_codes>
   License <https://github.com/cristinacampi/3D-BrAIn_codes/blob/master/LICENSE>


