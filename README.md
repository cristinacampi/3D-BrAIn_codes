# 3D-BrAIn: 3D Brain Activity In vitro Network

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Docs](https://img.shields.io/badge/docs-GitHub%20Pages-blue)](https://cristinacampi.github.io/3D-BrAIn_codes/)

Advanced spike sorting, clustering, and deep learning-based signal synthesis for microelectrode array (MEA) recordings of neural networks.

## Features

- **Spike Detection & Sorting**: Multiple algorithms for detecting and clustering spike events
- **Advanced Clustering**: K-means, Hierarchical Clustering, Leiden graph-based clustering
- **Distance Metrics**: DTW, WDTW, LCSS, EDR, Minkowski, correlation-based distances
- **Template Matching**: Pearson correlation-based template matching with greedy matching
- **GAN-based Synthesis**: VAE-GAN architecture for generating synthetic neural signals
- **Comprehensive Analysis**: Burst detection, network metrics, cross-correlations
- **Parallel Processing**: Multi-threaded and GPU-accelerated computations
- **File I/O**: Support for BRW (3Brain), BXR formats, HDF5, and CSV

## Quick Start

### Installation

#### Using pip
```bash
pip install 3D-BrAIn
```

#### Using conda
```bash
conda env create -f environment.yml
conda activate 3d-brain
```

#### From source
```bash
git clone https://github.com/cristinacampi/3D-BrAIn_codes.git
cd 3D-BrAIn_codes
pip install -e .
```

#### Using Docker
```bash
docker build -t 3d-brain:latest .
docker run -it -v $(pwd):/app 3d-brain:latest
```

## Project Structure

```
3D-BrAIn_codes/
├── src/
│   └── brain_3d/              # Main package
│       ├── __init__.py
│       ├── spike_sorting.py           # Spike detection and sorting
│       ├── merging_tree.py            # Hierarchical merging tree
│       ├── gan_functions.py           # GAN models and training
│       ├── vaegan_functions.py        # VAE-GAN implementation
│       ├── brw_functions.py           # BRW file I/O
│       ├── bxr_functions.py           # BXR file operations
│       ├── stratification.py          # Stratification algorithms
│       └── FCM.py                     # Fuzzy C-Means clustering
├── docs/                      # Sphinx documentation
│   ├── source/
│   └── build/
├── requirements.txt           # Python dependencies
├── environment.yml            # Conda environment
├── setup.py                   # Package installation
├── Dockerfile                 # Docker configuration
├── docker-compose.yml         # Docker Compose setup
└── README.md                  # This file
```


GPU acceleration available with CUDA.

Build locally:
```bash
cd docs
../.venv/bin/python -m sphinx -b html source build/html
# Open build/html/index.html
```

## Publish docs on GitHub Pages (Sphinx)

This repository includes an automated workflow at `.github/workflows/sphinx-gh-pages.yml`.

1. Push to the `main` branch.
2. In GitHub, open **Settings → Pages**.
3. Under **Build and deployment**, set **Source** to **GitHub Actions**.

After the workflow completes, docs are published at:

`https://cristinacampi.github.io/3D-BrAIn_codes/`


## License

This project is licensed under the MIT License - see the LICENSE file for details.

