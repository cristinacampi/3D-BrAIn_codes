# 3D-BrAIn: 3D Brain Activity In vitro Network

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.your-doi.svg)](https://doi.org/10.5281/zenodo.your-doi)

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
git clone https://github.com/your-repo/3D-BrAIn.git
cd 3D-BrAIn
pip install -e .
```

#### Using Docker
```bash
docker build -t 3d-brain:latest .
docker run -it -v $(pwd):/app 3d-brain:latest
```

## Project Structure

```
3D-BrAIn/
├── src/
│   └── tD_BrAIn/              # Main package
│       ├── __init__.py
│       ├── spikes_sorting.py          # Spike detection and sorting
│       ├── clustering.py              # Clustering algorithms
│       ├── merging_tree.py            # Hierarchical merging tree
│       ├── gan_functions.py           # GAN models and training
│       ├── vaegan_functions.py        # VAE-GAN implementation
│       ├── brw_functions.py           # BRW file I/O
│       ├── bxr_functions.py           # BXR file operations
│       ├── stratification.py          # Stratification algorithms
│       └── FCM.py                     # Fuzzy C-Means clustering
├── main/                      # Example scripts
│   ├── main_clustering_realdata.py
│   ├── main_gan.py
│   ├── main_vaegan.py
│   ├── main_extract_spike.py
│   ├── main_compute_spike_rates.py
│   └── plot_figure.py
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
make html
# Open _build/html/index.html
```

## Publish docs on GitHub Pages (Sphinx)

This repository includes an automated workflow at `.github/workflows/sphinx-gh-pages.yml`.

1. Push to the `main` branch.
2. In GitHub, open **Settings → Pages**.
3. Under **Build and deployment**, set **Source** to **GitHub Actions**.

After the workflow completes, docs are published at:

`https://<your-username>.github.io/<your-repository>/`



If you use 3D-BrAIn in your research, please cite:

```bibtex
@software{sacchi2024_3dbrain,
  author       = {},
  title        = {},
  year         = {},
  url          = {},
  doi          = {}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

