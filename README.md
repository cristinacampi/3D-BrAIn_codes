# 3D-BrAIn

3D-BrAIn is a Python package for working with 3Brain MEA recordings. It includes utilities for BRW/BXR file access, spike and burst analysis, spike sorting, clustering, merging-tree visualization, and GAN/VAE-GAN signal synthesis experiments.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## Main Modules

- `brain_3d.BrwFunctions`: read BRW recordings, decode raw data, filter traces, detect peaks, and convert recordings to data frames.
- `brain_3d.BxrFunctions`: read BXR spike, burst, waveform, and false-positive outputs.
- `brain_3d.SpikeSorting`: spike detection, template extraction, template matching, cross-correlograms, and channel-level sorting helpers.
- `brain_3d.Stratification`: distance metrics, normalization, dimensionality reduction, hierarchical clustering, k-means, k-shape, Leiden clustering, and recursive clustering.
- `brain_3d.Fcm`: fuzzy C-means clustering.
- `brain_3d.MergingTree`: community merging tree construction and visualization.
- `brain_3d.GanFunctions` and `brain_3d.VaeganFunctions`: PyTorch model components and training utilities for signal synthesis.

## Installation

Use Python 3.8 or newer.

```bash
git clone https://github.com/cristinacampi/3D-BrAIn_codes.git
cd 3D-BrAIn_codes
python3 -m pip install -e .
```

For documentation only:

```bash
python3 -m pip install -e ".[docs]"
```

For development work:

```bash
python3 -m pip install -e ".[dev]"
```

GPU support depends on a PyTorch build compatible with your CUDA installation.


## Documentation

The Sphinx documentation lives in `docs/source`.

Build it locally with:

```bash
cd docs
python3 -m sphinx -b html source build/html
```

Open `docs/build/html/index.html` after the build completes.

## Docker

Build the development image and open a shell:

```bash
docker compose build brain-3d
docker compose run --rm brain-3d
```

Run Jupyter locally:

```bash
docker compose up jupyter
```

Open `http://127.0.0.1:8888` and use the default token `brain`, or set a custom token:

```bash
JUPYTER_TOKEN=my-token docker compose up jupyter
```

Build the Sphinx documentation in Docker:

```bash
docker compose run --rm docs
```

## Repository Layout

```text
src/brain_3d/       Package source code
docs/source/        Sphinx documentation source
requirements.txt    Runtime dependencies
environment.yml     Conda environment
setup.py            Package metadata
```

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE).
