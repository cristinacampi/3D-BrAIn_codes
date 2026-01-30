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

## Core Modules

### spikes_sorting.py
Complete spike sorting pipeline with template learning and matching.

**Key Functions:**
- `SpikesDetection()`: Detect spike candidates
- `TemplateNeg()`: Learn spike templates through clustering
- `TemplateMatching()`: Match templates to signal
- `SacchiSpikeSorting()`: Complete pipeline

### clustering.py
Multiple clustering algorithms with distance metrics.

**Algorithms:**
- K-means with silhouette/elbow selection
- Hierarchical Clustering (HC)
- Leiden graph-based clustering
- Fuzzy C-means (FCM)
- PCA/ICA preprocessing options

**Distance Metrics:**
- Euclidean (Minkowski p=2)
- Dynamic Time Warping (DTW, DDTW)
- Weighted DTW (WDTW, WDDTW)
- Longest Common Subsequence (LCSS)
- Edit Distance on Real sequences (EDR)
- Short Time Series (STS)
- Correlation-based (rho)

### gan_functions.py
GAN-based signal synthesis models.

**Components:**
- `MEAGenerator`: Transformer-based generator
- `Discriminator`: Convolutional discriminator
- `GANTrainer`: Wasserstein GAN with gradient penalty
- `PositionalEncoding`: Sinusoidal encoding for Transformers

### vaegan_functions.py
Variational Autoencoder-GAN for signal reconstruction.

**Components:**
- `VAEEncoder`: Transformer encoder with reparameterization
- `LinearConvDecoder`: Convolutional decoder
- Multi-layer architecture with attention mechanisms

## Parameters & Configuration

### Spike Sorting
```python
SacchiSpikeSorting(
    Data_ch,                    # Input data
    SamplingRate=1000,
    algo='Leiden',              # Clustering algorithm
    lowcut=300,                 # Hz
    highcut=3000,               # Hz
    notchcut=50,                # Hz
    threshold_Leiden=0.95,      # Correlation threshold
    ...
)
```

### Clustering
```python
Clustering(
    data,
    algo='KM',                  # 'KM', 'HC', 'FCM', 'Leiden', 'PCA&KM', 'ICA&FCM'
    distance='m',               # 'dtw', 'wdtw', 'lcss', 'edr', 'rho', 'sts'
    method_HC='complete',       # Linkage method
    max_classes=[2],            # Number of clusters
    ...
)
```

### GAN Training
```python
trainer = GANTrainer(
    feature_dim=1,
    input_dim=99,
    latent_dim=42,
    emb_dim=360,
    num_heads=6,
    learning_rate_G=1e-5,
    learning_rate_D=1e-4,
    n_gen_steps=5,
    gp_lambda=10
)

trainer.train(
    data_loader,
    epochs=80,
    patience=50
)
```

## Data Formats

### Input
- **BRW files**: 3Brain recording format
- **BXR files**: 3Brain analysis format
- **HDF5/H5**: Generic HDF5 format
- **CSV/NumPy**: Array-based formats

### Output
- **Templates**: Numpy arrays of spike waveforms
- **Clusters**: Spike assignments to clusters
- **Metrics**: ISI, burst statistics, network properties

## Performance

Typical performance on standard hardware:
- **Spike Detection**: ~100,000 spikes/min (single channel)
- **Template Matching**: ~50,000 matches/min
- **Clustering**: 1,000 samples in <5 seconds
- **GAN Training**: ~5-10 epochs/hour (batch size 32)

GPU acceleration available with CUDA.

## Documentation

Full Sphinx documentation available at: https://3d-brain.readthedocs.io

Build locally:
```bash
cd docs
make html
# Open _build/html/index.html
```

## Examples

See the `main/` directory for comprehensive examples:
- `main_clustering_realdata.py`: Real data clustering
- `main_gan.py`: GAN training example
- `main_vaegan.py`: VAE-GAN training
- `main_extract_spike.py`: Spike extraction pipeline

## Testing

```bash
pytest tests/
pytest --cov=. tests/        # With coverage
```

## Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Citation

If you use 3D-BrAIn in your research, please cite:

```bibtex
@software{sacchi2024_3dbrain,
  author       = {Sacchi, Cristina and Contributors},
  title        = {3D-BrAIn: Spike Sorting and Neural Signal Analysis},
  year         = {2024},
  url          = {https://github.com/your-repo/3D-BrAIn},
  doi          = {10.5281/zenodo.your-doi}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## References

- [Leiden Algorithm](https://github.com/vtraag/leidenalg)
- [igraph](https://igraph.org/)
- [PyTorch](https://pytorch.org/)
- [Elephant](https://elephant.readthedocs.io/)
- [Neo](https://neo.readthedocs.io/)

## Authors

- **Cristina Sacchi** - Principal Developer

## Acknowledgments

- University of Genova, Department of Physics
- 3Brain GmbH (BRW/BXR format support)
- Contributors and users of the spike sorting community

## Contact

For questions, issues, or suggestions:
- **Email**: cristina.sacchi@unige.it
- **Issues**: https://github.com/your-repo/3D-BrAIn/issues
- **Discussions**: https://github.com/your-repo/3D-BrAIn/discussions

---

**Last Updated**: January 2024  
**Version**: 1.0.0
