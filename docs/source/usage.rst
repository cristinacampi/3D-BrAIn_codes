Usage Guide
===========

Advanced Topics and Detailed Usage Patterns

File Input/Output
-----------------

Working with BRW Files (3Brain Format)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from brain_3d import brw_functions
    
    # Load BRW file
    data, metadata = brw_functions.load_brw_data('recording.brw')
    
    # data: numpy array of shape (num_channels, num_samples)
    # metadata: dict containing sampling_rate, num_electrodes, etc.
    
    # Save processed data back to BRW
    brw_functions.save_brw_data('processed.brw', data, metadata)

Working with BXR Files
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from brain_3d import bxr_functions
    
    # Load BXR file
    data = bxr_functions.load_bxr_data('file.bxr')
    
    # Export to different formats
    bxr_functions.export_to_hdf5(data, 'output.h5')
    bxr_functions.export_to_csv(data, 'output.csv')

Spike Sorting in Detail
-----------------------

Threshold-based Detection
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from brain_3d.spike_sorting import detect_spikes
    import numpy as np
    
    # Basic spike detection with custom threshold
    spike_times, amplitudes = detect_spikes(
        recording_data,
        threshold=4.5,  # in units of standard deviation
        method='positive'  # 'positive', 'negative', or 'both'
    )
    
    print(f"Detected {len(spike_times)} spikes")

Advanced Sorting Algorithm
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from brain_3d.spike_sorting import sort_spikes
    from brain_3d.merging_tree import hierarchical_merging
    
    # Initial spike sorting
    initial_clusters = sort_spikes(
        spike_waveforms,
        method='kmeans',
        n_clusters=10
    )
    
    # Hierarchical merging with custom distance metric
    final_clusters = hierarchical_merging(
        initial_clusters,
        distance_metric='dtw',  # Dynamic Time Warping
        merge_threshold=0.85
    )

Clustering Methods
------------------

K-Means Clustering
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from sklearn.cluster import KMeans
    import brain_3d.spike_sorting as ss
    
    # K-means clustering wrapped for spike data
    kmeans = KMeans(n_clusters=5, random_state=42)
    labels = kmeans.fit_predict(spike_features)

Fuzzy C-Means
~~~~~~~~~~~~~~

.. code-block:: python

    from brain_3d.FCM import fuzzy_c_means
    
    centers, membership, iterations = fuzzy_c_means(
        data,
        n_clusters=5,
        m=2.0,  # Fuzziness parameter
        max_iterations=100,
        tolerance=1e-5
    )

Distance Metrics
~~~~~~~~~~~~~~~~

The package supports multiple distance metrics for clustering:

- **DTW** (Dynamic Time Warping): Handles temporal distortions
- **WDTW** (Weighted DTW): DTW with weights for specific regions
- **LCSS** (Longest Common Subsequence): Robust to noise
- **EDR** (Edit Distance on Real sequence): Edit distance variant
- **Euclidean**: Standard Euclidean distance
- **Correlation**: Pearson correlation distance
- **Minkowski**: Generalized distance metric

.. code-block:: python

    from brain_3d.spike_sorting import compute_pairwise_distances
    
    distances = compute_pairwise_distances(
        spike_waveforms,
        metric='dtw'
    )

Generative Models
-----------------

Training a GAN
~~~~~~~~~~~~~~

.. code-block:: python

    from brain_3d.gan_functions import GAN, train_gan
    from torch.utils.data import DataLoader
    
    # Prepare data
    train_loader = DataLoader(training_data, batch_size=32)
    
    # Train GAN
    generator, discriminator = train_gan(
        train_loader,
        latent_dim=100,
        epochs=100,
        learning_rate=0.0002,
        beta1=0.5
    )
    
    # Generate new samples
    z = torch.randn(10, 100)
    generated_samples = generator(z)

Training VAE-GAN
~~~~~~~~~~~~~~~~

.. code-block:: python

    from brain_3d.vaegan_functions import VAEGANTrainer
    
    trainer = VAEGANTrainer(
        latent_dim=10,
        learning_rate=1e-3,
        device='cuda'  # Use GPU if available
    )
    
    # Train the model
    trainer.train(
        training_data,
        epochs=50,
        batch_size=64,
        validate_every=5
    )
    
    # Generate synthetic signals
    synthetic = trainer.generate(num_samples=100)

Network Analysis
----------------

Stratification Analysis
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from brain_3d.stratification import stratify_neurons
    
    # Classify neurons by recording depth
    pyramidal, oblique = stratify_neurons(
        spike_data,
        depth_info,
        threshold=50  # micrometers
    )

Cross-Correlation Analysis
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from brain_3d import spike_sorting
    
    # Compute pairwise cross-correlations
    xcorr_matrix = spike_sorting.compute_cross_correlation(
        spike_times,
        max_lag=100,
        bin_size=1
    )

Burst Detection and Statistics
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from brain_3d.spike_sorting import detect_bursts, burst_statistics
    
    # Detect bursts
    bursts = detect_bursts(
        spike_times,
        min_spikes=4,
        max_interval=100  # ms
    )
    
    # Compute statistics
    stats = burst_statistics(
        spike_times,
        bursts,
        sampling_rate=20000
    )
    
    print(f"Burst frequency: {stats['burst_freq']:.2f} Hz")
    print(f"Intra-burst frequency: {stats['intra_burst_freq']:.2f} Hz")

Performance Optimization
------------------------

Using GPU Acceleration
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    import torch
    from brain_3d import gan_functions
    
    # Check GPU availability
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Models automatically use GPU when available
    model = gan_functions.GAN(device=device)

Parallel Processing
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from brain_3d import spike_sorting
    from multiprocessing import cpu_count
    
    # Use all available CPU cores for processing
    spike_times = spike_sorting.detect_spikes_parallel(
        recording_data,
        n_jobs=cpu_count(),
        threshold=4.5
    )

Troubleshooting
---------------

Memory Issues with Large Files
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For very large recordings:

.. code-block:: python

    # Process in chunks
    chunk_size = 100000  # samples
    all_spikes = []
    
    for i in range(0, len(data), chunk_size):
        chunk = data[i:i+chunk_size]
        spikes, amps = detect_spikes(chunk, threshold=4.5)
        all_spikes.extend(spikes)

GPU Out of Memory
~~~~~~~~~~~~~~~~~~

Reduce batch size or use CPU for preprocessing:

.. code-block:: python

    import torch
    
    # Reduce batch size
    batch_size = 16  # instead of 128
    
    # Clear GPU cache
    torch.cuda.empty_cache()
