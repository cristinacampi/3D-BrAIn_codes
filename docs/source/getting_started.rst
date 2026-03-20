Getting Started
===============

Basic Usage
-----------

This guide will help you get up and running with 3D-BrAIn's core features.

Importing the Package
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    import brain_3d
    from brain_3d import spike_sorting, brw_functions, gan_functions

Loading and Preprocessing Data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Load MEA recordings from supported formats:

.. code-block:: python

    # Load from BRW file (3Brain format)
    from brain_3d.brw_functions import load_brw_data
    data = load_brw_data('path/to/file.brw')
    
    # Load from BXR file
    from brain_3d.bxr_functions import load_bxr_data
    data = load_bxr_data('path/to/file.bxr')

Spike Detection and Sorting
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Detect and sort spikes using available algorithms:

.. code-block:: python

    from brain_3d.spike_sorting import detect_spikes, sort_spikes
    
    # Detect spikes
    spike_times, spike_amplitudes = detect_spikes(
        data,
        threshold=4.5,  # in standard deviations
        method='positive'  # 'positive', 'negative', or 'both'
    )
    
    # Sort spikes into clusters
    sorted_spikes = sort_spikes(spike_amplitudes)

Clustering Analysis
~~~~~~~~~~~~~~~~~~~

Apply clustering algorithms to organize spike data:

.. code-block:: python

    from brain_3d.FCM import fuzzy_c_means
    from brain_3d.merging_tree import hierarchical_merging
    
    # Fuzzy C-Means clustering
    centers, membership = fuzzy_c_means(
        spike_amplitudes,
        n_clusters=5,
        max_iterations=100
    )
    
    # Hierarchical tree merging
    merged_clusters = hierarchical_merging(spike_data)

Template Matching
~~~~~~~~~~~~~~~~~

Match detected spikes against templates:

.. code-block:: python

    from brain_3d.spike_sorting import template_matching
    
    matches, correlations = template_matching(
        detected_spikes,
        templates,
        threshold=0.7
    )

GAN-based Signal Synthesis
~~~~~~~~~~~~~~~~~~~~~~~~~~

Generate synthetic neural signals using generative models:

.. code-block:: python

    from brain_3d.gan_functions import train_gan, generate_signals
    from brain_3d.vaegan_functions import train_vaegan
    
    # Train a VAE-GAN model
    model = train_vaegan(
        training_data,
        epochs=100,
        batch_size=32,
        latent_dim=10
    )
    
    # Generate synthetic signals
    synthetic_signals = generate_signals(
        model,
        num_samples=1000,
        signal_length=1000
    )

Burst Detection
~~~~~~~~~~~~~~~

Analyze burst activity patterns:

.. code-block:: python

    from brain_3d.spike_sorting import detect_bursts
    
    bursts = detect_bursts(
        spike_times,
        min_spikes=4,
        max_interval=100  # milliseconds
    )

Network Analysis
~~~~~~~~~~~~~~~~

Calculate network-wide metrics:

.. code-block:: python

    from brain_3d.stratification import compute_network_metrics
    
    metrics = compute_network_metrics(
        spike_data,
        connectivity_matrix,
        sampling_rate=20000
    )

Common Workflows
----------------

Complete Spike Sorting Pipeline
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from brain_3d.brw_functions import load_brw_data
    from brain_3d.spike_sorting import detect_spikes, sort_spikes
    from brain_3d.merging_tree import hierarchical_merging
    
    # Load data
    raw_data = load_brw_data('recording.brw')
    
    # Detect spikes
    spike_times, spike_amps = detect_spikes(raw_data, threshold=4.0)
    
    # Sort and merge
    sorted_data = sort_spikes(spike_amps)
    final_clusters = hierarchical_merging(sorted_data)
    
    print(f"Found {len(final_clusters)} neuron clusters")

Next Steps
----------

- Read the :doc:`usage` guide for detailed examples
- Explore the :doc:`modules` API reference
- Check out tutorials and examples in the GitHub repository
