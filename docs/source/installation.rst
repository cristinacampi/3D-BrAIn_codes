Installation
=============

Choose the installation method that best suits your needs.

Quick Install with pip
----------------------

The easiest way to install 3D-BrAIn is using pip:

.. code-block:: bash

    pip install 3D-BrAIn

Installation from Conda
------------------------

If you prefer using conda, you can create an environment from the provided environment file:

.. code-block:: bash

    conda env create -f environment.yml
    conda activate 3d-brain

Installation from Source
-------------------------

For development or to access the latest features:

.. code-block:: bash

    git clone https://github.com/cristinacampi/3D-BrAIn_codes.git
    cd 3D-BrAIn_codes
    pip install -e .

This installs the package in editable mode, allowing you to make changes to the source code.

Docker Installation
-------------------

If you prefer containerized environments:

.. code-block:: bash

    docker build -t 3d-brain:latest .
    docker run -it -v $(pwd):/app 3d-brain:latest

GPU Support
-----------

For GPU acceleration with CUDA, ensure you have:

1. **NVIDIA CUDA Toolkit** (version 11.0 or later)
2. **cuDNN** installed and properly configured

Then install PyTorch with CUDA support:

.. code-block:: bash

    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

After PyTorch installation, install 3D-BrAIn as usual.

Dependencies
------------

The package requires:

- Python 3.8+
- NumPy
- SciPy
- scikit-learn
- PyTorch (for GAN/VAE-GAN functionality)
- Pandas
- HDF5 support libraries

All dependencies are automatically installed with pip or conda.

Verification
------------

To verify the installation, open a Python terminal and run:

.. code-block:: python

    import brain_3d
    print(brain_3d.__version__)

You should see the version number printed without errors.
