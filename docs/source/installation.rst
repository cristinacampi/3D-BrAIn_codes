Installation
============

3D-BrAIn requires Python 3.8 or newer.

From a local checkout:

.. code-block:: bash

   git clone https://github.com/cristinacampi/3D-BrAIn_codes.git
   cd 3D-BrAIn_codes
   python3 -m pip install -e .

For documentation and development tools:

.. code-block:: bash

   python3 -m pip install -e ".[dev]"

The package depends on scientific Python packages such as NumPy, pandas, SciPy, scikit-learn, h5py, PyWavelets, igraph, Leidenalg, tslearn, and PyTorch. Some workflows also require domain packages such as Elephant, Neo, and Quantities.

Documentation Build
-------------------

Install Sphinx dependencies, then build HTML documentation:

.. code-block:: bash

   cd docs
   python3 -m sphinx -b html source build/html

The generated site is written to ``docs/build/html``.
