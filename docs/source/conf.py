# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys
sys.path.insert(0, os.path.abspath('../../src'))

project = '3D-BrAIn'
copyright = '2026, Cristina Campi, Lorenzo Sacchi, Maurits Unkel'
author = 'Cristina Campi, Lorenzo Sacchi, Maurits Unkel'
release = '1.0.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
]

templates_path = ['_templates']

# -- Autodoc configuration ---------------------------------------------------
autodoc_default_options = {
    'members': '',
    'member-order': 'bysource',
    'special-members': '__init__',
    'undoc-members': '',
    'show-inheritance': '',
}

autodoc_mock_imports = [
    'scipy',
    'scipy.signal',
    'sklearn',
    'sklearn.cluster',
    'psutil',
    'h5py',
    'pywt',
    'plotly',
    'plotly.express',
    'seaborn',
    'fcmeans',
    'igraph',
    'leidenalg',
    'elephant',
    'neo',
    'quantities',
    'pyclustering',
    'kneed',
]

# -- Napoleon configuration ---------------------------------------------------
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = False
napoleon_use_admonition_for_examples = True
napoleon_use_admonition_for_notes = True
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = False
napoleon_type_aliases = None
napoleon_attr_annotations = True



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

#html_theme = 'alabaster'
html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

exclude_patterns = ['setup.py']