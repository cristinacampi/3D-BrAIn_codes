"""
3D-BrAIn: 3D Brain Activity In vitro Network
A package for spike sorting, clustering, and GAN-based signal synthesis for MEA data
"""

__version__ = '1.0.0'
__author__ = 'Cristina Campi, Lorenzo Sacchi, Maurits Unkel'

from . import brw_functions
from . import bxr_functions
from . import FCM
from . import gan_functions
from . import merging_tree
from . import spike_sorting
from . import stratification
from . import vaegan_functions

__all__ = [
    'brw_functions',
    'bxr_functions',
    'FCM',
    'gan_functions',
    'merging_tree',
    'spike_sorting',
    'stratification',
    'vaegan_functions',
]
