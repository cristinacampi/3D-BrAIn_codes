"""
3D-BrAIn: 3D Brain Activity In vitro Network
A package for spike sorting, clustering, and GAN-based signal synthesis for MEA data
"""

from importlib import import_module

__version__ = '1.0.0'
__author__ = '3D-BrAIn Team'

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


def __getattr__(name):
    """Lazily import submodules to avoid loading optional heavy dependencies."""
    if name in __all__:
        module = import_module(f'.{name}', __name__)
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
