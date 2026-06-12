"""
3D-BrAIn: 3D Brain Activity In vitro Network
A package for spike sorting, clustering, and GAN-based signal synthesis for MEA data
"""

from importlib import import_module

__version__ = '1.0.0'
__author__ = '3D-BrAIn Team'

__all__ = [
    'BrwFunctions',
    'BxrFunctions',
    'Classification',
    'Fcm',
    'GanFunctions',
    'MergingTree',
    'SpikeSorting',
    'Stratification',
    'VaeganFunctions',
]


def __getattr__(name):
    """Lazily import submodules to avoid loading optional heavy dependencies."""
    if name in __all__:
        Module = import_module(f'.{name}', __name__)
        globals()[name] = Module
        return Module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
