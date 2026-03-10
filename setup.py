"""
Setup configuration for 3D-BrAIn spike sorting and signal analysis package.
"""

from setuptools import setup, find_packages
import os

# Read the contents of README file
this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

with open(os.path.join(this_directory, 'requirements.txt'), encoding='utf-8') as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]

setup(
    name='3D-BrAIn',
    version='1.0.0',
    author='Cristina Campi, Lorenzo Sacchi, Maurits Unkel',
    author_email='cristina.campi@unige.it',
    description='3D Brain Activity In vitro Network: Spike sorting, clustering, and GAN-based signal synthesis for MEA data',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/your-repo/3D-BrAIn',
    project_urls={
        'Bug Tracker': 'https://github.com/your-repo/3D-BrAIn/issues',
        'Documentation': 'https://3d-brain.readthedocs.io',
        'Source Code': 'https://github.com/your-repo/3D-BrAIn',
    },
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Bio-Informatics',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.8',
    install_requires=requirements,
    extras_require={
        'dev': [
            'pytest>=6.2.0',
            'pytest-cov>=2.12.0',
            'black>=21.0.0',
            'flake8>=3.9.0',
            'sphinx>=4.5.0',
        ],
        'gpu': [
            'torch>=1.10.0,!=1.12.0[cu116]',
            'torchvision>=0.11.0[cu116]',
        ],
    },
    include_package_data=True,
    zip_safe=False,
    keywords='spike-sorting clustering MEA neural-recording deep-learning GAN',
    entry_points={
        'console_scripts': [
            '3d-brain=main_clustering_realdata:main',
        ],
    },
)
