"""Tensorflow datasets of astronomical time series."""
import astro_datasets.time_series.checksums
import astro_datasets.time_series.spcc
import astro_datasets.time_series.plasticc

builders = [
    'spcc',
    'plasticc'
]

__version__ = '0.1.0'
