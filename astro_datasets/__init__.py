# Tensorflow datasets of astronomical time series
import astro_datasets.time_series.spcc
import astro_datasets.time_series.plasticc

# Tensorflow datasets of astronomical images
import astro_datasets.images.mirabest
import astro_datasets.images.cmd
import astro_datasets.images.mlsst
import astro_datasets.images.slc

builders = [
    'spcc',
    'plasticc',
    'mirabest',
    'cmd',
    'mlsst',
    'slc',
]

__version__ = '0.0.17'
