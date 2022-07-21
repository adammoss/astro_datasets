# Tensorflow datasets of astronomical time series
import astro_datasets.time_series.spcc
import astro_datasets.time_series.plasticc

# Tensorflow datasets of astronomical images
import astro_datasets.images.mirabest
import astro_datasets.images.cmd

builders = [
    'spcc',
    'plasticc',
    'mirabest',
    'mirabest_confident',
    'cmd',
]

__version__ = '0.0.6'
