# Tensorflow datasets of astronomical time series
import astro_datasets.time_series.spcc
import astro_datasets.time_series.plasticc

# Tensorflow datasets of astronomical images
import astro_datasets.images.mirabest

builders = [
    'spcc',
    'plasticc',
    'mirabest',
    'mirabest_confident',
]

__version__ = '0.0.6'
