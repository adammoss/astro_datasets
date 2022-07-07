==================
Astronomy datasets
==================

This module contains the implementation of multiple astronomy datasets
following the tensorflow dataset API.

Currently implemented time series datasets are:

- ``spcc`` (Supernova Photometric Classification Challenge)
- ``plasticc`` (Photometric LSST Astronomical Time-series Classification Challenge)

Currently implemented image datasets are:

- ``mirabest``, ``mirabest_confident``

This is based on the medical datasets repository https://github.com/ExpectationMax/medical_ts_datasets

Example usage
-------------

In order to get a tensorflow dataset representation of one of the datasets simply
import ``tensorflow_datasets`` and this module.  The datasets can then be accessed
like any other tensorflow dataset.

.. code-block:: python

    import tensorflow_datasets as tfds
    import astro_datasets

    dataset, info = tfds.load(name='spcc', split='train', with_info=True)


Times series instance structure
-------------------------------

Each instance in the dataset is represented as a nested directory of the following
structure:

- ``static``: Static variables such as photometric redshift
- ``static_errors``: Static variable errors
- ``time``: Scalar time variable containing the observation time
- ``values``: Observation values of time series, these by default contain `NaN` for
  modalities which were not observed for the given timepoint.
- ``value_errors``: Observation values of time series errors, these by default contain `NaN` for
  modalities which were not observed for the given timepoint.
- ``targets``: Directory of potential target values, the available endpoints are
  dataset specific.
- ``metadata``: Directory of metadata on an individual object
