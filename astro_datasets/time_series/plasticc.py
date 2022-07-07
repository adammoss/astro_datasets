"""Module containing the Photometric LSST Astronomical Time Series Classification Challenge 2018."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from collections.abc import Sequence

import numpy as np
import pandas as pd
import tensorflow_datasets as tfds

from astro_datasets.time_series.util import AstroTsDatasetBuilder, AstroTsDatasetInfo

RESOURCES = os.path.join(
    os.path.dirname(__file__), 'resources', 'plasticc')

_CITATION = """
@article{Kessler:2019qge,
    author = "Kessler, R. and others",
    collaboration = "LSST Dark Energy Science, Transient, Variable Stars Science",
    title = "{Models and Simulations for the Photometric LSST Astronomical Time Series Classification Challenge (PLAsTiCC)}",
    eprint = "1903.11756",
    archivePrefix = "arXiv",
    primaryClass = "astro-ph.HE",
    doi = "10.1088/1538-3873/ab26f1",
    journal = "Publ. Astron. Soc. Pac.",
    volume = "131",
    number = "1003",
    pages = "094501",
    year = "2019"
}
"""

_DESCRIPTION = """
"""


class PlasticcDataReader(Sequence):
    """Reader class for Plasticc dataset."""

    static_features = [
        'hostgal_photoz', 'mwebv', 'galactic'
    ]

    ts_features = [
        'lsstu_flux', 'lsstg_flux', 'lsstr_flux', 'lssti_flux', 'lsstz_flux', 'lssty_flux'
    ]

    # Remove any specific instances
    blacklist = [
    ]

    class_keys = {
        6: 0,
        15: 1,
        16: 2,
        42: 3,
        52: 4,
        53: 5,
        62: 6,
        64: 7,
        65: 8,
        67: 9,
        88: 10,
        90: 11,
        92: 12,
        95: 13,
        99: 14,
    }

    # Time quantisation in days
    time_quantisation = 1.0

    def __init__(self, data_files, metadata_file, remove_no_timeseries=True):
        """Load instances from the Plasticc challenge.

        Args:
            data_path: Path containing the records.
            metadata_file: File containing the metadata definitions

        """

        self.static_error_features = [feature + '_error' for feature in self.static_features]
        self.ts_error_features = [feature + '_error' for feature in self.ts_features]
        self.data = pd.concat([pd.read_csv(data_file, header=0, sep=',') for data_file in data_files])
        self.data = self.data.rename(columns={'mjd': 'time', 'flux_err': 'flux_error', })
        self.data = self.data.replace(
            {'passband': {0: 'lsstu', 1: 'lsstg', 2: 'lsstr', 3: 'lssti', 4: 'lsstz', 5: 'lssty', }})
        self.data = pd.melt(self.data, id_vars=['object_id', 'time', 'passband'], value_vars=['flux', 'flux_error'],
                            var_name='parameter', value_name='value')
        self.data['parameter'] = self.data['passband'] + '_' + self.data['parameter']
        self.data.drop('passband', inplace=True, axis=1)
        metadata = pd.read_csv(metadata_file, header=0, sep=',')
        self.metadata = metadata[~metadata['object_id'].isin(self.blacklist)]
        if remove_no_timeseries:
            # Remove an objects we do not have lightcurves for
            self.metadata = metadata[metadata['object_id'].isin(self.data.object_id.unique())]
        self.metadata = self.metadata.rename(columns={'hostgal_photoz_err': 'hostgal_photoz_error'}, )
        self.metadata['galactic'] = (self.metadata['hostgal_photoz'] > 0).astype(float)
        for feature in self.static_error_features:
            if feature not in self.metadata:
                self.metadata[feature] = np.nan


    def _quantise_time(self, values):
        return self.time_quantisation * np.round(values / self.time_quantisation)

    def __getitem__(self, index):
        """Get instance at position index of metadata file."""
        instance = self.metadata.iloc[index]
        static = instance[self.static_features]
        static_errors = instance[self.static_error_features]
        object_id = int(instance['object_id'])
        # Read data
        timeseries = self._read_timeseries(object_id)
        time = timeseries['time']
        values = timeseries[self.ts_features]
        value_errors = timeseries[self.ts_error_features]

        if instance['true_target'] in self.class_keys:
            true_target = self.class_keys[instance['true_target']]
        else:
            true_target = self.class_keys[99]

        return object_id, {
            'static': static,
            'static_errors' : static_errors,
            'time': time,
            'values': values,
            'value_errors': value_errors,
            'targets': {
                'class': true_target,
                'sn1a': true_target == 11
            },
            'metadata': {
                'object_id': object_id,
                'redshift': instance['true_z']
            }
        }

    def _read_timeseries(self, object_id):
        data = self.data[self.data['object_id'] == object_id].copy()
        data['time'] = self._quantise_time(data['time'])
        timeseries = data.pivot_table(index='time', columns='parameter', values='value')
        timeseries = timeseries.reindex(columns=self.ts_features + self.ts_error_features).reset_index()
        return timeseries

    def __len__(self):
        """Return number of instances in the dataset."""
        return len(self.metadata)


class Plasticc(AstroTsDatasetBuilder):
    """Dataset of Plasticc"""

    VERSION = tfds.core.Version('1.0.10')
    has_metadata = True
    has_timeseries = True
    default_target = 'class'

    def _info(self):
        return AstroTsDatasetInfo(
            builder=self,
            targets={
                'class': tfds.features.ClassLabel(num_classes=15),
                'sn1a': tfds.features.ClassLabel(num_classes=2),
            },
            default_target='class',
            static_names=PlasticcDataReader.static_features,
            timeseries_names=PlasticcDataReader.ts_features,
            description=_DESCRIPTION,
            homepage='https://zenodo.org/record/2539456#.XzsWWxNKibs',
            citation=_CITATION
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Return SplitGenerators."""
        paths = dl_manager.download({
            'train_ts': 'https://zenodo.org/record/2539456/files/plasticc_train_lightcurves.csv.gz',  # noqa: E501
            'train_meta': 'https://zenodo.org/record/2539456/files/plasticc_train_metadata.csv.gz',  # noqa: E501
            'test_ts_1': 'https://zenodo.org/record/2539456/files/plasticc_test_lightcurves_01.csv.gz',  # noqa: E501
            'test_meta': 'https://zenodo.org/record/2539456/files/plasticc_test_metadata.csv.gz',  # noqa: E501
        })

        return [
            tfds.core.SplitGenerator(
                name=tfds.Split.TRAIN,
                gen_kwargs={
                    'data_files': [paths['train_ts']],
                    'metadata_file': paths['train_meta']
                },
            ),
            tfds.core.SplitGenerator(
                name=tfds.Split.TEST,
                gen_kwargs={
                    'data_files': [paths['test_ts_1']],
                    'metadata_file': paths['test_meta']
                }
            )
        ]

    def _generate_examples(self, data_files, metadata_file):
        """Yield examples."""
        reader = PlasticcDataReader(data_files, metadata_file)
        for instance in reader:
            yield instance
