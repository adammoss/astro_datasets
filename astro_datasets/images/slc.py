"""Module containing the strong lensing challenge dataset."""

import tensorflow as tf
import tensorflow_datasets as tfds
import os
import collections
import h5py
import numpy as np

_SLC_IMAGE_SIZE = 101

_DATA_OPTIONS = ['ground', 'space']


class SLCConfig(tfds.core.BuilderConfig):
    """BuilderConfig for SLC"""

    def __init__(self, *, data=None, num_channels=1, **kwargs):
        """Constructs a SLCConfig.
        Args:
          data: `str`, one of `_DATA_OPTIONS`.
          **kwargs: keyword arguments forwarded to super.
        """
        if data not in _DATA_OPTIONS:
            raise ValueError("data must be one of %s" % _DATA_OPTIONS)

        super(SLCConfig, self).__init__(**kwargs)
        self.data = data
        self.image_shape = (_SLC_IMAGE_SIZE, _SLC_IMAGE_SIZE, num_channels)


class SLC(tfds.core.GeneratorBasedBuilder):
    """SLC"""

    VERSION = tfds.core.Version("1.0.0")

    BUILDER_CONFIGS = [
        SLCConfig(
            name='space',
            version=tfds.core.Version("1.0.0"),
            data='space',
            num_channels=1,
        ),
        SLCConfig(
            name='ground',
            version=tfds.core.Version("1.0.0"),
            data='ground',
            num_channels=4,
        ),
    ]

    def _info(self):
        return tfds.core.DatasetInfo(
            builder=self,
            description=("The strong lensing challenge dataset consists of 101x101 "
                         "images in 2 classes. There "
                         "are 20000 training images and 20000 test images."),
            features=tfds.features.FeaturesDict({
                "image": tfds.features.Tensor(shape=self.builder_config.image_shape, dtype=tf.float32),
                "label": tfds.features.ClassLabel(num_classes=2),
            }),
            supervised_keys=("image", "label"),
            homepage="http://metcalf1.difa.unibo.it/blf-portal/gg_challenge.html",
        )

    @property
    def _slc_info(self):
        return SLCInfo(
            name=self.name,
            url="https://storage.googleapis.com/strong-lensing-challenge",
            space_file="strong-lensing-space-based-challenge1.tar.gz",
            ground_file="strong-lensing-space-based-challenge1.tar.gz",
            train_files=[
                "strong-lensing-space-based-challenge1/train1.h5",
            ],
            test_files=[
                "strong-lensing-space-based-challenge1/test1.h5"
            ],
            image_key="data",
            label_key="is_lens",
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""

        if self.builder_config.name == "space":

            slc_path = dl_manager.download_and_extract(os.path.join(self._slc_info.url, self._slc_info.space_file))

        elif self.builder_config.name == "ground":

            slc_path = dl_manager.download_and_extract(os.path.join(self._slc_info.url, self._slc_info.ground_file))

        slc_info = self._slc_info

        # Define the splits
        def gen_filenames(filenames):
            for f in filenames:
                yield os.path.join(slc_path, f)

        return {
            'train': self._generate_examples(filepaths=gen_filenames(slc_info.train_files)),
            'test': self._generate_examples(filepaths=gen_filenames(slc_info.test_files)),
        }

    def _generate_examples(self, filepaths):
        for path in filepaths:
            with h5py.File(path, "r") as f:
                images = f[self._slc_info.image_key][:]
                labels = f[self._slc_info.label_key][:]
            for i, (image, label) in enumerate(zip(images, labels)):
                if len(image.shape) == 2:
                    image = np.expand_dims(image, -1)
                record = {
                    "image": image,
                    "label": label,
                }
                yield i, record


class SLCInfo(
    collections.namedtuple("_SLCInfo", [
        "name",
        "url",
        "space_file",
        "ground_file",
        "train_files",
        "test_files",
        "image_key",
        "label_key",
    ])):
    """Contains the information necessary to generate a SLC dataset.
       Attributes:
         name (str): name of dataset.
         url (str): data URL.
         train_files (list<str>): name of training files within `prefix`.
         label_files (list<str>): names of the label files in the data.
         label_keys (list<str>): names of the label keys in the data.
       """
