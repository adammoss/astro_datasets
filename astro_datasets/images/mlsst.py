import tensorflow_datasets as tfds
import numpy as np
import tensorflow as tf
import os
import collections

_Y10_IMAGES_TRAIN_URL = "images_Y10_train.npy?download=1"
_Y1_IMAGES_TRAIN_URL = "images_Y1_train.npy?download=1"
_LABELS_TRAIN_URL = "labels_train.npy?download=1"

_Y10_IMAGES_VALID_URL = "images_Y10_valid.npy?download=1"
_Y1_IMAGES_VALID_URL = "images_Y1_valid.npy?download=1"
_LABELS_VALID_URL = "labels_valid.npy?download=1"

_Y10_IMAGES_TEST_URL = "images_Y10_test.npy?download=1"
_Y1_IMAGES_TEST_URL = "images_Y1_test.npy?download=1"
_LABELS_TEST_URL = "labels_test.npy?download=1"

_MLSST_IMAGE_SIZE = 100
_MLSST_IMAGE_SHAPE = (3, _MLSST_IMAGE_SIZE, _MLSST_IMAGE_SIZE)

_URL = "https://zenodo.org/record/5514180/files/"

_DATA_OPTIONS = ['Y1', 'Y10']


class MLSSTConfig(tfds.core.BuilderConfig):
    """BuilderConfig for MLSST"""

    def __init__(self, *, data=None, **kwargs):
        """Constructs a MLSSTConfig.
        Args:
          data: `str`, one of `_DATA_OPTIONS`.
          **kwargs: keyword arguments forwarded to super.
        """
        if data not in _DATA_OPTIONS:
            raise ValueError("data must be one of %s" % _DATA_OPTIONS)

        super(MLSSTConfig, self).__init__(**kwargs)
        self.data = data


class MLSST(tfds.core.GeneratorBasedBuilder):
    """Mock LSST dataset"""

    BUILDER_CONFIGS = [
        MLSSTConfig(
            name=config_name,
            version=tfds.core.Version("1.0.0"),
            data=config_name,
        ) for config_name in _DATA_OPTIONS
    ]

    num_classes = 3

    def _info(self):
        return tfds.core.DatasetInfo(
            builder=self,
            description=("The Mock LSST dataset consists of 2 sets of survey-emulating images, by applying an  "
                         "exposure time directly to raw images: low-noise 10 year (Y10) survey, high-noise 1 year (Y1)."
                         "The data is split into 3 sets: training, validation and testing."),
            features=tfds.features.FeaturesDict({
                "image": tfds.features.Tensor(shape=_MLSST_IMAGE_SHAPE, dtype=tf.float64),
                "label": tfds.features.ClassLabel(num_classes=self.num_classes),
            }),
            supervised_keys=("image", "label"),
            homepage="https://zenodo.org/record/5514180#.Yt-gRnbMIjJ",
        )

    @property
    def _mlsst_info(self):
        return MLSSTInfo(
            name=self.name,
            url=_URL,
            Y10_train_files=_Y10_IMAGES_TRAIN_URL,
            Y1_train_files=_Y1_IMAGES_TRAIN_URL,
            Y10_validation_files=_Y10_IMAGES_VALID_URL,
            Y1_validation_files=_Y1_IMAGES_VALID_URL,
            Y10_test_files=_Y10_IMAGES_TEST_URL,
            Y1_test_files=_Y1_IMAGES_TEST_URL,
            train_label_files=_LABELS_TRAIN_URL,
            validation_label_files=_LABELS_VALID_URL,
            test_label_files=_LABELS_TEST_URL,
            label_keys=["label"],
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""

        if self.builder_config.name == "Y10":

            train_image_url = self._mlsst_info.Y10_train_files
            valid_image_url = self._mlsst_info.Y10_validation_files
            test_image_url = self._mlsst_info.Y10_test_files

        elif self.builder_config.name == "Y1":

            train_image_url = self._mlsst_info.Y1_train_files
            valid_image_url = self._mlsst_info.Y1_validation_files
            test_image_url = self._mlsst_info.Y1_test_files

        train_label_url = self._mlsst_info.train_label_files
        valid_label_url = self._mlsst_info.validation_label_files
        test_label_url = self._mlsst_info.test_label_files

        return {
            'train': self._generate_examples(
                image_path=dl_manager.download(os.path.join(self._mlsst_info.url, train_image_url)),
                label_path=dl_manager.download(os.path.join(self._mlsst_info.url, train_label_url)),
            ),
            'validation': self._generate_examples(
                image_path=dl_manager.download(os.path.join(self._mlsst_info.url, valid_image_url)),
                label_path=dl_manager.download(os.path.join(self._mlsst_info.url, valid_label_url)),
            ),
            'test': self._generate_examples(
                image_path=dl_manager.download(os.path.join(self._mlsst_info.url, test_image_url)),
                label_path=dl_manager.download(os.path.join(self._mlsst_info.url, test_label_url)),
            ),
        }

    def _generate_examples(self, image_path, label_path):
        with tf.io.gfile.GFile(image_path, "rb") as f:
            images = np.load(f)
        with tf.io.gfile.GFile(label_path, "rb") as f:
            labels = np.argmax(np.load(f), axis=1)
        for i, (image, label) in enumerate(zip(images, labels)):
            record = {
                "image": image,
                "label": label,
            }
            yield i, record


class MLSSTInfo(
    collections.namedtuple("_MLSSTInfo", [
        "name",
        "url",
        "Y10_train_files",
        "Y1_train_files",
        "Y10_validation_files",
        "Y1_validation_files",
        "Y10_test_files",
        "Y1_test_files",
        "train_label_files",
        "validation_label_files",
        "test_label_files",
        "label_keys"
    ])):
    """Contains the information necessary to generate a CIFAR dataset.
    Attributes:
      name (str): name of dataset.
      url (str): data URL.

      label_keys (list<str>): names of the label keys in the data.
    """
