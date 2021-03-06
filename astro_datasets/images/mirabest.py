"""Module containing the Mirabest dataset."""

import tensorflow_datasets as tfds
import os
import collections
import pickle
import numpy as np

_MIRABEST_IMAGE_SIZE = 150
_MIRABEST_IMAGE_SHAPE = (_MIRABEST_IMAGE_SIZE, _MIRABEST_IMAGE_SIZE, 1)


class MIRABEST(tfds.core.GeneratorBasedBuilder):
    """MiraBest"""

    VERSION = tfds.core.Version("1.0.0")
    num_classes = 10
    class_keys = {
        0: 0,
        1: 1,
        2: 2,
        3: 3,
        4: 4,
        5: 5,
        6: 6,
        7: 7,
        8: 8,
        9: 9,
    }
    label_names = None

    def _info(self):
        return tfds.core.DatasetInfo(
            builder=self,
            description=("The MiraBest dataset consists of 1256 150x150 "
                         "images in 10 classes. There "
                         "are 1099 training images and 157 test images."),
            features=tfds.features.FeaturesDict({
                "id": tfds.features.Text(),
                "image": tfds.features.Image(shape=_MIRABEST_IMAGE_SHAPE),
                "label": tfds.features.ClassLabel(num_classes=self.num_classes),
            }),
            supervised_keys=("image", "label"),
            homepage="https://zenodo.org/record/5588282#.Yr7X8HbMIjJ",
        )

    @property
    def _mirabest_info(self):
        return MiraBestInfo(
            name=self.name,
            url="http://www.jb.man.ac.uk/research/MiraBest/full_dataset/MiraBest_full_batches.tar.gz",
            train_files=[
                "data_batch_1", "data_batch_2", "data_batch_3",
                "data_batch_4", "data_batch_5", "data_batch_6",
                "data_batch_7"
            ],
            test_files=["test_batch"],
            prefix="batches",
            label_files=["batches.meta"],
            label_keys=["label"],
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        mirabest_path = dl_manager.download_and_extract(self._mirabest_info.url)
        mirabest_info = self._mirabest_info

        mirabest_path = os.path.join(mirabest_path, mirabest_info.prefix)

        # Load the label names
        for label_key, label_file in zip(mirabest_info.label_keys,
                                         mirabest_info.label_files):
            labels_path = os.path.join(mirabest_path, label_file)
            with open(labels_path, 'rb') as f:
                data = pickle.load(f, encoding='latin1')
            if self.label_names is not None:
                self.info.features[label_key].names = self.label_names
            else:
                self.info.features[label_key].names = data['label_names']

        # Define the splits
        def gen_filenames(filenames):
            for f in filenames:
                yield os.path.join(mirabest_path, f)

        return {
            tfds.Split.TRAIN:
                self._generate_examples("train_",
                                        gen_filenames(mirabest_info.train_files)),
            tfds.Split.TEST:
                self._generate_examples("test_",
                                        gen_filenames(mirabest_info.test_files)),
        }

    def _generate_examples(self, split_prefix, filepaths):
        label_keys = self._mirabest_info.label_keys
        index = 0  # Using index as key since data is always loaded in same order.
        for path in filepaths:
            for label, np_image in _load_data(path):
                if label not in self.class_keys:
                    continue
                record = dict(zip(label_keys, [self.class_keys[label]]))
                # Note: "id" is only provided for the user convenience. To shuffle the
                # dataset we use `index`, so that the sharding is compatible with
                # earlier versions.
                record["id"] = "{}{:05d}".format(split_prefix, index)
                record["image"] = np_image
                yield index, record
                index += 1


class MiraBestInfo(
    collections.namedtuple("_MiraBestInfo", [
        "name",
        "url",
        "prefix",
        "train_files",
        "test_files",
        "label_files",
        "label_keys",
        ])):
    """Contains the information necessary to generate a CIFAR dataset.
    Attributes:
      name (str): name of dataset.
      url (str): data URL.
      prefix (str): path prefix within the downloaded and extracted file to look
        for `train_files` and `test_files`.
      train_files (list<str>): name of training files within `prefix`.
      test_files (list<str>): name of test files within `prefix`.
      label_files (list<str>): names of the label files in the data.
      label_keys (list<str>): names of the label keys in the data.
    """


def _load_data(path):
    """Yields (label, np_image) tuples."""
    with open(path, 'rb') as f:
        data = pickle.load(f)
    for i in range(len(data['labels'])):
        yield data['labels'][i], np.expand_dims(data['data'][i], -1)


class MIRABESTConfident(MIRABEST):
    num_classes = 2
    class_keys = {
        0: 0,
        1: 0,
        2: 0,
        5: 1,
        6: 1
    }
    label_names = ['FR1', 'FR2']
