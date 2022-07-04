import tensorflow_datasets as tfds
import os
import tensorflow as tf
import collections

# Testing MiraBest data

_MIRABEST_IMAGE_SIZE = 150
_MIRABEST_IMAGE_SHAPE = (_MIRABEST_IMAGE_SIZE, _MIRABEST_IMAGE_SIZE, 3)  # 3 channels?


class MIRABEST(tfds.core.GeneratorBasedBuilder):
    """MiraBest"""

    VERSION = tfds.core.Version("3.0.2")

    def _info(self):
        return tfds.core.DatasetInfo(
            builder=self,
            description=("The MiraBest dataset consists of 1256 150x150 colour? "  # Unsure if colour
                         "images in 24 classes. There "
                         "are 1099 training images and 157 test images."),
            features=tfds.features.FeaturesDict({
                "id": tfds.features.Text(),
                "image": tfds.features.Image(shape=_MIRABEST_IMAGE_SHAPE),
                "label": tfds.features.ClassLabel(num_classes=24),
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
                            "data_batch_1.bin", "data_batch_2.bin", "data_batch_3.bin",
                            "data_batch_4.bin", "data_batch_5.bin", "data_batch_6.bin",
                            "data_batch_7.bin"
                        ],
            test_files=["test_batch.bin"],
            prefix="batches",
            label_files=["batches.meta"],
            label_keys=["label"],
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        print(self._mirabest_info.url)
        mirabest_path = dl_manager.download_and_extract(self._mirabest_info.url)
        mirabest_info = self._mirabest_info

        mirabest_path = os.path.join(mirabest_path, mirabest_info.prefix)

        # Load the label names
        for label_key, label_file in zip(mirabest_info.label_keys,
                                         mirabest_info.label_files):
            labels_path = os.path.join(mirabest_path, label_file)
            with tf.io.gfile.GFile(labels_path) as label_f:
                label_names = [name for name in label_f.read().split("\n") if name]
            self.info.features[label_key].names = label_names

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
        """Generate CIFAR examples as dicts.
        Shared across CIFAR-{10, 100}. Uses self._cifar_info as
        configuration.
        Args:
          split_prefix (str): Prefix that identifies the split (e.g. "tr" or "te").
          filepaths (list[str]): The files to use to generate the data.
        Yields:
          The cifar examples, as defined in the dataset info features.
        """
        label_keys = self._mirabest_info.label_keys
        index = 0  # Using index as key since data is always loaded in same order.
        for path in filepaths:
            for labels, np_image in _load_data(path, len(label_keys)):
                record = dict(zip(label_keys, labels))
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

