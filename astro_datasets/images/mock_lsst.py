import tensorflow_datasets as tfds
import numpy as np
import tensorflow as tf


_Y10_IMAGES_TRAIN_URL = "https://zenodo.org/record/5514180/files/images_Y10_train.npy?download=1"
_Y1_IMAGES_TRAIN_URL = "https://zenodo.org/record/5514180/files/images_Y1_train.npy?download=1"
_LABELS_TRAIN_URL = "https://zenodo.org/record/5514180/files/labels_train.npy?download=1"


_Y10_IMAGES_VALID_URL = "https://zenodo.org/record/5514180/files/images_Y10_valid.npy?download=1"
_Y1_IMAGES_VALID_URL = "https://zenodo.org/record/5514180/files/images_Y1_valid.npy?download=1"
_LABELS_VALID_URL = "https://zenodo.org/record/5514180/files/labels_valid.npy?download=1"


_Y10_IMAGES_TEST_URL = "https://zenodo.org/record/5514180/files/images_Y10_test.npy?download=1"
_Y1_IMAGES_TEST_URL = "https://zenodo.org/record/5514180/files/images_Y1_test.npy?download=1"
_LABELS_TEST_URL = "https://zenodo.org/record/5514180/files/labels_test.npy?download=1"


_MLSST_IMAGE_SIZE = 100
_MLSST_IMAGE_SHAPE = (_MLSST_IMAGE_SIZE, _MLSST_IMAGE_SIZE, 3)


class MOCK_LSST(tfds.core.GeneratorBasedBuilder):
    """Mock LSST dataset"""

    VERSION = tfds.core.Version("1.0.0")
    num_classes = 3

    def __init__(self, data_type, *kwargs):
        self.data_type = data_type
        assert self.data_type == "Y10" or self.data_type == "Y1", "Incorrect data type entered"
        super().__init__(*kwargs)

    def _info(self):
        return tfds.core.DatasetInfo(
            builder=self,
            description=("The Mock LSST dataset consists of 2 sets of survey-emulating images, by applying an  "
                         "exposure time directly to raw images: low-noise 10 year (Y10) survey, high-noise 1 year (Y1)."
                         "The data is split into 3 sets: training, validation and testing."),
            features=tfds.features.FeaturesDict({
                "image": tfds.features.Image(shape=_MIRABEST_IMAGE_SHAPE),
                "label": tfds.features.ClassLabel(num_classes=self.num_classes),
            }),
            supervised_keys=("image", "label"),
            homepage="https://zenodo.org/record/5514180#.Yt-gRnbMIjJ",
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""

        if self.data_type == "Y10":

            train_image_url = _Y10_IMAGES_TRAIN_URL.format(self.builder_config.data)
            valid_image_url = _Y10_IMAGES_VALID_URL.format(self.builder_config.data)
            test_image_url = _Y10_IMAGES_TEST_URL.format(self.builder_config.data)

        elif self.data_type == "Y1":

            train_image_url = _Y1_IMAGES_TRAIN_URL.format(self.builder_config.data)
            valid_image_url = _Y1_IMAGES_VALID_URL.format(self.builder_config.data)
            test_image_url = _Y1_IMAGES_TEST_URL.format(self.builder_config.data)

        train_label_url = _LABELS_TRAIN_URL.format(self.builder_config.name)
        valid_label_url = _LABELS_VALID_URL.format(self.builder_config.name)
        test_label_url = _LABELS_TEST_URL.format(self.builder_config.name)

        train_image_path, train_label_path = dl_manager.download([
            train_image_url,
            train_label_url,
        ])

        valid_image_path, valid_label_path = dl_manager.download([
            valid_image_url,
            valid_label_url,
        ])

        test_image_path, test_label_path = dl_manager.download([
            test_image_url,
            test_label_url,
        ])

        return [
            tfds.core.SplitGenerator(
                name=tfds.Split.TRAIN,
                gen_kwargs={
                    "image_path": train_image_path,
                    "label_path": train_label_path,
                }),

            tfds.core.SplitGenerator(
                name=tfds.Split.VALIDATION,
                gen_kwargs={
                    "image_path": valid_image_path,
                    "label_path": valid_label_path,
                }),

            tfds.core.SplitGenerator(
                name=tfds.Split.TEST,
                gen_kwargs={
                    "image_path": test_image_path,
                    "label_path": test_label_path,
                }),
        ]

    def _generate_examples(self, image_path, label_path):
        with tf.io.gfile.GFile(image_path, "rb") as f:
            images = np.load(f)
        with tf.io.gfile.GFile(label_path, "rb") as f:
            labels = np.load(f)
        for i, (image, label) in enumerate(zip(images, labels)):
            record = {
                "image": image,
                "label": label,
            }
            yield i, record














