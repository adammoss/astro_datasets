"""Module containing the galaxy zoo 2 dataset."""

import tensorflow as tf
import tensorflow_datasets as tfds
import os
import collections
import h5py
import numpy as np
import csv

_GZ2_IMAGE_SIZE = 424
_GZ2_IMAGE_SHAPE = (_GZ2_IMAGE_SIZE, _GZ2_IMAGE_SIZE, 3)

_GZ2_LABELS_PATH = "https://gz2hart.s3.amazonaws.com/gz2_hart16.csv.gz"
_GZ2_IMAGES_PATH = "https://zenodo.org/record/3565489/files/images_gz2.zip?download=1"
_GZ2_FILENAME_MAPPING_PATH = "https://zenodo.org/record/3565489/files/gz2_filename_mapping.csv?download=1"





# Have to map the objid from labels to mapping to get asset_id to get correct image

# Below

import csv

objid = []
labels = []

with open('gz2_hart16.csv', newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        #print(row['dr7objid'])
        objid.append(row['dr7objid'])
        labels.append([row['t01_smooth_or_features_a01_smooth_count'],
                          row['t01_smooth_or_features_a02_features_or_disk_count'],
                          row['t03_bar_a06_bar_count'],
                          row['t03_bar_a07_no_bar_count']])


img_id = []

with open('gz2_filename_mapping.csv', newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        img_id.append(row['objid'])

import numpy as np

objid = np.array(objid)
img_id = np.array(img_id)
labels = np.array(labels)

match = np.in1d(img_id,objid)

ids = np.nonzero(match)[0]

correct = img_id[match]




with open('gz2_filename_mapping.csv', newline='') as csvfile:
    reader=csv.DictReader(csvfile)
    id_row=[row for idx, row in enumerate(reader) if idx in ids]


# id_row is list of all rows containing the ids of images we need
# The ids in id_row do not match the ids of the images that I have





import matplotlib.pyplot as plt

images = []

for a in range(np.size(id_row)):

    asset_id = id_row[a]['asset_id'] + ".jpg"

    try:
        images.append(plt.imread("E:\\images\\images\\" + asset_id)) #need correct path

    except FileNotFoundError:
        print(str(asset_id) + ' not available')
        np.delete(labels, a, axis=0)  # remove labels for that missing image - maintains order of images and labels

images = np.asarray(images)







# TAKES MUCH LONGER

import zipfile
import matplotlib.pyplot as plt


images = []


for a in range(np.size(id_row)):

    asset_id = id_row[a]['asset_id'] + ".jpg"

    with zipfile.ZipFile('images.zip') as myzip:
        with myzip.open('images/'+asset_id) as myfile:
            images.append(myfile)



images = np.asarray(images)







#_DATA_OPTIONS = ['space']


class GZ2Config(tfds.core.BuilderConfig):
    """BuilderConfig for GZ2"""

    def __init__(self, *, data=None, num_channels=1, **kwargs):
        """Constructs a SLCConfig.
        Args:
          data: `str`, one of `_DATA_OPTIONS`.
          **kwargs: keyword arguments forwarded to super.
        """
        #if data not in _DATA_OPTIONS:
           #raise ValueError("data must be one of %s" % _DATA_OPTIONS)

        super(GZ2Config, self).__init__(**kwargs)
        self.data = data



class GZ2(tfds.core.GeneratorBasedBuilder):
    """SLC"""

    VERSION = tfds.core.Version("1.0.0")

    #BUILDER_CONFIGS = [
        #SLCConfig(
            #name='space',
            #version=tfds.core.Version("1.0.0"),
            #data='space',
            #num_channels=1,
        #),
    #]

    num_classes = 2

    def _info(self):
        return tfds.core.DatasetInfo(
            builder=self,
            description=("The galaxy zoo 2 dataset consists of 424x424 "
                         "images in ? classes. There "
                         "are ? training images and ? test images."),
            features=tfds.features.FeaturesDict({
                "image": tfds.features.Tensor(shape=_GZ2_IMAGE_SHAPE, dtype=tf.float32),
                "label": tfds.features.ClassLabel(num_classes=self.num_classes),
            }),
            supervised_keys=("image", "label"),
            homepage="",
        )

    @property
    def _slc_info(self):
        return SLCInfo(
            name=self.name,
            url="",
            #space_file="strong-lensing-space-based-challenge1.tar.gz",
            #ground_file="",
            #train_files=[
                #"strong-lensing-space-based-challenge1/train1.h5",
            #],
            #test_files=[
                #"strong-lensing-space-based-challenge1/test1.h5"
            #],
            #image_key="data",
            #label_key="is_lens",
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""

        #if self.builder_config.name == "space":

            #slc_path = dl_manager.download_and_extract(os.path.join(self._slc_info.url, self._slc_info.space_file))

        #slc_info = self._slc_info

        # Define the splits
        #def gen_filenames(filenames):
            #for f in filenames:
                #yield os.path.join(slc_path, f)

        #return {
           #'train': self._generate_examples(filepaths=gen_filenames(slc_info.train_files)),
            #'test': self._generate_examples(filepaths=gen_filenames(slc_info.test_files)),
        #}

    def _generate_examples(self, filepaths):
        #for path in filepaths:
            #with h5py.File(path, "r") as f:
                #images = f[self._slc_info.image_key][:]
                #labels = f[self._slc_info.label_key][:]
            #for i, (image, label) in enumerate(zip(images, labels)):
                #if len(image.shape) == 2:
                    #image = np.expand_dims(image, -1)
                #record = {
                    #"image": image,
                    #"label": label,
                #}
                #yield i, record


class GZ2Info(
    collections.namedtuple("_GZ2Info", [
        "name",
        "url",
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
