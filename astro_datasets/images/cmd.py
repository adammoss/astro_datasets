import tensorflow_datasets as tfds
from tensorflow_datasets.core.constants import DATA_DIR
import os
import collections
import numpy as np
import tensorflow as tf

# CMD constants
_CMD_URL = "https://users.flatironinstitute.org/~fvillaescusa/priv/DEPnzxoWlaTQ6CjrXqsm0vYi8L7Jy/CMD/2D_maps/data/"
# SIMBA and IllustrisTNG parameters
_CMD_SIMBA_PARAMS_FILENAME = "params_SIMBA.txt"
_CMD_ILLUS_PARAMS_FILENAME = "params_IllustrisTNG.txt"
# N-body corresponding parameters
_CMD_N_SIMBA_PARAMS_FILENAME = "params_Nbody_SIMBA.txt"
_CMD_N_ILLUS_PARAMS_FILENAME = "params_Nbody_IllustrisTNG.txt"
# N-body total matter density data
_CMD_N_ILLUS_CV_MTOT_DATA_FILENAME = "Maps_Mtot_Nbody_IllustrisTNG_CV_z=0.00.npy"
_CMD_N_ILLUS_LH_MTOT_DATA_FILENAME = "Maps_Mtot_Nbody_IllustrisTNG_LH_z=0.00.npy"
_CMD_N_SIMBA_CV_MTOT_DATA_FILENAME = "Maps_Mtot_Nbody_SIMBA_CV_z=0.00.npy"
_CMD_N_SIMBA_LH_MTOT_DATA_FILENAME = "Maps_Mtot_Nbody_SIMBA_LH_z=0.00.npy"
# IllustrisTNG magenetic field data
_CMD_ILLUS_CV_B_DATA_FILENAME = "Maps_B_IllustrisTNG_CV_z=0.00.npy"
_CMD_ILLUS_LH_B_DATA_FILENAME = "Maps_B_IllustrisTNG_LH_z=0.00.npy"
# Neutral hydrogen density field data
_CMD_ILLUS_CV_HI_DATA_FILENAME = "Maps_HI_IllustrisTNG_CV_z=0.00.npy"
_CMD_ILLUS_LH_HI_DATA_FILENAME = "Maps_HI_IllustrisTNG_LH_z=0.00.npy"
_CMD_SIMBA_CV_HI_DATA_FILENAME = "Maps_HI_SIMBA_CV_z=0.00.npy"
_CMD_SIMBA_LH_HI_DATA_FILENAME = "Maps_HI_SIMBA_LH_z=0.00.npy"
# Dark matter density field data
_CMD_ILLUS_CV_MCDM_DATA_FILENAME = "Maps_Mcdm_IllustrisTNG_CV_z=0.00.npy"
_CMD_ILLUS_LH_MCDM_DATA_FILENAME = "Maps_Mcdm_IllustrisTNG_LH_z=0.00.npy"
_CMD_SIMBA_CV_MCDM_DATA_FILENAME = "Maps_Mcdm_SIMBA_CV_z=0.00.npy"
_CMD_SIMBA_LH_MCDM_DATA_FILENAME = "Maps_Mcdm_SIMBA_LH_z=0.00.npy"
# Gas density field data
_CMD_ILLUS_CV_MGAS_DATA_FILENAME = "Maps_Mgas_IllustrisTNG_CV_z=0.00.npy"
_CMD_ILLUS_LH_MGAS_DATA_FILENAME = "Maps_Mgas_IllustrisTNG_LH_z=0.00.npy"
_CMD_SIMBA_CV_MGAS_DATA_FILENAME = "Maps_Mgas_SIMBA_CV_z=0.00.npy"
_CMD_SIMBA_LH_MGAS_DATA_FILENAME = "Maps_Mgas_SIMBA_LH_z=0.00.npy"
# Magnesium over iron field data
_CMD_ILLUS_CV_MGFE_DATA_FILENAME = "Maps_MgFe_IllustrisTNG_CV_z=0.00.npy"
_CMD_ILLUS_LH_MGFE_DATA_FILENAME = "Maps_MgFe_IllustrisTNG_LH_z=0.00.npy"
_CMD_SIMBA_CV_MGFE_DATA_FILENAME = "Maps_MgFe_SIMBA_CV_z=0.00.npy"
_CMD_SIMBA_LH_MGFE_DATA_FILENAME = "Maps_MgFe_SIMBA_LH_z=0.00.npy"
# Stellar mass density field data
_CMD_ILLUS_CV_MSTAR_DATA_FILENAME = "Maps_Mstar_IllustrisTNG_CV_z=0.00.npy"
_CMD_ILLUS_LH_MSTAR_DATA_FILENAME = "Maps_Mstar_IllustrisTNG_LH_z=0.00.npy"
_CMD_SIMBA_CV_MSTAR_DATA_FILENAME = "Maps_Mstar_SIMBA_CV_z=0.00.npy"
_CMD_SIMBA_LH_MSTAR_DATA_FILENAME = "Maps_Mstar_SIMBA_LH_z=0.00.npy"
# Total mass density field data
_CMD_ILLUS_CV_MTOT_DATA_FILENAME = "Maps_Mtot_IllustrisTNG_CV_z=0.00.npy"
_CMD_ILLUS_LH_MTOT_DATA_FILENAME = "Maps_Mtot_IllustrisTNG_LH_z=0.00.npy"
_CMD_SIMBA_CV_MTOT_DATA_FILENAME = "Maps_Mtot_SIMBA_CV_z=0.00.npy"
_CMD_SIMBA_LH_MTOT_DATA_FILENAME = "Maps_Mtot_SIMBA_LH_z=0.00.npy"
# Electron number density field data
_CMD_ILLUS_CV_NE_DATA_FILENAME = "Maps_ne_IllustrisTNG_CV_z=0.00.npy"
_CMD_ILLUS_LH_NE_DATA_FILENAME = "Maps_ne_IllustrisTNG_LH_z=0.00.npy"
_CMD_SIMBA_CV_NE_DATA_FILENAME = "Maps_ne_SIMBA_CV_z=0.00.npy"
_CMD_SIMBA_LH_NE_DATA_FILENAME = "Maps_ne_SIMBA_LH_z=0.00.npy"
# Gas pressure field data
_CMD_ILLUS_CV_P_DATA_FILENAME = "Maps_P_IllustrisTNG_CV_z=0.00.npy"
_CMD_ILLUS_LH_P_DATA_FILENAME = "Maps_P_IllustrisTNG_LH_z=0.00.npy"
_CMD_SIMBA_CV_P_DATA_FILENAME = "Maps_P_SIMBA_CV_z=0.00.npy"
_CMD_SIMBA_LH_P_DATA_FILENAME = "Maps_P_SIMBA_LH_z=0.00.npy"
# Gas temperature field data
_CMD_ILLUS_CV_T_DATA_FILENAME = "Maps_T_IllustrisTNG_CV_z=0.00.npy"
_CMD_ILLUS_LH_T_DATA_FILENAME = "Maps_T_IllustrisTNG_LH_z=0.00.npy"
_CMD_SIMBA_CV_T_DATA_FILENAME = "Maps_T_SIMBA_CV_z=0.00.npy"
_CMD_SIMBA_LH_T_DATA_FILENAME = "Maps_T_SIMBA_LH_z=0.00.npy"
# Dark matter velocity field data
_CMD_ILLUS_CV_VCDM_DATA_FILENAME = "Maps_Vcdm_IllustrisTNG_CV_z=0.00.npy"
_CMD_ILLUS_LH_VCDM_DATA_FILENAME = "Maps_Vcdm_IllustrisTNG_LH_z=0.00.npy"
_CMD_SIMBA_CV_VCDM_DATA_FILENAME = "Maps_Vcdm_SIMBA_CV_z=0.00.npy"
_CMD_SIMBA_LH_VCDM_DATA_FILENAME = "Maps_Vcdm_SIMBA_LH_z=0.00.npy"
# Gas velocity field data
_CMD_ILLUS_CV_VGAS_DATA_FILENAME = "Maps_Vgas_IllustrisTNG_CV_z=0.00.npy"
_CMD_ILLUS_LH_VGAS_DATA_FILENAME = "Maps_Vgas_IllustrisTNG_LH_z=0.00.npy"
_CMD_SIMBA_CV_VGAS_DATA_FILENAME = "Maps_Vgas_SIMBA_CV_z=0.00.npy"
_CMD_SIMBA_LH_VGAS_DATA_FILENAME = "Maps_Vgas_SIMBA_LH_z=0.00.npy"
# Gas metallicity field data
_CMD_ILLUS_CV_Z_DATA_FILENAME = "Maps_Z_IllustrisTNG_CV_z=0.00.npy"
_CMD_ILLUS_LH_Z_DATA_FILENAME = "Maps_Z_IllustrisTNG_LH_z=0.00.npy"
_CMD_SIMBA_CV_Z_DATA_FILENAME = "Maps_Z_SIMBA_CV_z=0.00.npy"
_CMD_SIMBA_LH_Z_DATA_FILENAME = "Maps_Z_SIMBA_LH_z=0.00.npy"
# CMD image sizes
_CMD_IMAGE_SIZE = 256
_CMD_IMAGE_SHAPE = (_CMD_IMAGE_SIZE, _CMD_IMAGE_SIZE, 1)

_CMD_PARAMETERS = ['omegam', 'sigma8', 'asn1', 'aagn1', 'asn2', 'aagn2']
_CMD_FIELDS = ['Mtot', 'B', 'HI', 'Mcdm', 'Mgas', 'MgFe', 'Mstar', 'Mtot', 'ne', 'P', 'T', 'Vcdm', 'Vgas', 'Z']
_CMD_SIMULATIONS = ['IllustrisTNG', 'SIMBA']
_CMD_SIM_SET = ['CV', 'LH']


class CMD(tfds.core.GeneratorBasedBuilder):
    """CAMELS Multifield Dataset"""

    URL = _CMD_URL
    VERSION = tfds.core.Version("1.0.0")

    def __init__(self, simulation, sim_set, field, parameters=None, *kwargs, **kwds):
        assert simulation in _CMD_SIMULATIONS, 'Not a valid simulation'
        assert sim_set in _CMD_SIM_SET, 'Not a valid sim set'
        assert field in _CMD_FIELDS, 'Not a valid field'
        self.simulation = simulation
        self.sim_set = sim_set
        self.field = field
        if parameters is not None:
            assert len(parameters) > 0, 'No parameters given'
            for p in parameters:
                assert p in _CMD_PARAMETERS, 'Not a valid parameter'
            self.parameters = parameters
        else:
            self.parameters = _CMD_PARAMETERS
        self.parameter_indices = [_CMD_PARAMETERS.index(p) for p in self.parameters]
        kwds['data_dir'] = os.path.join(DATA_DIR, '_'.join([simulation[:1], sim_set, field] + self.parameters))
        super().__init__(*kwargs, **kwds)

    def _info(self):
        return tfds.core.DatasetInfo(
            builder=self,
            description=("The CAMELS Multifield dataset consists of 3 groups "
                         "indicating the type of simulation used to create the data."
                         "IllustrisTNG are magneto-hydrodynamic simulations."
                         "SIMBA are hydrodynamic simulations."
                         "There are corresponding N-body simulations for each"
                         "(magneto-)hydrodynamic simulation."),
            features=tfds.features.FeaturesDict({
                "image": tfds.features.Tensor(shape=_CMD_IMAGE_SHAPE, dtype=tf.float32),
                'label': tfds.features.Tensor(shape=(len(self.parameters),), dtype=tf.float64),
            }),
            supervised_keys=("image", "label"),
            homepage="https://camels-multifield-dataset.readthedocs.io/en/latest/index.html",
        )

    @property
    def _cmd_info(self):
        return CMDInfo(
            name=self.name,
            url="https://users.flatironinstitute.org/~fvillaescusa/priv/DEPnzxoWlaTQ6CjrXqsm0vYi8L7Jy/CMD/2D_maps/data/",
            train_files=[_CMD_N_ILLUS_CV_MTOT_DATA_FILENAME, _CMD_N_ILLUS_LH_MTOT_DATA_FILENAME,
                         _CMD_N_SIMBA_CV_MTOT_DATA_FILENAME, _CMD_N_SIMBA_LH_MTOT_DATA_FILENAME,
                         _CMD_ILLUS_CV_B_DATA_FILENAME, _CMD_ILLUS_LH_B_DATA_FILENAME, _CMD_ILLUS_CV_HI_DATA_FILENAME,
                         _CMD_ILLUS_LH_HI_DATA_FILENAME, _CMD_SIMBA_CV_HI_DATA_FILENAME, _CMD_SIMBA_LH_HI_DATA_FILENAME,
                         _CMD_ILLUS_CV_MCDM_DATA_FILENAME, _CMD_ILLUS_LH_MCDM_DATA_FILENAME,
                         _CMD_SIMBA_CV_MCDM_DATA_FILENAME, _CMD_SIMBA_LH_MCDM_DATA_FILENAME,
                         _CMD_ILLUS_CV_MGAS_DATA_FILENAME, _CMD_ILLUS_LH_MGAS_DATA_FILENAME,
                         _CMD_SIMBA_CV_MGAS_DATA_FILENAME, _CMD_SIMBA_LH_MGAS_DATA_FILENAME,
                         _CMD_ILLUS_CV_MGFE_DATA_FILENAME, _CMD_ILLUS_LH_MGFE_DATA_FILENAME,
                         _CMD_SIMBA_CV_MGFE_DATA_FILENAME, _CMD_SIMBA_LH_MGFE_DATA_FILENAME,
                         _CMD_ILLUS_CV_MSTAR_DATA_FILENAME, _CMD_ILLUS_LH_MSTAR_DATA_FILENAME,
                         _CMD_SIMBA_CV_MSTAR_DATA_FILENAME, _CMD_SIMBA_LH_MSTAR_DATA_FILENAME,
                         _CMD_ILLUS_CV_MTOT_DATA_FILENAME, _CMD_ILLUS_LH_MTOT_DATA_FILENAME,
                         _CMD_SIMBA_CV_MTOT_DATA_FILENAME, _CMD_SIMBA_LH_MTOT_DATA_FILENAME,
                         _CMD_ILLUS_CV_NE_DATA_FILENAME, _CMD_ILLUS_LH_NE_DATA_FILENAME, _CMD_SIMBA_CV_NE_DATA_FILENAME,
                         _CMD_SIMBA_LH_NE_DATA_FILENAME,
                         _CMD_ILLUS_CV_P_DATA_FILENAME, _CMD_ILLUS_LH_P_DATA_FILENAME, _CMD_SIMBA_CV_P_DATA_FILENAME,
                         _CMD_SIMBA_LH_P_DATA_FILENAME,
                         _CMD_ILLUS_CV_T_DATA_FILENAME, _CMD_ILLUS_LH_T_DATA_FILENAME, _CMD_SIMBA_CV_T_DATA_FILENAME,
                         _CMD_SIMBA_LH_T_DATA_FILENAME,
                         _CMD_ILLUS_CV_VCDM_DATA_FILENAME, _CMD_ILLUS_LH_VCDM_DATA_FILENAME,
                         _CMD_SIMBA_CV_VCDM_DATA_FILENAME, _CMD_SIMBA_LH_VGAS_DATA_FILENAME,
                         _CMD_ILLUS_CV_VGAS_DATA_FILENAME, _CMD_ILLUS_LH_VGAS_DATA_FILENAME,
                         _CMD_SIMBA_CV_VGAS_DATA_FILENAME, _CMD_SIMBA_LH_VGAS_DATA_FILENAME,
                         _CMD_ILLUS_CV_Z_DATA_FILENAME, _CMD_ILLUS_LH_Z_DATA_FILENAME, _CMD_SIMBA_CV_Z_DATA_FILENAME,
                         _CMD_SIMBA_LH_Z_DATA_FILENAME],
            label_files=[_CMD_SIMBA_PARAMS_FILENAME, _CMD_ILLUS_PARAMS_FILENAME,
                         _CMD_N_SIMBA_PARAMS_FILENAME, _CMD_N_ILLUS_PARAMS_FILENAME],
            label_keys=["label"],
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        params_file = 'params_' + self.simulation + '.txt'
        assert params_file in self._cmd_info.label_files, 'Invalid parameter file'
        cmd_label_path = dl_manager.download(os.path.join(_CMD_URL, params_file))
        data_file = 'Maps_' + self.field + '_' + self.simulation + '_' + self.sim_set + '_z=0.00.npy'
        assert data_file in self._cmd_info.train_files, 'Invalid data file'
        cmd_data_path = dl_manager.download(os.path.join(_CMD_URL, data_file))
        return {
            'train': self._generate_examples(
                images_path=cmd_data_path,
                label_path=cmd_label_path,
            ),
        }

    def _generate_examples(self, images_path, label_path):
        with tf.io.gfile.GFile(images_path, "rb") as f:
            images = np.load(f)
            images = np.expand_dims(images, axis=-1)
        with tf.io.gfile.GFile(label_path, "rb") as f:
            labels = np.loadtxt(f)
            labels = np.repeat(labels, 15, axis=0)
            labels = labels[:, self.parameter_indices]
        for i, (image, label) in enumerate(zip(images, labels)):
            record = {
                "image": image,
                "label": label,
            }
            yield i, record


class CMDInfo(
    collections.namedtuple("_CMDInfo", [
        "name",
        "url",
        "train_files",
        "label_files",
        "label_keys",
    ])):
    """Contains the information necessary to generate a CMD dataset.
    Attributes:
      name (str): name of dataset.
      url (str): data URL.
      train_files (list<str>): name of training files within `prefix`.
      label_files (list<str>): names of the label files in the data.
      label_keys (list<str>): names of the label keys in the data.
    """
