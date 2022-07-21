import tensorflow_datasets as tfds
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
# IllustrisTNG magnetic field data
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


class CMD(tfds.core.GeneratorBasedBuilder):
    """CAMELS Multifield Dataset"""

    URL = _CMD_URL
    VERSION = tfds.core.Version("1.0.0")

    def __init__(self, simulation, sim_set, field, parameters, *kwargs, **kwds):
        # Allow default target to be changed when loading dataset
        self.simulation = simulation
        self.sim_set = sim_set
        self.field = field
        self.parameters = parameters
        kwds['data_dir'] = ''.join([simulation[:1], sim_set, field] + parameters)
        super().__init__(*kwargs, **kwds)

    def _info(self):
        return tfds.core.DatasetInfo(
            builder=self,
            description=("The CAMELS Multifield dataset consists of 3 groups "
                         "indicating the type of simulation used to create the data."
                         "IllustrisTNG are magneto-hydrodynamic simulations."
                         "SIMBA are hydrodynamic simulations."
                         "There are corresponding N-body simulations for  (magneto-)hydrodynamic simulation."
                         "There are 15000 256x256 greyscale images, covering a periodic area of (25 h^-1 Mpc)^2, "
                         "each with 6 corresponding parameters:"
                         "'omega_m', sigma_8', 'a_sn1', 'a_agn1', 'a_sn2', 'a_agn2'."
                         "The data consists of 12 different fields and a magnetic field file for IllustrisTNG only:"
                         "gas density 'Mgas', gas velocity 'Vgas', gas temperature 'T', gas pressure 'P',"
                         "gas metallicity 'Z', neutral hydrogen density 'HI', electron number density 'ne',"
                         "magnetic fields 'B', magnesium over iron 'MgFe', dark matter density 'Mcdm',"
                         "dark matter velocity 'Vcdm', stellar mass density 'Mstar', total matter density 'Mtot'."),
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

        assert 'params_'+self.simulation+'.txt' in self._cmd_info.label_files, 'Invalid parameter file'
        assert 'Maps_'+self.field+'_'+self.simulation+'_'+self.sim_set+'_z=0.00.npy' in self._cmd_info.train_files, 'Invalid data file'

        cmd_label_path = dl_manager.download(_CMD_URL+'params_'+self.simulation+'.txt')
        cmd_data_path = dl_manager.download(_CMD_URL+'Maps_'+self.field+'_'+self.simulation+'_'+self.sim_set+'_z=0.00.npy')
        cmd_info = self._cmd_info

        return {
            'train': self._generate_examples(
                images_path=cmd_data_path,
                label_path=cmd_label_path,
            ),
        }

    def _generate_examples(self, images_path, label_path):
        # Resize parameters to match the array size of images
        file = np.loadtxt(label_path)
        file = np.repeat(file, 15, axis=0)

        parameters = ['omega_m', 'sigma_8', 'a_sn1', 'a_agn1', 'a_asn2', 'a_agn2']
        choice = self.parameters
        final = np.zeros(len(parameters), dtype=bool)
        for y in range(len(choice)):
            for x in range(len(parameters)):
                if final[x] == False:
                    final[x] = choice[y] in parameters[x]
                elif final[x] == True:
                    final[x] = True

        a = [i for i, x in enumerate(final) if x]

        assert len(a) != 0, 'Invalid parameters' #not sure why this is != when it should be ==?
                                                # Also this doesn't ensure the user can't enter an incorrect value
        file = file[:, a]

        with tf.io.gfile.GFile(images_path, "rb") as f:
            images = np.load(f)
            images = np.expand_dims(images, -1)
        for i, (image, label) in enumerate(zip(images, file)):
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
      prefix (str): path prefix within the downloaded and extracted file to look
        for `train_files`.
      train_files (list<str>): name of training files within `prefix`.
      label_files (list<str>): names of the label files in the data.
      label_keys (list<str>): names of the label keys in the data.
    """


