import tensorflow_datasets as tfds
import os
import collections
import numpy as np
import pickle

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


class CMD(tfds.core.GeneratorBasedBuilder):
    """CAMELS Multifield Dataset"""

    URL = _CMD_URL

    VERSION = tfds.core.Version("1.0.0")

    # num_classes =

    def __init__(self, **kwargs):
        # Allow default target to be changed when loading dataset
        if 'target' in kwargs:
            self.default_target = kwargs['target']
            kwargs.pop('target')
        super().__init__(*kwargs)
        # To call: ds, info = tfds.load(name='camels', split='train', with_info=True, as_supervised=True, builder_kwargs={'target': 'SIMBA'})

    def _info(self):
        return tfds.core.DatasetInfo(
            builder=self,
            description=("The CAMELS Multifield dataset consists of 3 groups "
                         "indicating the type of simualtion used to create the data."
                         "IllustrisTNG are magneto-hydrodynamic simulations."
                         "SIMBA are hydrodynamic simulations."
                         "There are corresponding N-body simulations for each"
                         "(magneto-)hydrodynamic simulation."),
            features=tfds.features.FeaturesDict({
                # "id": tfds.features.Text(),
                "image": tfds.features.Image(shape=_CMD_IMAGE_SHAPE),
                # "label": tfds.features.ClassLabel(num_classes=self.num_classes),
            }),
            # supervised_keys=("image", "label"),
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
            # test_files=["test_batch"],
            # prefix="batches",
            label_files=[_CMD_SIMBA_PARAMS_FILENAME, _CMD_ILLUS_PARAMS_FILENAME,
                         _CMD_N_SIMBA_PARAMS_FILENAME, _CMD_N_ILLUS_PARAMS_FILENAME],
            label_key=["label"],
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        cmd_path = dl_manager.download_and_extract(self._cmd_info.url)
        cmd_info = self._cmd_info

        # mirabest_path = os.path.join(mirabest_path, mirabest_info.prefix)

        simulation = self.builder_kwargs[0]
        sim_set = self.builder_kwargs[1]
        field = self.builder_kwargs[2]

        if field == 'B' and sim_set == 'CV':
            params = _CMD_ILLUS_PARAMS_FILENAME
            data = _CMD_ILLUS_CV_B_DATA_FILENAME
            print('Magnetic field is IllustrisTNG simulation data only')

        elif field == 'B' and sim_set == 'LH':
            params = _CMD_ILLUS_PARAMS_FILENAME
            data = _CMD_ILLUS_LH_B_DATA_FILENAME
            print('Magnetic field is IllustrisTNG simulation data only')

        elif simulation == 'Nbody_IllustrisTNG' and sim_set == 'CV':
            params = _CMD_N_ILLUS_PARAMS_FILENAME
            data = _CMD_N_ILLUS_CV_MTOT_DATA_FILENAME
            print('N-body field data is only MTOT')

        elif simulation == 'Nbody_IllustrisTNG' and sim_set == 'LH':
            params = _CMD_N_ILLUS_PARAMS_FILENAME
            data = _CMD_N_ILLUS_LH_MTOT_DATA_FILENAME
            print('N-body field data is only MTOT')

        elif simulation == 'Nbody_SIMBA' and sim_set == 'CV':
            params = _CMD_N_SIMBA_PARAMS_FILENAME
            data = _CMD_N_SIMBA_CV_MTOT_DATA_FILENAME
            print('N-body field data is only MTOT')

        elif simulation == 'Nbody_SIMBA' and sim_set == 'LH':
            params = _CMD_N_SIMBA_PARAMS_FILENAME
            data = _CMD_N_SIMBA_LH_MTOT_DATA_FILENAME
            print('N-body field data is only MTOT')

        elif field == 'HI' and simulation == 'IllustrisTNG' and sim_set == 'CV':
            params = _CMD_ILLUS_PARAMS_FILENAME
            data = _CMD_ILLUS_CV_HI_DATA_FILENAME

        elif sim_set == 'HI' and simulation == 'IllustrisTNG' and sim_set == 'LH':
            params = _CMD_ILLUS_PARAMS_FILENAME
            data = _CMD_ILLUS_LH_HI_DATA_FILENAME

        elif sim_set == 'HI' and simulation == 'SIMBA' and sim_set == 'CV':
            params = _CMD_SIMBA_PARAMS_FILENAME
            data = _CMD_SIMBA_CV_HI_DATA_FILENAME

        elif sim_set == 'HI' and simulation == 'SIMBA' and sim_set == 'LH':
            params = _CMD_SIMBA_PARAMS_FILENAME
            data = _CMD_SIMBA_LH_HI_DATA_FILENAME

        elif sim_set == 'MCDM' and simulation == 'IllustrisTNG' and sim_set == 'CV':
            params = _CMD_ILLUS_PARAMS_FILENAME
            data = _CMD_ILLUS_CV_MCDM_DATA_FILENAME

        elif sim_set == 'MCDM' and simulation == 'IllustrisTNG' and sim_set == 'LH':
            params = _CMD_ILLUS_PARAMS_FILENAME
            data = _CMD_ILLUS_LH_MCDM_DATA_FILENAME

        elif sim_set == 'MCDM' and simulation == 'SIMBA' and sim_set == 'CV':
            params = _CMD_SIMBA_PARAMS_FILENAME
            data = _CMD_SIMBA_CV_MCDM_DATA_FILENAME

        elif sim_set == 'MCDM' and simulation == 'SIMBA' and sim_set == 'LH':
            params = _CMD_SIMBA_PARAMS_FILENAME
            data = _CMD_SIMBA_LH_MCDM_DATA_FILENAME

        elif sim_set == 'MGAS' and simulation == 'IllustrisTNG' and sim_set == 'CV':
            params = _CMD_ILLUS_PARAMS_FILENAME
            data = _CMD_ILLUS_CV_MGAS_DATA_FILENAME

        elif sim_set == 'MGAS' and simulation == 'IllustrisTNG' and sim_set == 'LH':
            params = _CMD_ILLUS_PARAMS_FILENAME
            data = _CMD_ILLUS_LH_MGAS_DATA_FILENAME

        elif sim_set == 'MGAS' and simulation == 'SIMBA' and sim_set == 'CV':
            params = _CMD_SIMBA_PARAMS_FILENAME
            data = _CMD_SIMBA_CV_MGAS_DATA_FILENAME

        elif sim_set == 'MGAS' and simulation == 'SIMBA' and sim_set == 'LH':
            params = _CMD_SIMBA_PARAMS_FILENAME
            data = _CMD_SIMBA_LH_MGAS_DATA_FILENAME

        elif sim_set == 'MGFE' and simulation == 'IllustrisTNG' and sim_set == 'CV':
            params = _CMD_ILLUS_PARAMS_FILENAME
            data = _CMD_ILLUS_CV_MGFE_DATA_FILENAME

        elif sim_set == 'MGFE' and simulation == 'IllustrisTNG' and sim_set == 'LH':
            params = _CMD_ILLUS_PARAMS_FILENAME
            data = _CMD_ILLUS_LH_MGFE_DATA_FILENAME

        elif sim_set == 'MGFE' and simulation == 'SIMBA' and sim_set == 'CV':
            params = _CMD_SIMBA_PARAMS_FILENAME
            data = _CMD_SIMBA_CV_MGFE_DATA_FILENAME

        elif sim_set == 'MGFE' and simulation == 'SIMBA' and sim_set == 'LH':
            params = _CMD_SIMBA_PARAMS_FILENAME
            data = _CMD_SIMBA_LH_MGFE_DATA_FILENAME

        elif sim_set == 'MSTAR' and simulation == 'IllustrisTNG' and sim_set == 'CV':
            params = _CMD_ILLUS_PARAMS_FILENAME
            data = _CMD_ILLUS_CV_MSTAR_DATA_FILENAME

        elif sim_set == 'MSTAR' and simulation == 'IllustrisTNG' and sim_set == 'LH':
            params = _CMD_ILLUS_PARAMS_FILENAME
            data = _CMD_ILLUS_LH_MSTAR_DATA_FILENAME

        elif sim_set == 'MSTAR' and simulation == 'SIMBA' and sim_set == 'CV':
            params = _CMD_SIMBA_PARAMS_FILENAME
            data = _CMD_SIMBA_CV_MSTAR_DATA_FILENAME

        elif sim_set == 'MSTAR' and simulation == 'SIMBA' and sim_set == 'LH':
            params = _CMD_SIMBA_PARAMS_FILENAME
            data = _CMD_SIMBA_LH_MSTAR_DATA_FILENAME

        elif sim_set == 'MTOT' and simulation == 'IllustrisTNG' and sim_set == 'CV':
            params = _CMD_ILLUS_PARAMS_FILENAME
            data = _CMD_ILLUS_CV_MTOT_DATA_FILENAME

        elif sim_set == 'MTOT' and simulation == 'IllustrisTNG' and sim_set == 'LH':
            params = _CMD_ILLUS_PARAMS_FILENAME
            data = _CMD_ILLUS_LH_MTOT_DATA_FILENAME

        elif sim_set == 'MTOT' and simulation == 'SIMBA' and sim_set == 'CV':
            params = _CMD_SIMBA_PARAMS_FILENAME
            data = _CMD_SIMBA_CV_MTOT_DATA_FILENAME

        elif sim_set == 'MTOT' and simulation == 'SIMBA' and sim_set == 'LH':
            params = _CMD_SIMBA_PARAMS_FILENAME
            data = _CMD_SIMBA_LH_MTOT_DATA_FILENAME

        elif sim_set == 'NE' and simulation == 'IllustrisTNG' and sim_set == 'CV':
            params = _CMD_ILLUS_PARAMS_FILENAME
            data = _CMD_ILLUS_CV_NE_DATA_FILENAME

        elif sim_set == 'NE' and simulation == 'IllustrisTNG' and sim_set == 'LH':
            params = _CMD_ILLUS_PARAMS_FILENAME
            data = _CMD_ILLUS_LH_NE_DATA_FILENAME

        elif sim_set == 'NE' and simulation == 'SIMBA' and sim_set == 'CV':
            params = _CMD_SIMBA_PARAMS_FILENAME
            data = _CMD_SIMBA_CV_NE_DATA_FILENAME

        elif sim_set == 'NE' and simulation == 'SIMBA' and sim_set == 'LH':
            params = _CMD_SIMBA_PARAMS_FILENAME
            data = _CMD_SIMBA_LH_NE_DATA_FILENAME

        elif sim_set == 'P' and simulation == 'IllustrisTNG' and sim_set == 'CV':
            params = _CMD_ILLUS_PARAMS_FILENAME
            data = _CMD_ILLUS_CV_P_DATA_FILENAME

        elif sim_set == 'P' and simulation == 'IllustrisTNG' and sim_set == 'LH':
            params = _CMD_ILLUS_PARAMS_FILENAME
            data = _CMD_ILLUS_LH_P_DATA_FILENAME

        elif sim_set == 'P' and simulation == 'SIMBA' and sim_set == 'CV':
            params = _CMD_SIMBA_PARAMS_FILENAME
            data = _CMD_SIMBA_CV_P_DATA_FILENAME

        elif sim_set == 'P' and simulation == 'SIMBA' and sim_set == 'LH':
            params = _CMD_SIMBA_PARAMS_FILENAME
            data = _CMD_SIMBA_LH_P_DATA_FILENAME

        elif sim_set == 'T' and simulation == 'IllustrisTNG' and sim_set == 'CV':
            params = _CMD_ILLUS_PARAMS_FILENAME
            data = _CMD_ILLUS_CV_T_DATA_FILENAME

        elif sim_set == 'T' and simulation == 'IllustrisTNG' and sim_set == 'LH':
            params = _CMD_ILLUS_PARAMS_FILENAME
            data = _CMD_ILLUS_LH_T_DATA_FILENAME

        elif sim_set == 'T' and simulation == 'SIMBA' and sim_set == 'CV':
            params = _CMD_SIMBA_PARAMS_FILENAME
            data = _CMD_SIMBA_CV_T_DATA_FILENAME

        elif sim_set == 'T' and simulation == 'SIMBA' and sim_set == 'LH':
            params = _CMD_SIMBA_PARAMS_FILENAME
            data = _CMD_SIMBA_LH_T_DATA_FILENAME

        elif sim_set == 'VCDM' and simulation == 'IllustrisTNG' and sim_set == 'CV':
            params = _CMD_ILLUS_PARAMS_FILENAME
            data = _CMD_ILLUS_CV_VCDM_DATA_FILENAME

        elif sim_set == 'VCDM' and simulation == 'IllustrisTNG' and sim_set == 'LH':
            params = _CMD_ILLUS_PARAMS_FILENAME
            data = _CMD_ILLUS_LH_VCDM_DATA_FILENAME

        elif sim_set == 'VCDM' and simulation == 'SIMBA' and sim_set == 'CV':
            params = _CMD_SIMBA_PARAMS_FILENAME
            data = _CMD_SIMBA_CV_VCDM_DATA_FILENAME

        elif sim_set == 'VCDM' and simulation == 'SIMBA' and sim_set == 'LH':
            params = _CMD_SIMBA_PARAMS_FILENAME
            data = _CMD_SIMBA_LH_VCDM_DATA_FILENAME

        elif sim_set == 'VGAS' and simulation == 'IllustrisTNG' and sim_set == 'CV':
            params = _CMD_ILLUS_PARAMS_FILENAME
            data = _CMD_ILLUS_CV_VGAS_DATA_FILENAME

        elif sim_set == 'VGAS' and simulation == 'IllustrisTNG' and sim_set == 'LH':
            params = _CMD_ILLUS_PARAMS_FILENAME
            data = _CMD_ILLUS_LH_VGAS_DATA_FILENAME

        elif sim_set == 'VGAS' and simulation == 'SIMBA' and sim_set == 'CV':
            params = _CMD_SIMBA_PARAMS_FILENAME
            data = _CMD_SIMBA_CV_VGAS_DATA_FILENAME

        elif sim_set == 'VGAS' and simulation == 'SIMBA' and sim_set == 'LH':
            params = _CMD_SIMBA_PARAMS_FILENAME
            data = _CMD_SIMBA_LH_VGAS_DATA_FILENAME

        elif sim_set == 'Z' and simulation == 'IllustrisTNG' and sim_set == 'CV':
            params = _CMD_ILLUS_PARAMS_FILENAME
            data = _CMD_ILLUS_CV_Z_DATA_FILENAME

        elif sim_set == 'Z' and simulation == 'IllustrisTNG' and sim_set == 'LH':
            params = _CMD_ILLUS_PARAMS_FILENAME
            data = _CMD_ILLUS_LH_Z_DATA_FILENAME

        elif sim_set == 'Z' and simulation == 'SIMBA' and sim_set == 'CV':
            params = _CMD_SIMBA_PARAMS_FILENAME
            data = _CMD_SIMBA_CV_Z_DATA_FILENAME

        elif sim_set == 'Z' and simulation == 'SIMBA' and sim_set == 'LH':
            params = _CMD_SIMBA_PARAMS_FILENAME
            data = _CMD_SIMBA_LH_Z_DATA_FILENAME

        else:
            print('Error: Incorrect arguments')

        # Load the label names
        # for label_key, label_file in zip(mirabest_info.label_keys,
        # mirabest_info.label_files):
        # labels_path = os.path.join(mirabest_path, label_file)
        # with open(labels_path, 'rb') as f:
        # data = pickle.load(f)
        # if self.label_names is not None:
        # self.info.features[label_key].names = self.label_names
        # else:
        # self.info.features[label_key].names = data['label_names']

        y = np.ones([15000, 6])

        for j in range(1000):
            y[(15 * j):15 * (j + 1) + j, :] = params[j, :]

        for label_keys, label_files in zip(cmd_info.label_keys,
                                           cmd_info.label_files):
            labels_path = os.path.join(cmd_path, label_files)
            with open(labels_path, 'rb') as f:
                data_store = pickle.load(f)
            self.info.features[label_keys].names = y

        # Define the splits
        def gen_filenames(filenames):
            for f in filenames:
                yield os.path.join(cmd_path, f)

        return {
            tfds.Split.TRAIN:
                self._generate_examples("train_",
                                        gen_filenames(cmd_info.data)),
        }

    def _generate_examples(self, split_prefix, filepaths):
        label_keys = self._cmd_info.label_keys
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


class CMDInfo(
    collections.namedtuple("_CMDInfo", [
        "name",
        "url",
        "prefix",
        "train_files",
        "test_files",
        "label_files",
        "label_keys",
    ])):
    """Contains the information necessary to generate a CMD dataset.
    Attributes:
      name (str): name of dataset.
      url (str): data URL.
      prefix (str): path prefix within the downloaded and extracted file to look
        for `train_files` and `test_files`.
      train_files (list<str>): name of training files within `prefix`.

      label_files (list<str>): names of the label files in the data.
      label_keys (list<str>): names of the label keys in the data.
    """


def _load_data(path):
    """Yields (label, np_image) tuples."""
    with open(path, 'rb') as f:
        data_store = pickle.load(f)
    for i in range(len(data_store['labels'])):
        yield data_store['labels'][i], np.expand_dims(data_store['data'][i], -1)
