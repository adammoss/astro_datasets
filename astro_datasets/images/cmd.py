import tensorflow_datasets as tfds
import os
import collections
import numpy as np
import pickle
import csv

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

params = ''
data = ''

class CMD(tfds.core.GeneratorBasedBuilder):
    """CAMELS Multifield Dataset"""

    URL = _CMD_URL

    VERSION = tfds.core.Version("1.0.0")

    # num_classes =

    def __init__(self, simulation, sim_set, field, *kwargs, **kwds):
        # Allow default target to be changed when loading dataset
        self.simulation = simulation
        self.sim_set = sim_set
        self.field = field
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
            label_keys=["label"],
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""


        # mirabest_path = os.path.join(mirabest_path, mirabest_info.prefix)

        global params
        global data

        #simulation = self.simulation
        #sim_set = self.sim_set
        #field = self.field

        if self.field == 'B' and self.sim_set == 'CV':
            params = _CMD_ILLUS_PARAMS_FILENAME
            data = _CMD_ILLUS_CV_B_DATA_FILENAME

            #params = label_files[1]
            #data = train_files[5]
            #label_files = params
            #train_files = data

            print('Magnetic field is IllustrisTNG simulation data only')

        elif self.field == 'B' and self.sim_set == 'LH':
            params = _CMD_ILLUS_PARAMS_FILENAME
            data = _CMD_ILLUS_LH_B_DATA_FILENAME
            print('Magnetic field is IllustrisTNG simulation data only')

        elif self.simulation == 'Nbody_IllustrisTNG' and self.sim_set == 'CV':
            params = _CMD_N_ILLUS_PARAMS_FILENAME
            data = _CMD_N_ILLUS_CV_MTOT_DATA_FILENAME
            print('N-body field data is only MTOT')

        elif self.simulation == 'Nbody_IllustrisTNG' and self.sim_set == 'LH':
            params = _CMD_N_ILLUS_PARAMS_FILENAME
            data = _CMD_N_ILLUS_LH_MTOT_DATA_FILENAME
            print('N-body field data is only MTOT')

        elif self.simulation == 'Nbody_SIMBA' and self.sim_set == 'CV':
            params = _CMD_N_SIMBA_PARAMS_FILENAME
            data = _CMD_N_SIMBA_CV_MTOT_DATA_FILENAME
            print('N-body field data is only MTOT')

        elif self.simulation == 'Nbody_SIMBA' and self.sim_set == 'LH':
            params = _CMD_N_SIMBA_PARAMS_FILENAME
            data = _CMD_N_SIMBA_LH_MTOT_DATA_FILENAME
            print('N-body field data is only MTOT')

        elif self.field == 'HI' and self.simulation == 'IllustrisTNG' and self.sim_set == 'CV':
            params = _CMD_ILLUS_PARAMS_FILENAME
            data = _CMD_ILLUS_CV_HI_DATA_FILENAME

        elif self.sim_set == 'HI' and self.simulation == 'IllustrisTNG' and self.sim_set == 'LH':
            params = _CMD_ILLUS_PARAMS_FILENAME
            data = _CMD_ILLUS_LH_HI_DATA_FILENAME

        elif self.sim_set == 'HI' and self.simulation == 'SIMBA' and self.sim_set == 'CV':
            params = _CMD_SIMBA_PARAMS_FILENAME
            data = _CMD_SIMBA_CV_HI_DATA_FILENAME

        elif self.sim_set == 'HI' and self.simulation == 'SIMBA' and self.sim_set == 'LH':
            params = _CMD_SIMBA_PARAMS_FILENAME
            data = _CMD_SIMBA_LH_HI_DATA_FILENAME

        elif self.sim_set == 'MCDM' and self.simulation == 'IllustrisTNG' and self.sim_set == 'CV':
            params = _CMD_ILLUS_PARAMS_FILENAME
            data = _CMD_ILLUS_CV_MCDM_DATA_FILENAME

        elif self.sim_set == 'MCDM' and self.simulation == 'IllustrisTNG' and self.sim_set == 'LH':
            params = _CMD_ILLUS_PARAMS_FILENAME
            data = _CMD_ILLUS_LH_MCDM_DATA_FILENAME

        elif self.sim_set == 'MCDM' and self.simulation == 'SIMBA' and self.sim_set == 'CV':
            params = _CMD_SIMBA_PARAMS_FILENAME
            data = _CMD_SIMBA_CV_MCDM_DATA_FILENAME

        elif self.sim_set == 'MCDM' and self.simulation == 'SIMBA' and self.sim_set == 'LH':
            params = _CMD_SIMBA_PARAMS_FILENAME
            data = _CMD_SIMBA_LH_MCDM_DATA_FILENAME

        elif self.sim_set == 'MGAS' and self.simulation == 'IllustrisTNG' and self.sim_set == 'CV':
            params = _CMD_ILLUS_PARAMS_FILENAME
            data = _CMD_ILLUS_CV_MGAS_DATA_FILENAME

        elif self.sim_set == 'MGAS' and self.simulation == 'IllustrisTNG' and self.sim_set == 'LH':
            params = _CMD_ILLUS_PARAMS_FILENAME
            data = _CMD_ILLUS_LH_MGAS_DATA_FILENAME

        elif self.sim_set == 'MGAS' and self.simulation == 'SIMBA' and self.sim_set == 'CV':
            params = _CMD_SIMBA_PARAMS_FILENAME
            data = _CMD_SIMBA_CV_MGAS_DATA_FILENAME

        elif self.sim_set == 'MGAS' and self.simulation == 'SIMBA' and self.sim_set == 'LH':
            params = _CMD_SIMBA_PARAMS_FILENAME
            data = _CMD_SIMBA_LH_MGAS_DATA_FILENAME

        elif self.sim_set == 'MGFE' and self.simulation == 'IllustrisTNG' and self.sim_set == 'CV':
            params = _CMD_ILLUS_PARAMS_FILENAME
            data = _CMD_ILLUS_CV_MGFE_DATA_FILENAME

        elif self.sim_set == 'MGFE' and self.simulation == 'IllustrisTNG' and self.sim_set == 'LH':
            params = _CMD_ILLUS_PARAMS_FILENAME
            data = _CMD_ILLUS_LH_MGFE_DATA_FILENAME

        elif self.sim_set == 'MGFE' and self.simulation == 'SIMBA' and self.sim_set == 'CV':
            params = _CMD_SIMBA_PARAMS_FILENAME
            data = _CMD_SIMBA_CV_MGFE_DATA_FILENAME

        elif self.sim_set == 'MGFE' and self.simulation == 'SIMBA' and self.sim_set == 'LH':
            params = _CMD_SIMBA_PARAMS_FILENAME
            data = _CMD_SIMBA_LH_MGFE_DATA_FILENAME

        elif self.sim_set == 'MSTAR' and self.simulation == 'IllustrisTNG' and self.sim_set == 'CV':
            params = _CMD_ILLUS_PARAMS_FILENAME
            data = _CMD_ILLUS_CV_MSTAR_DATA_FILENAME

        elif self.sim_set == 'MSTAR' and self.simulation == 'IllustrisTNG' and self.sim_set == 'LH':
            params = _CMD_ILLUS_PARAMS_FILENAME
            data = _CMD_ILLUS_LH_MSTAR_DATA_FILENAME

        elif self.sim_set == 'MSTAR' and self.simulation == 'SIMBA' and self.sim_set == 'CV':
            params = _CMD_SIMBA_PARAMS_FILENAME
            data = _CMD_SIMBA_CV_MSTAR_DATA_FILENAME

        elif self.sim_set == 'MSTAR' and self.simulation == 'SIMBA' and self.sim_set == 'LH':
            params = _CMD_SIMBA_PARAMS_FILENAME
            data = _CMD_SIMBA_LH_MSTAR_DATA_FILENAME

        elif self.sim_set == 'MTOT' and self.simulation == 'IllustrisTNG' and self.sim_set == 'CV':
            params = _CMD_ILLUS_PARAMS_FILENAME
            data = _CMD_ILLUS_CV_MTOT_DATA_FILENAME

        elif self.sim_set == 'MTOT' and self.simulation == 'IllustrisTNG' and self.sim_set == 'LH':
            params = _CMD_ILLUS_PARAMS_FILENAME
            data = _CMD_ILLUS_LH_MTOT_DATA_FILENAME

        elif self.sim_set == 'MTOT' and self.simulation == 'SIMBA' and self.sim_set == 'CV':
            params = _CMD_SIMBA_PARAMS_FILENAME
            data = _CMD_SIMBA_CV_MTOT_DATA_FILENAME

        elif self.sim_set == 'MTOT' and self.simulation == 'SIMBA' and self.sim_set == 'LH':
            params = _CMD_SIMBA_PARAMS_FILENAME
            data = _CMD_SIMBA_LH_MTOT_DATA_FILENAME

        elif self.sim_set == 'NE' and self.simulation == 'IllustrisTNG' and self.sim_set == 'CV':
            params = _CMD_ILLUS_PARAMS_FILENAME
            data = _CMD_ILLUS_CV_NE_DATA_FILENAME

        elif self.sim_set == 'NE' and self.simulation == 'IllustrisTNG' and self.sim_set == 'LH':
            params = _CMD_ILLUS_PARAMS_FILENAME
            data = _CMD_ILLUS_LH_NE_DATA_FILENAME

        elif self.sim_set == 'NE' and self.simulation == 'SIMBA' and self.sim_set == 'CV':
            params = _CMD_SIMBA_PARAMS_FILENAME
            data = _CMD_SIMBA_CV_NE_DATA_FILENAME

        elif self.sim_set == 'NE' and self.simulation == 'SIMBA' and self.sim_set == 'LH':
            params = _CMD_SIMBA_PARAMS_FILENAME
            data = _CMD_SIMBA_LH_NE_DATA_FILENAME

        elif self.sim_set == 'P' and self.simulation == 'IllustrisTNG' and self.sim_set == 'CV':
            params = _CMD_ILLUS_PARAMS_FILENAME
            data = _CMD_ILLUS_CV_P_DATA_FILENAME

        elif self.sim_set == 'P' and self.simulation == 'IllustrisTNG' and self.sim_set == 'LH':
            params = _CMD_ILLUS_PARAMS_FILENAME
            data = _CMD_ILLUS_LH_P_DATA_FILENAME

        elif self.sim_set == 'P' and self.simulation == 'SIMBA' and self.sim_set == 'CV':
            params = _CMD_SIMBA_PARAMS_FILENAME
            data = _CMD_SIMBA_CV_P_DATA_FILENAME

        elif self.sim_set == 'P' and self.simulation == 'SIMBA' and self.sim_set == 'LH':
            params = _CMD_SIMBA_PARAMS_FILENAME
            data = _CMD_SIMBA_LH_P_DATA_FILENAME

        elif self.sim_set == 'T' and self.simulation == 'IllustrisTNG' and self.sim_set == 'CV':
            params = _CMD_ILLUS_PARAMS_FILENAME
            data = _CMD_ILLUS_CV_T_DATA_FILENAME

        elif self.sim_set == 'T' and self.simulation == 'IllustrisTNG' and self.sim_set == 'LH':
            params = _CMD_ILLUS_PARAMS_FILENAME
            data = _CMD_ILLUS_LH_T_DATA_FILENAME

        elif self.sim_set == 'T' and self.simulation == 'SIMBA' and self.sim_set == 'CV':
            params = _CMD_SIMBA_PARAMS_FILENAME
            data = _CMD_SIMBA_CV_T_DATA_FILENAME

        elif self.sim_set == 'T' and self.simulation == 'SIMBA' and self.sim_set == 'LH':
            params = _CMD_SIMBA_PARAMS_FILENAME
            data = _CMD_SIMBA_LH_T_DATA_FILENAME

        elif self.sim_set == 'VCDM' and self.simulation == 'IllustrisTNG' and self.sim_set == 'CV':
            params = _CMD_ILLUS_PARAMS_FILENAME
            data = _CMD_ILLUS_CV_VCDM_DATA_FILENAME

        elif self.sim_set == 'VCDM' and self.simulation == 'IllustrisTNG' and self.sim_set == 'LH':
            params = _CMD_ILLUS_PARAMS_FILENAME
            data = _CMD_ILLUS_LH_VCDM_DATA_FILENAME

        elif self.sim_set == 'VCDM' and self.simulation == 'SIMBA' and self.sim_set == 'CV':
            params = _CMD_SIMBA_PARAMS_FILENAME
            data = _CMD_SIMBA_CV_VCDM_DATA_FILENAME

        elif self.sim_set == 'VCDM' and self.simulation == 'SIMBA' and self.sim_set == 'LH':
            params = _CMD_SIMBA_PARAMS_FILENAME
            data = _CMD_SIMBA_LH_VCDM_DATA_FILENAME

        elif self.sim_set == 'VGAS' and self.simulation == 'IllustrisTNG' and self.sim_set == 'CV':
            params = _CMD_ILLUS_PARAMS_FILENAME
            data = _CMD_ILLUS_CV_VGAS_DATA_FILENAME

        elif self.sim_set == 'VGAS' and self.simulation == 'IllustrisTNG' and self.sim_set == 'LH':
            params = _CMD_ILLUS_PARAMS_FILENAME
            data = _CMD_ILLUS_LH_VGAS_DATA_FILENAME

        elif self.sim_set == 'VGAS' and self.simulation == 'SIMBA' and self.sim_set == 'CV':
            params = _CMD_SIMBA_PARAMS_FILENAME
            data = _CMD_SIMBA_CV_VGAS_DATA_FILENAME

        elif self.sim_set == 'VGAS' and self.simulation == 'SIMBA' and self.sim_set == 'LH':
            params = _CMD_SIMBA_PARAMS_FILENAME
            data = _CMD_SIMBA_LH_VGAS_DATA_FILENAME

        elif self.sim_set == 'Z' and self.simulation == 'IllustrisTNG' and self.sim_set == 'CV':
            params = _CMD_ILLUS_PARAMS_FILENAME
            data = _CMD_ILLUS_CV_Z_DATA_FILENAME

        elif self.sim_set == 'Z' and self.simulation == 'IllustrisTNG' and self.sim_set == 'LH':
            params = _CMD_ILLUS_PARAMS_FILENAME
            data = _CMD_ILLUS_LH_Z_DATA_FILENAME

        elif self.sim_set == 'Z' and self.simulation == 'SIMBA' and self.sim_set == 'CV':
            params = _CMD_SIMBA_PARAMS_FILENAME
            data = _CMD_SIMBA_CV_Z_DATA_FILENAME

        elif self.sim_set == 'Z' and self.simulation == 'SIMBA' and self.sim_set == 'LH':
            params = _CMD_SIMBA_PARAMS_FILENAME
            data = _CMD_SIMBA_LH_Z_DATA_FILENAME

        else:
            print('Incorrect arguments')



        cmd_data_path = dl_manager.download(os.path.join(self._cmd_info.url, data))
        cmd_label_path = dl_manager.download(os.path.join(self._cmd_info.url, params))
        cmd_info = self._cmd_info

        return {
            'train': self._generate_examples(
                images_path=cmd_data_path,
                label_path=cmd_label_path,
            ),
        }


    # y = np.ones([15000, 6])

    # for j in range(1000):
        # y[(15 * j):15 * (j + 1) + j, :] = cmd_label_path[j, :]


    def _generate_examples(self, images_path, label_path):

        #file = np.loadtxt(label_path)
        #y = np.ones([15000, 6])

        #for j in range(1000):
            #y[(15 * j):15 * (j + 1) + j, :] = file[j, :]

        print(label_path)

        with label_path.open() as f:
            #file1 = np.open(f)
            #file2 = np.read(f)
            #print(file1)
            #print(file2)
            #print(f)
            for row in csv.DictReader(f):
                print(row)
                #y = np.ones([15000, 6])
                #y[0:14, :] = row           FOR ALL 1000 ROWS BUT THE FILE IS NOT THE PARAMETERS ANYMORE

                #for j in range(1000):
                    #y[(15 * j):15 * (j + 1) + j, :] = f[j, :]


                image_id = row['image_id'] #NEEDS TO BE ROW NUMBER THERE IS NO IMAGE ID
                # And yield (key, feature_dict)
                yield image_id, {
                    'image_description': row['description'],
                    'image': images_path / f'{image_id}.jpeg',
                    'label': row['label'],
                }



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
