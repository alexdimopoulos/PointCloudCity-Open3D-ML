import numpy as np
import os, argparse, pickle, sys
from os.path import exists, join, isfile, dirname, abspath, split
import logging

from sklearn.neighbors import KDTree
import yaml

from .base_dataset import BaseDataset, BaseDatasetSplit
from .utils import DataProcessing
from ..utils import make_dir, DATASET

logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s - %(asctime)s - %(module)s - %(message)s',
)
log = logging.getLogger(__name__)


class SemanticKITTI_PCC(BaseDataset):
    """This class is used to create a dataset based on the SemanticKitti
    dataset, and used in visualizer, training, or testing.

    The dataset is best for semantic scene understanding.
    """

    def __init__(self,
                 dataset_path='/data/PCC_Open3DML/dev_repo/skitti_PCC',
                 name='SemanticKITTI_PCC',
                 cache_dir='./logs/cache',
                 use_cache=False,
                 class_weights=[3745937, 158410323, 52152331, 45737069, 4095093, 1861458, 1261051, 260402, 242560, 2325584, 1845428, 1365698, 856921, 555396, 16565622, 10325419, 6727019, 221218, 3855813],
                 ignored_label_inds=[0],
                 test_result_folder='./test',
                 test_split=['60', '61', '62', '63', '64', '65', '66', '67', '68', '69', '60', '61', '62', '63', '64', '65', '66', '67', '68', '69', '70', '71', '72', '73', '74', '75', '76', '77', '78', '79', '80', '81', '82', '83', '84', '85', '86', '87', '88', '89', '90', '91', '92', '93', '94', '95', '96', '97', '98', '99', '100', '101', '102', '103', '104', '105', '106', '107', '108', '109', '110', '111', '112', '113', '114', '115', '116', '117', '134', '135', '136', '137', '138', '139', '140', '141', '142', '143', '144', '145', '146', '147', '148', '149'],
                 training_split=['00', '01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31', '32', '33', '34', '35', '36', '37', '38', '39', '40', '41', '42', '43', '44', '45', '46', '47', '48', '49', '50', '51', '52', '53', '54', '55', '56', '57', '58', '118', '119', '120', '121', '122', '123', '124', '125', '126', '127', '128', '129', '130', '131', '132', '133'],
                 all_split=['00', '01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31', '32', '33', '34', '35', '36', '37', '38', '39', '40', '41', '42', '43', '44', '45', '46', '47', '48', '49', '50', '51', '52', '53', '54', '55', '56', '57', '58', '59', '60', '61', '62', '63', '64', '65', '66', '67', '68', '69', '60', '61', '62', '63', '64', '65', '66', '67', '68', '69', '70', '71', '72', '73', '74', '75', '76', '77', '78', '79', '80', '81', '82', '83', '84', '85', '86', '87', '88', '89', '90', '91', '92', '93', '94', '95', '96', '97', '98', '99', '100', '101', '102', '103', '104', '105', '106', '107', '108', '109', '110', '111', '112', '113', '114', '115', '116', '117', '118', '119', '120', '121', '122', '123', '124', '125', '126', '127', '128', '129', '130', '131', '132', '133', '134', '135', '136', '137', '138', '139', '140', '141', '142', '143', '144', '145', '146', '147', '148', '149'],
                 validation_split=['59'],
                 **kwargs):
        """Initialize the function by passing the dataset and other details.

        Args:
            dataset_path: The path to the dataset to use.
            name: The name of the dataset (Semantic3D in this case).
            cache_dir: The directory where the cache is stored.
            use_cache: Indicates if the dataset should be cached.
            class_weights: The class weights to use in the dataset.
            ignored_label_inds: A list of labels that should be ignored in the dataset.
            test_result_folder: The folder where the test results should be stored.

        Returns:
            class: The corresponding class.
        """
        super().__init__(dataset_path=dataset_path,
                         name=name,
                         cache_dir=cache_dir,
                         use_cache=use_cache,
                         class_weights=class_weights,
                         ignored_label_inds=ignored_label_inds,
                         test_result_folder=test_result_folder,
                         test_split=test_split,
                         training_split=training_split,
                         validation_split=validation_split,
                         all_split=all_split,
                         **kwargs)

        cfg = self.cfg

        self.label_to_names = self.get_label_to_names()
        self.num_classes = len(self.label_to_names)

        data_config = join(dirname(abspath(__file__)), '_resources/',
                           'semantic-kitti-pcc.yaml')
        DATA = yaml.safe_load(open(data_config, 'r'))
        remap_dict = DATA["learning_map_inv"]

        # make lookup table for mapping
        max_key = max(remap_dict.keys())
        remap_lut = np.zeros((max_key + 100), dtype=np.int32)
        remap_lut[list(remap_dict.keys())] = list(remap_dict.values())

        remap_dict_val = DATA["learning_map"]
        max_key = max(remap_dict_val.keys())
        remap_lut_val = np.zeros((max_key + 100), dtype=np.int32)
        remap_lut_val[list(remap_dict_val.keys())] = list(
            remap_dict_val.values())

        self.remap_lut_val = remap_lut_val
        self.remap_lut = remap_lut

    @staticmethod
    def get_label_to_names():
        """Returns a label to names dictionary object.

        Returns:
            A dict where keys are label numbers and
            values are the corresponding names.
        """
        label_to_names = {
            0: 'unassigned',
            1: 'stairway',
            2: 'door',
            3: 'non exit door',
            4: 'fire door',
            5: 'window',
            6: 'roof access',
            7: 'exit sign',
            8: 'emergency lighting',
            9: 'smoke detector',
            10: 'extinguisher',
            11: 'fire alarm',
            12: 'person',
            13: 'AED',
            14: 'sprinkler',
            15: 'standpipe',
            16: 'utility shutoffs - electric',
            17: 'elevator',
            18: 'hydrant',
            19: 'gas shutoff'
        }
        return label_to_names

    def get_split(self, split):
        """Returns a dataset split.

        Args:
            split: A string identifying the dataset split that is usually one of
            'training', 'test', 'validation', or 'all'.

        Returns:
            A dataset split object providing the requested subset of the data.
        """
        return SemanticKITTISplit(self, split=split)

    def is_tested(self, attr):
        """Checks if a datum in the dataset has been tested.

        Args:
            attr: The attribute that needs to be checked.

        Returns:
            If the datum attribute is tested, then return the path where the
                attribute is stored; else, returns false.
        """
        cfg = self.cfg
        name = attr['name']
        name_seq, name_points = name.split("_")
        test_path = join(cfg.test_result_folder, 'sequences')
        save_path = join(test_path, name_seq, 'predictions')
        test_file_name = name_points
        store_path = join(save_path, name_points + '.label')
        if exists(store_path):
            print("{} already exists.".format(store_path))
            return True
        else:
            return False

    def save_test_result(self, results, attr):
        """Saves the output of a model.

        Args:
            results: The output of a model for the datum associated with the attribute passed.
            attr: The attributes that correspond to the outputs passed in results.
        """
        cfg = self.cfg
        name = attr['name']
        name_seq, name_points = name.split("_")

        test_path = join(cfg.test_result_folder, 'sequences')
        make_dir(test_path)
        save_path = join(test_path, name_seq, 'predictions')
        make_dir(save_path)
        test_file_name = name_points
        pred = results['predict_labels']

        for ign in cfg.ignored_label_inds:
            pred[pred >= ign] += 1

        store_path = join(save_path, name_points + '.label')

        pred = self.remap_lut[pred].astype(np.uint32)
        pred.tofile(store_path)

    def save_test_result_kpconv(self, results, inputs):
        cfg = self.cfg
        for j in range(1):
            name = inputs['attr']['name']
            name_seq, name_points = name.split("_")

            test_path = join(cfg.test_result_folder, 'sequences')
            make_dir(test_path)
            save_path = join(test_path, name_seq, 'predictions')
            make_dir(save_path)

            test_file_name = name_points

            proj_inds = inputs['data'].reproj_inds[0]
            probs = results[proj_inds, :]

            pred = np.argmax(probs, 1)

            store_path = join(save_path, name_points + '.label')
            pred = pred + 1
            pred = self.remap_lut[pred].astype(np.uint32)
            pred.tofile(store_path)

    def get_split_list(self, split):
        """Returns a dataset split.

        Args:
            split: A string identifying the dataset split that is usually one of
            'training', 'test', 'validation', or 'all'.

        Returns:
            A dataset split object providing the requested subset of the data.

        Raises:
            ValueError: Indicates that the split name passed is incorrect. The split name should be one of
            'training', 'test', 'validation', or 'all'.
        """
        cfg = self.cfg
        dataset_path = cfg.dataset_path
        file_list = []

        if split in ['train', 'training']:
            seq_list = cfg.training_split
        elif split in ['test', 'testing']:
            seq_list = cfg.test_split
        elif split in ['val', 'validation']:
            seq_list = cfg.validation_split
        elif split in ['all']:
            seq_list = cfg.all_split
        else:
            raise ValueError("Invalid split {}".format(split))

        for seq_id in seq_list:
            pc_path = join(dataset_path, 'dataset', 'sequences', seq_id,
                           'velodyne')
            file_list.append(
                [join(pc_path, f) for f in np.sort(os.listdir(pc_path))])

        file_list = np.concatenate(file_list, axis=0)

        return file_list


class SemanticKITTISplit(BaseDatasetSplit):

    def __init__(self, dataset, split='training'):
        super().__init__(dataset, split=split)
        log.info("Found {} pointclouds for {}".format(len(self.path_list),
                                                      split))
        self.remap_lut_val = dataset.remap_lut_val

    def __len__(self):
        return len(self.path_list)

    def get_data(self, idx):
        pc_path = self.path_list[idx]
        points = DataProcessing.load_pc_kitti(pc_path)

        dir, file = split(pc_path)
        label_path = join(dir, '../labels', file[:-4] + '.label')
        if not exists(label_path):
            labels = np.zeros(np.shape(points)[0], dtype=np.int32)
            if self.split not in ['test', 'all']:
                raise FileNotFoundError(f' Label file {label_path} not found')

        else:
            labels = DataProcessing.load_label_kitti(
                label_path, self.remap_lut_val).astype(np.int32)

        data = {
            'point': points[:, 0:3],
            'feat': None,
            'label': labels,
        }

        return data

    def get_attr(self, idx):
        pc_path = self.path_list[idx]
        dir, file = split(pc_path)
        _, seq = split(split(dir)[0])
        name = '{}_{}'.format(seq, file[:-4])

        pc_path = str(pc_path)
        attr = {'idx': idx, 'name': name, 'path': pc_path, 'split': self.split}
        return attr


DATASET._register_module(SemanticKITTI_PCC)
