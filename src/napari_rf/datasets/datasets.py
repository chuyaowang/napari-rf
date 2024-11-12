import os.path

from napari_rf.datasets.folder_structure_dataset import FolderStructureDataset
from napari_rf.datasets.nd2_dataset import Nd2Dataset
from napari_rf.datasets.single_image_dataset import SingleImageDataset


def get_dataset(cfg):
    if cfg['dataset_path'].lower().endswith('nd2'):
        print('opening nd2 stack')
        return Nd2Dataset(cfg)
    elif os.path.isdir(cfg['dataset_path']):
        print('opening folders')
        return FolderStructureDataset(cfg)
    elif cfg['dataset_path'].upper().endswith('TIF') | cfg['dataset_path'].upper().endswith('TIFF'):
        print('fetching single image')
        return SingleImageDataset(cfg)
    return ValueError('dataset not found')