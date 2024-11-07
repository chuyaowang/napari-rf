import sys, os
import hydra
from omegaconf import DictConfig
from typing import Optional
from joblib import load
from skimage import io
from tqdm import tqdm
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from napari_rf.features import FeatureCreator
from napari_rf.nd2_dataset import Nd2Dataset
def make_output_folder(working_dir, position, z_level, channel,):
    os.makedirs(f"{working_dir}/position_{position}/z_level_{z_level}/channel_{channel}", exist_ok=True)

def save_img(img, working_dir, position, z_level, channel, frame):
    io.imsave(f"{working_dir}/position_{position}/z_level_{z_level}/channel_{channel}/frame_{frame}.tif", img)

@hydra.main(version_base='1.2', config_path='../../config', config_name='batch_classify_config.yaml')
def main(cfg: DictConfig) -> Optional[float]:

    working_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir

    dataset = Nd2Dataset(cfg)
    paths = dataset.get_save_structure()
    for p in paths:
        print(p)
        os.makedirs(f"{working_dir}/{p}", exist_ok=True)

    clf = load(cfg['classifier'])
    feature_creator = FeatureCreator()

    for img, save_path in tqdm(dataset):
        features = feature_creator.make_simple_features(img)
        out = clf.predict_segmenter(features)
        print(f"{working_dir}/{save_path}")
        io.imsave(f"{working_dir}/{save_path}", np.argmax(out, axis=0))


if __name__ == "__main__":
    # sys.argv.append('hydra.run.dir=/media/philipp/WD1/sac6/segmentation')
    main()