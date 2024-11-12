import os
import hydra
from omegaconf import DictConfig
from typing import Optional
from skimage import io
import warnings
warnings.filterwarnings('ignore')
from napari_rf.datasets.datasets import get_dataset
from napari_rf.features import FeatureCreator
from joblib import load
import numpy as np
from tqdm import tqdm


@hydra.main(version_base='1.2', config_path='../../config', config_name='batch_classify_config.yaml')
def main(cfg: DictConfig) -> Optional[float]:
    working_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    print(working_dir)
    dataset = get_dataset(cfg)
    paths = dataset.get_save_structure()
    for p in paths:
        os.makedirs(f"{working_dir}/{p}", exist_ok=True)

    clf = load(cfg['classifier'])
    feature_creator = FeatureCreator()

    for img, save_path in tqdm(dataset):
        if not isinstance(img, list):
            img = [img]
        features = feature_creator.make_simple_features(*img)
        out = clf.predict_segmenter(features)
        io.imsave(f"{working_dir}/{save_path}", np.argmax(out, axis=0))


if __name__ == "__main__":
    # sys.argv.append('hydra.run.dir=/media/philipp/WD1/sac6/segmentation')
    main()