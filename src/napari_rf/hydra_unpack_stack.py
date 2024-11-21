import os, sys
import hydra
from omegaconf import DictConfig
from typing import Optional
from skimage import io
import warnings
warnings.filterwarnings('ignore')
from napari_rf.datasets.datasets import get_dataset
from tqdm import tqdm

@hydra.main(version_base='1.2', config_path='../../config', config_name='batch_classify_config.yaml')
def main(cfg: DictConfig) -> Optional[float]:
    working_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir

    dataset = get_dataset(cfg)
    paths = dataset.get_save_structure()
    for p in paths:
        os.makedirs(f"{working_dir}/{p}", exist_ok=True)

    for img, save_path in tqdm(dataset):
        io.imsave(f"{working_dir}/{save_path}", img)


if __name__ == "__main__":
    # sys.argv.append('hydra.run.dir=/media/philipp/seagate5tb/hydra2/')
    main()
