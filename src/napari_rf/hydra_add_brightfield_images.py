import os
from skimage import io
import warnings
warnings.filterwarnings('ignore')

from tqdm import tqdm
import yaml
from pims import ND2_Reader as nd2
import re
from pathlib import Path

def main(working_dir):
    for path, subdir, files in os.walk(working_dir):
        if files:
            for file in tqdm(files):
                filepath = f"{path}/{file}"
                if 'config.yaml' in file:
                    with open(filepath) as f:
                        d = yaml.safe_load(f)
                        stack = nd2(d['dataset_path'])
                if file.endswith('tif'):
                    match = re.findall('position_(\d+).*z_level_(\d+).*frame_(\d+)', filepath)
                    if match:
                        m, z, t = match[0]
                        savepath = filepath.replace('channels_pooled', 'bf')
                        os.makedirs(Path(savepath).parent, exist_ok=True)
                        img = stack.get_frame_2D(m=int(m), z=int(z), t=int(t), c=0)
                        io.imsave(savepath, img)


if __name__ == "__main__":
    working_dir = '/media/philipp/seagate5tb/hydra2/'
    main(working_dir)
