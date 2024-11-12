from skimage import io
from pathlib import Path

class SingleImageDataset():
    def __init__(self, cfg):

        self.parse_config(cfg)

        self.normalise = cfg['normalise_img']
        self.indices = [0]
        self.path = cfg['dataset_path']
        self.get_save_structure()


    def parse_config(self, cfg):
        pass


    def get_save_structure(self):
        save_name = Path(self.path).stem + Path(self.path).suffix
        self.save_name = save_name

        return ['']

    def normalise_image(self, img):
        return (img / 65535) - 0.5


    def __len__(self):
        return len(self.indices)


    def __getitem__(self, index):
        img = io.imread(self.path)
        if self.normalise:
            img = self.normalise_image(img)

        return img, self.save_name


    def __iter__(self):
        for i in range(len(self.indices)):
            yield self.__getitem__(i)
