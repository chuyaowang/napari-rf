from omegaconf import ListConfig
from pims import ND2_Reader as nd2


class PooledNd2Dataset:
    def __init__(self, cfg):

        self.stack = nd2(cfg["dataset_path"])
        self.normalise = cfg["normalise_img"]

        if isinstance(cfg["positions"], ListConfig):
            self.positions = cfg["positions"]
        elif isinstance(cfg["positions"], int):
            self.positions = [cfg["positions"]]
        else:
            self.positions = list(range(self.stack.sizes["m"]))

        if isinstance(cfg["frames"], ListConfig):
            self.frames = cfg["frames"]
        elif isinstance(cfg["frames"], int):
            self.frames = [cfg["frames"]]
        else:
            self.frames = list(range(self.stack.sizes["t"]))

        if isinstance(cfg["z_levels"], ListConfig):
            self.z_levels = cfg["z_levels"]
        elif isinstance(cfg["z_levels"], int):
            self.z_levels = [cfg["z_levels"]]
        else:
            try:
                self.z_levels = list(range(self.stack.sizes["z"]))
            except KeyError:
                self.z_levels = [0]

        if isinstance(cfg["channels_to_segment"], ListConfig):
            self.channels = cfg["channels_to_segment"]
        elif isinstance(cfg["channels_to_segment"], int):
            self.channels = [cfg["channels_to_segment"]]
        else:
            self.channels = list(range(self.stack.sizes["c"]))

        self.indices = []
        for position in self.positions:
            for z_level in self.z_levels:
                for frame in self.frames:
                    self.indices.append(
                        [position, z_level, self.channels, frame]
                    )

        self.fills = {a: len(str(b)) for a, b in self.stack.sizes.items()}
        for axis in "zmtxy":
            if axis not in self.fills:
                self.fills[axis] = 0

    def get_save_structure(self):
        paths = []

        for position in self.positions:
            for z_level in self.z_levels:
                m = str(position).zfill(self.fills["m"])
                z = str(z_level).zfill(self.fills["z"])
                paths.append(f"position_{m}/z_level_{z}/channels_pooled")
        return paths

    def normalise_image(self, img):
        return (img / 65535) - 0.5

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        [position, z_level, channel, frame] = self.indices[index]
        imgs = []
        for c in channel:
            imgs.append(
                self.stack.get_frame_2D(m=position, z=z_level, c=c, t=frame)[:]
            )

        m = str(position).zfill(self.fills["m"])
        z = str(z_level).zfill(self.fills["z"])
        t = str(frame).zfill(self.fills["t"])

        if self.normalise:
            imgs = [self.normalise_image(img) for img in imgs]

        return imgs, f"position_{m}/z_level_{z}/channels_pooled/frame_{t}.tif"

    def __iter__(self):
        for i in range(len(self.indices)):
            yield self.__getitem__(i)
