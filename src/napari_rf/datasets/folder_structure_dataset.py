import os
import re

from omegaconf import ListConfig
from skimage import io


class FolderStructureDataset:
    def __init__(self, cfg):

        self.parse_config(cfg)

        self.normalise = cfg["normalise_img"]

        self.fills = {
            "m": [],
            "z": [],
            "t": [],
            "c": [],
        }

        paths = []
        for p, dirnames, filenames in os.walk(self.root):
            if filenames:
                for f in filenames:

                    path = f"{p}/{f}"

                    if "BACKGR" not in path.upper():
                        if f.upper().endswith("TIF") or f.upper().endswith(
                            "TIFF"
                        ):
                            paths.append(path)

        temp = []
        for path in paths:
            match = re.findall(r"channel_(\d+)", path)
            if match:
                channel = int(match[0])
            else:
                channel = None

            match = re.findall(r"frame_(\d+)", path)
            if match:
                frame = int(match[0])
            else:
                frame = None

            match = re.findall(r"z_level_(\d+)", path)
            if match:
                z = int(match[0])
            else:
                z = None

            match1 = re.findall(r"position_(\d+)", path)
            match2 = re.findall(r"pos_(\d+)", path)
            if match1:
                position = int(match1[0])
            elif match2:
                position = int(match2[0])
            else:
                position = None

            if (
                (not self.channels)
                | (channel is None)
                | (channel in self.channels)
            ):
                if (
                    (not self.frames)
                    | (frame is None)
                    | (frame in self.frames)
                ):
                    if (
                        (not self.positions)
                        | (position is None)
                        | (position in self.positions)
                    ):
                        if (
                            (not self.z_levels)
                            | (z is None)
                            | (z in self.z_levels)
                        ):
                            if z is None:
                                z = 0
                            if position is None:
                                position = 0
                            if channel is None:
                                channel = 0
                            if frame is None:
                                frame = 0

                            self.fills["z"].append(z)
                            self.fills["m"].append(position)
                            self.fills["c"].append(channel)
                            self.fills["t"].append(frame)
                            temp.append(
                                [
                                    path,
                                    f"position_{position}/z_level_{z}/channel_{channel}/frame_{frame}.tif",
                                ]
                            )

        self.indices = temp

        self.positions = list(set(self.fills["m"]))
        self.channels = list(set(self.fills["c"]))
        self.frames = list(set(self.fills["t"]))
        self.z_levels = list(set(self.fills["z"]))
        self.fills = {a: len(str(max(b))) for a, b in self.fills.items()}

    def parse_config(self, cfg):

        self.root = cfg["dataset_path"]
        if isinstance(cfg["channels_to_segment"], ListConfig):
            self.channels = cfg["channels_to_segment"]
        elif isinstance(cfg["channels_to_segment"], int):
            self.channels = [cfg["channels_to_segment"]]
        else:
            self.channels = []

        if isinstance(cfg["positions"], ListConfig):
            self.positions = cfg["positions"]
        elif isinstance(cfg["positions"], int):
            self.positions = [cfg["positions"]]
        else:
            self.positions = []

        if isinstance(cfg["frames"], ListConfig):
            self.frames = cfg["frames"]
        elif isinstance(cfg["frames"], int):
            self.frames = [cfg["frames"]]
        else:
            self.frames = []

        if isinstance(cfg["z_levels"], ListConfig):
            self.z_levels = cfg["z_levels"]
        elif isinstance(cfg["z_levels"], int):
            self.z_levels = [cfg["z_levels"]]
        else:
            self.z_levels = []

    def get_save_structure(self):
        paths = []

        for position in self.positions:
            for z_level in self.z_levels:
                for channel in self.channels:
                    m = str(position).zfill(self.fills["m"])
                    z = str(z_level).zfill(self.fills["z"])
                    c = str(channel).zfill(self.fills["c"])

                    paths.append(f"position_{m}/z_level_{z}/channel_{c}")
        return paths

    def normalise_image(self, img):
        return (img / 65535) - 0.5

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        path, savepath = self.indices[index]
        img = io.imread(path)
        if self.normalise:
            img = self.normalise_image(img)

        return img, savepath

    def __iter__(self):
        for i in range(len(self.indices)):
            yield self.__getitem__(i)
