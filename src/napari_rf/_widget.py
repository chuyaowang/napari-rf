"""
"""
from typing import TYPE_CHECKING

from qtpy.QtWidgets import QVBoxLayout, QPushButton, QWidget

if TYPE_CHECKING:
    import napari


class RFWidget(QWidget):
    # your QWidget.__init__ can optionally request the napari viewer instance
    # use a type annotation of 'napari.viewer.Viewer' for any parameter
    def __init__(self, viewer: "napari.viewer.Viewer"):
        super().__init__()
        self.viewer = viewer

        btn_create_features = QPushButton("create features")
        btn_create_features.clicked.connect(self.create_features)

        btn_train = QPushButton("train random forest")
        btn_train.clicked.connect(self.train)

        btn_load = QPushButton("load classifier")
        btn_load.clicked.connect(self.load)

        btn_save = QPushButton("save classifier")
        btn_save.clicked.connect(self.save)

        self.setLayout(QVBoxLayout())
        self.layout().addWidget(btn_create_features)
        self.layout().addWidget(btn_train)
        self.layout().addWidget(btn_load)
        self.layout().addWidget(btn_save)

    def create_features(self):
        pass

    def train(self):
        pass

    def load(self):
        pass

    def save(self):
        pass