"""
"""
from typing import TYPE_CHECKING

from qtpy.QtWidgets import QVBoxLayout, QPushButton, QWidget
from qtpy.QtWidgets import QFileDialog

from joblib import load, dump

if TYPE_CHECKING:
    import napari


class RFWidget(QWidget):
    # your QWidget.__init__ can optionally request the napari viewer instance
    # use a type annotation of 'napari.viewer.Viewer' for any parameter
    def __init__(self, viewer: "napari.viewer.Viewer"):
        super().__init__()
        self.viewer = viewer

        self.clf = None

        btn_create_features = QPushButton("create features")
        btn_create_features.clicked.connect(self.create_features)

        btn_train = QPushButton("train random forest")
        btn_train.clicked.connect(self.train)

        btn_load = QPushButton("load classifier")
        btn_load.clicked.connect(self.load)

        btn_save = QPushButton("save classifier")
        btn_save.clicked.connect(self.save)
        btn_save.setDisabled(True)

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
        source_file = QFileDialog.getOpenFileName(
            self,
            'open classifier',
            '/home/philipp/',
            'classifiers (*.joblib)'
        )
        self.clf = load(source_file[0])
        print(f"loaded {source_file[0]}")

    def save(self):
        save_path = QFileDialog.getSaveFileName(
            self,
            'save file',
            '',
            'all files (*)'
        )
        if self.clf is not None:
            dump(self.clf, save_path[0])