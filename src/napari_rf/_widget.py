"""
"""
from typing import TYPE_CHECKING

from qtpy.QtWidgets import QVBoxLayout, QPushButton, QWidget
from qtpy.QtWidgets import QFileDialog

import numpy as np

from joblib import load, dump

from napari_rf.RF import RF
from napari_rf.features import FeatureCreator

if TYPE_CHECKING:
    import napari


class RFWidget(QWidget):
    # your QWidget.__init__ can optionally request the napari viewer instance
    # use a type annotation of 'napari.viewer.Viewer' for any parameter
    def __init__(self, viewer: "napari.viewer.Viewer"):
        super().__init__()
        self.viewer = viewer

        self.clf = RF()
        self.feature_creator = FeatureCreator()

        self.btn_create_features = QPushButton("create features")
        self.btn_create_features.clicked.connect(self.create_features)

        self.btn_train = QPushButton("train random forest")
        self.btn_train.clicked.connect(self.train)

        self.btn_apply_rf = QPushButton("apply random forest")
        self.btn_apply_rf.clicked.connect(self.apply_rf)
        self.btn_apply_rf.setDisabled(True)

        self.btn_load = QPushButton("load classifier")
        self.btn_load.clicked.connect(self.load)

        self.btn_save = QPushButton("save classifier")
        self.btn_save.clicked.connect(self.save)
        self.btn_save.setDisabled(True)

        self.setLayout(QVBoxLayout())
        self.layout().addWidget(self.btn_create_features)
        self.layout().addWidget(self.btn_train)
        self.layout().addWidget(self.btn_apply_rf)
        self.layout().addWidget(self.btn_load)
        self.layout().addWidget(self.btn_save)

    def create_features(self):
        img = self.viewer.layers.selection.active.data
        features = self.feature_creator.make_simple_features(img)
        self.viewer.add_image(np.moveaxis(features, -1, 0), name='features')

    def train(self):
        if 'Labels' in self.viewer.layers:
            training_labels = self.viewer.layers['Labels'].data
        else:
            raise Exception('training labels must be in a layer called "Labels"')

        if not 'features' in self.viewer.layers:
            self.create_features()

        features = np.moveaxis(self.viewer.layers['features'].data, 0, -1)

        result = self.clf.train(training_labels, features)

        self.btn_save.setDisabled(False)
        self.btn_apply_rf.setDisabled(False)

        self.viewer.add_image(result, name=f'segmentation probabilities')


    def apply_rf(self):
        if not 'features' in self.viewer.layers:
            self.create_features()
        features = np.moveaxis(self.viewer.layers['features'].data, 0, -1)
        result = self.clf.predict_segmenter(features)
        self.viewer.add_image(result, name=f'segmentation probabilities')

    def load(self):
        source_file = QFileDialog.getOpenFileName(
            self,
            'open classifier',
            '/home/philipp/',
            'classifiers (*.joblib)'
        )
        self.clf = load(source_file[0])
        print(f"loaded {source_file[0]}")
        self.btn_save.setDisabled(False)
        self.btn_apply_rf.setDisabled(False)

    def save(self):
        save_path = QFileDialog.getSaveFileName(
            self,
            'save file',
            '',
            'all files (*)'
        )
        if self.clf is not None:
            dump(self.clf, save_path[0])