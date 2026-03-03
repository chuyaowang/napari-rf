"""
"""

from typing import TYPE_CHECKING

import numpy as np
from joblib import dump, load
from qtpy.QtWidgets import QFileDialog, QPushButton, QVBoxLayout, QWidget

from napari_rf.features import FeatureCreator
from napari_rf.RF import RF
from napari.qt.threading import thread_worker
from napari.utils import progress

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
        self.features = None

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
        
        self.btn_create_features.setEnabled(False)
        self.btn_create_features.setText("Creating features...")

        # Initialize progress bar on main thread
        pbar = progress(desc="Creating Features")

        @thread_worker
        def _create_features_worker():
            # The creator now yields progress info or the final result
            gen = self.feature_creator.make_simple_features(img)
            for val in gen:
                yield val

        def _on_yielded(val):
            if isinstance(val, tuple):
                step, total, desc = val
                pbar.total = total
                pbar.set_description(desc)
                pbar.update(1)
            else:
                # Final result (the features array)
                self.features = val
                self.viewer.add_image(np.moveaxis(self.features, -1, 0), name="features")

        def _on_finished():
            pbar.close()
            self.btn_create_features.setEnabled(True)
            self.btn_create_features.setText("create features")

        worker = _create_features_worker()
        worker.yielded.connect(_on_yielded)
        worker.finished.connect(_on_finished)
        worker.start()

    def train(self):
        if "Labels" in self.viewer.layers:
            training_labels = self.viewer.layers["Labels"].data
        else:
            raise Exception(
                'training labels must be in a layer called "Labels"'
            )

        if self.features is None:
            self.create_features()

        # Check if labels are 3D (Z, Y, X) but features are 2D (Y, X, C)
        # This happens if the Labels layer was created based on the Features layer shape.
        if training_labels.ndim == 3 and self.features.ndim == 3:
            if training_labels.shape[0] == self.features.shape[-1]:
                training_labels = np.max(training_labels, axis=0)

        result = self.clf.train(training_labels, self.features)

        self.btn_save.setDisabled(False)
        self.btn_apply_rf.setDisabled(False)

        self.viewer.add_image(result, name="segmentation probabilities")

    def apply_rf(self):
        if self.features is None:
            self.create_features()
        result = self.clf.predict_segmenter(self.features)
        self.viewer.add_image(result, name="segmentation probabilities")

    def load(self):
        source_file = QFileDialog.getOpenFileName(
            self, "open classifier", "/home/philipp/", "classifiers (*.joblib)"
        )
        self.clf = load(source_file[0])
        print(f"loaded {source_file[0]}")
        self.btn_save.setDisabled(False)
        self.btn_apply_rf.setDisabled(False)

    def save(self):
        save_path = QFileDialog.getSaveFileName(
            self, "save file", "classifier.joblib", "classifiers (*.joblib)"
        )
        if self.clf is not None:
            dump(self.clf, save_path[0])
