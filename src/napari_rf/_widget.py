"""
"""
import os
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from joblib import dump, load
from qtpy.QtWidgets import QFileDialog, QPushButton, QVBoxLayout, QWidget
from skimage.io import imsave

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
        self.image_path = None
        self.image_name = "image"

        self.btn_create_features = QPushButton("Create Features")
        self.btn_create_features.setToolTip("Extract multiscale features (intensity, edges, texture) from the active image layer.")
        self.btn_create_features.clicked.connect(self.create_features)

        self.btn_train = QPushButton("Train Random Forest")
        self.btn_train.setToolTip("Train the classifier using the 'Labels' and 'Features' layers.")
        self.btn_train.clicked.connect(self.train)

        self.btn_apply_rf = QPushButton("Apply Random Forest")
        self.btn_apply_rf.setToolTip("Apply the current classifier to the 'Features' layer.")
        self.btn_apply_rf.clicked.connect(self.apply_rf)
        self.btn_apply_rf.setDisabled(True)

        self.btn_load = QPushButton("Load Classifier")
        self.btn_load.setToolTip("Load a previously saved .joblib classifier.")
        self.btn_load.clicked.connect(self.load)

        self.btn_save = QPushButton("Save Classifier")
        self.btn_save.setToolTip("Save the current trained classifier to a .joblib file.")
        self.btn_save.clicked.connect(self.save)
        self.btn_save.setDisabled(True)

        self.btn_save_labels = QPushButton("Save Labels")
        self.btn_save_labels.setToolTip("Save the manually drawn labels as a TIFF file in a subfolder.")
        self.btn_save_labels.clicked.connect(self.save_labels)

        self.btn_save_preds = QPushButton("Save Predictions")
        self.btn_save_preds.setToolTip("Save predicted class labels and probability maps as TIFF files in a subfolder.")
        self.btn_save_preds.clicked.connect(self.save_predictions)

        self.setLayout(QVBoxLayout())
        self.layout().addWidget(self.btn_create_features)
        self.layout().addWidget(self.btn_train)
        self.layout().addWidget(self.btn_apply_rf)
        self.layout().addWidget(self.btn_load)
        self.layout().addWidget(self.btn_save)
        self.layout().addWidget(self.btn_save_labels)
        self.layout().addWidget(self.btn_save_preds)

        # Connect to layer events to manage Save button states
        self.viewer.layers.events.inserted.connect(self._update_button_states)
        self.viewer.layers.events.removed.connect(self._update_button_states)
        self._update_button_states()

    def _update_button_states(self, event=None):
        """Enable/disable save buttons based on layer existence."""
        self.btn_save_labels.setEnabled("Labels" in self.viewer.layers)
        self.btn_save_preds.setEnabled("Segmentation Probabilities" in self.viewer.layers)

    def create_features(self):
        active_layer = self.viewer.layers.selection.active
        if active_layer is None:
            return
        
        img = active_layer.data
        
        # Track image path and clean name (stem)
        self.image_path = getattr(active_layer, "source", None)
        if self.image_path:
            self.image_path = getattr(self.image_path, "path", None)
            self.image_name = Path(self.image_path).stem
        else:
            self.image_name = active_layer.name
        
        self.btn_create_features.setEnabled(False)
        self.btn_create_features.setText("Creating Features...")

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
                self.viewer.add_image(np.moveaxis(self.features, -1, 0), name="Features")

        def _on_finished():
            pbar.close()
            self.btn_create_features.setEnabled(True)
            self.btn_create_features.setText("Create Features")

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

        self.viewer.add_image(result, name="Segmentation Probabilities")

    def apply_rf(self):
        if self.features is None:
            self.create_features()
        result = self.clf.predict_segmenter(self.features)
        self.viewer.add_image(result, name="Segmentation Probabilities")

    def load(self):
        source_file = QFileDialog.getOpenFileName(
            self, "Open Classifier", "/home/philipp/", "classifiers (*.joblib)"
        )
        if source_file[0]:
            self.clf = load(source_file[0])
            print(f"Loaded {source_file[0]}")
            self.btn_save.setDisabled(False)
            self.btn_apply_rf.setDisabled(False)

    def save(self):
        save_path = QFileDialog.getSaveFileName(
            self, "Save File", "classifier.joblib", "classifiers (*.joblib)"
        )
        if self.clf is not None and save_path[0]:
            dump(self.clf, save_path[0])

    def _get_save_dir(self):
        """Create and return a subfolder named after the image."""
        if self.image_path:
            parent = Path(self.image_path).parent
            save_dir = parent / self.image_name
        else:
            save_dir = Path.home() / self.image_name
        
        save_dir.mkdir(parents=True, exist_ok=True)
        return save_dir

    def save_labels(self):
        if "Labels" not in self.viewer.layers:
            print("No 'Labels' layer found.")
            return
        
        labels = self.viewer.layers["Labels"].data
        save_dir = self._get_save_dir()
        save_name = f"{self.image_name}_labels.tif"
        save_path = save_dir / save_name
        
        # Using uint8 for labels is sufficient and reduces low-contrast warnings
        imsave(str(save_path), labels.astype(np.uint8), check_contrast=False)
        print(f"Saved labels to {save_path}")

    def save_predictions(self):
        if "Segmentation Probabilities" not in self.viewer.layers:
            print("No 'Segmentation Probabilities' layer found.")
            return
            
        probs = self.viewer.layers["Segmentation Probabilities"].data
        # probs shape is (C, Y, X) or (C, Z, Y, X)
        # class labels are the argmax over channel dimension (axis 0)
        class_labels = np.argmax(probs, axis=0).astype(np.uint8)
        
        save_dir = self._get_save_dir()
        
        # Save class labels as uint8
        labels_path = save_dir / f"{self.image_name}_predictions_class.tif"
        imsave(str(labels_path), class_labels, check_contrast=False)
        
        # Save probabilities as float32
        probs_path = save_dir / f"{self.image_name}_predictions_probs.tif"
        imsave(str(probs_path), probs.astype(np.float32), check_contrast=False)
        
        print(f"Saved predictions to {save_dir}")
