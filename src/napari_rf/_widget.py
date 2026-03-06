"""
"""
import os
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from joblib import dump, load
from qtpy.QtWidgets import QComboBox, QLabel, QFileDialog, QPushButton, QVBoxLayout, QWidget
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
        self._last_processed_layer = None
        self._clf_ready = False

        # Layer Selection Drop-down
        self.layer_combo = QComboBox()
        self.layer_combo.setToolTip("Select the image layer to use for training and prediction.")
        self.layer_combo.currentIndexChanged.connect(self._on_layer_change)
        
        self.btn_train = QPushButton("Train Random Forest")
        self.btn_train.setToolTip("Train the classifier using the 'Labels' and features from the selected image.")
        self.btn_train.clicked.connect(self.train)

        self.btn_apply_rf = QPushButton("Apply Random Forest")
        self.btn_apply_rf.setToolTip("Apply the current classifier to the selected image stack.")
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

        # 2D Save button
        self.btn_save_preds = QPushButton("Save Predictions")
        self.btn_save_preds.setToolTip("Save predicted class labels and probability maps.")
        self.btn_save_preds.clicked.connect(self.save_predictions)

        # 3D Specific Save buttons
        self.btn_save_training_preds = QPushButton("Save Training Predictions")
        self.btn_save_training_preds.clicked.connect(self.save_training_predictions)
        self.btn_save_full_preds = QPushButton("Save Full Stack Predictions")
        self.btn_save_full_preds.clicked.connect(self.save_predictions)

        # Reset All button
        self.btn_reset = QPushButton("Reset All")
        self.btn_reset.setToolTip("Reset internal model, features, and caches to original state.")
        self.btn_reset.clicked.connect(self.reset_all)

        self.setLayout(QVBoxLayout())
        self.layout().addWidget(QLabel("Select Image Layer:"))
        self.layout().addWidget(self.layer_combo)
        self.layout().addWidget(self.btn_train)
        self.layout().addWidget(self.btn_apply_rf)
        self.layout().addWidget(self.btn_load)
        self.layout().addWidget(self.btn_save)
        self.layout().addWidget(self.btn_save_labels)
        self.layout().addWidget(self.btn_save_preds)
        self.layout().addWidget(self.btn_save_training_preds)
        self.layout().addWidget(self.btn_save_full_preds)
        self.layout().addWidget(self.btn_reset)

        # Connect to layer events to manage UI state
        self.viewer.layers.events.inserted.connect(self._on_layer_change)
        self.viewer.layers.events.removed.connect(self._on_layer_change)
        self._on_layer_change()

    def _on_layer_change(self, event=None):
        """Update the layer combo box and save button states."""
        import napari
        
        # 1. Update the Drop-down
        current_selection = self.layer_combo.currentText()
        self.layer_combo.blockSignals(True)
        self.layer_combo.clear()
        image_layers = [
            l.name for l in self.viewer.layers 
            if isinstance(l, napari.layers.Image) 
            and not any(x in l.name for x in ["Features", "Probabilities"])
        ]
        self.layer_combo.addItems(image_layers)
        if current_selection in image_layers:
            self.layer_combo.setCurrentText(current_selection)
        elif image_layers:
            self.layer_combo.setCurrentIndex(0)
        self.layer_combo.blockSignals(False)

        # 2. Detect dimensionality
        active_layer = self.get_selected_layer()
        is_3d = active_layer is not None and active_layer.data.ndim == 3
        
        # 3. Dynamic UI Renaming and Visibility
        if is_3d:
            self.btn_apply_rf.setText("Apply RF to All Slices")
            self.btn_save_preds.setVisible(False)
            self.btn_save_training_preds.setVisible(True)
            self.btn_save_full_preds.setVisible(True)
        else:
            self.btn_apply_rf.setText("Apply Random Forest")
            self.btn_save_preds.setVisible(True)
            self.btn_save_training_preds.setVisible(False)
            self.btn_save_full_preds.setVisible(False)

        # 4. Enable/Disable logic
        has_layers = active_layer is not None
        has_labels = "Labels" in self.viewer.layers
        
        self.btn_train.setEnabled(has_labels and has_layers)
        self.btn_save_labels.setEnabled(has_labels)
        # Apply button only enabled if a model exists AND there's an image to apply it to
        self.btn_apply_rf.setEnabled(self._clf_ready and has_layers)
        
        # Detect multi-channel layers
        has_full_preds = any("Segmentation Probabilities" in l.name for l in self.viewer.layers)
        has_train_preds = any("Training Probabilities" in l.name for l in self.viewer.layers)
        
        self.btn_save_preds.setEnabled(has_full_preds)
        self.btn_save_training_preds.setEnabled(has_train_preds)
        self.btn_save_full_preds.setEnabled(has_full_preds)

    def reset_all(self):
        """Reset internal state to default."""
        self.clf = RF()
        self.features = None
        self.image_path = None
        self.image_name = "image"
        self._last_processed_layer = None
        self._clf_ready = False
        self.btn_save.setDisabled(True)
        self._on_layer_change()
        print("Internal model and feature caches have been reset.")

    def get_selected_layer(self):
        """Returns the napari layer selected in the drop-down."""
        layer_name = self.layer_combo.currentText()
        if layer_name in self.viewer.layers:
            return self.viewer.layers[layer_name]
        return None

    def create_features(self, callback=None, indices=None):
        active_layer = self.get_selected_layer()
        if active_layer is None:
            return
        
        img = active_layer.data
        self.image_path = getattr(active_layer, "source", None)
        if self.image_path:
            self.image_path = getattr(self.image_path, "path", None)
            self.image_name = Path(self.image_path).stem
        else:
            self.image_name = active_layer.name
        
        self.btn_train.setEnabled(False)
        self.btn_apply_rf.setEnabled(False)

        pbar = progress(desc="Generating Features")

        @thread_worker
        def _create_features_worker():
            gen = self.feature_creator.make_simple_features(img, indices=indices)
            for val in gen:
                yield val

        def _on_yielded(val):
            if isinstance(val, tuple):
                step, total, desc = val
                pbar.total = total
                pbar.set_description(desc)
                pbar.update(1)
            else:
                self.features = val
                self._last_processed_layer = active_layer
                display_feats = self.features[-1] if self.features.ndim == 4 else self.features
                if "Features" in self.viewer.layers:
                    self.viewer.layers["Features"].data = np.moveaxis(display_feats, -1, 0)
                else:
                    self.viewer.add_image(np.moveaxis(display_feats, -1, 0), name="Features")

        def _on_finished():
            pbar.close()
            self._on_layer_change()
            if callback:
                callback()

        worker = _create_features_worker()
        worker.yielded.connect(_on_yielded)
        worker.finished.connect(_on_finished)
        worker.start()

    def train(self):
        active_layer = self.get_selected_layer()
        if active_layer is None or "Labels" not in self.viewer.layers:
            return

        training_labels = self.viewer.layers["Labels"].data
        is_3d = active_layer.data.ndim == 3
        
        # Robust Dimensionality Handling:
        # Handle labels created matching the 'Features' layer (multi-channel)
        # instead of the source Image layer.
        if not is_3d and training_labels.ndim == 3:
            # 2D image but labels are (C, Y, X)
            training_labels = np.max(training_labels, axis=0)
        elif is_3d and training_labels.ndim == 4:
            # 3D stack but labels are (Z, C, Y, X)
            training_labels = np.max(training_labels, axis=1)

        indices = None
        if is_3d:
            indices = np.where(np.any(training_labels > 0, axis=(1, 2)))[0].tolist()
            if not indices:
                print("No labels found. Please draw labels first.")
                return

        def _do_train():
            # Use the corrected labels (filtered for 3D sparse)
            labels = training_labels
            if is_3d:
                labels = labels[indices]

            # RF returns (C, Z, Y, X) for 3D or (C, Y, X) for 2D
            result = self.clf.train(labels, self.features)
            self._clf_ready = True
            self.btn_save.setEnabled(True)
            
            if is_3d:
                # Transpose to (Z, C, Y, X) for composed 3D display
                result = np.moveaxis(result, 0, 1)
                layer_name = "Training Probabilities"
            else:
                layer_name = "Segmentation Probabilities"
            
            if layer_name in self.viewer.layers:
                self.viewer.layers.remove(layer_name)
            self.viewer.add_image(result, name=layer_name)
            
            if not is_3d:
                self.apply_rf()
            else:
                self._on_layer_change()

        self.create_features(callback=_do_train, indices=indices)

    def apply_rf(self):
        active_layer = self.get_selected_layer()
        if active_layer is None or not self._clf_ready:
            return
        
        img = active_layer.data
        is_3d = img.ndim == 3
        self.btn_train.setEnabled(False)
        self.btn_apply_rf.setEnabled(False)

        pbar = progress(desc="Applying Random Forest")

        @thread_worker
        def _apply_rf_worker():
            if is_3d:
                total_slices = img.shape[0]
                results_buffer = None
                for z in range(total_slices):
                    gen = self.feature_creator.make_simple_features(img, indices=[z])
                    for val in gen:
                        if isinstance(val, tuple):
                            yield (z, total_slices, f"Slice {z+1}/{total_slices}: {val[2]}")
                        else:
                            feats = val
                    
                    prob_slice = self.clf.predict_segmenter(feats)
                    if results_buffer is None:
                        results_buffer = np.zeros((total_slices, prob_slice.shape[0], *img.shape[1:]), dtype=np.float32)
                    results_buffer[z] = prob_slice[:, 0]
                    if z == total_slices - 1:
                        self.features = feats
                yield results_buffer
            else:
                if self.features is None or self._last_processed_layer != active_layer or self.features.ndim == 4:
                    gen = self.feature_creator.make_simple_features(img)
                    for val in gen:
                        if isinstance(val, tuple): yield (0, 1, val[2])
                        else: self.features = val
                yield self.clf.predict_segmenter(self.features)

        def _on_yielded(val):
            if isinstance(val, tuple):
                z, total, desc = val
                pbar.total, pbar.n = total, z
                pbar.set_description(desc)
                pbar.refresh()
            else:
                layer_name = "Segmentation Probabilities"
                if layer_name in self.viewer.layers:
                    self.viewer.layers.remove(layer_name)
                self.viewer.add_image(val, name=layer_name)

                # Update Features layer
                display_feats = self.features[-1] if self.features.ndim == 4 else self.features
                if "Features" in self.viewer.layers:
                    self.viewer.layers["Features"].data = np.moveaxis(display_feats, -1, 0)
                else:
                    self.viewer.add_image(np.moveaxis(display_feats, -1, 0), name="Features")

        def _on_finished():
            pbar.close()
            self._on_layer_change()

        worker = _apply_rf_worker()
        worker.yielded.connect(_on_yielded)
        worker.finished.connect(_on_finished)
        worker.start()

    def load(self):
        source_file = QFileDialog.getOpenFileName(self, "Open Classifier", "", "classifiers (*.joblib)")
        if source_file[0]:
            self.clf = load(source_file[0])
            self._clf_ready = True
            self.btn_save.setEnabled(True)
            self._on_layer_change()

    def save(self):
        save_path = QFileDialog.getSaveFileName(self, "Save File", "classifier.joblib", "classifiers (*.joblib)")
        if self.clf is not None and save_path[0]:
            dump(self.clf, save_path[0])

    def _get_save_dir(self):
        if self.image_path:
            save_dir = Path(self.image_path).parent / self.image_name
        else:
            save_dir = Path.home() / self.image_name
        save_dir.mkdir(parents=True, exist_ok=True)
        return save_dir

    def save_labels(self):
        if "Labels" in self.viewer.layers:
            labels = self.viewer.layers["Labels"].data
            # Re-apply dimensionality handling before saving
            is_3d = self.get_selected_layer() is not None and self.get_selected_layer().data.ndim == 3
            if not is_3d and labels.ndim == 3:
                labels = np.max(labels, axis=0)
            elif is_3d and labels.ndim == 4:
                labels = np.max(labels, axis=1)
                
            save_path = self._get_save_dir() / f"{self.image_name}_labels.tif"
            imsave(str(save_path), labels.astype(np.uint8), check_contrast=False)

    def save_predictions(self):
        self._save_layer_stack("Segmentation Probabilities", "full")

    def save_training_predictions(self):
        self._save_layer_stack("Training Probabilities", "training")

    def _save_layer_stack(self, base_name, suffix):
        layers = [l for l in self.viewer.layers if base_name in l.name]
        if not layers: return
        probs = layers[0].data
        argmax_axis = 1 if probs.ndim == 4 else 0
        class_labels = np.argmax(probs, axis=argmax_axis).astype(np.uint8)
        save_dir = self._get_save_dir()
        imsave(str(save_dir / f"{self.image_name}_{suffix}_class.tif"), class_labels, check_contrast=False)
        imsave(str(save_dir / f"{self.image_name}_{suffix}_probs.tif"), probs.astype(np.float32), check_contrast=False)
        print(f"Saved {suffix} predictions to {save_dir}")
