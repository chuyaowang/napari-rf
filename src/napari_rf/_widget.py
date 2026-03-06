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

        # Layer Selection Drop-down
        self.layer_combo = QComboBox()
        self.layer_combo.setToolTip("Select the image layer to use for training and prediction.")
        
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

        self.btn_save_preds = QPushButton("Save Predictions")
        self.btn_save_preds.setToolTip("Save predicted class labels and probability maps as TIFF files in a subfolder.")
        self.btn_save_preds.clicked.connect(self.save_predictions)

        self.setLayout(QVBoxLayout())
        self.layout().addWidget(QLabel("Select Image Layer:"))
        self.layout().addWidget(self.layer_combo)
        self.layout().addWidget(self.btn_train)
        self.layout().addWidget(self.btn_apply_rf)
        self.layout().addWidget(self.btn_load)
        self.layout().addWidget(self.btn_save)
        self.layout().addWidget(self.btn_save_labels)
        self.layout().addWidget(self.btn_save_preds)

        # Connect to layer events to manage UI state
        self.viewer.layers.events.inserted.connect(self._on_layer_change)
        self.viewer.layers.events.removed.connect(self._on_layer_change)
        self._on_layer_change()

    def _on_layer_change(self, event=None):
        """Update the layer combo box and save button states."""
        import napari
        # 1. Update the Drop-down
        current_selection = self.layer_combo.currentText()
        self.layer_combo.clear()
        
        image_layers = [
            l.name for l in self.viewer.layers 
            if isinstance(l, napari.layers.Image) 
            and l.name not in ["Features", "Segmentation Probabilities"]
        ]
        self.layer_combo.addItems(image_layers)
        
        # Restore selection if it still exists
        if current_selection in image_layers:
            self.layer_combo.setCurrentText(current_selection)
        elif image_layers:
            self.layer_combo.setCurrentIndex(0)

        # 2. Update button states
        has_labels = "Labels" in self.viewer.layers
        self.btn_train.setEnabled(has_labels)
        self.btn_save_labels.setEnabled(has_labels)
        self.btn_save_preds.setEnabled("Segmentation Probabilities" in self.viewer.layers)

    def get_selected_layer(self):
        """Returns the napari layer selected in the drop-down."""
        layer_name = self.layer_combo.currentText()
        if layer_name in self.viewer.layers:
            return self.viewer.layers[layer_name]
        return None

    def create_features(self, callback=None, indices=None):
        """
        Modified to work as a silent background task triggered by Train/Apply.
        Calls 'callback' with the features when finished.
        
        Parameters
        ----------
        callback : callable, optional
            Function to call when feature generation is complete.
        indices : list of int, optional
            Specific slices to process for 3D stacks.
        """
        active_layer = self.get_selected_layer()
        if active_layer is None:
            return
        
        img = active_layer.data
        
        # Track image metadata
        self.image_path = getattr(active_layer, "source", None)
        if self.image_path:
            self.image_path = getattr(self.image_path, "path", None)
            self.image_name = Path(self.image_path).stem
        else:
            self.image_name = active_layer.name
        
        self.btn_train.setEnabled(False)
        self.btn_apply_rf.setEnabled(False)

        # Initialize progress bar on main thread
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
                # Cache results and update the display (2D or last slice only)
                self.features = val
                self._last_processed_layer = active_layer
                
                # Display logic: If indices were used, 'val' is (len(indices), Y, X, C)
                # We display the last processed slice of the set.
                display_feats = self.features[-1] if self.features.ndim == 4 else self.features
                
                if "Features" in self.viewer.layers:
                    self.viewer.layers["Features"].data = np.moveaxis(display_feats, -1, 0)
                else:
                    self.viewer.add_image(np.moveaxis(display_feats, -1, 0), name="Features")

        def _on_finished():
            pbar.close()
            self.btn_train.setEnabled(True)
            self.btn_apply_rf.setEnabled(True)
            if callback:
                callback()

        worker = _create_features_worker()
        worker.yielded.connect(_on_yielded)
        worker.finished.connect(_on_finished)
        worker.start()

    def train(self):
        if "Labels" not in self.viewer.layers:
            raise Exception('training labels must be in a layer called "Labels"')
        
        active_layer = self.get_selected_layer()
        if active_layer is None:
            return

        training_labels = self.viewer.layers["Labels"].data
        indices = None

        # 3D Memory Efficiency: Find labeled slices
        if training_labels.ndim == 3:
            # Find indices where at least one pixel is labeled (> 0)
            indices = np.where(np.any(training_labels > 0, axis=(1, 2)))[0].tolist()
            if not indices:
                print("No labels found in any slice. Please draw labels first.")
                return
            print(f"Training on {len(indices)} labeled slices: {indices}")

        def _do_train():
            labels = self.viewer.layers["Labels"].data
            if indices is not None:
                # Filter labels to match the generated feature slices
                labels = labels[indices]

            result = self.clf.train(labels, self.features)
            self.btn_save.setDisabled(False)
            self.btn_apply_rf.setDisabled(False)
            
            # If 3D, result will be probabilities for the labeled slices.
            # We add it as a new layer.
            self.viewer.add_image(result, name="Segmentation Probabilities (Training Result)")

        # For training, we ALWAYS regenerate features for the specific labeled slices
        # to ensure memory efficiency and correct label alignment.
        self.create_features(callback=_do_train, indices=indices)

    def apply_rf(self):
        active_layer = self.get_selected_layer()
        if active_layer is None:
            return
        
        img = active_layer.data
        self.btn_train.setEnabled(False)
        self.btn_apply_rf.setEnabled(False)

        # Initialize progress bar on main thread
        pbar = progress(desc="Applying Random Forest")

        @thread_worker
        def _apply_rf_worker():
            if img.ndim == 3:
                # 3D: Slice-by-slice inference
                total_slices = img.shape[0]
                results_buffer = None
                
                for z in range(total_slices):
                    # 1. Generate features for one slice
                    # Note: We consume the generator directly here for speed
                    gen = self.feature_creator.make_simple_features(img, indices=[z])
                    for val in gen:
                        if isinstance(val, tuple):
                            step, total, desc = val
                            # Update global progress: each slice is (z/total_slices) + (val_step/val_total)/total_slices
                            yield (z, total_slices, f"Slice {z+1}/{total_slices}: {desc}")
                        else:
                            # Features for one slice: shape is (1, Y, X, C)
                            feats = val
                    
                    # 2. Predict for one slice
                    prob_slice = self.clf.predict_segmenter(feats) # (C, Y, X)
                    
                    # 3. Initialize buffer on first slice
                    if results_buffer is None:
                        num_classes = prob_slice.shape[0]
                        results_buffer = np.zeros((num_classes, total_slices, *img.shape[1:]), dtype=np.float32)
                    
                    results_buffer[:, z] = prob_slice
                    
                    # 4. Cache the last slice's features for display
                    if z == total_slices - 1:
                        self.features = feats
                        self._last_processed_layer = active_layer
                
                yield results_buffer
            else:
                # 2D: Standard inference
                # Ensure we have features first
                if self.features is None or self._last_processed_layer != active_layer:
                    gen = self.feature_creator.make_simple_features(img)
                    for val in gen:
                        if isinstance(val, tuple):
                            yield (0, 1, val[2])
                        else:
                            self.features = val
                            self._last_processed_layer = active_layer
                
                yield (0, 1, "Predicting...")
                yield self.clf.predict_segmenter(self.features)

        def _on_yielded(val):
            if isinstance(val, tuple):
                z, total, desc = val
                pbar.total = total
                pbar.n = z
                pbar.set_description(desc)
                pbar.refresh()
            else:
                # Result is the final probability stack/image
                self.viewer.add_image(val, name="Segmentation Probabilities")
                
                # Update Features layer with the last slice's features
                display_feats = self.features[-1] if self.features.ndim == 4 else self.features
                if "Features" in self.viewer.layers:
                    self.viewer.layers["Features"].data = np.moveaxis(display_feats, -1, 0)
                else:
                    self.viewer.add_image(np.moveaxis(display_feats, -1, 0), name="Features")

        def _on_finished():
            pbar.close()
            self.btn_train.setEnabled(True)
            self.btn_apply_rf.setEnabled(True)

        worker = _apply_rf_worker()
        worker.yielded.connect(_on_yielded)
        worker.finished.connect(_on_finished)
        worker.start()

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
