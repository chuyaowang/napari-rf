"""
"""
import os
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Any

import numpy as np
from joblib import dump, load
from qtpy.QtWidgets import QComboBox, QLabel, QFileDialog, QPushButton, QVBoxLayout, QWidget, QMessageBox
from skimage.io import imsave

from napari_rf.features import FeatureCreator
from napari_rf.RF import RF
from napari.qt.threading import thread_worker
from napari.utils import progress

if TYPE_CHECKING:
    import napari


class RFWidget(QWidget):
    def __init__(self, viewer: "napari.viewer.Viewer"):
        super().__init__()
        self.viewer = viewer

        self.clf = RF()
        self.feature_creator = FeatureCreator()
        
        # State management: Dictionary holding data and caches for each image
        # Key: napari.layers.Image object, Value: dict of state
        self.image_states: Dict["napari.layers.Image", Dict[str, Any]] = {}
        self._current_image = None
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

    def _init_image_state(self, image):
        """Initialize the state dictionary for a specific image."""
        if image is None or image in self.image_states:
            return
        
        path = None
        source = getattr(image, "source", None)
        raw_path = getattr(source, "path", None)
        if raw_path:
            path = str(raw_path[0] if isinstance(raw_path, (list, tuple)) else raw_path)
        
        self.image_states[image] = {
            "data": image.data,
            "ndim": image.data.ndim,
            "name": image.name,
            "path": path,
            "labeled_slices": [],
            "training_features": None,
            "prediction_features": None, # (Y, X, F) - stores last processed slice
            "training_probabilities": None, # Sparse subset for 3D, full for 2D
            "prediction_probabilities": None, # Full stack/image
        }
        print(f"[RF Plugin] Initialized state for image: {image.name}")

    def _on_layer_change(self, event=None):
        """Update the layer combo box and manage image state transitions."""
        import napari
        
        # 1. Update the Drop-down
        current_selection = self.layer_combo.currentText()
        self.layer_combo.blockSignals(True)
        self.layer_combo.clear()
        image_layers = [
            l for l in self.viewer.layers 
            if isinstance(l, napari.layers.Image) 
            and not any(x in l.name for x in ["Features", "Probabilities"])
        ]
        self.layer_combo.addItems([l.name for l in image_layers])
        
        # Restore selection. Prevents selection from being forgotten in the menu refresh
        for i, l in enumerate(image_layers):
            if l.name == current_selection:
                self.layer_combo.setCurrentIndex(i)
                break
        else:
            if image_layers:
                self.layer_combo.setCurrentIndex(0)
        self.layer_combo.blockSignals(False)

        # 2. Handle Image Switch
        active_image = self.get_selected_layer()
        if active_image != self._current_image:
            # Memory Management: Ask to clear previous image data
            if self._current_image and self._current_image in self.image_states:
                state = self.image_states[self._current_image]
                if any(state[x] is not None for x in ["training_features", "prediction_features", "training_probabilities", "prediction_probabilities"]):
                    reply = QMessageBox.question(
                        self, "Clear Memory?",
                        f"You are switching to a new image. Do you want to delete the feature and probability caches for '{state['name']}' to save RAM?",
                        QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes
                    )
                    if reply == QMessageBox.Yes:
                        del self.image_states[self._current_image]
                        print(f"[RF Plugin] Cleared state for: {state['name']}")

            self._current_image = active_image
            self._init_image_state(active_image)

        # 3. Dynamic UI Renaming and Visibility
        state = self.image_states.get(active_image)
        is_3d = state is not None and state["ndim"] == 3
        
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
        has_layers = active_image is not None
        has_labels = "Labels" in self.viewer.layers
        
        self.btn_train.setEnabled(has_labels and has_layers)
        self.btn_save_labels.setEnabled(has_labels)
        self.btn_apply_rf.setEnabled(self._clf_ready and has_layers)
        
        has_full_preds = state is not None and state["prediction_probabilities"] is not None
        has_train_preds = state is not None and state["training_probabilities"] is not None
        
        self.btn_save_preds.setEnabled(has_full_preds)
        self.btn_save_training_preds.setEnabled(has_train_preds)
        self.btn_save_full_preds.setEnabled(has_full_preds)

    def reset_all(self):
        """Reset internal state to default."""
        self.clf = RF()
        self.image_states.clear()
        self._current_image = None
        self._clf_ready = False
        self.btn_save.setDisabled(True)
        self._on_layer_change()
        print("[RF Plugin] Internal model and all feature caches have been reset.")

    def get_selected_layer(self):
        """Returns the napari layer selected in the drop-down."""
        layer_name = self.layer_combo.currentText()
        if layer_name in self.viewer.layers:
            return self.viewer.layers[layer_name]
        return None

    def create_features(self, callback=None, slice_indices=None, feature_type="prediction"):
        active_image = self.get_selected_layer()
        if active_image is None or active_image not in self.image_states:
            return
        
        state = self.image_states[active_image]
        img_data = state["data"]
        self.btn_train.setEnabled(False)
        self.btn_apply_rf.setEnabled(False)

        pbar = progress(desc="Generating Features")

        @thread_worker
        def _create_features_worker():
            gen = self.feature_creator.make_simple_features(img_data, indices=slice_indices)
            for val in gen:
                yield val

        def _on_yielded(val):
            if isinstance(val, tuple):
                step, total, desc = val
                pbar.total = total
                pbar.set_description(desc)
                pbar.update(1)
            else:
                # Update state caches explicitly based on purpose
                if feature_type == "training":
                    state["training_features"] = val
                else:
                    state["prediction_features"] = val
                
                # Display preview (last processed slice)
                display_feats = val[-1] if val.ndim == 4 else val
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
        active_image = self.get_selected_layer()
        if active_image is None or "Labels" not in self.viewer.layers:
            return

        state = self.image_states[active_image]
        training_labels = self.viewer.layers["Labels"].data
        is_3d = state["ndim"] == 3
        
        # Robust Dimensionality Handling:
        # Handle labels created matching the 'Features' layer (multi-channel)
        # instead of the source Image layer.
        if not is_3d and training_labels.ndim == 3:
            # 2D image but labels are (C, Y, X)
            training_labels = np.max(training_labels, axis=0)
        elif is_3d and training_labels.ndim == 4:
            # 3D stack but labels are (Z, C, Y, X)
            training_labels = np.max(training_labels, axis=1)

        # For 2D, labeled slices default to 0
        labeled_slices = [0]
        # Check which slices are labeled for 3D data
        if is_3d:
            labeled_slices = np.where(np.any(training_labels > 0, axis=(1, 2)))[0].tolist()
            if not labeled_slices:
                print("[RF Plugin] Training failed: No labels found.")
                return
        
        state["labeled_slices"] = labeled_slices
        print(f"[RF Plugin] Starting training on {state['name']}...")

        def _do_train():
            try:
                labels = training_labels
                if is_3d:
                    labels = labels[labeled_slices]

                result = self.clf.train(labels, state["training_features"])
                self._clf_ready = True
                self.btn_save.setEnabled(True)
                
                # Cache probabilities in state dict
                state["training_probabilities"] = result
                
                if is_3d:
                    display_result = np.moveaxis(result, 0, 1)
                    layer_name = "Training Probabilities"
                else:
                    state["prediction_probabilities"] = result
                    display_result = result
                    layer_name = "Segmentation Probabilities"
                
                if layer_name in self.viewer.layers:
                    self.viewer.layers.remove(layer_name)
                self.viewer.add_image(display_result, name=layer_name)
                
                print(f"[RF Plugin] Training finished successfully.")
                if not is_3d:
                    self.apply_rf()
                else:
                    self._on_layer_change()
            except Exception as e:
                print(f"[RF Plugin] Training failed: {e}")

        self.create_features(callback=_do_train, slice_indices=labeled_slices, feature_type="training")

    def apply_rf(self):
        active_image = self.get_selected_layer()
        if active_image is None or not self._clf_ready:
            return
        
        state = self.image_states[active_image]
        img_data = state["data"]
        is_3d = state["ndim"] == 3
        self.btn_train.setEnabled(False)
        self.btn_apply_rf.setEnabled(False)

        print(f"[RF Plugin] Applying Random Forest to full {state['name']}...")
        pbar = progress(desc="Applying Random Forest")

        @thread_worker
        def _apply_rf_worker():
            if is_3d:
                total_slices = img_data.shape[0]
                results_buffer = None
                for z in range(total_slices):
                    # Hybrid Workflow: Reuse training features if available, otherwise generate
                    if z in state["labeled_slices"] and state["training_features"] is not None:
                        idx = state["labeled_slices"].index(z)
                        feats = state["training_features"][idx][None, ...] # Add singleton Z dim
                    else:
                        gen = self.feature_creator.make_simple_features(img_data, indices=[z])
                        for val in gen:
                            if isinstance(val, tuple):
                                yield (z, total_slices, f"Slice {z+1}/{total_slices}: {val[2]}")
                            else:
                                feats = val
                    
                    prob_slice = self.clf.predict_segmenter(feats)
                    if results_buffer is None:
                        results_buffer = np.zeros((total_slices, prob_slice.shape[0], *img_data.shape[1:]), dtype=np.float32)
                    results_buffer[z] = prob_slice[:, 0]
                    
                    if z == total_slices - 1:
                        state["prediction_features"] = feats
                yield results_buffer
            else:
                # 2D: Use existing cache if valid
                if state["prediction_features"] is None:
                    gen = self.feature_creator.make_simple_features(img_data)
                    for val in gen:
                        if isinstance(val, tuple): yield (0, 1, val[2])
                        else: state["prediction_features"] = val
                yield self.clf.predict_segmenter(state["prediction_features"])

        def _on_yielded(val):
            if isinstance(val, tuple):
                z, total, desc = val
                pbar.total, pbar.n = total, z
                pbar.set_description(desc)
                pbar.refresh()
            else:
                # Cache prediction probabilities in state dict
                # If 3D, result_buffer is (Z, C, Y, X). RF expects (C, Z, Y, X) for consistent state.
                state["prediction_probabilities"] = np.moveaxis(val, 0, 1) if is_3d else val
                
                layer_name = "Segmentation Probabilities"
                if layer_name in self.viewer.layers:
                    self.viewer.layers.remove(layer_name)
                self.viewer.add_image(val, name=layer_name)

                # Update Features layer with the preview (last processed slice)
                disp_feats = state["prediction_features"]
                display_feats = disp_feats[-1] if disp_feats.ndim == 4 else disp_feats
                if "Features" in self.viewer.layers:
                    self.viewer.layers["Features"].data = np.moveaxis(display_feats, -1, 0)
                else:
                    self.viewer.add_image(np.moveaxis(display_feats, -1, 0), name="Features")

        def _on_finished():
            pbar.close()
            self._on_layer_change()
            print(f"[RF Plugin] Random Forest application complete.")

        worker = _apply_rf_worker()
        worker.yielded.connect(_on_yielded)
        worker.finished.connect(_on_finished)
        worker.start()

    def load(self):
        source_file = QFileDialog.getOpenFileName(self, "Open Classifier", "", "classifiers (*.joblib)")
        if source_file[0]:
            try:
                self.clf = load(source_file[0])
                self._clf_ready = True
                self.btn_save.setEnabled(True)
                self._on_layer_change()
                print(f"[RF Plugin] Loaded classifier from: {source_file[0]}")
            except Exception as e:
                print(f"[RF Plugin] Failed to load classifier: {e}")

    def save(self):
        active_image = self.get_selected_layer()
        state = self.image_states.get(active_image)
        save_dir = self._get_save_dir(state)
        default_path = str(save_dir / "classifier.joblib")
        
        save_path = QFileDialog.getSaveFileName(self, "Save File", default_path, "classifiers (*.joblib)")
        if self.clf is not None and save_path[0]:
            try:
                dump(self.clf, save_path[0])
                print(f"[RF Plugin] Saved classifier to: {save_path[0]}")
            except Exception as e:
                print(f"[RF Plugin] Failed to save classifier: {e}")

    def _get_save_dir(self, state):
        if state and state["path"]:
            save_dir = Path(state["path"]).parent / state["name"]
        elif state:
            save_dir = Path.home() / state["name"]
        else:
            save_dir = Path.home() / "napari_rf_export"
        
        save_dir.mkdir(parents=True, exist_ok=True)
        return save_dir

    def save_labels(self):
        if "Labels" in self.viewer.layers:
            try:
                active_image = self.get_selected_layer()
                state = self.image_states.get(active_image)
                labels = self.viewer.layers["Labels"].data
                
                is_3d = state["ndim"] == 3
                if not is_3d and labels.ndim == 3:
                    labels = np.max(labels, axis=0)
                elif is_3d and labels.ndim == 4:
                    labels = np.max(labels, axis=1)
                
                save_dir = self._get_save_dir(state)
                save_path = save_dir / f"{state['name']}_labels.tif"
                imsave(str(save_path), labels.astype(np.uint8), check_contrast=False)
                print(f"[RF Plugin] Saved labels to {save_path}")
            except Exception as e:
                print(f"[RF Plugin] Error in save_labels: {e}")
        else:
            print("[RF Plugin] No 'Labels' layer found to save.")

    def save_predictions(self):
        active_image = self.get_selected_layer()
        state = self.image_states.get(active_image)
        if state and state["prediction_probabilities"] is not None:
            self._save_state_probabilities(state, "prediction_probabilities", "full")

    def save_training_predictions(self):
        active_image = self.get_selected_layer()
        state = self.image_states.get(active_image)
        if state and state["training_probabilities"] is not None:
            self._save_state_probabilities(state, "training_probabilities", "training")

    def _save_state_probabilities(self, state, key, suffix):
        probs = state[key]
        # In state dict, probs are always (C, Y, X) or (C, Z, Y, X)
        is_3d = probs.ndim == 4
        argmax_axis = 1 if is_3d else 0
        
        class_labels = np.argmax(probs, axis=argmax_axis).astype(np.uint8)
        save_dir = self._get_save_dir(state)
        
        # If 3D, we transpose back to (Z, C, Y, X) for standard TIFF save shape
        save_probs = np.moveaxis(probs, 0, 1) if is_3d else probs
        
        imsave(str(save_dir / f"{state['name']}_{suffix}_class.tif"), class_labels, check_contrast=False)
        imsave(str(save_dir / f"{state['name']}_{suffix}_probs.tif"), save_probs.astype(np.float32), check_contrast=False)
        print(f"[RF Plugin] Saved {suffix} predictions to {save_dir}")
