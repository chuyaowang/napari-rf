# API Reference

## `napari_rf.RF.RF`
Core machine learning wrapper for Random Forest segmentation.

### `RF(clf=None)`
- **`train(training_labels, features)`**
    - Fits the classifier to the provided data. Supports sparse labels (0 = unlabelled).
    - **Input**:
        - `training_labels`: `(Y, X)` or `(Z, Y, X)` (Values > 0 are training pixels).
        - `features`: `(Y, X, F)` or `(Z, Y, X, F)` where `F` is the number of features.
    - **Output**: `(C, Y, X)` or `(C, Z, Y, X)` probabilities.
- **`predict_segmenter(features)`**
    - Generates a probability map for the input features.
    - **Input**: `features` array matching training dimensions.
    - **Output**: `(C, Y, X)` or `(C, Z, Y, X)` probability map.

---

## `napari_rf.features.FeatureCreator`
Generator-based feature extraction engine.

### `make_simple_features(*imgs, indices=None)`
A **generator** function that yields progress updates.
- **Parameters**:
    - `*imgs`: One or more raw images (ND-arrays).
    - `indices` (Optional): List of slice indices to process for 3D stacks.
- **Yields**: 
    - `(current_step, total_steps, description)` for UI monitoring.
    - Final `numpy.ndarray`: `(Y, X, F)` or `(Z_subset, Y, X, F)`.
- **Extracted Features**:
    - **Intensity**: Normalized original image.
    - **Structure**: Multiscale texture and edge features (skimage).
    - **Local Variance**: Local standard deviation (sigma=3).
    - **Blobs/Edges**: Difference of Gaussians (scales: 1-3, 3-5, 5-8).
    - **Curvature**: Hessian Matrix Determinant (sigma=1, 3) and Shape Index.
    - **Texture**: Local Binary Patterns (LBP).

---

## `napari_rf.RFWidget`
The Qt GUI interface for napari.

### Primary Methods
- `create_features(callback=None, slice_indices=None, feature_type="prediction")`: 
    - Launches background extraction. 
    - `feature_type`: `"training"` (sparse multi-slice) or `"prediction"` (single plane or 2D).
- `train()`: Orchestrates training. For 3D, detects labeled slices and generates sparse features.
- `apply_rf()`: Performs inference. For 3D, uses a hybrid slice-by-slice loop (reuses training features).
- `save_labels()` / `save_predictions()`: Exports results pull from `image_states`.
- `reset_all()`: Resets model instance and clears all `image_states`.

### State Management
- `self.clf`: The active `RF` instance.
- `self.image_states`: `Dict[Layer, Dict]` holding:
    - `data`: Raw numpy data.
    - `name` / `path`: Metadata.
    - `labeled_slices`: List of indices.
    - `training_features`: Cache for sparse training data.
    - `prediction_features`: Cache for current inference slice.
    - `training_probabilities`: Cached probability maps from training.
    - `prediction_probabilities`: Cached probability maps from full stack inference.
- `self._clf_ready`: Indicates if the model is ready for inference.
- `self._current_image`: Tracks the currently selected image layer for switch-detection.

---

## Data Modules (`src/napari_rf/datasets/`)
- `nd2_dataset.py`: Support for Nikon ND2 files.
- `folder_structure_dataset.py`: Directory-based batch processing.
- `single_image_dataset.py`: Lightweight image wrapper.
