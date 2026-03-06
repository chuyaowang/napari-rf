# API Reference

## `napari_rf.RF`
Core machine learning logic.

### `RF(clf=None)`
- **`train(training_labels, features)`**: Fits the classifier to the provided data. Supports sparse labels.
- **`predict_segmenter(features)`**: Generates a probability map (C, Y, X) for the input features.

---

## `napari_rf.features.FeatureCreator`
Generator-based feature extraction.

### `make_simple_features(*imgs)`
A **generator** function that yields progress updates.
- **Yields**: 
    - `(current_step, total_steps, description)` for progress monitoring.
    - Final `numpy.ndarray` (C, Y, X) containing the full feature stack.
- **Extracted Features**:
    - **Intensity**: Normalized original image.
    - **Structure**: Multiscale texture and edge features (skimage).
    - **Local Variance**: Local standard deviation (sigma=3).
    - **Blobs/Edges**: Difference of Gaussians (scales: 1-3, 3-5, 5-8).
    - **Curvature**: Hessian Determinant (sigma=1, 3) and Shape Index.
    - **Texture**: Local Binary Patterns (LBP).

---

## `napari_rf.RFWidget`
The Qt GUI for napari.

### Primary Methods
- `create_features()`: Launches a `thread_worker` to run the feature extraction generator and updates the activity bar.
- `train()`: Orchestrates the training process. Automatically handles 2D label layer to 3D feature layer alignment.
- `apply_rf()`: Applies the trained model to generate a "Segmentation Probabilities" layer.
- `save_labels()` / `save_predictions()`: Export work to TIFF files in an image-specific subfolder.
- `save()` / `load()`: Serializes/Deserializes the `RF` model instance via `joblib`.

### State Management
- `self.clf`: The active `RF` instance.
- `self.features`: Cache for extracted features.
- `self.image_path` / `self.image_name`: Metadata for the source image, used for automated file saving.

---

## Data Modules (`src/napari_rf/datasets/`)
Specialized loaders:
- `nd2_dataset.py`: Nikon ND2 support.
- `folder_structure_dataset.py`: Batch directory processing.
- `single_image_dataset.py`: General image wrapper.
