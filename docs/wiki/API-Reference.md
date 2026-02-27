# API Reference

## `napari_rf.RF`
The core machine learning wrapper.

### `RF(clf=None)`
- **`train(training_labels, features)`**: Trains the classifier. `training_labels` can be sparse.
- **`predict_segmenter(features)`**: Returns class probabilities for the input features.

---

## `napari_rf.features.FeatureCreator`
Logic for image-to-feature transformations.

### `make_simple_features(*imgs)`
- Generates a stack of features including:
    - Original intensity
    - Multiscale basic features (texture, edges)
    - Sobel filters
    - Difference of Gaussians
    - Laplacian of Gaussian

---

## `napari_rf.RFWidget`
The Qt-based GUI for napari.

### Methods
- `create_features()`: Triggers feature extraction on the active layer.
- `train()`: Orchestrates the training process using the 'Labels' and 'features' layers.
- `apply_rf()`: Runs prediction on the active layer.
- `save()` / `load()`: Serializes the `RF` object using `joblib`.

---

## Data Modules (`src/napari_rf/datasets/`)
Specialized classes for handling different data formats:
- `nd2_dataset.py`: Nikon ND2 file support.
- `folder_structure_dataset.py`: Handles images organized in directories.
- `single_image_dataset.py`: Simple 2D/3D image wrapper.
