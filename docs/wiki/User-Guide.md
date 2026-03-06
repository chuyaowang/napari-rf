# User Guide

This guide explains how to use the `RFWidget` to perform image segmentation with Random Forest classifiers.

## Workflow Overview

### 1. Feature Generation
Before training, the plugin must extract multiscale features from your image (e.g., textures, edges, blurred versions).
- Select your image layer in the napari layer list.
- Click **"Create Features"**.
- **Progress Tracking**: A progress bar will appear in the napari activity dock (bottom right). It will show the current processing step (e.g., "Slice 1/10: Multiscale Basic").
- **Improved Extraction**: The plugin now uses a robust pipeline:
    - **Normalization**: 0.5% - 99.5% percentile scaling to handle varied lighting and exposure.
    - **Multiscale Features**: Intensity, edges, and texture via `skimage`.
    - **Advanced Features**: Difference of Gaussians (DoG), Hessian Determinant (blobness), Local Standard Deviation, and Local Binary Patterns (LBP) for texture analysis.
- A new image layer named **"Features"** will be added. This is a multi-channel stack where each channel represents a different extracted feature.

### 2. Annotation
The classifier needs ground truth to learn.
- Add a new **Labels** layer to napari (must be named exactly `Labels`).
- Use the brush tool to paint regions corresponding to different classes (e.g., Label 1 for "Background", Label 2 for "Object").
- **Sparse Annotation**: You don't need to paint the whole image; sparse annotations are often sufficient.
- **Dimensionality Support**: If your features are multi-channel (3D) but your labels were created as a 2D layer, the plugin will automatically alignment them during training.

### 3. Training
- Once you have a `Labels` layer and a `Features` layer, click **"Train Random Forest"**.
- The plugin will use `scikit-image.future.fit_segmenter` to train the model.
- A **"Segmentation Probabilities"** layer will be added, showing the model's confidence for each class.

### 4. Prediction (Apply RF)
If you have a new image and a trained model:
- Select the new image.
- Ensure features are created for it (click "Create Features").
- Click **"Apply Random Forest"** to generate predictions.

### 5. Saving and Loading

#### Classifier Models
- Use **"Save Classifier"** to export your trained model as a `.joblib` file.
- Use **"Load Classifier"** to bring back a previously trained model.

#### Labels and Predictions
The plugin provides dedicated buttons to export your work directly to your data folders:
- **"Save Labels"**: Saves the current "Labels" layer as a TIFF file.
- **"Save Predictions"**: Saves two files:
    - `<image_name>_predictions_class.tif`: The predicted class for each pixel (most likely class).
    - `<image_name>_predictions_probs.tif`: The full multi-channel probability map.
- **Smart Directory Handling**: Files are automatically saved into a subfolder named after your image, located in the same directory as the original image file.
- **Optimized Data Types**: Labels are saved as compact `uint8` (8-bit) and probabilities as `float32` (32-bit).

## UI Features
- **Tooltips**: Hover over any button to see a brief description of what it does.
- **Dynamic States**: Saving buttons are disabled (grayed out) until the relevant layers ("Labels" or "Segmentation Probabilities") are present in the viewer.
