# User Guide

This guide explains how to use the `napari-rf` plugin to perform image segmentation with Random Forest classifiers. The plugin is optimized for both 2D images and large 3D stacks using a memory-efficient, on-demand feature generation pipeline.

## Workflow Overview

### 1. Image Selection
The plugin allows you to choose exactly which image to segment using a drop-down menu.
- **Select Image Layer**: Choose the source image from the list.
- **Auto-Sync**: The list updates automatically as you add or remove layers in napari.
- **Data Shapes**:
    - **2D Input**: `(Y, X)`
    - **3D Input**: `(Z, Y, X)`

### 2. Annotation (Labeling)
The classifier learns from examples you provide.
- Add a new **Labels** layer to napari (must be named exactly `Labels`).
- Use the brush tool to paint regions corresponding to different classes (e.g., Label 1 for "Background", Label 2 for "Object").
- **Sparse Annotation**: You don't need to paint the whole image; sparse annotations are often sufficient.
- **For 3D Stacks**: You only need to label a few representative slices. The plugin will detect these automatically and only train on them.
- **Dimensionality Robustness**: If you accidentally create labels on a multi-channel "Features" layer, the plugin will automatically project them back to spatial dimensions during training.

### 3. Training
Click **"Train Random Forest"**. This button is only active when a `Labels` layer is present.
- **On-Demand Features**: Feature generation is now auto-triggered. You no longer need to click a separate "Create Features" button.
- **Improved Extraction Pipeline**:
    - **Normalization**: 0.5% - 99.5% percentile scaling to handle varied lighting and exposure.
    - **Multiscale Features**: Intensity, edges, and texture via `skimage`.
    - **Advanced Features**: Difference of Gaussians (DoG), Hessian Determinant (blobness), Local Standard Deviation, and Local Binary Patterns (LBP).
- **Output**:
    - **2D**: A `Segmentation Probabilities` layer is created, and inference is automatically run on the whole image.
    - **3D**: A `Training Probabilities` layer is created showing results *only* for the labeled slices.
- **Training Result Data Shapes**:
    - **2D**: `(C, Y, X)` where `C` is the number of classes.
    - **3D**: `(Z_labeled, C, Y, X)` (Composed layer).

### 4. Prediction (Inference)
- **2D Workflow**: Happens automatically after training or when you click **"Apply Random Forest"**.
- **3D Workflow**: Click **"Apply RF to All Slices"**.
- **Slice-by-Slice Inference**: For 3D stacks, the plugin processes one slice at a time (Generate Features -> Predict -> Discard). This ensures memory usage stays constant even for huge stacks.
- **Output Data Shapes**:
    - **2D**: `(C, Y, X)`
    - **3D**: `(Z, C, Y, X)` (Composed layer).

### 5. Saving and Loading

#### Classifier Models
- **Save Classifier**: Export your model as a `.joblib` file.
- **Load Classifier**: Restore a previously trained model.

#### Exporting Results
The plugin automatically creates subfolders named after your image for organized export.
- **Save Labels**: Exports the current mask as a `uint8` TIFF.
- **Save Training Predictions (3D Only)**: Saves probabilities and class maps for the training subset.
- **Save Predictions / Save Full Stack Predictions**: Saves two files:
    - `..._class.tif`: The most likely class for each pixel (`uint8`).
    - `..._probs.tif`: The raw multi-channel probability map (`float32`).
- **Smart Directory Handling**: Files are saved into a subfolder named after your image, located in the same directory as the original image file.

### 6. Batch Training

The plugin comes with a command-line script to automate training a model from multiple annotated 2D or 3D images. 

**Folder Structure Requirement**:
- Original Image: `parent_folder/image_name.tif`
- Label Image: `parent_folder/image_name/*label*.tif`

**Usage**:
Once `napari-rf` is installed, you can call the script from the terminal:
```bash
napari-rf-batch-train /path/to/your/parent_folder
```
Optional: Set the output model name:
```bash
napari-rf-batch-train /path/to/your/parent_folder --output custom_model.joblib
```

The script will automatically detect whether each image is 2D or 3D, and for 3D stacks, it will efficiently extract features only for the slices that contain manual annotations before training the final aggregated Random Forest model.

### 7. Maintenance
- **Features Layer**: Shows the last processed feature stack (2D representation) to keep the workspace clean and RAM usage low.
- **Reset All**: Clears internal model, feature caches, and resets the UI to its initial state.

## UI Features
- **Tooltips**: Hover over any button to see a brief description of what it does.
- **Dynamic States**: Buttons are enabled/disabled based on the presence of relevant layers and model state.

## Summary of Data Flow

| Data Type | 2D Shape | 3D Shape |
| :--- | :--- | :--- |
| **Input Image** | `(Y, X)` | `(Z, Y, X)` |
| **Labels Layer** | `(Y, X)` | `(Z, Y, X)` |
| **Features Layer** | `(C_feat, Y, X)` | `(C_feat, Y, X)` (Last slice only) |
| **Probabilities** | `(C, Y, X)` | `(Z, C, Y, X)` |
| **Class Map** | `(Y, X)` | `(Z, Y, X)` |
