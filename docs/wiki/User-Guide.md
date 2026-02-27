# User Guide

This guide explains how to use the `RFWidget` to perform image segmentation.

## Workflow Overview

### 1. Feature Generation
Before training, the plugin must extract features from your image (e.g., textures, edges, blurred versions).
- Select your image layer in the napari layer list.
- Click **"create features"**.
- A new image layer named `features` will be added. This is a multi-channel stack where each channel represents a different extracted feature.

### 2. Annotation
The classifier needs ground truth to learn.
- Add a new **Labels** layer to napari (must be named exactly `Labels`).
- Use the brush tool to paint regions corresponding to different classes (e.g., Label 1 for "Background", Label 2 for "Object").
- You don't need to paint the whole image; sparse annotations are often sufficient.

### 3. Training
- Once you have a `Labels` layer and a `features` layer, click **"train random forest"**.
- The plugin will use `scikit-image.future.fit_segmenter` to train the model.
- A `segmentation probabilities` layer will be added, showing the model's confidence for each class.

### 4. Prediction (Apply RF)
If you have a new image and a trained model:
- Select the new image.
- Ensure features are created for it.
- Click **"apply random forest"** to generate predictions.

### 5. Saving and Loading
- Use **"save classifier"** to export your trained model as a `.joblib` file.
- Use **"load classifier"** to bring back a previously trained model.
