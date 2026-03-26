import argparse
from pathlib import Path

import numpy as np
from joblib import dump
from skimage.io import imread

from napari_rf.features import FeatureCreator
from napari_rf.RF import RF


def find_image_and_label(subfolder: Path) -> tuple[Path, Path]:
    """
    Locates the original image and the manually drawn label image based on the specific folder structure:
    - Original Image: parent_folder/subfolder_name.tif (or .tiff)
    - Label Image: parent_folder/subfolder_name/*label*.tif (or .tiff)

    Parameters
    ----------
    subfolder : Path
        The Path object to the subdirectory (which shares the name of the original image).

    Returns
    -------
    tuple[Path, Path]
        A tuple containing (original_image_path, label_image_path).

    Raises
    ------
    FileNotFoundError
        If the label or the original image cannot be unambiguously identified.
    """
    parent_dir = subfolder.parent
    
    # 1. Find the original image in the parent directory
    image_path = None
    possible_images = [
        parent_dir / f"{subfolder.name}.tif",
        parent_dir / f"{subfolder.name}.tiff"
    ]
    for p in possible_images:
        if p.is_file():
            image_path = p
            break
            
    if not image_path:
        raise FileNotFoundError(f"Could not find the original image file matching '{subfolder.name}' in {parent_dir}")

    # 2. Find the label image in the subfolder
    label_path = None
    tiff_files = list(subfolder.glob("*.tif")) + list(subfolder.glob("*.tiff"))
    for file_path in tiff_files:
        if "label" in file_path.name.lower():
            label_path = file_path
            break
            
    if not label_path:
        raise FileNotFoundError(f"Could not find a label file (containing 'label') in {subfolder}")
        
    return image_path, label_path


def extract_features_for_image(image: np.ndarray, feature_creator: FeatureCreator) -> np.ndarray:
    """
    Extracts features for a given 2D image using the provided FeatureCreator.

    Parameters
    ----------
    image : np.ndarray
        The 2D intensity image.
    feature_creator : FeatureCreator
        The initialized FeatureCreator instance from napari-rf.

    Returns
    -------
    np.ndarray
        The computed feature array of shape (Y, X, F), where F is the number of features.
    """
    gen = feature_creator.make_simple_features(image)
    features = None
    for val in gen:
        # The generator yields progress tuples, and finally the feature array
        if not isinstance(val, tuple):
            features = val
    return features


def batch_train_models(parent_folder: str | Path, output_filename: str = "batch_rf_model.joblib"):
    """
    Iterates through subdirectories of the parent folder, loads the corresponding 
    original images and labels, extracts features, and trains a Random Forest model.

    The model is then saved as a .joblib file in the parent folder.

    Parameters
    ----------
    parent_folder : str | Path
        Path to the parent directory containing the original images and their subfolders.
    output_filename : str, optional
        The name of the saved model file, by default "batch_rf_model.joblib".
    """
    parent_path = Path(parent_folder)
    if not parent_path.is_dir():
        print(f"Error: The directory {parent_path} does not exist.")
        return

    # Find all subdirectories
    subfolders = [d for d in parent_path.iterdir() if d.is_dir()]
    if not subfolders:
        print(f"No subdirectories found in {parent_path}.")
        return

    feature_creator = FeatureCreator()
    rf = RF()

    all_X = []
    all_y = []

    print(f"Found {len(subfolders)} subfolders. Starting feature extraction...")

    for i, subfolder in enumerate(subfolders):
        try:
            img_path, lbl_path = find_image_and_label(subfolder)
        except FileNotFoundError as e:
            print(f"[{i+1}/{len(subfolders)}] Skipping {subfolder.name}: {e}")
            continue

        print(f"[{i+1}/{len(subfolders)}] Processing {subfolder.name}...")
        
        # Load images
        img = imread(str(img_path))
        lbl = imread(str(lbl_path))

        if img.shape != lbl.shape:
            print(f"  -> Warning: Shape mismatch. Image: {img.shape}, Label: {lbl.shape}. Skipping.")
            continue

        # Extract features
        print(f"  -> Extracting features...")
        features = extract_features_for_image(img, feature_creator)

        # Flatten arrays for training
        # Background is labeled 1, Object is labeled 2, Unlabeled is 0.
        mask = lbl > 0
        if not np.any(mask):
            print(f"  -> Warning: No labels found (values > 0) in {lbl_path.name}. Skipping.")
            continue

        # Reshape features from (Y, X, F) to (N, F) where N is number of labeled pixels
        # and labels from (Y, X) to (N,)
        X_labeled = features[mask]
        y_labeled = lbl[mask]

        all_X.append(X_labeled)
        all_y.append(y_labeled)
        
        # Report class distribution for this image
        unique, counts = np.unique(y_labeled, return_counts=True)
        class_dist = dict(zip(unique, counts))
        print(f"  -> Extracted {len(y_labeled)} labeled pixels. Class distribution: {class_dist}")

    if not all_X:
        print("Error: No labeled pixels were found across any of the images. Training aborted.")
        return

    # Concatenate all labeled data
    X_train = np.concatenate(all_X, axis=0)
    y_train = np.concatenate(all_y, axis=0)

    print(f"\nTotal training samples: {len(y_train)} pixels with {X_train.shape[1]} features.")
    
    unique_all, counts_all = np.unique(y_train, return_counts=True)
    print(f"Total class distribution: {dict(zip(unique_all, counts_all))}")
    
    print("Training Random Forest model (this may take a while)...")

    # Fit the Random Forest classifier directly on the accumulated pixels
    rf.clf.fit(X_train, y_train)
    
    print("Training complete.")

    # Save the model
    output_path = parent_path / output_filename
    dump(rf, output_path)
    print(f"Model successfully saved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch train a napari-rf model from a folder of segmented 2D images.")
    parser.add_argument(
        "parent_folder", 
        type=str, 
        help="Path to the parent directory containing the original images and their subfolders."
    )
    parser.add_argument(
        "--output", 
        type=str, 
        default="batch_rf_model.joblib", 
        help="Name of the output model file (default: batch_rf_model.joblib)."
    )

    args = parser.parse_args()
    batch_train_models(args.parent_folder, args.output)
