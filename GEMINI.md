# Gemini Context: napari-rf

## Project Overview
`napari-rf` is a napari plugin designed for image segmentation using Random Forest classifiers. It leverages `scikit-image` for feature engineering and `scikit-learn` for machine learning. The plugin supports both 2D images and 3D stacks, with a specialized memory-efficient workflow for large 3D datasets.

### Main Technologies
- **Python**: Core language.
- **napari**: Multi-dimensional image viewer and plugin framework.
- **scikit-image**: Used for `fit_segmenter` and feature extraction.
- **scikit-learn**: Provides the underlying `RandomForestClassifier`.
- **Qt/qtpy**: GUI framework for the plugin widget.
- **joblib**: Used for serializing/deserializing trained models.

### Key Architecture
- **`RFWidget` (`src/napari_rf/_widget.py`)**: The main GUI component. Manages state, coordinates asynchronous feature generation, training, and inference.
- **`FeatureCreator` (`src/napari_rf/features.py`)**: A generator-based engine for extracting multi-scale image features. Supports sparse slice generation for 3D efficiency.
- **`RF` (`src/napari_rf/RF.py`)**: A wrapper around the `RandomForestClassifier`.

---

## Development Workflow & Preferences

### Git and Source Control
- **Branching**: Use descriptive feature branches (e.g., `feature/memory-efficient-3d`) for major updates.
- **Commit Logic**: Commit logically grouped changes. Do not wait for the very end of a task to commit if a sub-component is verified.
- **Commit Message Style**: Use **comprehensive and detailed** commit messages. Avoid brief one-liners for complex changes. Messages should include a summary and a bulleted list of specific technical improvements (e.g., bug fixes, UI changes, architectural decisions).

### Documentation and Wiki
- **"Code Wiki" Preference**: Maintain a technical "code wiki" in the `docs/wiki/` directory. This documentation is the source of truth for users and developers.
- **Comprehensiveness**: When updating the wiki, **never delete existing documentation** unless explicitly instructed. Integrate new features into the existing context to maintain a complete guide for the entire application.
- **Technical Detail**: Documentation should clearly specify:
    - Input/Output data shapes (e.g., `(Z, C, Y, X)`.
    - UI button behaviors and dynamic states.
    - Internal data flow and logic (including Mermaid diagrams).

---

## Building and Running

### Development Setup
1. Create a virtual environment.
2. Install dependencies: `pip install -e .`
3. Testing requirements: `pip install -e ".[testing]"`

### Running and Testing
- Launch: `napari`
- Direct Tests: `pytest`
- Tox: `tox`

---

## Technical Conventions

### 3D Memory Efficiency
- **Sparse Training**: Only generate features for slices with labels.
- **Slice-wise Inference**: Process full stacks slice-by-slice to maintain constant RAM usage.
- **Robustness**: Always project multi-channel label layers back to spatial dimensions before processing to handle common user interaction patterns.
