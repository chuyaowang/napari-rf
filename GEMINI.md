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

## Testing Standards & CI

### Test Organization
- **Logic Tests**: Place core algorithmic and data processing tests in the `src/napari_rf/_tests/` directory (e.g., `test_logic.py`). These tests **must not** require a `napari.Viewer` or any GUI elements, allowing them to run in headless CLI environments.
- **GUI Tests**: Place tests that require a `napari.Viewer` or Qt widgets in the `src/napari_rf/_tests/gui/` subdirectory. 

### Headless Compatibility
- **Pytest Configuration**: To prevent crashes in environments without a display (like remote SSH or basic CI), the `gui/` directory is excluded from default `pytest` runs via `pyproject.toml` (`norecursedirs = ["gui"]`).
- **Running GUI Tests**: When a display is available, run GUI tests explicitly: `pytest src/napari_rf/_tests/gui/`.

### Updating Testing Setup
- **New Dependencies**: When adding libraries that are imported in the source code, always update `setup.cfg`'s `install_requires` section. If the library is only needed for testing, add it to `options.extras_require.testing`.
- **CI Synchronization**: Ensure that `tox.ini` and `.github/workflows/test_and_deploy.yml` are updated if new Python versions are supported or if specific system-level dependencies (like OpenGL libs) are required.
- **Mocking**: For logic tests that involve complex `napari` objects, prefer mocking the objects or testing the underlying functions with simple `numpy` arrays to maintain headless compatibility.

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

### Coding Style & Logic
- **Explicit State Management**: Prefer centralized state dictionaries (`image_states`) to track data and caches. Avoid "magic number" shape checks (e.g., `ndim == 4`) to infer state.
- **Standardized Terminology**:
    - **`image`**: Refers to the data source or image object.
    - **`slice`**: Refers to a specific 2D plane within a 3D stack.
    - **`layer`**: Reserved specifically for napari UI layer components.
- **Condition Flags**: Use explicit function arguments (e.g., `feature_type="training"`) to communicate intent instead of checking variable properties (like list lengths) to infer logic.
- **Status Reporting**: Provide clear console reports for the lifecycle of operations:
    - **Success/Failure**: Report the outcome of training, prediction, and I/O.
    - **Metadata**: Report when image selection or paths are updated.
    - **I/O Actions**: Explicitly print the target path when saving or loading models and labels.
