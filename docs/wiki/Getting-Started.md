# Getting Started

## Installation

### From Source
1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/napari-rf.git
   cd napari-rf
   ```
2. Install the package in editable mode:
   ```bash
   pip install -e .
   ```

### Requirements
- **Python**: >= 3.8
- **Core Dependencies**: `numpy`, `scikit-image`, `magicgui`, `qtpy`
- **GUI**: `napari`, `PyQt5` (or `PySide2`)

## Launching the Plugin
1. Open napari:
   ```bash
   napari
   ```
2. Navigate to the menu: `Plugins` -> `napari rf: RFWidget`.
3. The widget will appear in the right-hand sidebar.

## Loading Sample Data
You can test the plugin using the built-in sample data:
`File` -> `Open Sample` -> `napari rf` -> `napari rf`.
