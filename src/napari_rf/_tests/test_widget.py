import numpy as np
import pytest
from napari_rf._widget import RFWidget

def test_rf_widget_init(make_napari_viewer):
    viewer = make_napari_viewer()
    widget = RFWidget(viewer)
    assert widget.viewer == viewer
    assert widget.layer_combo.count() == 0
    assert len(widget.image_states) == 0

def test_rf_widget_layer_sync(make_napari_viewer):
    viewer = make_napari_viewer()
    widget = RFWidget(viewer)
    
    # Add an image layer
    viewer.add_image(np.random.random((10, 10)), name="test_image")
    assert widget.layer_combo.count() == 1
    assert widget.layer_combo.currentText() == "test_image"
    # Current image is initialized upon layer change via _on_layer_change()
    assert any(l.name == "test_image" for l in widget.image_states.keys())
    
    # Add another image
    viewer.add_image(np.random.random((10, 10)), name="another_image")
    assert widget.layer_combo.count() == 2
    
    # Remove a layer
    viewer.layers.remove("test_image")
    assert widget.layer_combo.count() == 1
    assert widget.layer_combo.currentText() == "another_image"

def test_rf_widget_reset(make_napari_viewer):
    viewer = make_napari_viewer()
    widget = RFWidget(viewer)
    
    viewer.add_image(np.random.random((10, 10)), name="image")
    assert len(widget.image_states) == 1
    
    widget.reset_all()
    assert len(widget.image_states) == 0
    assert widget._current_image is None

def test_init_image_state(make_napari_viewer):
    viewer = make_napari_viewer()
    widget = RFWidget(viewer)
    
    img = viewer.add_image(np.random.random((10, 10)), name="test")
    # Triggered automatically by _on_layer_change via dropdown event
    assert img in widget.image_states
    state = widget.image_states[img]
    assert state["name"] == "test"
    assert state["ndim"] == 2
    assert state["training_features"] is None
