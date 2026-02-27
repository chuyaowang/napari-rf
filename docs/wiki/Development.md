# Development Guide

## Environment Setup

To set up a development environment with all testing dependencies:

```bash
pip install -e ".[testing]"
```

## Running Tests

We use `pytest` for unit and integration testing.

### Local Execution
```bash
pytest
```

### With Coverage
```bash
pytest --cov=napari_rf
```

### Multi-Environment Testing
We use `tox` to ensure compatibility across different Python versions.
```bash
tox
```

## Testing GUI Components
When writing tests for `RFWidget`, use the `make_napari_viewer` fixture provided by `pytest-napari`.

Example (see `src/napari_rf/_tests/test_widget.py`):
```python
def test_rf_widget(make_napari_viewer):
    viewer = make_napari_viewer()
    widget = RFWidget(viewer)
    # Perform assertions
```

## Continuous Integration
GitHub Actions are configured to run tests on every push and pull request across Linux, macOS, and Windows. See `.github/workflows/test_and_deploy.yml`.

## Style Guide
- Follow [PEP 8](https://pep8.org/).
- Use `pre-commit` to ensure code quality:
  ```bash
  pip install pre-commit
  pre-commit install
  ```
