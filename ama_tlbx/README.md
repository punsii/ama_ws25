# AMA Toolbox

A comprehensive Python toolbox for Applied Multivariate Analysis (AMA), providing dataset handlers, analysis tools, and visualization utilities.

## Features

- **Data Handling**: Robust dataset classes with built-in preprocessing and standardization
- **Analysis Tools**:
  - Correlation analysis with multiple visualization options
  - PCA (Principal Component Analysis) with loading vectors and geometric operations
  - Outlier detection using multiple strategies (IQR, Z-Score, Isolation Forest)
- **Visualization**: Separated plotting module with publication-ready figures
- **Modular Architecture**: Clean separation between data, analysis, and visualization

## Requirements

Only required if using `Option A`:
- [uv package manager](https://docs.astral.sh/uv/getting-started/installation/)

## Setup Instructions

### Option A: Using `uv`

Install the package in editable mode with dependencies:

```bash
cd ama_tlbx

# Choose either:
uv sync # necessary dependencies only
uv sync --extra dev # dev tools
uv sync --extra notebook # Jupyter support
uv sync --all-extras # install both dev and notebook extras
```

This will also create a virtual environment `.venv` in the project directory, which is automatically used when running commands with `uv run ...`, it can also be activated manually.

```bash
source .venv/bin/activate
```

### Option B: Using `pip`

**Optional**: Create Conda env:

```bash
conda create -n ama python=3.13 -y
conda activate ama

which pip
```

The following will install the `ama-tlbx` to the currently active python env. Extras can be added by listing them within the dependency specifier, following the name of the package.

```bash
cd ama_tlbx

# Install all dependencies
pip install -e ".[dev,notebook,docs]"

# Install only the notebook extra
pip install -e ".[notebook]"

# Verify
pip list | grep ama-tlbx
```
