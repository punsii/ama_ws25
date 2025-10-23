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

- [uv package manager](https://docs.astral.sh/uv/getting-started/installation/)

## Setup Instructions

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

If you want to work inside Jupyter notebooks, you might want to register the Jupyter kernel:

```bash
uv run ipython kernel install --user --env VIRTUAL_ENV "$(pwd)/.venv" --name ama-venv
```
