import os
from pathlib import Path
from typing import Literal


__all__ = ["get_data_dir", "get_dataset_path"]


_DATASET_MAP: dict[str, str] = {
    "life_expectancy": "life_expectancy_data.csv",
    "life_expectancy_updated": "Life-Expectancy-Data-Updated.csv",
    "milk": "milk.csv",
    "siebenkampf": "siebenkampf.csv",
    "undp_hdr": "HDR25_Composite_indices_complete_time_series.csv",
}


def get_data_dir() -> Path:
    """Get the path to the data directory.

    Returns:
        Path to the data directory
    """
    data_dir_env = os.getenv("DATA_DIR")
    data_dir = Path(data_dir_env) if data_dir_env is not None else (Path(__file__).parents[3] / "_data").resolve()
    assert data_dir.exists(), f"Data directory not found at {data_dir}"
    return data_dir


def get_dataset_path(
    filename: Literal["life_expectancy", "life_expectancy_updated", "milk", "siebenkampf", "undp_hdr"] | str,
) -> Path:
    """Get the full path to a dataset file in the data directory.

    Args:
        filename: Key to known dataset or custom filename

    Returns:
        Full path to the dataset file relative to the data directory of the project

    Supported: life_expectancy_data.csv milk.csv                 siebenkampf.csv
    """
    data_dir = get_data_dir()
    ds_path = data_dir / _DATASET_MAP.get(filename, filename)
    assert ds_path.exists(), f"Dataset file '{filename}' not found at {ds_path}"

    return ds_path
