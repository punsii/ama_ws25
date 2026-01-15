"""Utility path resolution tests."""

from ama_tlbx.utils.paths import get_data_dir, get_dataset_path


def test_get_dataset_path_life_expectancy() -> None:
    """Ensure real dataset path resolution works."""
    data_dir = get_data_dir()
    le_path = get_dataset_path("life_expectancy")

    assert le_path.exists()
    assert le_path.parent == data_dir
    assert le_path.name.endswith("life_expectancy_data.csv")
