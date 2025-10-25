"""Data module for dataset classes."""

from .life_expectancy_columns import LifeExpectancyColumn as LECol
from .life_expectancy_dataset import LifeExpectancyDataset


__all__ = ["LECol", "LifeExpectancyDataset"]
