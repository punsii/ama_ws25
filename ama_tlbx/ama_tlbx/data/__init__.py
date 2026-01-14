"""Data module for dataset classes."""

from .life_expectancy_columns import LifeExpectancyColumn as LECol
from .life_expectancy_dataset import LifeExpectancyDataset
from .undp_hdr_columns import UNDPHDRColumn as UNDPCol
from .undp_hdr_dataset import UNDPHDRDataset


__all__ = ["LECol", "LifeExpectancyDataset", "UNDPCol", "UNDPHDRDataset"]
