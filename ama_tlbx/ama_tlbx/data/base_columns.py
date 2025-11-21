"""Base column definitions and metadata structures."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    from pandas import Series


@dataclass(frozen=True)
class ColumnMetadata:
    """Metadata for a dataset column.

    Attributes:
        cleaned_name: Standardized column name used in DataFrames.
        dtype: Expected Python/pandas data type as a string.
        pretty_name: Human-readable name for use in plots and visualizations.
        transform: Optional callable to transform the raw Series (e.g., log1p).
    """

    original_name: str
    """Column name as it appears in the raw CSV file."""
    cleaned_name: str
    dtype: str
    pretty_name: str
    transform: Callable[[Series], Series] | None = None


class BaseColumn(StrEnum):
    """Base class for dataset column enums.

    All derived column enums must define a TARGET member to specify
    the target variable for the dataset.

    Subclasses must implement:
    - metadata(): Return ColumnMetadata for each enum member
    - numeric_columns(): Return list of numeric column names
    - identifier_columns(): Return list of identifier column names
    """

    TARGET: str

    def metadata(self) -> ColumnMetadata:
        """Get metadata for this column.

        Returns:
            ColumnMetadata instance with original name, cleaned name, dtype, and pretty name.

        Raises:
            NotImplementedError: If not implemented by subclass.
        """
        raise NotImplementedError(f"{self.__class__.__name__} must implement metadata() method")

    @classmethod
    def numeric_columns(cls) -> list[str]:
        """Get all numeric column names.

        Returns:
            List of numeric column names.

        Raises:
            NotImplementedError: If not implemented by subclass.
        """
        raise NotImplementedError(f"{cls.__name__} must implement numeric_columns() method")

    @classmethod
    def identifier_columns(cls) -> list[str]:
        """Get identifier column names.

        Returns:
            List of identifier column names (country, year, status).

        Raises:
            NotImplementedError: If not implemented by subclass.
        """
        raise NotImplementedError(f"{cls.__name__} must implement identifier_columns() method")

    @classmethod
    def feature_columns(cls, *, exclude_target: bool = False) -> list[str]:
        """Get all feature column names.

        Args:
            exclude_target: If True, exclude life_expectancy from features.

        Returns:
            List of feature column names.
        """
        features = cls.numeric_columns()
        if exclude_target:
            features = list(filter(lambda f: f != cls.TARGET, features))
        return features

    @property
    def pretty_name(self) -> str:
        """Get the human-readable name for plots and visualizations."""
        return self.metadata().pretty_name

    @property
    def original_name(self) -> str:
        """Get the original column name from the CSV file."""
        return self.metadata().original_name

    @property
    def dtype_name(self) -> str:
        """Get the expected data type as a string."""
        return self.metadata().dtype

    @property
    def transform(self) -> Callable[[Series], Series] | None:
        """Get the optional transformation function for this column."""
        return self.metadata().transform
