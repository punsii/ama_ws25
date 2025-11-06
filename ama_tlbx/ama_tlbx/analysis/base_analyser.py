"""Base analyzer class for all analysis components in the AMA module."""

from abc import ABC, abstractmethod
from typing import Any


class BaseAnalyser(ABC):
    """Abstract base class for data analysis components in the AMA module.

    All analyzers must:
    1. Accept a DatasetView in their constructor
    2. Implement fit() to perform the analysis and return self for chaining
    3. Implement result() to return a frozen dataclass with results

    The result() method must return a frozen @dataclass with all analysis outputs.
    This ensures immutability and proper type checking.
    """

    @abstractmethod
    def fit(self) -> "BaseAnalyser":
        """Fit the analyzer to the data.

        Returns:
            Self for method chaining.
        """
        ...

    @abstractmethod
    def result(self) -> Any:
        """Return analysis results as a frozen dataclass instance.

        Returns:
            A frozen @dataclass containing all analysis results.

        Raises:
            ValueError: If fit() has not been called yet.
        """
        ...
