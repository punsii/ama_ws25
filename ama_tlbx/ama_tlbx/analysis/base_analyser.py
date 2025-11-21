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


    ---


    ### Adding a New Analyzer

    **1. Create analyzer class** (in `analysis/my_analyzer.py`):

    ```python
    from dataclasses import dataclass
    from ama_tlbx.data.views import DatasetView

    @dataclass(frozen=True)
    class MyAnalysisResult:
        '''Results package for MyAnalyzer.'''
        summary: pd.DataFrame
        # ... other result fields ...

    class MyAnalyzer:
        '''Pure computation analyzer (no plotting!).'''

        def __init__(self, view: DatasetView):
            self._view = view
            self._fitted = False

        def fit(self) -> "MyAnalyzer":
            '''Compute the analysis.'''
            # ... computation logic ...
            self._fitted = True
            return self

        def result(self) -> MyAnalysisResult:
            '''Return packaged results.'''
            if not self._fitted:
                raise ValueError("Call fit() first")
            return MyAnalysisResult(...)
    ```

    **2. Add factory method** to `BaseDataset`:

    ```python
    def my_analyzer(
        self,
        *,
        columns: Iterable[str] | None = None,
        standardized: bool = True,
    ) -> MyAnalyzer:
        '''Instantiate MyAnalyzer for this dataset.'''
        from ama_tlbx.analysis.my_analyzer import MyAnalyzer
        view = self.analyzer_view(columns=columns, standardized=standardized)
        return MyAnalyzer(view)
    ```

    ### Adding Visualization Functions

    **In `plotting/my_plots.py`**:

    ```python
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure
    from ama_tlbx.analysis.my_analyzer import MyAnalysisResult

    def plot_my_analysis(
        result: MyAnalysisResult,
        figsize: tuple[int, int] = (10, 6),
        **kwargs,
    ) -> Figure:
        '''Plot results from MyAnalyzer.

    Args:
            result: Analysis results from MyAnalyzer.fit().result()
            figsize: Figure dimensions in inches.
            **kwargs: Additional keyword arguments passed to plotting function.

    Returns:
            Matplotlib Figure object.
        '''

    fig, ax = plt.subplots(figsize=figsize)
    # ... plotting logic using result ...
    fig.tight_layout()
    return fig
    ```

    **Key principles:**

    - Accept `*Result` dataclasses (ensures metadata like pretty names are available)
    - Return `Figure` objects (not tuples unless multiple axes needed)
    - Accept `**kwargs` to forward to underlying plot functions
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
