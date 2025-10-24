"""Tests for LifeExpectancyDataset."""

from pathlib import Path

import pandas as pd
import pytest

from ama_tlbx.data_handling.column_definitions import LifeExpectancyColumn as Col
from ama_tlbx.data_handling.life_expectancy_dataset import LifeExpectancyDataset


class TestLifeExpectancyDataset:
    """Test LifeExpectancyDataset functionality."""

    @pytest.fixture
    def sample_df(self) -> pd.DataFrame:
        """Create a sample DataFrame for testing."""
        return pd.DataFrame(
            {
                "Country": ["USA", "Canada", "Mexico"],
                "Year": ["2015", "2015", "2015"],
                "Status": ["Developed", "Developed", "Developing"],
                "Life expectancy ": [78.5, 82.0, 77.0],
                "GDP": [56000.0, 45000.0, 10000.0],
                " BMI ": [28.5, 26.0, 27.0],
                "Adult Mortality": [100.0, 80.0, 120.0],
            },
        )

    @pytest.fixture
    def sample_dataset(self, sample_df: pd.DataFrame, tmp_path: Path) -> LifeExpectancyDataset:
        """Create a sample dataset from CSV."""
        csv_path = tmp_path / "test_data.csv"
        sample_df.to_csv(csv_path, index=False)
        return LifeExpectancyDataset.from_csv(
            csv_path,
            aggregate_by_country=False,
            drop_missing_target=False,
        )

    def test_normalize_col_names(self, sample_df: pd.DataFrame) -> None:
        """Test column name normalization."""
        normalized = LifeExpectancyDataset._normalize_col_names(sample_df)

        # Check that columns are properly cleaned
        assert "country" in normalized.columns
        assert "life_expectancy" in normalized.columns
        assert "bmi" in normalized.columns
        assert "gdp" in normalized.columns

        # Check that original messy names are gone
        assert "Life expectancy " not in normalized.columns
        assert " BMI " not in normalized.columns

    def test_convert_data_types(self, sample_df: pd.DataFrame) -> None:
        """Test data type conversion."""
        normalized = LifeExpectancyDataset._normalize_col_names(sample_df)
        converted = LifeExpectancyDataset._convert_data_types(normalized)

        # Check that year is datetime
        assert pd.api.types.is_datetime64_any_dtype(converted["year"])

        # Check that numeric columns are numeric
        assert pd.api.types.is_numeric_dtype(converted["life_expectancy"])
        assert pd.api.types.is_numeric_dtype(converted["gdp"])
        assert pd.api.types.is_numeric_dtype(converted["bmi"])

    def test_from_csv_creates_dataset(self, sample_dataset: LifeExpectancyDataset) -> None:
        """Test that from_csv creates a dataset."""
        assert isinstance(sample_dataset, LifeExpectancyDataset)
        assert len(sample_dataset.df) > 0

    def test_from_csv_normalizes_columns(self, sample_dataset: LifeExpectancyDataset) -> None:
        """Test that from_csv normalizes column names."""
        df = sample_dataset.df
        assert "country" in df.columns
        assert "life_expectancy" in df.columns
        assert "gdp" in df.columns

    def test_aggregate_by_country(self, sample_df: pd.DataFrame, tmp_path: Path) -> None:
        """Test aggregation by country."""
        # Create data with multiple years
        multi_year_df = pd.concat(
            [
                sample_df,
                sample_df.assign(Year="2016"),
            ],
        )
        csv_path = tmp_path / "multi_year.csv"
        multi_year_df.to_csv(csv_path, index=False)

        dataset = LifeExpectancyDataset.from_csv(
            csv_path,
            aggregate_by_country=True,
        )

        # Should have 3 countries (one row per country)
        assert len(dataset.df) == 3
        assert set(dataset.df["country"]) == {"USA", "Canada", "Mexico"}

    def test_drop_missing_target(self, tmp_path: Path) -> None:
        """Test dropping rows with missing target values."""
        df_with_missing = pd.DataFrame(
            {
                "Country": ["USA", "Canada", "Mexico"],
                "Year": ["2015", "2015", "2015"],
                "Status": ["Developed", "Developed", "Developing"],
                "Life expectancy ": [78.5, None, 77.0],
                "GDP": [56000.0, 45000.0, 10000.0],
            },
        )
        csv_path = tmp_path / "missing_target.csv"
        df_with_missing.to_csv(csv_path, index=False)

        dataset = LifeExpectancyDataset.from_csv(
            csv_path,
            aggregate_by_country=False,
            drop_missing_target=True,
        )

        # Should only have 2 rows (Canada dropped)
        assert len(dataset.df) == 2
        assert not dataset.df["life_expectancy"].isna().any()

    def test_get_pretty_name(self, sample_dataset: LifeExpectancyDataset) -> None:
        """Test getting pretty names for columns."""
        assert sample_dataset.get_pretty_name("life_expectancy") == "Life Expectancy (years)"
        assert sample_dataset.get_pretty_name("gdp") == "GDP per Capita (USD)"
        assert sample_dataset.get_pretty_name("bmi") == "BMI (Average)"

    def test_get_pretty_name_fallback(self, sample_dataset: LifeExpectancyDataset) -> None:
        """Test fallback for unknown column names."""
        pretty = sample_dataset.get_pretty_name("unknown_column")
        assert pretty == "Unknown Column"

    def test_feature_columns(self, sample_dataset: LifeExpectancyDataset) -> None:
        """Test getting feature columns."""
        features = sample_dataset.feature_columns(include_target=False)

        assert "life_expectancy" not in features
        assert "gdp" in features
        assert "bmi" in features
        assert "adult_mortality" in features

    def test_feature_columns_include_target(self, sample_dataset: LifeExpectancyDataset) -> None:
        """Test getting feature columns including target."""
        features = sample_dataset.feature_columns(include_target=True)

        assert "life_expectancy" in features
        assert "gdp" in features

    def test_feature_columns_exclude_extra(self, sample_dataset: LifeExpectancyDataset) -> None:
        """Test excluding additional columns from features."""
        features = sample_dataset.feature_columns(
            include_target=False,
            extra_exclude=["gdp"],
        )

        assert "gdp" not in features
        assert "bmi" in features

    def test_identifier_columns(self, sample_dataset: LifeExpectancyDataset) -> None:
        """Test identifier columns attribute."""
        assert "country" in sample_dataset.identifier_columns
        assert "status" in sample_dataset.identifier_columns
        assert "year" in sample_dataset.identifier_columns

    def test_default_target_column(self, sample_dataset: LifeExpectancyDataset) -> None:
        """Test default target column."""
        assert sample_dataset.default_target_column == "life_expectancy"
