"""Tests for LifeExpectancyDataset."""

from pathlib import Path

import pandas as pd
import pytest

from ama_tlbx.data import LECol, LifeExpectancyDataset


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
            csv_path=csv_path,
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
        df_le = sample_dataset.df
        assert "country" in df_le.columns
        assert "life_expectancy" in df_le.columns
        assert "gdp" in df_le.columns

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
            csv_path=csv_path,
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
            csv_path=csv_path,
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
        identifier_cols = sample_dataset.Col.identifier_columns()
        assert "country" in identifier_cols
        assert "status" in identifier_cols
        assert "year" in identifier_cols

    def test_default_target_column(self, sample_dataset: LifeExpectancyDataset) -> None:
        """Test default target column."""
        assert sample_dataset.Col.TARGET == "life_expectancy"

    def test_select_representative_year_filters_outliers(self, tmp_path: Path) -> None:
        """Representative year skips year-level outliers."""
        df = pd.DataFrame(
            {
                "Country": ["A", "B", "C", "D", "E"],
                "Year": ["2012", "2013", "2014", "2015", "2016"],
                "Status": ["Developing"] * 5,
                "Life expectancy ": [69.5, 71.0, 72.5, 74.0, 88.0],
                "GDP": [2800.0, 3000.0, 3200.0, 3400.0, 7800.0],
                " BMI ": [23.5, 24.5, 25.0, 26.0, 30.0],
                "Adult Mortality": [155.0, 150.0, 140.0, 135.0, 75.0],
            },
        )
        csv_path = tmp_path / "representative_year.csv"
        df.to_csv(csv_path, index=False)

        dataset = LifeExpectancyDataset.from_csv(
            csv_path=csv_path,
            aggregate_by_country=False,
            drop_missing_target=False,
        )

        year = dataset.select_representative_year()
        assert year == 2014

    def test_select_representative_year_prefer_recent_tiebreak(self, tmp_path: Path) -> None:
        """prefer_recent resolves ties toward the latest year."""
        df = pd.DataFrame(
            {
                "Country": ["A", "B", "C", "D", "E"],
                "Year": ["2011", "2012", "2013", "2014", "2015"],
                "Status": ["Developing"] * 5,
                "Life expectancy ": [69.0, 71.0, 72.0, 72.0, 88.0],
                "GDP": [2600.0, 3000.0, 3200.0, 3200.0, 7800.0],
                " BMI ": [23.0, 24.5, 25.0, 25.0, 31.0],
                "Adult Mortality": [160.0, 145.0, 140.0, 140.0, 80.0],
            },
        )
        csv_path = tmp_path / "representative_year_tie.csv"
        df.to_csv(csv_path, index=False)

        dataset = LifeExpectancyDataset.from_csv(
            csv_path=csv_path,
            aggregate_by_country=False,
            drop_missing_target=False,
        )

        default_year = dataset.select_representative_year()
        recent_year = dataset.select_representative_year(prefer_recent=True)

        assert default_year == 2013
        assert recent_year == 2014
