"""Tests for LifeExpectancyDataset."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ama_tlbx.data import LECol, LifeExpectancyDataset
from ama_tlbx.utils.paths import get_dataset_path


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
            aggregate_by_country="mean",
            drop_missing_target=False,
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
        assert "year" in identifier_cols

    def test_default_target_column(self, sample_dataset: LifeExpectancyDataset) -> None:
        """Test default target column."""
        assert sample_dataset.Col.TARGET == "life_expectancy"

    def test_from_csv_defaults_to_latest_year(self) -> None:
        """Default aggregation should pick the most recent available year in the CSV."""
        csv_path = get_dataset_path("life_expectancy")
        raw = pd.read_csv(csv_path)

        expected_year = int(raw.dropna(subset=["Life expectancy "])["Year"].max())

        ds = LifeExpectancyDataset.from_csv(csv_path=csv_path)

        year_series = ds.df[LECol.YEAR]
        years = (
            year_series.dt.year.unique()
            if pd.api.types.is_datetime64_any_dtype(year_series)
            else pd.Series(year_series, copy=False).astype(int).unique()
        )
        assert set(years) == {expected_year}
        assert ds.df.index.name == LECol.COUNTRY

    def test_from_csv_respects_explicit_year_override(self) -> None:
        """Passing an explicit year should override the latest-year default."""
        csv_path = get_dataset_path("life_expectancy")
        raw = pd.read_csv(csv_path)

        chosen_year = int(raw["Year"].min())  # earliest year present to ensure coverage

        ds = LifeExpectancyDataset.from_csv(csv_path=csv_path, aggregate_by_country=chosen_year)

        year_series = ds.df[LECol.YEAR]
        years = (
            year_series.dt.year.unique()
            if pd.api.types.is_datetime64_any_dtype(year_series)
            else pd.Series(year_series, copy=False).astype(int).unique()
        )
        assert set(years) == {chosen_year}

    def test_latest_valid_year_skips_missing_target(self) -> None:
        """Latest-year inference should ignore rows where the target is missing when requested."""
        df = pd.DataFrame(
            {
                LECol.COUNTRY: ["A", "A"],
                LECol.YEAR: pd.to_datetime(["2014", "2015"], format="%Y"),
                LECol.STATUS: [0, 0],
                LECol.TARGET: [1.0, float("nan")],
            },
        )

        assert LifeExpectancyDataset._latest_valid_year(df, target_col=LECol.TARGET) == 2014
        assert LifeExpectancyDataset._latest_valid_year(df, target_col=None) == 2015

    def test_tf_only_applies_transforms_without_standardization(self, sample_dataset: LifeExpectancyDataset) -> None:
        """tf_only should apply transforms but leave numeric scales unchanged."""
        raw = sample_dataset.df

        tf_only = sample_dataset.tf_only()
        assert np.isclose(tf_only[LECol.GDP].iloc[0], np.log1p(raw[LECol.GDP].iloc[0]))
        assert tf_only[LECol.TARGET].equals(raw[LECol.TARGET])
        assert "status_developed" in tf_only.columns
        assert LECol.STATUS not in tf_only.columns
        assert set(tf_only["status_developed"].unique()) <= {0, 1}

        tf_norm = sample_dataset.tf_and_norm()
        assert abs(float(tf_norm[LECol.GDP].mean())) < 1e-8
        assert np.isclose(float(tf_norm[LECol.GDP].std(ddof=0)), 1.0)
        assert tf_norm[LECol.TARGET].equals(raw[LECol.TARGET])
        assert "status_developed" in tf_norm.columns
        assert set(tf_norm["status_developed"].unique()) <= {0, 1}
