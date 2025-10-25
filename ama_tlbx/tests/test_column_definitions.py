"""Tests for column definition modules."""

import pytest

from ama_tlbx.data.base_columns import BaseColumn, ColumnMetadata
from ama_tlbx.data.life_expectancy_columns import LifeExpectancyColumn
from ama_tlbx.data.siebenkampf_columns import SiebenkampfColumn


class TestColumnMetadata:
    """Test ColumnMetadata dataclass."""

    def test_column_metadata_creation(self) -> None:
        """Test creating column metadata."""
        metadata = ColumnMetadata(
            original_name="Test Name",
            cleaned_name="test_name",
            dtype="float64",
            pretty_name="Test Name (units)",
        )
        assert metadata.original_name == "Test Name"
        assert metadata.cleaned_name == "test_name"
        assert metadata.dtype == "float64"
        assert metadata.pretty_name == "Test Name (units)"

    def test_column_metadata_is_frozen(self) -> None:
        """Test that ColumnMetadata is immutable."""
        metadata = ColumnMetadata(
            original_name="Test",
            cleaned_name="test",
            dtype="str",
            pretty_name="Test",
        )
        with pytest.raises(AttributeError):
            metadata.original_name = "Changed"  # type: ignore[misc]


class TestLifeExpectancyColumn:
    """Test LifeExpectancyColumn enum."""

    def test_target_column_exists(self) -> None:
        """Test that TARGET column is defined."""
        assert LifeExpectancyColumn.TARGET.value == "life_expectancy"

    def test_enum_values_are_snake_case(self) -> None:
        """Test that all enum values are valid snake_case identifiers."""
        for col in LifeExpectancyColumn:
            assert col.value.islower()
            assert " " not in col.value

    def test_metadata_access(self) -> None:
        """Test accessing metadata for columns."""
        col = LifeExpectancyColumn.GDP
        metadata = col.metadata()
        assert isinstance(metadata, ColumnMetadata)
        assert metadata.cleaned_name == "gdp"
        assert metadata.pretty_name == "GDP per Capita (USD)"

    def test_original_name_property(self) -> None:
        """Test original_name property."""
        assert LifeExpectancyColumn.HIV_AIDS.original_name == " HIV/AIDS"
        assert LifeExpectancyColumn.TARGET.original_name == "Life expectancy "

    def test_dtype_name_property(self) -> None:
        """Test dtype_name property."""
        assert LifeExpectancyColumn.GDP.dtype_name == "float64"
        assert LifeExpectancyColumn.COUNTRY.dtype_name == "str"

    def test_pretty_name_property(self) -> None:
        """Test pretty_name property."""
        assert LifeExpectancyColumn.SCHOOLING.pretty_name == "Schooling (years)"
        assert LifeExpectancyColumn.BMI.pretty_name == "BMI (Average)"

    def test_numeric_columns(self) -> None:
        """Test numeric_columns class method."""
        numeric = LifeExpectancyColumn.numeric_columns()
        assert "life_expectancy" in numeric
        assert "gdp" in numeric
        assert "country" not in numeric
        assert "status" not in numeric
        assert "year" not in numeric

    def test_feature_columns_include_target(self) -> None:
        """Test feature_columns with target included."""
        features = LifeExpectancyColumn.feature_columns(exclude_target=False)
        assert "life_expectancy" in features
        assert "gdp" in features

    def test_feature_columns_exclude_target(self) -> None:
        """Test feature_columns with target excluded."""
        features = LifeExpectancyColumn.feature_columns(exclude_target=True)
        assert "life_expectancy" not in features
        assert "gdp" in features
        assert "bmi" in features

    def test_identifier_columns(self) -> None:
        """Test identifier_columns class method."""
        identifiers = LifeExpectancyColumn.identifier_columns()
        assert identifiers == ["country", "year", "status"]


class TestSiebenkampfColumn:
    """Test SiebenkampfColumn enum."""

    def test_target_column_exists(self) -> None:
        """Test that TARGET column is defined."""
        assert SiebenkampfColumn.TARGET.value == "punkte85"

    def test_discipline_columns(self) -> None:
        """Test discipline_columns returns all 7 disciplines."""
        disciplines = SiebenkampfColumn.discipline_columns()
        assert len(disciplines) == 7
        assert "zeit_100m_huerden" in disciplines
        assert "hochsprung" in disciplines
        assert "kugelstossen" in disciplines
        assert "zeit_200m_lauf" in disciplines
        assert "weitsprung" in disciplines
        assert "speerwurf" in disciplines
        assert "zeit_800m_lauf" in disciplines

    def test_discipline_columns_order(self) -> None:
        """Test that disciplines are in competition order."""
        disciplines = SiebenkampfColumn.discipline_columns()
        expected_order = [
            "zeit_100m_huerden",
            "hochsprung",
            "kugelstossen",
            "zeit_200m_lauf",
            "weitsprung",
            "speerwurf",
            "zeit_800m_lauf",
        ]
        assert disciplines == expected_order

    def test_numeric_columns(self) -> None:
        """Test numeric_columns includes disciplines and target."""
        numeric = SiebenkampfColumn.numeric_columns()
        assert "punkte85" in numeric
        assert "zeit_100m_huerden" in numeric
        assert len(numeric) == 8  # 7 disciplines + target

    def test_feature_columns_include_target(self) -> None:
        """Test feature_columns with target included."""
        features = SiebenkampfColumn.feature_columns(exclude_target=False)
        assert "punkte85" in features
        assert len(features) == 8

    def test_feature_columns_exclude_target(self) -> None:
        """Test feature_columns with target excluded."""
        features = SiebenkampfColumn.feature_columns(exclude_target=True)
        assert "punkte85" not in features
        assert len(features) == 7

    def test_identifier_columns(self) -> None:
        """Test identifier_columns."""
        identifiers = SiebenkampfColumn.identifier_columns()
        assert "platzierung" in identifiers
        assert "name" in identifiers
        assert "land" in identifiers
        assert "jahr" in identifiers
        assert "wettkampf" in identifiers

    def test_metadata_access(self) -> None:
        """Test accessing metadata."""
        col = SiebenkampfColumn.HOCHSPRUNG
        metadata = col.metadata()
        assert metadata.cleaned_name == "hochsprung"
        assert metadata.pretty_name == "High Jump (m)"
        assert metadata.dtype == "float64"
