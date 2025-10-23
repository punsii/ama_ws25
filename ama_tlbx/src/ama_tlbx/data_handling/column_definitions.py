"""Column definitions and types for the Life Expectancy dataset. Created by GitHub Copilot, Claude Sonnet 4.5."""

from dataclasses import dataclass
from enum import Enum
from typing import Literal


@dataclass(frozen=True)
class ColumnMetadata:
    """Metadata for a dataset column.

    Attributes:
        original_name: Column name as it appears in the raw CSV file.
        cleaned_name: Standardized column name used in DataFrames.
        dtype: Expected Python/pandas data type as a string.
        pretty_name: Human-readable name for use in plots and visualizations.
    """

    original_name: str
    cleaned_name: str
    dtype: str
    pretty_name: str


class LifeExpectancyColumn(str, Enum):
    """Column names for the Life Expectancy dataset.

    This enum provides type-safe access to all column names in the dataset,
    ensuring consistency across the codebase and enabling IDE autocomplete.
    Each enum member's value is the cleaned column name used in DataFrames.
    """

    # Identifiers
    COUNTRY = "country"
    YEAR = "year"
    STATUS = "status"

    # Target variable
    LIFE_EXPECTANCY = "life_expectancy"

    # Mortality indicators
    ADULT_MORTALITY = "adult_mortality"
    INFANT_DEATHS = "infant_deaths"
    UNDER_FIVE_DEATHS = "under_five_deaths"

    # Disease and health indicators
    HIV_AIDS = "hiv_aids"
    MEASLES = "measles"

    # Immunization coverage (%)
    HEPATITIS_B = "hepatitis_b"
    POLIO = "polio"
    DIPHTHERIA = "diphtheria"

    # Nutrition and physical health
    BMI = "bmi"
    THINNESS_1_19_YEARS = "thinness_1_19_years"
    THINNESS_5_9_YEARS = "thinness_5_9_years"

    # Economic indicators
    GDP = "gdp"
    PERCENTAGE_EXPENDITURE = "percentage_expenditure"
    TOTAL_EXPENDITURE = "total_expenditure"

    # Social indicators
    INCOME_COMPOSITION = "income_composition_of_resources"
    SCHOOLING = "schooling"
    ALCOHOL = "alcohol"

    # Population
    POPULATION = "population"

    def metadata(self) -> ColumnMetadata:
        """Get metadata for this column.

        Returns:
            ColumnMetadata instance with original name, cleaned name, dtype, and pretty name.
        """
        return _COLUMN_METADATA[self]

    @property
    def original_name(self) -> str:
        """Get the original column name from the CSV file."""
        return self.metadata().original_name

    @property
    def dtype_name(self) -> str:
        """Get the expected data type as a string."""
        return self.metadata().dtype

    @property
    def pretty_name(self) -> str:
        """Get the human-readable name for plots and visualizations."""
        return self.metadata().pretty_name

    @classmethod
    def numeric_columns(cls) -> list[str]:
        """Get all numeric column names (excluding country, status, year).

        Returns:
            List of numeric column names.
        """
        return [col.value for col in cls if col not in {cls.COUNTRY, cls.STATUS, cls.YEAR}]

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
            features = [f for f in features if f != cls.LIFE_EXPECTANCY.value]
        return features

    @classmethod
    def identifier_columns(cls) -> list[str]:
        """Get identifier column names.

        Returns:
            List of identifier column names (country, year, status).
        """
        return [cls.COUNTRY.value, cls.YEAR.value, cls.STATUS.value]


# Column metadata mapping
_COLUMN_METADATA: dict[LifeExpectancyColumn, ColumnMetadata] = {
    # Identifiers
    LifeExpectancyColumn.COUNTRY: ColumnMetadata(
        original_name="Country",
        cleaned_name="country",
        dtype="str",
        pretty_name="Country",
    ),
    LifeExpectancyColumn.YEAR: ColumnMetadata(
        original_name="Year",
        cleaned_name="year",
        dtype="datetime64[ns]",
        pretty_name="Year",
    ),
    LifeExpectancyColumn.STATUS: ColumnMetadata(
        original_name="Status",
        cleaned_name="status",
        dtype="str",
        pretty_name="Development Status",
    ),
    # Target variable
    LifeExpectancyColumn.LIFE_EXPECTANCY: ColumnMetadata(
        original_name="Life expectancy ",  # Note: trailing space in CSV
        cleaned_name="life_expectancy",
        dtype="float64",
        pretty_name="Life Expectancy (years)",
    ),
    # Mortality indicators
    LifeExpectancyColumn.ADULT_MORTALITY: ColumnMetadata(
        original_name="Adult Mortality",
        cleaned_name="adult_mortality",
        dtype="float64",
        pretty_name="Adult Mortality (per 1000)",
    ),
    LifeExpectancyColumn.INFANT_DEATHS: ColumnMetadata(
        original_name="infant deaths",
        cleaned_name="infant_deaths",
        dtype="float64",
        pretty_name="Infant Deaths (per 1000)",
    ),
    LifeExpectancyColumn.UNDER_FIVE_DEATHS: ColumnMetadata(
        original_name="under-five deaths ",  # Note: trailing space in CSV
        cleaned_name="under_five_deaths",
        dtype="float64",
        pretty_name="Under-5 Deaths (per 1000)",
    ),
    # Disease and health indicators
    LifeExpectancyColumn.HIV_AIDS: ColumnMetadata(
        original_name=" HIV/AIDS",  # Note: leading space in CSV
        cleaned_name="hiv_aids",
        dtype="float64",
        pretty_name="HIV/AIDS Deaths (per 1000 births)",
    ),
    LifeExpectancyColumn.MEASLES: ColumnMetadata(
        original_name="Measles ",  # Note: trailing space in CSV
        cleaned_name="measles",
        dtype="float64",
        pretty_name="Measles Cases (per 1000)",
    ),
    # Immunization coverage (%)
    LifeExpectancyColumn.HEPATITIS_B: ColumnMetadata(
        original_name="Hepatitis B",
        cleaned_name="hepatitis_b",
        dtype="float64",
        pretty_name="Hepatitis B Coverage (%)",
    ),
    LifeExpectancyColumn.POLIO: ColumnMetadata(
        original_name="Polio",
        cleaned_name="polio",
        dtype="float64",
        pretty_name="Polio Coverage (%)",
    ),
    LifeExpectancyColumn.DIPHTHERIA: ColumnMetadata(
        original_name="Diphtheria ",  # Note: trailing space in CSV
        cleaned_name="diphtheria",
        dtype="float64",
        pretty_name="Diphtheria Coverage (%)",
    ),
    # Nutrition and physical health
    LifeExpectancyColumn.BMI: ColumnMetadata(
        original_name=" BMI ",  # Note: spaces in CSV
        cleaned_name="bmi",
        dtype="float64",
        pretty_name="BMI (Average)",
    ),
    LifeExpectancyColumn.THINNESS_1_19_YEARS: ColumnMetadata(
        original_name=" thinness  1-19 years",  # Note: spaces in CSV
        cleaned_name="thinness_1_19_years",
        dtype="float64",
        pretty_name="Thinness 10-19 Years (%)",
    ),
    LifeExpectancyColumn.THINNESS_5_9_YEARS: ColumnMetadata(
        original_name=" thinness 5-9 years",  # Note: leading space in CSV
        cleaned_name="thinness_5_9_years",
        dtype="float64",
        pretty_name="Thinness 5-9 Years (%)",
    ),
    # Economic indicators
    LifeExpectancyColumn.GDP: ColumnMetadata(
        original_name="GDP",
        cleaned_name="gdp",
        dtype="float64",
        pretty_name="GDP per Capita (USD)",
    ),
    LifeExpectancyColumn.PERCENTAGE_EXPENDITURE: ColumnMetadata(
        original_name="percentage expenditure",
        cleaned_name="percentage_expenditure",
        dtype="float64",
        pretty_name="Health Expenditure (% of GDP per capita)",
    ),
    LifeExpectancyColumn.TOTAL_EXPENDITURE: ColumnMetadata(
        original_name="Total expenditure",
        cleaned_name="total_expenditure",
        dtype="float64",
        pretty_name="Total Health Expenditure (% of govt expenditure)",
    ),
    # Social indicators
    LifeExpectancyColumn.INCOME_COMPOSITION: ColumnMetadata(
        original_name="Income composition of resources",
        cleaned_name="income_composition_of_resources",
        dtype="float64",
        pretty_name="Income Composition (HDI)",
    ),
    LifeExpectancyColumn.SCHOOLING: ColumnMetadata(
        original_name="Schooling",
        cleaned_name="schooling",
        dtype="float64",
        pretty_name="Schooling (years)",
    ),
    LifeExpectancyColumn.ALCOHOL: ColumnMetadata(
        original_name="Alcohol",
        cleaned_name="alcohol",
        dtype="float64",
        pretty_name="Alcohol Consumption (liters per capita)",
    ),
    # Population
    LifeExpectancyColumn.POPULATION: ColumnMetadata(
        original_name="Population",
        cleaned_name="population",
        dtype="float64",
        pretty_name="Population",
    ),
}


# Type aliases for column name validation
LifeExpectancyColumnName = Literal[
    "country",
    "year",
    "status",
    "life_expectancy",
    "adult_mortality",
    "infant_deaths",
    "under_five_deaths",
    "hiv_aids",
    "measles",
    "hepatitis_b",
    "polio",
    "diphtheria",
    "bmi",
    "thinness_1_19_years",
    "thinness_5_9_years",
    "gdp",
    "percentage_expenditure",
    "total_expenditure",
    "income_composition_of_resources",
    "schooling",
    "alcohol",
    "population",
]

DevelopmentStatus = Literal["Developing", "Developed"]
