"""Column definitions for the Life Expectancy dataset. Created by GitHub Copilot, Claude Sonnet 4.5."""

from typing import Literal

from .base_columns import BaseColumn, ColumnMetadata


class LifeExpectancyColumn(BaseColumn):
    """Column names for the Life Expectancy dataset.

    Columns:
    - ``country``: str - Country name
    - ``year``: datetime - Year of observation
    - ``status``: str - Development status (Developing/Developed)
    - ``life_expectancy``: float - Life expectancy in years (target variable)
    - ``adult_mortality``: float - Adult mortality rate per 1000 population
    - ``infant_deaths``: int - Number of infant deaths per 1000 population
    - ``alcohol``: float - Alcohol consumption per capita (liters)
    - ``percentage_expenditure``: float - Health expenditure as % of GDP per capita
    - ``hepatitis_b``: float - Hepatitis B immunization coverage (%)
    - ``measles``: int - Number of measles cases per 1000 population
    - ``bmi``: float - Average Body Mass Index
    - ``under_five_deaths``: int - Deaths of children under 5 per 1000 population
    - ``polio``: float - Polio immunization coverage (%)
    - ``total_expenditure``: float - Government health expenditure (% of total govt expenditure)
    - ``diphtheria``: float - Diphtheria immunization coverage (%)
    - ``hiv_aids``: float - Deaths per 1000 live births due to HIV/AIDS (0-4 years)
    - ``gdp``: float - Gross Domestic Product per capita (USD)
    - ``population``: int - Population of the country
    - ``thinness_1_19_years``: float - Prevalence of thinness among children 10-19 years (%)
    - ``thinness_5_9_years``: float - Prevalence of thinness among children 5-9 years (%)
    - ``income_composition_of_resources``: float - Human Development Index (0-1)
    - ``schooling``: float - Average years of schooling
    """

    # Target variable
    TARGET = "life_expectancy"

    # Identifiers
    COUNTRY = "country"
    YEAR = "year"
    STATUS = "status"

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
        return _COLUMN_METADATA_LIFE_EXPECTANCY[self]

    @classmethod
    def numeric_columns(cls) -> list[str]:
        return [col.value for col in cls if col not in {cls.COUNTRY, cls.STATUS, cls.YEAR}]

    @classmethod
    def identifier_columns(cls) -> list[str]:
        """Get identifier column names.

        Returns:
            List of identifier column names (country, year, status).
        """
        return [cls.COUNTRY, cls.YEAR, cls.STATUS]


# Column metadata mapping
_COLUMN_METADATA_LIFE_EXPECTANCY: dict[LifeExpectancyColumn, ColumnMetadata] = {
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
    LifeExpectancyColumn.TARGET: ColumnMetadata(
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
