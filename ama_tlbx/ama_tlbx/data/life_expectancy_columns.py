"""Column definitions for the Life Expectancy dataset. Created by GitHub Copilot, Claude Sonnet 4.5."""

from __future__ import annotations

import numpy as np
import pandas as pd
from patsy import bs

from .base_columns import BaseColumn, ColumnMetadata


def _log1p_under_coverage(series: pd.Series) -> pd.Series:
    return np.log1p(100 - series)


def _status_dummies(series: pd.Series, *, drop_first: bool = True) -> pd.DataFrame:
    """One-hot encode development status with stable, formula-safe names.

    Args:
        series: Status values (0/1 or Developing/Developed).
        drop_first: Drop the first category to avoid collinearity (default: True).

    Returns:
        DataFrame with columns ``status_developed`` (default) or both
        ``status_developing`` and ``status_developed`` when ``drop_first=False``.
    """
    s = series
    if pd.api.types.is_numeric_dtype(s):
        s = s.astype("Int64")
        if s.isna().any():
            mode = s.mode(dropna=True)
            fill_value = int(mode.iloc[0]) if not mode.empty else 0
            s = s.fillna(fill_value)
        labels = s.map({0: "developing", 1: "developed"}).astype("string")
    else:
        labels = s.astype("string").str.strip().str.lower()
        labels = labels.replace({"developing": "developing", "developed": "developed"})

    categories = ["developing", "developed"]
    cat = pd.Categorical(labels, categories=categories, ordered=True)
    cat_series = pd.Series(cat, index=series.index, name="status")
    return pd.get_dummies(cat_series, prefix="status", prefix_sep="_", drop_first=drop_first).astype(int)


def _spline(series: pd.Series, df: int = 2, degree: int = 2, prefix: str | None = None) -> pd.DataFrame:
    """Return spline basis with stable, column-specific names."""
    # Handle missing by median-imputing locally to avoid all-NaN basis
    s = series.fillna(series.median())
    basis = bs(s, df=df, degree=degree, include_intercept=False)
    name = prefix or (series.name if series.name is not None else "spline")
    return pd.DataFrame(
        basis,
        index=series.index,
        columns=[f"{name}_bs{i + 1}" for i in range(basis.shape[1])],
    )


class LifeExpectancyColumn(BaseColumn):
    """Column names for the Life Expectancy dataset as per [Life Expectancy (WHO) on Kaggle](https://www.kaggle.com/datasets/kumarajarshi/life-expectancy-who).

    Columns:
    - ``country``: str - Country name
    - ``year``: datetime - Year of observation
    - ``status``: str - Development status (Developing/Developed)
    - ``life_expectancy``: float - Life expectancy in years (target variable)
    - ``adult_mortality``: float - Adult mortality rate per 1000 population (probability of dying between 15 and 60 years)
    - ``infant_deaths``: int - Number of infant deaths per 1000 population
    - ``alcohol``: float - Alcohol, recorded per capita (15+) consumption (in litres of pure alcohol)
    - ``percentage_expenditure``: float - Expenditure on health as a percentage of GDP per capita (%)
    - ``hepatitis_b``: float - Hepatitis B (HepB) immunization coverage among 1-year-olds (%)
    - ``measles``: int - Number of measles cases per 1000 population
    - ``bmi``: float - Average Body Mass Index
    - ``under_five_deaths``: int - Deaths of children under 5 per 1000 population
    - ``polio``: float - Polio immunization coverage among 1-year-olds (%)
    - ``total_expenditure``: float - Government health expenditure (% of total govt expenditure)
    - ``diphtheria``: float - Diphtheria tetanus toxoid and pertussis (DTP3) immunization coverage among 1-year-olds (%)
    - ``hiv_aids``: float - Deaths per 1000 live births due to HIV/AIDS (0-4 years)
    - ``gdp``: float - Gross Domestic Product per capita (USD)
    - ``population``: int - Population of the country
    - ``thinness_1_19_years``: float - Prevalence of thinness among children 10-19 years (%)
    - ``thinness_5_9_years``: float - Prevalence of thinness among children 5-9 years (%)
    - ``human_development_index``: float - Human Development Index (HDI, UNDP composite; 0-1)
    - ``schooling``: float - Average years of schooling
    """

    # Target variable
    TARGET = "life_expectancy"
    """Life expectancy in years (target variable)."""
    LIFE_EXPECTANCY = TARGET

    # Identifiers
    COUNTRY = "country"
    """Country name."""
    YEAR = "year"
    """Year of observation."""

    # Binary indicator
    STATUS = "status"
    """Development status: 0 = Developing, 1 = Developed (binary indicator)."""

    # Mortality indicators
    ADULT_MORTALITY = "adult_mortality"
    """Adult mortality rate per 1000 population (probability of dying between 15 and 60 years)."""
    INFANT_DEATHS = "infant_deaths"
    """Number of infant deaths per 1000 population."""
    UNDER_FIVE_DEATHS = "under_five_deaths"
    """Deaths of children under 5 per 1000 population."""

    # Disease and health indicators
    HIV_AIDS = "hiv_aids"
    """Deaths per 1000 live births due to HIV/AIDS (0-4 years)."""
    MEASLES = "measles"
    """Number of measles cases per 1000 population."""

    # Immunization coverage (%)
    HEPATITIS_B = "hepatitis_b"
    """Hepatitis B (HepB) immunization coverage among 1-year-olds (%)."""
    POLIO = "polio"
    """Polio immunization coverage (%)."""
    DIPHTHERIA = "diphtheria"
    """Diphtheria immunization coverage (%)."""

    # Nutrition and physical health
    BMI = "bmi"
    """Average Body Mass Index."""
    THINNESS_1_19_YEARS = "thinness_1_19_years"
    """Prevalence of thinness among children 10-19 years (%)."""
    THINNESS_5_9_YEARS = "thinness_5_9_years"
    """Prevalence of thinness among children 5-9 years (%)."""

    # Economic indicators
    GDP = "gdp"
    """Gross Domestic Product per capita (USD)."""
    PERCENTAGE_EXPENDITURE = "percentage_expenditure"
    """Expenditure on health as a percentage of GDP per capita (%)."""
    TOTAL_EXPENDITURE = "total_expenditure"
    """Government health expenditure (% of total govt expenditure)."""

    # Social indicators
    HDI = "human_development_index"
    """Human Development Index (HDI, UNDP composite index; 0-1).

    Note: HDI combines health (life expectancy), education, and standard of living components.
    When using `life_expectancy` as the target, this variable can act as a proxy for the target itself;
    interpret coefficients cautiously and consider excluding HDI in models focused on causal explanation.
    """
    SCHOOLING = "schooling"
    """Average years of schooling."""
    ALCOHOL = "alcohol"
    """Alcohol, recorded per capita (15+) consumption (in litres of pure alcohol)."""

    # Population
    POPULATION = "population"
    """Population of the country."""

    def metadata(self) -> ColumnMetadata:
        """Get metadata for this column.

        Returns:
            ColumnMetadata instance with original name, cleaned name, dtype, and pretty name.
        """
        return _COLUMN_METADATA_LIFE_EXPECTANCY[self]

    @classmethod
    def numeric_columns(cls) -> list[str]:
        return [col.value for col in cls if col not in {cls.COUNTRY, cls.YEAR}]

    @classmethod
    def identifier_columns(cls) -> list[str]:
        """Get identifier column names.

        Returns:
            List of identifier column names (country, year).
        """
        return [cls.COUNTRY, cls.YEAR]


# Column metadata mapping
_COLUMN_METADATA_LIFE_EXPECTANCY: dict[LifeExpectancyColumn, ColumnMetadata] = {
    # Identifiers
    LifeExpectancyColumn.COUNTRY: ColumnMetadata(
        original_name="Country",
        cleaned_name="country",
        dtype="str",
        pretty_name="Country",
        transform=None,
    ),
    LifeExpectancyColumn.YEAR: ColumnMetadata(
        original_name="Year",
        cleaned_name="year",
        dtype="datetime64[ns]",
        pretty_name="Year",
        transform=None,
    ),
    LifeExpectancyColumn.STATUS: ColumnMetadata(
        original_name="Status",
        cleaned_name="status",
        dtype="int64",
        pretty_name="Development Status (0=Developing, 1=Developed)",
        transform=_status_dummies,
    ),
    # Target variable
    LifeExpectancyColumn.TARGET: ColumnMetadata(
        original_name="Life expectancy ",
        cleaned_name="life_expectancy",
        dtype="float64",
        pretty_name="Life Expectancy (years)",
        transform=None,
    ),
    # Mortality indicators
    LifeExpectancyColumn.ADULT_MORTALITY: ColumnMetadata(
        original_name="Adult Mortality",
        cleaned_name="adult_mortality",
        dtype="float64",
        pretty_name="Adult Mortality (per 1000)",
        transform=None,
    ),
    LifeExpectancyColumn.INFANT_DEATHS: ColumnMetadata(
        original_name="infant deaths",
        cleaned_name="infant_deaths",
        dtype="float64",
        pretty_name="Infant Deaths (per 1000)",
        transform=np.log1p,
    ),
    LifeExpectancyColumn.UNDER_FIVE_DEATHS: ColumnMetadata(
        original_name="under-five deaths ",
        cleaned_name="under_five_deaths",
        dtype="float64",
        pretty_name="Under-5 Deaths (per 1000)",
        transform=np.log1p,
    ),
    # Disease and health indicators
    LifeExpectancyColumn.HIV_AIDS: ColumnMetadata(
        original_name=" HIV/AIDS",
        cleaned_name="hiv_aids",
        dtype="float64",
        pretty_name="HIV/AIDS Deaths (per 1000 births)",
        transform=np.log1p,
    ),
    LifeExpectancyColumn.MEASLES: ColumnMetadata(
        original_name="Measles ",
        cleaned_name="measles",
        dtype="float64",
        pretty_name="Measles Cases (per 1000)",
        transform=np.log1p,
    ),
    # Immunization coverage (%)
    LifeExpectancyColumn.HEPATITIS_B: ColumnMetadata(
        original_name="Hepatitis B",
        cleaned_name="hepatitis_b",
        dtype="float64",
        pretty_name="Hepatitis B Coverage (%)",
        transform=_log1p_under_coverage,
    ),
    LifeExpectancyColumn.POLIO: ColumnMetadata(
        original_name="Polio",
        cleaned_name="polio",
        dtype="float64",
        pretty_name="Polio Coverage (%)",
        transform=_log1p_under_coverage,
    ),
    LifeExpectancyColumn.DIPHTHERIA: ColumnMetadata(
        original_name="Diphtheria ",
        cleaned_name="diphtheria",
        dtype="float64",
        pretty_name="Diphtheria Coverage (%)",
        transform=_log1p_under_coverage,
    ),
    # Nutrition and physical health
    LifeExpectancyColumn.BMI: ColumnMetadata(
        original_name=" BMI ",
        cleaned_name="bmi",
        dtype="float64",
        pretty_name="BMI (Average)",
        transform=None,
    ),
    LifeExpectancyColumn.THINNESS_1_19_YEARS: ColumnMetadata(
        original_name=" thinness  1-19 years",
        cleaned_name="thinness_1_19_years",
        dtype="float64",
        pretty_name="Thinness 10-19 Years (%)",
        transform=np.log1p,
    ),
    LifeExpectancyColumn.THINNESS_5_9_YEARS: ColumnMetadata(
        original_name=" thinness 5-9 years",
        cleaned_name="thinness_5_9_years",
        dtype="float64",
        pretty_name="Thinness 5-9 Years (%)",
        transform=np.log1p,
    ),
    # Economic indicators
    LifeExpectancyColumn.GDP: ColumnMetadata(
        original_name="GDP",
        cleaned_name="gdp",
        dtype="float64",
        pretty_name="GDP per Capita (USD)",
        transform=np.log1p,
    ),
    LifeExpectancyColumn.PERCENTAGE_EXPENDITURE: ColumnMetadata(
        original_name="percentage expenditure",
        cleaned_name="percentage_expenditure",
        dtype="float64",
        pretty_name="Health Expenditure (% of GDP per capita)",
        transform=np.log1p,
    ),
    LifeExpectancyColumn.TOTAL_EXPENDITURE: ColumnMetadata(
        original_name="Total expenditure",
        cleaned_name="total_expenditure",
        dtype="float64",
        pretty_name="Total Health Expenditure (% of govt expenditure)",
        transform=None,
    ),
    # Social indicators
    LifeExpectancyColumn.HDI: ColumnMetadata(
        original_name="Income composition of resources",
        cleaned_name="income_composition_of_resources",
        dtype="float64",
        pretty_name="Human Development Index (HDI, 0-1)",
        transform=None,
    ),
    LifeExpectancyColumn.SCHOOLING: ColumnMetadata(
        original_name="Schooling",
        cleaned_name="schooling",
        dtype="float64",
        pretty_name="Schooling (years)",
        transform=None,
    ),
    LifeExpectancyColumn.ALCOHOL: ColumnMetadata(
        original_name="Alcohol",
        cleaned_name="alcohol",
        dtype="float64",
        pretty_name="Alcohol Consumption (liters per capita)",
        transform=np.log1p,
    ),
    # Population
    LifeExpectancyColumn.POPULATION: ColumnMetadata(
        original_name="Population",
        cleaned_name="population",
        dtype="float64",
        pretty_name="Population",
        transform=np.log1p,
    ),
}
