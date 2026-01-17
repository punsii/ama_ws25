"""Column definitions for the UNDP HDR time series dataset."""

from __future__ import annotations

from .base_columns import BaseColumn, ColumnMetadata


class UNDPHDRColumn(BaseColumn):
    """Column names for the UNDP HDR time series dataset.

    Summary of series (UNDP HDR time series):
    - hdi: Human Development Index (HDI)
    - gnipc: Gross National Income per capita (PPP$)
    - le: Life expectancy at birth (years)
    - eys: Expected years of schooling (years)
    - mys: Mean years of schooling (years)
    - pop_total: Total population
    - gdi: Gender Development Index (GDI)
    - gdi_group: GDI group classification
    - hdi_rank: HDI rank
    - gii: Gender Inequality Index (GII)
    - gii_rank: GII rank
    - hdi_f / hdi_m: HDI (female / male)
    - le_f / le_m: Life expectancy (female / male)
    - eys_f / eys_m: Expected years of schooling (female / male)
    - mys_f / mys_m: Mean years of schooling (female / male)
    - gni_pc_f / gni_pc_m: GNI per capita (female / male, PPP$)
    - ihdi: Inequality-adjusted HDI (IHDI)
    - coef_ineq: Coefficient of inequality (HDI)
    - loss: Loss due to inequality (%)
    - ineq_le / ineq_edu / ineq_inc: Inequality in life expectancy / education / income
    - rankdiff_hdi_phdi: Rank difference between HDI and PHDI
    - phdi: Planetary pressures-adjusted HDI (PHDI)
    - diff_hdi_phdi: HDI minus PHDI
    - co2_prod: CO2 production (UNDP series)
    - mmr: Maternal mortality ratio
    - abr: Adolescent birth rate
    - se_f / se_m: Population with at least secondary education (female / male, %)
    - pr_f / pr_m: Parliamentary representation (female / male, % seats)
    - lfpr_f / lfpr_m: Labor force participation rate (female / male, %)
    - mf: UNDP series code MF (see HDR metadata)
    """

    ISO3 = "iso3"
    COUNTRY = "country"
    HDICODE = "hdicode"
    REGION = "region"
    YEAR = "year"
    HDI = "hdi"
    GNI_PC = "gnipc"
    LE = "le_undp"
    EYS = "eys"
    MYS = "mys"
    POP_TOTAL = "pop_total"
    GDI = "gdi"
    GDI_GROUP = "gdi_group"
    HDI_RANK = "hdi_rank"
    GII = "gii"
    GII_RANK = "gii_rank"
    HDI_F = "hdi_f"
    HDI_M = "hdi_m"
    LE_F = "le_f"
    LE_M = "le_m"
    EYS_F = "eys_f"
    EYS_M = "eys_m"
    MYS_F = "mys_f"
    MYS_M = "mys_m"
    GNI_PC_F = "gni_pc_f"
    GNI_PC_M = "gni_pc_m"
    IHDI = "ihdi"
    COEF_INEQ = "coef_ineq"
    LOSS = "loss"
    INEQ_LE = "ineq_le"
    INEQ_EDU = "ineq_edu"
    INEQ_INC = "ineq_inc"
    RANKDIFF_HDI_PHDI = "rankdiff_hdi_phdi"
    PHDI = "phdi"
    DIFF_HDI_PHDI = "diff_hdi_phdi"
    CO2_PROD = "co2_prod"
    MMR = "mmr"
    ABR = "abr"
    SE_F = "se_f"
    SE_M = "se_m"
    PR_F = "pr_f"
    PR_M = "pr_m"
    LFPR_F = "lfpr_f"
    LFPR_M = "lfpr_m"
    MF = "mf"

    TARGET = HDI

    def metadata(self) -> ColumnMetadata:
        mapping: dict[UNDPHDRColumn, ColumnMetadata] = {
            UNDPHDRColumn.ISO3: ColumnMetadata(
                original_name="iso3",
                cleaned_name="iso3",
                dtype="str",
                pretty_name="ISO3",
            ),
            UNDPHDRColumn.COUNTRY: ColumnMetadata(
                original_name="country",
                cleaned_name="country",
                dtype="str",
                pretty_name="Country",
            ),
            UNDPHDRColumn.HDICODE: ColumnMetadata(
                original_name="hdicode",
                cleaned_name="hdicode",
                dtype="str",
                pretty_name="HDI Code",
            ),
            UNDPHDRColumn.REGION: ColumnMetadata(
                original_name="region",
                cleaned_name="region",
                dtype="str",
                pretty_name="Region",
            ),
            UNDPHDRColumn.YEAR: ColumnMetadata(
                original_name="year",
                cleaned_name="year",
                dtype="int",
                pretty_name="Year",
            ),
            UNDPHDRColumn.HDI: ColumnMetadata(
                original_name="hdi",
                cleaned_name="hdi",
                dtype="float",
                pretty_name="Human Development Index (HDI)",
            ),
            UNDPHDRColumn.GNI_PC: ColumnMetadata(
                original_name="gnipc",
                cleaned_name="gnipc",
                dtype="float",
                pretty_name="GNI per Capita (PPP$)",
            ),
            UNDPHDRColumn.LE: ColumnMetadata(
                original_name="le_undp",
                cleaned_name="le_undp",
                dtype="float",
                pretty_name="Life Expectancy (UNDP)",
            ),
            UNDPHDRColumn.EYS: ColumnMetadata(
                original_name="eys",
                cleaned_name="eys",
                dtype="float",
                pretty_name="Expected Years of Schooling (UNDP)",
            ),
            UNDPHDRColumn.MYS: ColumnMetadata(
                original_name="mys",
                cleaned_name="mys",
                dtype="float",
                pretty_name="Mean Years of Schooling (UNDP)",
            ),
            UNDPHDRColumn.POP_TOTAL: ColumnMetadata(
                original_name="pop_total",
                cleaned_name="pop_total",
                dtype="float",
                pretty_name="Population (Total)",
            ),
            UNDPHDRColumn.GDI: ColumnMetadata(
                original_name="gdi",
                cleaned_name="gdi",
                dtype="float",
                pretty_name="Gender Development Index (GDI)",
            ),
            UNDPHDRColumn.GDI_GROUP: ColumnMetadata(
                original_name="gdi_group",
                cleaned_name="gdi_group",
                dtype="float",
                pretty_name="GDI Group",
            ),
            UNDPHDRColumn.HDI_RANK: ColumnMetadata(
                original_name="hdi_rank",
                cleaned_name="hdi_rank",
                dtype="float",
                pretty_name="HDI Rank",
            ),
            UNDPHDRColumn.GII: ColumnMetadata(
                original_name="gii",
                cleaned_name="gii",
                dtype="float",
                pretty_name="Gender Inequality Index (GII)",
            ),
            UNDPHDRColumn.GII_RANK: ColumnMetadata(
                original_name="gii_rank",
                cleaned_name="gii_rank",
                dtype="float",
                pretty_name="GII Rank",
            ),
            UNDPHDRColumn.HDI_F: ColumnMetadata(
                original_name="hdi_f",
                cleaned_name="hdi_f",
                dtype="float",
                pretty_name="HDI (Female)",
            ),
            UNDPHDRColumn.HDI_M: ColumnMetadata(
                original_name="hdi_m",
                cleaned_name="hdi_m",
                dtype="float",
                pretty_name="HDI (Male)",
            ),
            UNDPHDRColumn.LE_F: ColumnMetadata(
                original_name="le_f",
                cleaned_name="le_f",
                dtype="float",
                pretty_name="Life Expectancy (Female)",
            ),
            UNDPHDRColumn.LE_M: ColumnMetadata(
                original_name="le_m",
                cleaned_name="le_m",
                dtype="float",
                pretty_name="Life Expectancy (Male)",
            ),
            UNDPHDRColumn.EYS_F: ColumnMetadata(
                original_name="eys_f",
                cleaned_name="eys_f",
                dtype="float",
                pretty_name="Expected Years of Schooling (Female)",
            ),
            UNDPHDRColumn.EYS_M: ColumnMetadata(
                original_name="eys_m",
                cleaned_name="eys_m",
                dtype="float",
                pretty_name="Expected Years of Schooling (Male)",
            ),
            UNDPHDRColumn.MYS_F: ColumnMetadata(
                original_name="mys_f",
                cleaned_name="mys_f",
                dtype="float",
                pretty_name="Mean Years of Schooling (Female)",
            ),
            UNDPHDRColumn.MYS_M: ColumnMetadata(
                original_name="mys_m",
                cleaned_name="mys_m",
                dtype="float",
                pretty_name="Mean Years of Schooling (Male)",
            ),
            UNDPHDRColumn.GNI_PC_F: ColumnMetadata(
                original_name="gni_pc_f",
                cleaned_name="gni_pc_f",
                dtype="float",
                pretty_name="GNI per Capita (Female, PPP$)",
            ),
            UNDPHDRColumn.GNI_PC_M: ColumnMetadata(
                original_name="gni_pc_m",
                cleaned_name="gni_pc_m",
                dtype="float",
                pretty_name="GNI per Capita (Male, PPP$)",
            ),
            UNDPHDRColumn.IHDI: ColumnMetadata(
                original_name="ihdi",
                cleaned_name="ihdi",
                dtype="float",
                pretty_name="Inequality-adjusted HDI (IHDI)",
            ),
            UNDPHDRColumn.COEF_INEQ: ColumnMetadata(
                original_name="coef_ineq",
                cleaned_name="coef_ineq",
                dtype="float",
                pretty_name="Coefficient of Inequality (HDI)",
            ),
            UNDPHDRColumn.LOSS: ColumnMetadata(
                original_name="loss",
                cleaned_name="loss",
                dtype="float",
                pretty_name="Loss due to Inequality (%)",
            ),
            UNDPHDRColumn.INEQ_LE: ColumnMetadata(
                original_name="ineq_le",
                cleaned_name="ineq_le",
                dtype="float",
                pretty_name="Inequality in Life Expectancy",
            ),
            UNDPHDRColumn.INEQ_EDU: ColumnMetadata(
                original_name="ineq_edu",
                cleaned_name="ineq_edu",
                dtype="float",
                pretty_name="Inequality in Education",
            ),
            UNDPHDRColumn.INEQ_INC: ColumnMetadata(
                original_name="ineq_inc",
                cleaned_name="ineq_inc",
                dtype="float",
                pretty_name="Inequality in Income",
            ),
            UNDPHDRColumn.RANKDIFF_HDI_PHDI: ColumnMetadata(
                original_name="rankdiff_hdi_phdi",
                cleaned_name="rankdiff_hdi_phdi",
                dtype="float",
                pretty_name="Rank Difference: HDI vs PHDI",
            ),
            UNDPHDRColumn.PHDI: ColumnMetadata(
                original_name="phdi",
                cleaned_name="phdi",
                dtype="float",
                pretty_name="Planetary Pressures-adjusted HDI (PHDI)",
            ),
            UNDPHDRColumn.DIFF_HDI_PHDI: ColumnMetadata(
                original_name="diff_hdi_phdi",
                cleaned_name="diff_hdi_phdi",
                dtype="float",
                pretty_name="HDI minus PHDI",
            ),
            UNDPHDRColumn.CO2_PROD: ColumnMetadata(
                original_name="co2_prod",
                cleaned_name="co2_prod",
                dtype="float",
                pretty_name="CO2 Production (UNDP series)",
            ),
            UNDPHDRColumn.MMR: ColumnMetadata(
                original_name="mmr",
                cleaned_name="mmr",
                dtype="float",
                pretty_name="Maternal Mortality Ratio",
            ),
            UNDPHDRColumn.ABR: ColumnMetadata(
                original_name="abr",
                cleaned_name="abr",
                dtype="float",
                pretty_name="Adolescent Birth Rate",
            ),
            UNDPHDRColumn.SE_F: ColumnMetadata(
                original_name="se_f",
                cleaned_name="se_f",
                dtype="float",
                pretty_name="Secondary Education (Female, %)",
            ),
            UNDPHDRColumn.SE_M: ColumnMetadata(
                original_name="se_m",
                cleaned_name="se_m",
                dtype="float",
                pretty_name="Secondary Education (Male, %)",
            ),
            UNDPHDRColumn.PR_F: ColumnMetadata(
                original_name="pr_f",
                cleaned_name="pr_f",
                dtype="float",
                pretty_name="Parliamentary Representation (Female, %)",
            ),
            UNDPHDRColumn.PR_M: ColumnMetadata(
                original_name="pr_m",
                cleaned_name="pr_m",
                dtype="float",
                pretty_name="Parliamentary Representation (Male, %)",
            ),
            UNDPHDRColumn.LFPR_F: ColumnMetadata(
                original_name="lfpr_f",
                cleaned_name="lfpr_f",
                dtype="float",
                pretty_name="Labor Force Participation (Female, %)",
            ),
            UNDPHDRColumn.LFPR_M: ColumnMetadata(
                original_name="lfpr_m",
                cleaned_name="lfpr_m",
                dtype="float",
                pretty_name="Labor Force Participation (Male, %)",
            ),
            UNDPHDRColumn.MF: ColumnMetadata(
                original_name="mf",
                cleaned_name="mf",
                dtype="float",
                pretty_name="MF (UNDP series)",
            ),
        }
        return mapping[self]

    @classmethod
    def numeric_columns(cls) -> list[str]:
        return list(cls.time_series_prefix_map().values())

    @classmethod
    def identifier_columns(cls) -> list[str]:
        return [cls.ISO3, cls.COUNTRY, cls.HDICODE, cls.REGION, cls.YEAR]

    @classmethod
    def time_series_prefix_map(cls) -> dict[str, UNDPHDRColumn]:
        return {
            "hdi": cls.HDI,
            "gnipc": cls.GNI_PC,
            "le": cls.LE,
            "eys": cls.EYS,
            "mys": cls.MYS,
            "pop_total": cls.POP_TOTAL,
            "gdi": cls.GDI,
            "gdi_group": cls.GDI_GROUP,
            "hdi_rank": cls.HDI_RANK,
            "gii": cls.GII,
            "gii_rank": cls.GII_RANK,
            "hdi_f": cls.HDI_F,
            "hdi_m": cls.HDI_M,
            "le_f": cls.LE_F,
            "le_m": cls.LE_M,
            "eys_f": cls.EYS_F,
            "eys_m": cls.EYS_M,
            "mys_f": cls.MYS_F,
            "mys_m": cls.MYS_M,
            "gni_pc_f": cls.GNI_PC_F,
            "gni_pc_m": cls.GNI_PC_M,
            "ihdi": cls.IHDI,
            "coef_ineq": cls.COEF_INEQ,
            "loss": cls.LOSS,
            "ineq_le": cls.INEQ_LE,
            "ineq_edu": cls.INEQ_EDU,
            "ineq_inc": cls.INEQ_INC,
            "rankdiff_hdi_phdi": cls.RANKDIFF_HDI_PHDI,
            "phdi": cls.PHDI,
            "diff_hdi_phdi": cls.DIFF_HDI_PHDI,
            "co2_prod": cls.CO2_PROD,
            "mmr": cls.MMR,
            "abr": cls.ABR,
            "se_f": cls.SE_F,
            "se_m": cls.SE_M,
            "pr_f": cls.PR_F,
            "pr_m": cls.PR_M,
            "lfpr_f": cls.LFPR_F,
            "lfpr_m": cls.LFPR_M,
            "mf": cls.MF,
        }
