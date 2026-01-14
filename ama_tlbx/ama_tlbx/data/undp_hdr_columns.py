"""Column definitions for the UNDP HDR time series dataset."""

from __future__ import annotations

from .base_columns import BaseColumn, ColumnMetadata


class UNDPHDRColumn(BaseColumn):
    """Column names for the UNDP HDR time series dataset."""

    ISO3 = "iso3"
    COUNTRY = "country"
    YEAR = "year"
    HDI = "hdi"
    GNI_PC = "gnipc"
    LE = "le_undp"

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
        }
        return mapping[self]

    @classmethod
    def numeric_columns(cls) -> list[str]:
        return [cls.HDI, cls.GNI_PC, cls.LE]

    @classmethod
    def identifier_columns(cls) -> list[str]:
        return [cls.ISO3, cls.COUNTRY, cls.YEAR]
