"""Column definitions for the Siebenkampf (Heptathlon) dataset. Created by GitHub Copilot, Claude Sonnet 4.5."""

from typing import Literal

from .base_columns import BaseColumn, ColumnMetadata


class SiebenkampfColumn(BaseColumn):
    """Column names for the Siebenkampf (Heptathlon) dataset.

    This enum provides type-safe access to all column names in the dataset,
    ensuring consistency across the codebase and enabling IDE autocomplete.
    Each enum member's value is the cleaned column name used in DataFrames.

    The dataset contains results from Olympic Games heptathlon competitions,
    including seven disciplines (100m hurdles, high jump, shot put, 200m sprint,
    long jump, javelin throw, and 800m run).
    """

    # Target variable (points based on 1985 scoring system)
    TARGET = "punkte85"

    # Identifiers
    PLATZIERUNG = "platzierung"
    NAME = "name"
    LAND = "land"
    JAHR = "jahr"
    WETTKAMPF = "wettkampf"
    QUELLE = "quelle"

    # Date information
    GEBURTSDATUM = "geburtsdatum"

    # Disciplines (7 events of heptathlon)
    ZEIT_100M_HUERDEN = "zeit_100m_huerden"
    HOCHSPRUNG = "hochsprung"
    KUGELSTOSSEN = "kugelstossen"
    ZEIT_200M_LAUF = "zeit_200m_lauf"
    WEITSPRUNG = "weitsprung"
    SPEERWURF = "speerwurf"
    ZEIT_800M_LAUF = "zeit_800m_lauf"

    def metadata(self) -> ColumnMetadata:
        """Get metadata for this column.

        Returns:
            ColumnMetadata instance with original name, cleaned name, dtype, and pretty name.
        """
        return _COLUMN_METADATA_SIEBENKAMPF[self]

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
    def discipline_columns(cls) -> list[str]:
        """Get all seven discipline column names.

        Returns:
            List of discipline column names in competition order.
        """
        return [
            cls.ZEIT_100M_HUERDEN.value,
            cls.HOCHSPRUNG.value,
            cls.KUGELSTOSSEN.value,
            cls.ZEIT_200M_LAUF.value,
            cls.WEITSPRUNG.value,
            cls.SPEERWURF.value,
            cls.ZEIT_800M_LAUF.value,
        ]

    @classmethod
    def numeric_columns(cls) -> list[str]:
        """Get all numeric column names (disciplines + target).

        Returns:
            List of numeric column names.
        """
        return [*cls.discipline_columns(), cls.TARGET.value]

    @classmethod
    def feature_columns(cls, *, exclude_target: bool = False) -> list[str]:
        """Get all feature column names (disciplines).

        Args:
            exclude_target: If True, exclude punkte85 from features.

        Returns:
            List of feature column names (disciplines).
        """
        features = cls.discipline_columns()
        if not exclude_target:
            features = [*features, cls.TARGET.value]
        return features

    @classmethod
    def identifier_columns(cls) -> list[str]:
        """Get identifier column names.

        Returns:
            List of identifier column names.
        """
        return [
            cls.PLATZIERUNG.value,
            cls.NAME.value,
            cls.LAND.value,
            cls.JAHR.value,
            cls.WETTKAMPF.value,
        ]


# Column metadata mapping
_COLUMN_METADATA_SIEBENKAMPF: dict[SiebenkampfColumn, ColumnMetadata] = {
    # Target variable
    SiebenkampfColumn.TARGET: ColumnMetadata(
        original_name="Punkte85",
        cleaned_name="punkte85",
        dtype="float64",
        pretty_name="Points (1985 System)",
    ),
    # Identifiers
    SiebenkampfColumn.PLATZIERUNG: ColumnMetadata(
        original_name="Platzierung",
        cleaned_name="platzierung",
        dtype="int64",
        pretty_name="Placement",
    ),
    SiebenkampfColumn.NAME: ColumnMetadata(
        original_name="Name",
        cleaned_name="name",
        dtype="str",
        pretty_name="Athlete Name",
    ),
    SiebenkampfColumn.LAND: ColumnMetadata(
        original_name="Land",
        cleaned_name="land",
        dtype="str",
        pretty_name="Country",
    ),
    SiebenkampfColumn.JAHR: ColumnMetadata(
        original_name="jahr",
        cleaned_name="jahr",
        dtype="datetime64[ns]",
        pretty_name="Year",
    ),
    SiebenkampfColumn.WETTKAMPF: ColumnMetadata(
        original_name="wettkamp",
        cleaned_name="wettkampf",
        dtype="str",
        pretty_name="Competition",
    ),
    SiebenkampfColumn.QUELLE: ColumnMetadata(
        original_name="Quelle",
        cleaned_name="quelle",
        dtype="str",
        pretty_name="Source",
    ),
    # Date information
    SiebenkampfColumn.GEBURTSDATUM: ColumnMetadata(
        original_name="Gebutsdatum",
        cleaned_name="geburtsdatum",
        dtype="datetime64[ns]",
        pretty_name="Birth Date",
    ),
    # Disciplines
    SiebenkampfColumn.ZEIT_100M_HUERDEN: ColumnMetadata(
        original_name="Zeit_100m_Huerden",
        cleaned_name="zeit_100m_huerden",
        dtype="float64",
        pretty_name="100m Hurdles (s)",
    ),
    SiebenkampfColumn.HOCHSPRUNG: ColumnMetadata(
        original_name="Hochsprung",
        cleaned_name="hochsprung",
        dtype="float64",
        pretty_name="High Jump (m)",
    ),
    SiebenkampfColumn.KUGELSTOSSEN: ColumnMetadata(
        original_name="Kugelstoßen",
        cleaned_name="kugelstossen",
        dtype="float64",
        pretty_name="Shot Put (m)",
    ),
    SiebenkampfColumn.ZEIT_200M_LAUF: ColumnMetadata(
        original_name="Zeit_200m_Lauf",
        cleaned_name="zeit_200m_lauf",
        dtype="float64",
        pretty_name="200m Sprint (s)",
    ),
    SiebenkampfColumn.WEITSPRUNG: ColumnMetadata(
        original_name="Weitsprung",
        cleaned_name="weitsprung",
        dtype="float64",
        pretty_name="Long Jump (m)",
    ),
    SiebenkampfColumn.SPEERWURF: ColumnMetadata(
        original_name="Speerwurf",
        cleaned_name="speerwurf",
        dtype="float64",
        pretty_name="Javelin Throw (m)",
    ),
    SiebenkampfColumn.ZEIT_800M_LAUF: ColumnMetadata(
        original_name="Zeit_800m_Lauf",
        cleaned_name="zeit_800m_lauf",
        dtype="float64",
        pretty_name="800m Run (s)",
    ),
}


# Type aliases for column name validation
SiebenkampfColumnName = Literal[
    "platzierung",
    "name",
    "punkte85",
    "land",
    "geburtsdatum",
    "jahr",
    "wettkampf",
    "zeit_100m_huerden",
    "hochsprung",
    "kugelstossen",
    "zeit_200m_lauf",
    "weitsprung",
    "speerwurf",
    "zeit_800m_lauf",
    "quelle",
]
