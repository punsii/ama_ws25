"""Column definitions and types for all datasets. Created by GitHub Copilot, Claude Sonnet 4.5.

This module provides a unified interface to column definitions across all datasets
in the ama_tlbx package. It re-exports all column enums and metadata structures
for backward compatibility.

For new code, prefer importing from the specific modules:
    - base_columns: BaseColumn, ColumnMetadata
    - life_expectancy_columns: LifeExpectancyColumn, LifeExpectancyColumnName, DevelopmentStatus
    - siebenkampf_columns: SiebenkampfColumn, SiebenkampfColumnName
"""

from .base_columns import BaseColumn, ColumnMetadata
from .life_expectancy_columns import (
    DevelopmentStatus,
    LifeExpectancyColumn,
    LifeExpectancyColumnName,
)
from .siebenkampf_columns import SiebenkampfColumn, SiebenkampfColumnName


__all__ = [
    "BaseColumn",
    "ColumnMetadata",
    "DevelopmentStatus",
    "LifeExpectancyColumn",
    "LifeExpectancyColumnName",
    "SiebenkampfColumn",
    "SiebenkampfColumnName",
]
