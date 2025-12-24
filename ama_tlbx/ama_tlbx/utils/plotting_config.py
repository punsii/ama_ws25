"""Shared plotting configuration (style, palette, font sizes)."""

from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

import matplotlib as mpl
import plotly.io as pio
import seaborn as sns


@dataclass
class PlottingConfig:
    """Reusable plotting style that can be applied across figures."""

    style: str = "whitegrid"
    palette: str | list[str] = "tab10"
    font_family: str = "DejaVu Sans"
    font_scale: float = 1.0
    title_size: int = 14
    label_size: int = 12
    tick_size: int = 10
    figure_dpi: int = 100
    context: str = "notebook"
    plotly_template: str = "plotly_white"
    plotly_colorway: list[str] | None = None
    seaborn_kwargs: dict[str, Any] = field(default_factory=dict)

    def apply_global(self) -> None:
        """Apply plotting style globally (no automatic restore).

        This is intended for notebooks and Quarto documents where we want a
        consistent global plotting style set once at the top of the document.

        For temporary styling (with automatic restoration), use
        :meth:`apply` instead.
        """
        palette_colors = sns.color_palette(self.palette)

        sns.set_theme(style=self.style, palette=palette_colors, context=self.context, **self.seaborn_kwargs)
        sns.set_theme(font_scale=self.font_scale)
        mpl.rcParams.update(
            {
                "axes.titlesize": self.title_size,
                "axes.labelsize": self.label_size,
                "xtick.labelsize": self.tick_size,
                "ytick.labelsize": self.tick_size,
                "figure.dpi": self.figure_dpi,
                "axes.prop_cycle": mpl.cycler(color=palette_colors),
                "font.family": [self.font_family],
            },
        )

        pio.templates.default = self.plotly_template
        if self.plotly_colorway is not None:
            pio.templates[self.plotly_template].layout.colorway = self.plotly_colorway

    @contextmanager
    def apply(self) -> Generator[None]:
        """Apply style within a context, restoring previous rcParams afterwards."""
        keys = [
            "axes.titlesize",
            "axes.labelsize",
            "xtick.labelsize",
            "ytick.labelsize",
            "figure.dpi",
        ]
        prev = {k: mpl.rcParams.get(k) for k in keys}
        prev_prop_cycle = mpl.rcParams.get("axes.prop_cycle")
        prev_font_family = mpl.rcParams.get("font.family")
        prev_plotly_template = pio.templates.default
        # Colorway may be None on some templates; guard access
        prev_plotly_colorway = getattr(pio.templates[prev_plotly_template].layout, "colorway", None)
        target_plotly_colorway = getattr(pio.templates[self.plotly_template].layout, "colorway", None)

        palette_colors = sns.color_palette(self.palette)

        sns.set_theme(style=self.style, palette=palette_colors, context=self.context, **self.seaborn_kwargs)
        sns.set_theme(font_scale=self.font_scale)
        mpl.rcParams.update(
            {
                "axes.titlesize": self.title_size,
                "axes.labelsize": self.label_size,
                "xtick.labelsize": self.tick_size,
                "ytick.labelsize": self.tick_size,
                "figure.dpi": self.figure_dpi,
                "axes.prop_cycle": mpl.cycler(color=palette_colors),
                "font.family": [self.font_family],
            },
        )
        pio.templates.default = self.plotly_template
        if self.plotly_colorway is not None:
            pio.templates[self.plotly_template].layout.colorway = self.plotly_colorway
        try:
            yield
        finally:
            pio.templates.default = prev_plotly_template
            pio.templates[prev_plotly_template].layout.colorway = prev_plotly_colorway
            if self.plotly_colorway is not None:
                pio.templates[self.plotly_template].layout.colorway = target_plotly_colorway
            mpl.rcParams.update(prev)
            mpl.rcParams["axes.prop_cycle"] = prev_prop_cycle
            mpl.rcParams["font.family"] = prev_font_family


# Default configuration used across plotting functions
DEFAULT_PLOT_CFG = PlottingConfig()


__all__ = ["DEFAULT_PLOT_CFG", "PlottingConfig"]
