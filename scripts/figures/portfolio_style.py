"""Shared styling for this repository's figures, in a light and a dark palette.

Call `apply("light")` or `apply("dark")` before drawing; every colour constant
below is rebound to the chosen palette, and `save()` writes to `name.png` for
light and `name-dark.png` for dark, so a figure-drawing function never has to
know which theme it is running under -- it just reads `ps.BLUE`, `ps.INK`, and
so on, same as before.
"""

from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt

PALETTES = {
    "light": {
        "PAPER": "#FFFFFF",
        "INK": "#111827",
        "MUTED": "#4B5563",
        "FAINT": "#9CA3AF",
        "HAIR": "#E5E7EB",
        "BLUE": "#2563EB",
        "BLUE_SOFT": "#BFDBFE",
        "GREEN": "#059669",
        "GREEN_SOFT": "#A7F3D0",
        "AMBER": "#D97706",
        "AMBER_SOFT": "#FDE68A",
        "RED": "#DC2626",
        "RED_SOFT": "#FECACA",
        "SLATE": "#64748B",
        "SLATE_SOFT": "#CBD5E1",
        "SEVEN": ["#2563EB", "#059669", "#D97706", "#64748B", "#DC2626", "#7C3AED", "#0891B2"],
    },
    "dark": {
        "PAPER": "none",
        "INK": "#E6EDF3",
        "MUTED": "#9BA7B4",
        "FAINT": "#6E7681",
        "HAIR": "#30363D",
        "BLUE": "#58A6FF",
        "BLUE_SOFT": "#1F3B57",
        "GREEN": "#3FB950",
        "GREEN_SOFT": "#1B3A2A",
        "AMBER": "#D29922",
        "AMBER_SOFT": "#3A2E10",
        "RED": "#F85149",
        "RED_SOFT": "#3D1F1E",
        "SLATE": "#94A3B8",
        "SLATE_SOFT": "#2B3744",
        "SEVEN": ["#58A6FF", "#3FB950", "#D29922", "#94A3B8", "#F85149", "#A78BFA", "#39C5CF"],
    },
}

FONTS = ["Segoe UI", "DejaVu Sans", "Helvetica", "Arial", "sans-serif"]

_THEME = "light"

#: Bound here so every name below exists before `apply()` is first called;
#: `apply()` reassigns them, it does not introduce them.
PAPER = PALETTES["light"]["PAPER"]
INK = PALETTES["light"]["INK"]
MUTED = PALETTES["light"]["MUTED"]
FAINT = PALETTES["light"]["FAINT"]
HAIR = PALETTES["light"]["HAIR"]
BLUE = PALETTES["light"]["BLUE"]
BLUE_SOFT = PALETTES["light"]["BLUE_SOFT"]
GREEN = PALETTES["light"]["GREEN"]
GREEN_SOFT = PALETTES["light"]["GREEN_SOFT"]
AMBER = PALETTES["light"]["AMBER"]
AMBER_SOFT = PALETTES["light"]["AMBER_SOFT"]
RED = PALETTES["light"]["RED"]
RED_SOFT = PALETTES["light"]["RED_SOFT"]
SLATE = PALETTES["light"]["SLATE"]
SLATE_SOFT = PALETTES["light"]["SLATE_SOFT"]
SEVEN = PALETTES["light"]["SEVEN"]


def apply(theme: str = "light") -> None:
    global _THEME
    _THEME = theme
    p = PALETTES[theme]
    globals().update(p)

    mpl.rcParams.update(
        {
            "figure.facecolor": p["PAPER"],
            "savefig.facecolor": p["PAPER"],
            "savefig.transparent": theme == "dark",
            "axes.facecolor": "none",
            "savefig.dpi": 200,
            "figure.dpi": 110,
            "font.family": "sans-serif",
            "font.sans-serif": FONTS,
            "text.color": p["INK"],
            "axes.labelcolor": p["MUTED"],
            "axes.edgecolor": p["HAIR"],
            "xtick.color": p["MUTED"],
            "ytick.color": p["MUTED"],
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.5,
            "axes.titlesize": 11.5,
            "axes.labelsize": 10.5,
            "legend.frameon": False,
            "axes.prop_cycle": mpl.cycler(
                color=[p["BLUE"], p["GREEN"], p["AMBER"], p["SLATE"], p["RED"]]
            ),
        }
    )


def clean(ax, *, left: bool = True, bottom: bool = True, grid_axis: str = "y") -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_visible(left)
    ax.spines["bottom"].set_visible(bottom)
    ax.tick_params(length=0)
    if grid_axis in ("x", "y", "both"):
        ax.grid(True, axis=grid_axis, color=HAIR, lw=0.9, zorder=0)
    ax.set_axisbelow(True)


def fits(fig, artist, *, margin: float = 0.02) -> bool:
    """True if a drawn text artist stays inside the canvas."""
    fig.canvas.draw()
    box = artist.get_window_extent(fig.canvas.get_renderer())
    return box.x1 <= fig.bbox.x1 * (1.0 - margin) and box.x0 >= 0.0


def title_block(
    fig, title: str, subtitle: str = "", *, y: float = 0.96, size: float = 20, x: float = 0.065
) -> None:
    fig.text(x, y, title, fontsize=size, color=INK, fontweight="600", va="top")
    if subtitle:
        gap = (size * 1.45) / (fig.get_figheight() * 72.0)
        fig.text(x, y - gap, subtitle, fontsize=11.2, color=MUTED, va="top", linespacing=1.55)


def footnote(fig, lines, *, y: float = 0.085, x: float = 0.065, size: float = 9.4) -> None:
    step = (size * 1.75) / (fig.get_figheight() * 72.0)
    for i, line in enumerate(lines):
        artist = fig.text(
            x, y - i * step, line, fontsize=size, color=FAINT, va="top", linespacing=1.5
        )
        if not fits(fig, artist):
            print(f"    caption overflows the canvas: {line[:60]}...")


def save(fig, out_dir, name: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = "" if _THEME == "light" else "-dark"
    fig.savefig(out_dir / f"{name}{suffix}.png")
    plt.close(fig)
    print(f"  wrote {name}{suffix}.png")
