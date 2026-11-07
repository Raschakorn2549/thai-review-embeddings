"""Shared matplotlib setup so every figure in the report looks the same."""
import matplotlib.pyplot as plt

from src import config


def setup():
    plt.rcParams["font.family"] = config.PLOT_FONT
    # Leelawadee UI has no U+2212, so fall back to the ASCII hyphen
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.dpi"] = 110
    plt.rcParams["savefig.bbox"] = "tight"
    plt.rcParams["axes.grid"] = True
    plt.rcParams["grid.alpha"] = 0.3


def save(fig, name):
    for directory in (config.FIGURE_DIR, config.REPORT_FIGURE_DIR):
        directory.mkdir(parents=True, exist_ok=True)
        fig.savefig(directory / f"{name}.png", dpi=config.FIGURE_DPI)
