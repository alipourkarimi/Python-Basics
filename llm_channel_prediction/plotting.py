"""Matplotlib figures reproducing Fig. 4, Fig. 5, and Table I of the paper.

Series colors use a validated colorblind-safe categorical palette applied
in fixed slot order (identity is also carried by distinct line markers, so
the figures survive grayscale printing and CVD)."""

from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# validated categorical palette, fixed slot order (never cycled)
PALETTE = {
    "proposed":    "#2a78d6",   # blue
    "no_sensing":  "#1baf7a",   # aqua
    "lstm":        "#eda100",   # yellow
    "transformer": "#4a3aa7",   # violet
    "cnn":         "#e34948",   # red
}
MARKERS = {
    "proposed": "o", "no_sensing": "s", "lstm": "^",
    "transformer": "D", "cnn": "v",
}
LABELS = {
    "proposed": "Proposed LLM-based (with sensing)",
    "no_sensing": "No sensing assistance",
    "lstm": "LSTM",
    "transformer": "Transformer",
    "cnn": "CNN",
    "wo_channel_attention": "W/o channel attention",
    "wo_cross_attention": "W/o cross attention",
    "wo_llm": "W/o LLM",
}

_INK = "#0b0b0b"
_MUTED = "#898781"
_GRID = "#e1e0d9"
_SURFACE = "#fcfcfb"


def _style_axes(ax, xlabel: str, ylabel: str, title: str):
    ax.set_facecolor(_SURFACE)
    ax.set_xlabel(xlabel, color=_INK)
    ax.set_ylabel(ylabel, color=_INK)
    ax.set_title(title, color=_INK, pad=12)
    ax.grid(True, color=_GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(_MUTED)
    ax.tick_params(colors=_MUTED)


def plot_nmse_sweep(results: dict, x_values, xlabel: str, title: str,
                    path: str, log_y: bool = True):
    """results: {model_name: [nmse per x-value]}"""
    fig, ax = plt.subplots(figsize=(7.2, 4.8), dpi=150)
    fig.patch.set_facecolor(_SURFACE)
    for name in PALETTE:          # fixed slot order
        if name not in results:
            continue
        ax.plot(x_values, results[name],
                color=PALETTE[name], marker=MARKERS[name],
                markersize=6, linewidth=2, label=LABELS[name])
    if log_y:
        ax.set_yscale("log")
    _style_axes(ax, xlabel, "NMSE", title)
    ax.legend(frameon=False, fontsize=9, labelcolor=_INK)
    fig.tight_layout()
    fig.savefig(path, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"saved {path}")


def save_ablation_table(nmse_by_model: dict, path: str):
    """Table I: ablation NMSE over all testing velocities."""
    order = ["proposed", "no_sensing", "wo_channel_attention",
             "wo_cross_attention", "wo_llm"]
    lines = ["| Metric | " + " | ".join(
        LABELS.get(m, m) for m in order if m in nmse_by_model) + " |"]
    lines.append("|" + "---|" * (1 + sum(m in nmse_by_model
                                         for m in order)))
    lines.append("| NMSE | " + " | ".join(
        f"{nmse_by_model[m]:.3f}" for m in order if m in nmse_by_model)
        + " |")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"saved {path}")
    print("\n".join(lines))


def save_results_json(obj: dict, path: str):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
    print(f"saved {path}")
