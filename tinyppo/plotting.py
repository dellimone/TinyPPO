from __future__ import annotations
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .logger import StepLogger


# Default set of metric keys and their plot labels / colors
_DEFAULT_PANELS = [
    ("reward",       "Mean Reward",             "tab:green"),
    ("kl",           "KL (log ratio)",          "tab:red"),
    ("kl_approx",    "KL Approx (≥0)",          "tab:pink"),
    ("beta",         "KL Coefficient (beta)",   "tab:purple"),
    ("entropy",      "Entropy",                 "tab:blue"),
    ("policy_loss",  "Policy Loss",             "tab:orange"),
    ("value_loss",   "Value Loss",              "tab:cyan"),
]


def plot_training_curves(
    source: "StepLogger | str",
    keys: list[str] | None = None,
    target_kl: float | None = None,
    save_path: str | None = None,
    show: bool = True,
):
    """Plot training curves from a StepLogger or a saved metrics.jsonl.

    Args:
        source: StepLogger instance or path to a metrics.jsonl file.
        keys: List of metric keys to plot. None = default 6-panel layout.
              Pass a custom list to plot any subset or ablation-specific metrics.
        target_kl: If given, draw a horizontal dashed reference line on the KL panel.
        save_path: If given, save the figure to this path (e.g. "outputs/curves.png").
        show: Whether to call plt.show(). Set False when saving only.

    Returns:
        matplotlib.figure.Figure
    """
    import matplotlib.pyplot as plt
    import matplotlib

    # ── Load data ──────────────────────────────────────────────────────────
    if isinstance(source, str):
        from .logger import StepLogger
        source = StepLogger.load(source)

    # ── Determine panels ───────────────────────────────────────────────────
    if keys is None:
        panels = [(k, label, color) for k, label, color in _DEFAULT_PANELS
                  if source.get_metric(k)]
    else:
        # Use provided keys; assign labels and colors from defaults if available
        default_map = {k: (label, color) for k, label, color in _DEFAULT_PANELS}
        panels = []
        for k in keys:
            label, color = default_map.get(k, (k, "tab:blue"))
            if source.get_metric(k):
                panels.append((k, label, color))

    if not panels:
        print("[plotting] No data found for the requested metrics.")
        return None

    # ── Layout ─────────────────────────────────────────────────────────────
    n = len(panels)
    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows), squeeze=False)
    fig.suptitle("PPO-RLHF Training", fontsize=14)

    for idx, (key, label, color) in enumerate(panels):
        row, col = divmod(idx, ncols)
        ax = axes[row][col]
        values = source.get_metric(key)
        steps = range(len(values))
        ax.plot(steps, values, color=color)
        ax.set_title(label)
        ax.set_xlabel("step")

        # Special decorations
        if key == "kl" and target_kl is not None:
            ax.axhline(y=target_kl, color="gray", linestyle="--",
                       label=f"target={target_kl}")
            ax.legend()

    # Hide unused axes
    for idx in range(len(panels), nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row][col].set_visible(False)

    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    if show:
        plt.show()

    return fig


# ══════════════════════════════════════════════════════════════════════════
#  Ablation analysis helpers
# ══════════════════════════════════════════════════════════════════════════
#
#  These helpers power the post-hoc ablation analysis in `analysis.ipynb`.
#  They live here so the notebook itself can stay focused on storytelling
#  rather than plumbing.
#
#  Layout:
#    - IO          : load_metrics, load_group
#    - Smoothing   : smooth
#    - Labeling    : LR_DISPLAY, LR_STYLES, GROUP_DISPLAY, run_sort_key, pretty_label
#    - Plotting    : plot_panels, plot_lr_panel, plot_final_bar
#    - Export      : save_svg
#
#  All functions are pure: they read DataFrames / dicts and return matplotlib
#  Figures. None of them write files unless explicitly asked (save_svg).

import re
from typing import Iterable


# ── IO ──────────────────────────────────────────────────────────────────────

def load_metrics(path):
    """Load a metrics.jsonl file into a pandas DataFrame."""
    import json
    import pandas as pd
    records = []
    with open(path) as f:
        for line in f:
            records.append(json.loads(line))
    return pd.DataFrame(records)


def load_group(group_dir, outputs_dir="outputs"):
    """Load every run inside an experiment group, sorted by `run_sort_key`.

    Returns a dict mapping run_name → DataFrame, ordered so that legends and
    bar charts come out in a meaningful (numeric) order rather than asciibetical.
    """
    group_path = Path(outputs_dir) / group_dir
    runs = {}
    for sub in sorted(group_path.iterdir(), key=lambda p: run_sort_key(p.name)):
        mf = sub / "metrics.jsonl"
        if mf.exists() and mf.stat().st_size > 0:
            runs[sub.name] = load_metrics(mf)
    return runs


# ── Smoothing ───────────────────────────────────────────────────────────────

def smooth(series, window=5):
    """Centered moving average. Used to make noisy curves visually readable."""
    return series.rolling(window, min_periods=1).mean()


# ── Labeling / sorting ──────────────────────────────────────────────────────
#
#  The run-directory naming scheme is compact (e.g. `tkl_10_lr1e6`) so that
#  filesystem listings stay tidy. These helpers translate that scheme into
#  human-readable legend labels and a numeric sort order.

LR_DISPLAY = {"lr1e6": "1e-6", "lr5e6": "5e-6", "lr1e5": "1e-5", "lr5e5": "5e-5"}
LR_STYLES = {"lr1e6": "-", "lr5e6": "--"}

GROUP_DISPLAY = {
    "lr":           "Learning Rate",
    "target_kl":    "Target KL",
    "kl_mode":      "KL Mode",
    "clip":         "PPO Clip",
    "adv_norm":     "Adv. Norm.",
    "ppo_epochs":   "PPO Epochs",
    "noisy_reward": "Noisy Reward",
    "gamma":        "Gamma",
    "lam":          "GAE Lambda",
    "critic":       "VF Coeff",
}


def _fmt_lr(lr_tag):
    return LR_DISPLAY.get(lr_tag, lr_tag)


def _fmt_decimal(val_str):
    """Convert compact int strings to decimals: '095' → '0.95', '08' → '0.8'."""
    if val_str == "1":
        return "1.0"
    if len(val_str) >= 2 and "." not in val_str and val_str[0] == "0":
        return "0." + val_str[1:]
    return val_str


def run_sort_key(name):
    """Numeric sort key for run directory names.

    Returns a tuple `(primary, lr_order)` so that runs sort first by their
    ablation parameter and then by learning rate within each parameter value.
    """
    # LR sweep itself: lr_1e-6, lr_5e-5 …
    m = re.match(r"^lr_(.+)$", name)
    if m:
        try:
            return (float(m.group(1)),)
        except ValueError:
            return (0.0,)

    # Strip trailing _lr<tag> to get prefix and learning-rate order
    idx = name.rfind("_lr")
    if idx != -1:
        prefix, lr_tag = name[:idx], name[idx + 1:]
    else:
        prefix, lr_tag = name, ""

    lr_order = {"lr1e6": 0, "lr5e6": 1, "lr1e5": 2, "lr5e5": 3}.get(lr_tag, 99)

    # tkl_<int>
    m = re.match(r"^tkl_(\d+)$", prefix)
    if m:
        return (int(m.group(1)), lr_order)

    # ppo_e<int>
    m = re.match(r"^ppo_e(\d+)$", prefix)
    if m:
        return (int(m.group(1)), lr_order)

    # noise_<float>
    m = re.match(r"^noise_(.+)$", prefix)
    if m:
        try:
            return (float(m.group(1)), lr_order)
        except ValueError:
            return (0.0, lr_order)

    # gamma / lam / vf — compact decimal strings
    for pat in [r"^gamma_(.+)$", r"^lam_(.+)$", r"^vf_(.+)$"]:
        m = re.match(pat, prefix)
        if m:
            try:
                return (float(_fmt_decimal(m.group(1))), lr_order)
            except ValueError:
                return (0.0, lr_order)

    # clip: numeric values first, "none" last
    m = re.match(r"^clip_(.+)$", prefix)
    if m:
        val = m.group(1)
        if val == "none":
            return (float("inf"), lr_order)
        try:
            return (float(_fmt_decimal(val)), lr_order)
        except ValueError:
            return (0.0, lr_order)

    # kl_mode: adaptive first, fixed_b<value> by value, none last
    m = re.match(r"^kl_(.+)$", prefix)
    if m:
        mode = m.group(1)
        if mode == "adaptive":
            return (-1.0, lr_order)
        if mode == "none":
            return (float("inf"), lr_order)
        m2 = re.match(r"^fixed_b(.+)$", mode)
        if m2:
            try:
                return (float(m2.group(1)), lr_order)
            except ValueError:
                return (0.0, lr_order)

    # adv_norm: global → batch → none
    m = re.match(r"^adv_(.+)$", prefix)
    if m:
        return ({"global": 0, "batch": 1, "none": 2}.get(m.group(1), 99), lr_order)

    return (0, lr_order)


def pretty_label(name):
    """Convert raw run-directory names to human-readable legend labels."""
    # LR sweep itself
    m = re.match(r"^lr_(.+)$", name)
    if m:
        return f"LR = {m.group(1)}"

    idx = name.rfind("_lr")
    if idx == -1:
        return name
    prefix, lr_tag = name[:idx], name[idx + 1:]
    lr_str = _fmt_lr(lr_tag)

    m = re.match(r"^tkl_(\d+)$", prefix)
    if m:
        return f"target_kl={m.group(1)},  lr={lr_str}"

    m = re.match(r"^kl_(.+)$", prefix)
    if m:
        mode = m.group(1)
        m2 = re.match(r"^fixed_b(.+)$", mode)
        if m2:
            return f"fixed β={m2.group(1)},  lr={lr_str}"
        return {"adaptive": f"adaptive,  lr={lr_str}",
                "none":     f"no KL,  lr={lr_str}"}.get(mode, f"{mode},  lr={lr_str}")

    m = re.match(r"^clip_(.+)$", prefix)
    if m:
        val = m.group(1)
        if val == "none":
            return f"no clip,  lr={lr_str}"
        return f"ε={_fmt_decimal(val)},  lr={lr_str}"

    m = re.match(r"^adv_(.+)$", prefix)
    if m:
        mode_map = {"batch": "batch-norm", "global": "global-norm", "none": "no-norm"}
        return f"{mode_map.get(m.group(1), m.group(1))},  lr={lr_str}"

    m = re.match(r"^ppo_e(\d+)$", prefix)
    if m:
        return f"epochs={m.group(1)},  lr={lr_str}"

    m = re.match(r"^noise_(.+)$", prefix)
    if m:
        return f"noise_std={m.group(1)},  lr={lr_str}"

    m = re.match(r"^gamma_(.+)$", prefix)
    if m:
        return f"γ={_fmt_decimal(m.group(1))},  lr={lr_str}"

    m = re.match(r"^lam_(.+)$", prefix)
    if m:
        return f"λ={_fmt_decimal(m.group(1))},  lr={lr_str}"

    m = re.match(r"^vf_(.+)$", prefix)
    if m:
        return f"vf_coef={_fmt_decimal(m.group(1))},  lr={lr_str}"

    return name


def _get_lr_style(name):
    for key, style in LR_STYLES.items():
        if key in name:
            return style
    return "-"


# ── Plotting ────────────────────────────────────────────────────────────────

def plot_panels(runs, metrics, titles, group_title, figsize=None, ylims=None,
                smooth_window=5, legend_loc="best"):
    """Multi-panel training curves: one panel per metric, all runs overlaid.

    Used for ablations where every run goes on the same axes (one color per
    run, dashed line for the lr=5e-6 variant).
    """
    import matplotlib.pyplot as plt
    import numpy as np

    n = len(metrics)
    if figsize is None:
        figsize = (14, 4 * n)
    fig, axes = plt.subplots(n, 1, figsize=figsize, sharex=True)
    if n == 1:
        axes = [axes]
    fig.suptitle(group_title, fontsize=14, fontweight="bold", y=1.01)

    colors = plt.cm.tab10(np.linspace(0, 1, len(runs)))

    for i, (metric, title) in enumerate(zip(metrics, titles)):
        ax = axes[i]
        for (name, df), color in zip(runs.items(), colors):
            if metric in df.columns:
                ls = _get_lr_style(name)
                ax.plot(df["step"], smooth(df[metric], smooth_window),
                        label=pretty_label(name), color=color,
                        linestyle=ls, linewidth=1.5)
        ax.set_ylabel(title)
        if ylims and i < len(ylims) and ylims[i] is not None:
            ax.set_ylim(ylims[i])
        ax.legend(fontsize=8, loc=legend_loc)

    axes[-1].set_xlabel("Step")
    plt.tight_layout()
    return fig


def plot_lr_panel(runs, metrics, titles, group_title, lr="lr1e6",
                  ylims=None, smooth_window=5, legend_loc="best"):
    """Single-LR slice of an ablation: one column of training curves.

    Useful when comparing the two learning-rate variants of an ablation
    side-by-side as separate figures (e.g. on a slide).
    """
    import matplotlib.pyplot as plt
    import numpy as np

    def _strip_lr(name):
        idx = name.rfind("_lr")
        return name[:idx] if idx != -1 else name

    def _short(name):
        return pretty_label(name).split(",")[0].strip()

    lr_runs = {n: df for n, df in runs.items() if lr in n}
    # Use the full set of params (across both LRs) so colors stay consistent
    params  = sorted({_strip_lr(n) for n in runs}, key=run_sort_key)
    palette = plt.cm.tab10(np.linspace(0, 1, max(len(params), 2)))
    pcolor  = {p: palette[i] for i, p in enumerate(params)}

    n = len(metrics)
    lr_label = {"lr1e6": "LR = 1e-6", "lr5e6": "LR = 5e-6"}.get(lr, lr)

    fig, axes = plt.subplots(n, 1, figsize=(6.5, 3.5 * n), sharex=True)
    if n == 1:
        axes = [axes]
    fig.suptitle(f"{group_title}  ·  {lr_label}", fontsize=13, fontweight="bold")

    for row, (metric, title) in enumerate(zip(metrics, titles)):
        ax = axes[row]
        for name, df in sorted(lr_runs.items(), key=lambda x: run_sort_key(x[0])):
            if metric in df.columns:
                ax.plot(df["step"], smooth(df[metric], smooth_window),
                        label=_short(name), color=pcolor[_strip_lr(name)],
                        linewidth=1.8)
        ax.set_ylabel(title)
        if ylims and row < len(ylims) and ylims[row] is not None:
            ax.set_ylim(ylims[row])
        ax.legend(loc=legend_loc)
        if row == n - 1:
            ax.set_xlabel("Step")

    plt.tight_layout()
    return fig


def plot_final_bar(runs, group_title):
    """Grouped bar chart of final reward, one pair of bars per ablation value
    (lr=1e-6 in blue, lr=5e-6 in red)."""
    import matplotlib.pyplot as plt
    import numpy as np

    def _strip_lr(name):
        idx = name.rfind("_lr")
        return name[:idx] if idx != -1 else name

    def _short(name):
        return pretty_label(name).split(",")[0].strip()

    params = sorted({_strip_lr(n) for n in runs}, key=run_sort_key)
    fig, ax = plt.subplots(figsize=(max(6, len(params) * 1.5), 4))
    w, xi = 0.35, np.arange(len(params))

    for i, p in enumerate(params):
        for side, (lr_key, clr, lbl) in enumerate([
                ("lr1e6", "steelblue", "lr = 1e-6"),
                ("lr5e6", "tomato",    "lr = 5e-6")]):
            full = f"{p}_{lr_key}"
            if full not in runs:
                continue
            v = float(runs[full]["reward"].iloc[-1])
            offset = (side - 0.5) * w
            ax.bar(i + offset, v, w, color=clr, alpha=0.85,
                   label=lbl if i == 0 else "_nolegend_")
            ax.text(i + offset, v + 0.012, f"{v:.2f}",
                    ha="center", va="bottom", fontsize=8)

    ax.set_xticks(xi)
    ax.set_xticklabels(
        [_short(f"{p}_lr1e6") if f"{p}_lr1e6" in runs else _short(f"{p}_lr5e6")
         for p in params],
        rotation=30, ha="right")
    ax.set_ylabel("Final Reward")
    ax.set_title(f"{group_title}  ·  Final Reward", fontweight="bold")
    ax.legend()
    ax.set_ylim(0, 1.0)
    plt.tight_layout()
    return fig


# ── Export ──────────────────────────────────────────────────────────────────

def save_svg(fig, name, save_dir="presentation/assets/plots"):
    """Save a figure as a slide-ready SVG with a transparent background."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    path = save_dir / f"{name}.svg"
    fig.savefig(path, format="svg", transparent=True, bbox_inches="tight")
    print(f"  saved → {path}")


# ══════════════════════════════════════════════════════════════════════════


def plot_samples(source: "StepLogger | str", save_path: str | None = None) -> None:
    """Print generated text samples from a StepLogger or metrics.jsonl.

    Args:
        source: StepLogger instance or path to metrics.jsonl.
        save_path: Optional path to write samples as a text file.
    """
    if isinstance(source, str):
        from .logger import StepLogger
        source = StepLogger.load(source)

    lines = []
    for step, samples in source._samples:
        lines.append(f"--- Step {step} ---")
        for s in samples:
            lines.append(f"  {s}")

    output = "\n".join(lines)
    print(output)

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w") as f:
            f.write(output + "\n")
