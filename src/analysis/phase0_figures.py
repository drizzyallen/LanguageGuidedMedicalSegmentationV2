"""Phase 0 required figures (from results/phase0_main_results.csv):

  1. reports/figures/phase0_qata_dice_ci.png      QaTa-COV19 Dice with 95% CI
  2. reports/figures/phase0_mosmed_dice_ci.png    MosMedData+ Dice with 95% CI
  3. reports/figures/phase0_seed_stability.png    three-seed test Dice for all methods

Usage: python src/analysis/phase0_figures.py
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT = os.path.join(ROOT, "reports", "figures")
METHODS = ["U-Net", "TransUNet", "nnU-Net"]
COLORS = {"U-Net": "#2a78d6", "TransUNet": "#eb6834", "nnU-Net": "#1baf7a"}  # fixed categorical order
SEEDS = (1001, 1002, 1003)
INK, INK_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"

plt.rcParams.update({
    "font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": INK_2, "xtick.color": INK_2, "ytick.color": INK_2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
})


def dice_ci_figure(df, dataset, filename):
    d = df[df.dataset == dataset].set_index("method").loc[METHODS] * 1
    fig, ax = plt.subplots(figsize=(6.4, 2.6))
    for y, method in enumerate(METHODS):
        r = d.loc[method]
        mean, lo, hi = 100 * r.dice_mean, 100 * r.dice_ci95_low, 100 * r.dice_ci95_high
        ax.plot([lo, hi], [y, y], color=COLORS[method], lw=2, solid_capstyle="round", zorder=2)
        ax.scatter([mean], [y], s=64, color=COLORS[method], edgecolor=SURFACE, linewidth=2, zorder=3)
        ax.text(hi + 0.25, y, f"{mean:.2f}  [{lo:.2f}, {hi:.2f}]", va="center", color=INK, fontsize=9)
    ax.set_yticks(range(len(METHODS)), METHODS, color=INK)
    ax.invert_yaxis()
    lo_all, hi_all = 100 * d.dice_ci95_low.min(), 100 * d.dice_ci95_high.max()
    ax.set_xlim(lo_all - 0.5, hi_all + 3.5)
    ax.set_xlabel("Test Dice (%) — mean over 3 seeds, 95% bootstrap CI")
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.tick_params(axis="y", length=0)
    ax.set_title(f"{dataset}: Dice with 95% CI ({int(d.n_test_cases.iloc[0])} test images)",
                 loc="left", color=INK, fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, filename), dpi=200)
    plt.close(fig)


def seed_stability_figure(df, filename):
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.2))
    for ax, dataset in zip(axes, ["QaTa-COV19", "MosMedData+"]):
        d = df[df.dataset == dataset].set_index("method").loc[METHODS]
        for x, method in enumerate(METHODS):
            r = d.loc[method]
            seeds = [100 * r[f"dice_seed{s}"] for s in SEEDS]
            ax.plot([x - 0.22, x + 0.22], [100 * r.dice_mean] * 2, color=INK_2, lw=2, solid_capstyle="round", zorder=2)
            ax.scatter([x] * 3, seeds, s=64, color=COLORS[method], edgecolor=SURFACE, linewidth=2, zorder=3)
            ax.text(x + 0.27, 100 * r.dice_mean, f"{100 * r.dice_mean:.2f} ± {100 * r.dice_sd:.2f}",
                    va="center", color=INK, fontsize=9)
        ax.set_xticks(range(len(METHODS)), METHODS, color=INK)
        ax.set_xlim(-0.5, len(METHODS) - 0.1)
        ax.grid(axis="y", color=GRID, lw=0.8)
        ax.tick_params(axis="x", length=0)
        ax.set_ylabel("Test Dice (%)")
        ax.set_title(dataset, loc="left", color=INK, fontsize=11)
    fig.suptitle("Three-seed stability: test Dice per seed (dots) and mean ± SD (line)",
                 x=0.01, ha="left", color=INK, fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, filename), dpi=200)
    plt.close(fig)


def main():
    os.makedirs(OUT, exist_ok=True)
    df = pd.read_csv(os.path.join(ROOT, "results", "phase0_main_results.csv"))
    dice_ci_figure(df, "QaTa-COV19", "phase0_qata_dice_ci.png")
    dice_ci_figure(df, "MosMedData+", "phase0_mosmed_dice_ci.png")
    seed_stability_figure(df, "phase0_seed_stability.png")
    print("figures written to", OUT)


if __name__ == "__main__":
    main()
