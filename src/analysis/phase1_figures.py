"""Phase 1 figures (from results/final_six_method_results.csv and the saved test predictions):

  1. reports/figures/phase1_qata_six_method_dice.png     six-method Dice with 95% CI, QaTa-COV19
  2. reports/figures/phase1_mosmed_six_method_dice.png   six-method Dice with 95% CI, MosMedData+
  3. reports/figures/phase1_seed_stability.png           three-seed test Dice of the language-guided methods
  4. reports/figures/phase1_example_predictions.png      optional: representative test predictions (seed 1001)

Example cases are not hand-picked: per dataset, the test images at the 25th, 50th and 75th percentile of the
seed-averaged Dice averaged over the five shown methods.

Usage (env lgms_phase0): python src/analysis/phase1_figures.py
"""
import os
import sys

import matplotlib

matplotlib.use("Agg")
import cv2  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.dataset import image_path, mask_path  # noqa: E402
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import phase0_results as P0  # noqa: E402

ROOT = P0.ROOT
OUT = os.path.join(ROOT, "reports", "figures")
IMAGE_ONLY = ["U-Net", "TransUNet", "nnU-Net"]
LANGUAGE = ["LViT-T", "RecLMIS", "DD-CMD"]
METHODS = IMAGE_ONLY + LANGUAGE
# Fixed categorical order (slots 1-6); slots 1-3 are the Phase 0 colors.
COLORS = {"U-Net": "#2a78d6", "TransUNet": "#eb6834", "nnU-Net": "#1baf7a", "LViT-T": "#eda100",
          "RecLMIS": "#e87ba4", "DD-CMD": "#008300"}
KEYS = {"U-Net": "unet", "TransUNet": "transunet", "nnU-Net": "nnunet", "LViT-T": "lvit", "RecLMIS": "reclmis",
        "DD-CMD": "ddcmd"}
SEEDS = (1001, 1002, 1003)
INK, INK_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"

plt.rcParams.update({
    "font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": INK_2, "xtick.color": INK_2, "ytick.color": INK_2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
})


def dice_ci_figure(df, dataset, filename):
    d = df[df.dataset == dataset].set_index("method").loc[METHODS]
    fig, ax = plt.subplots(figsize=(7.2, 3.9))
    for y, method in enumerate(METHODS):
        r = d.loc[method]
        mean, lo, hi = 100 * r.dice_mean, 100 * r.dice_ci95_low, 100 * r.dice_ci95_high
        ax.plot([lo, hi], [y, y], color=COLORS[method], lw=2, solid_capstyle="round", zorder=2)
        ax.scatter([mean], [y], s=64, color=COLORS[method], edgecolor=SURFACE, linewidth=2, zorder=3)
        ax.text(hi + 0.3, y, f"{mean:.2f}  [{lo:.2f}, {hi:.2f}]", va="center", color=INK, fontsize=9)
    ax.axhline(2.5, color=GRID, lw=1, zorder=1)
    ax.set_yticks(range(len(METHODS)), METHODS, color=INK)
    lo_all, hi_all = 100 * d.dice_ci95_low.min(), 100 * d.dice_ci95_high.max()
    ax.set_xlim(lo_all - 0.8, hi_all + 0.30 * (hi_all - lo_all) + 2)
    for y, label in ((-0.62, "image-only"), (2.62, "language-guided")):
        ax.text(0.0, y, label, transform=ax.get_yaxis_transform(), ha="left", va="center", color=INK_2,
                fontsize=8, style="italic")
    ax.set_ylim(len(METHODS) - 0.5, -0.9)
    ax.set_xlabel("Test Dice (%) — mean over 3 seeds, 95% bootstrap CI")
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.tick_params(axis="y", length=0)
    ax.set_title(f"{dataset}: six-method Dice with 95% CI ({int(d.n_test_cases.iloc[0])} test images)",
                 loc="left", color=INK, fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, filename), dpi=200)
    plt.close(fig)


def seed_stability_figure(df, filename):
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.2))
    for ax, dataset in zip(axes, ["QaTa-COV19", "MosMedData+"]):
        d = df[df.dataset == dataset].set_index("method").loc[LANGUAGE]
        for x, method in enumerate(LANGUAGE):
            r = d.loc[method]
            seeds = [100 * r[f"dice_seed{s}"] for s in SEEDS]
            ax.plot([x - 0.22, x + 0.22], [100 * r.dice_mean] * 2, color=INK_2, lw=2, solid_capstyle="round", zorder=2)
            ax.scatter([x] * 3, seeds, s=64, color=COLORS[method], edgecolor=SURFACE, linewidth=2, zorder=3)
            ax.annotate(f"{100 * r.dice_mean:.2f} ± {100 * r.dice_sd:.2f}", (x, max(seeds)), xytext=(0, 9),
                        textcoords="offset points", ha="center", va="bottom", color=INK, fontsize=9)
        ax.set_xticks(range(len(LANGUAGE)), LANGUAGE, color=INK)
        ax.set_xlim(-0.6, len(LANGUAGE) - 0.4)
        lo_y, hi_y = ax.get_ylim()
        ax.set_ylim(lo_y, hi_y + 0.12 * (hi_y - lo_y))
        ax.grid(axis="y", color=GRID, lw=0.8)
        ax.tick_params(axis="x", length=0)
        ax.set_ylabel("Test Dice (%)")
        ax.set_title(dataset, loc="left", color=INK, fontsize=11)
    fig.suptitle("Language-guided methods, three-seed stability: test Dice per seed (dots) and mean ± SD (line)",
                 x=0.01, ha="left", color=INK, fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, filename), dpi=200)
    plt.close(fig)


def example_figure(filename, shown=("U-Net", "nnU-Net", "LViT-T", "RecLMIS", "DD-CMD"), seed=1001):
    rows = []
    for dataset, name in P0.DATASETS.items():
        avg = pd.concat([P0.seed_averaged(KEYS[m], dataset)["dice"] for m in shown], axis=1).mean(axis=1)
        order = avg.sort_values(kind="stable")
        for q in (0.25, 0.5, 0.75):
            rows.append((dataset, name, order.index[int(round(q * (len(order) - 1)))], q))
    cols = ["Image", "Ground truth"] + list(shown)
    fig, axes = plt.subplots(len(rows), len(cols), figsize=(1.75 * len(cols), 1.95 * len(rows)))
    for i, (dataset, name, sid, q) in enumerate(rows):
        image = cv2.imread(image_path(dataset, "test", sid), cv2.IMREAD_GRAYSCALE)
        target = cv2.imread(mask_path(dataset, "test", sid), cv2.IMREAD_GRAYSCALE) > 0
        for j, col in enumerate(cols):
            ax = axes[i, j]
            ax.imshow(image, cmap="gray", vmin=0, vmax=255)
            if col == "Ground truth":
                mask, color, label = target, "#ffffff", ""
            elif col != "Image":
                pred_file = os.path.join(ROOT, "runs", KEYS[col], dataset, f"seed_{seed}", "test", "predictions",
                                         f"{sid}.png")
                mask = cv2.imread(pred_file, cv2.IMREAD_GRAYSCALE) > 0
                per_case = P0.per_case(KEYS[col], dataset, seed)
                color, label = COLORS[col], f"Dice {100 * per_case.loc[sid, 'dice']:.1f}"
            if col != "Image":
                overlay = np.zeros((*mask.shape, 4))
                overlay[mask] = matplotlib.colors.to_rgba(color, 0.55)
                ax.imshow(overlay)
                ax.contour(target, levels=[0.5], colors=["#ffffff"], linewidths=0.7)
                if label:
                    ax.text(0.03, 0.03, label, transform=ax.transAxes, color="#ffffff", fontsize=7.5,
                            bbox=dict(facecolor="#0b0b0b", alpha=0.6, pad=1.5, lw=0))
            ax.set_xticks([])
            ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
            if i == 0:
                ax.set_title(col, color=INK, fontsize=9)
            if j == 0:
                ax.set_ylabel(f"{name}\n{int(q * 100)}th pct", color=INK, fontsize=8)
    fig.suptitle(f"Representative test predictions (seed {seed}); white outline = ground truth. Cases at the "
                 "25th / 50th / 75th percentile of mean Dice over the shown methods", x=0.01, ha="left",
                 color=INK, fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, filename), dpi=170)
    plt.close(fig)
    return rows


def main():
    os.makedirs(OUT, exist_ok=True)
    df = pd.read_csv(os.path.join(ROOT, "results", "final_six_method_results.csv"))
    dice_ci_figure(df, "QaTa-COV19", "phase1_qata_six_method_dice.png")
    dice_ci_figure(df, "MosMedData+", "phase1_mosmed_six_method_dice.png")
    seed_stability_figure(df, "phase1_seed_stability.png")
    rows = example_figure("phase1_example_predictions.png")
    print("example cases:", [(r[1], r[2], r[3]) for r in rows])
    print("figures written to", OUT)


if __name__ == "__main__":
    main()
