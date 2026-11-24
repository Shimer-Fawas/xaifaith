"""Paper figures, from the runs already on disk.

    python figures.py                      # everything it can find
    python figures.py --out runs/v1 --bxaic runs/bxaic --seeds runs/seed1 runs/seed2 runs/seed3

Each figure is written as PNG (300 dpi) and PDF into <out>/figures/. A figure
whose inputs are missing is skipped with a note rather than failing, so this is
safe to run while jobs are still going.

  fig1_explainer_distributions  per-explainer faithfulness, with the random control
  fig2_risk_coverage            gating: realised faithfulness against coverage
  fig3_cell_heatmap             within (explainer, architecture) Spearman
  fig4_agreement_vs_accuracy    where faithfulness and plausibility diverge
  fig5_ne_vs_faithfulness       the null-explanation metric against faithfulness
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from typing import Dict, List, Optional, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

plt.rcParams.update({
    "figure.dpi": 120,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.5,
    "legend.frameon": False,
})

# One colour per explainer, kept identical across every figure so a reader can
# track a method between panels. Random is grey: it is a control, not a result.
PALETTE: Dict[str, str] = {
    "integrated_gradients": "#1f4e79",
    "smoothgrad": "#2e75b6",
    "gradient_input": "#5b9bd5",
    "grad_cam": "#c55a11",
    "gnn_explainer": "#ed7d31",
    "saliency": "#843c0c",
    "occlusion": "#7f7f7f",
    "random": "#bfbfbf",
}
SHORT = {
    "integrated_gradients": "IntGrad",
    "smoothgrad": "SmoothGrad",
    "gradient_input": "Grad×Input",
    "grad_cam": "Grad-CAM",
    "gnn_explainer": "GNNExplainer",
    "saliency": "Saliency",
    "occlusion": "Occlusion",
    "random": "Random",
}



def _annotate_spread(ax, items, fontsize: float = 7.5) -> None:
    """Label points so none of the labels overlap.

    Results land on top of each other (four B-XAIC tasks share AUROC 1.000),
    so labels must be displaced. Placement is global rather than per-cluster:
    a label pushed down out of one group would otherwise land on the group
    below it. Labels are placed top-down, each dropping until it clears every
    label already placed near its x, with a leader line when it has moved.
    """
    if not items:
        return
    xs = [x for x, _, _ in items]
    ys = [y for _, y, _ in items]
    span_x = max(max(xs) - min(xs), 1e-9)
    span_y = max(max(ys) - min(ys), 1e-9)
    gap = 0.075          # minimum vertical clearance, as a fraction of the y span
    near = 0.16          # how close in x two labels must be to be able to clash

    placed: List[tuple] = []
    for i in sorted(range(len(items)), key=lambda k: -ys[k]):
        x, y, text = items[i]
        target = y / span_y + 0.015
        while any(abs(x - px) / span_x < near and abs(target - py) < gap
                  for px, py in placed):
            target -= gap
        placed.append((x, target))
        y_label = target * span_y
        moved = abs(y_label - y) > 0.02 * span_y
        ax.annotate(text, xy=(x, y), xytext=(x + 0.035 * span_x, y_label),
                    textcoords="data", fontsize=fontsize, va="center",
                    arrowprops=dict(arrowstyle="-", linewidth=0.5, color="#999999",
                                    shrinkA=0, shrinkB=3) if moved else None)


def _save(fig: plt.Figure, out_dir: str, name: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out_dir, f"{name}.{ext}"))
    plt.close(fig)
    print(f"  wrote {os.path.join(out_dir, name)}.png/.pdf")


def _load_rows(run_dir: str) -> Optional[pd.DataFrame]:
    path = os.path.join(run_dir, "meta", "all_rows.parquet")
    if not os.path.exists(path):
        return None
    return pd.read_parquet(path)


def _order_by_mean(df: pd.DataFrame, column: str = "faithfulness") -> List[str]:
    means = df.groupby("explainer")[column].mean().sort_values(ascending=False)
    return [e for e in means.index if e != "random"] + (
        ["random"] if "random" in means.index else []
    )


# --------------------------------------------------------------------------- #

def fig1_explainer_distributions(df: pd.DataFrame, out_dir: str, seeds: Sequence[pd.DataFrame] = ()) -> None:
    """Faithfulness distribution per explainer. The random control anchors zero."""
    order = _order_by_mean(df)
    fig, ax = plt.subplots(figsize=(7.2, 3.6))

    data = [df.loc[df.explainer == e, "faithfulness"].dropna().to_numpy() for e in order]
    parts = ax.violinplot(data, showextrema=False, widths=0.85)
    for body, explainer in zip(parts["bodies"], order):
        body.set_facecolor(PALETTE.get(explainer, "#999999"))
        body.set_alpha(0.55)
        body.set_edgecolor("none")

    positions = np.arange(1, len(order) + 1)
    ax.boxplot(data, positions=positions, widths=0.12, showfliers=False,
               medianprops=dict(color="white", linewidth=1.2),
               boxprops=dict(facecolor="#333333", color="#333333"),
               whiskerprops=dict(color="#333333", linewidth=0.8),
               capprops=dict(linewidth=0), patch_artist=True)

    # Seed spread on the mean, when replication runs are available.
    if seeds:
        for i, explainer in enumerate(order, start=1):
            means = [s.loc[s.explainer == explainer, "faithfulness"].mean() for s in seeds]
            means = [m for m in means if not np.isnan(m)]
            if len(means) > 1:
                ax.errorbar(i, float(np.mean(means)), yerr=float(np.std(means)),
                            fmt="o", color="black", markersize=3, capsize=2, linewidth=1)

    ax.axhline(0, color="#cc0000", linestyle="--", linewidth=1, zorder=0)
    ax.text(0.55, 0.03, "no better than random", fontsize=7,
            color="#cc0000", va="bottom", ha="left")
    ax.set_xticks(positions)
    ax.set_xticklabels([SHORT.get(e, e) for e in order], rotation=20, ha="right")
    real = df[df.explainer != "random"]
    within = float(real.groupby("explainer")["faithfulness"].std().mean())
    between = float(real.groupby("explainer")["faithfulness"].mean().std())
    ax.set_ylabel("per-molecule faithfulness $F$")
    ax.set_title("Knowing which explainer ran tells you little about any one molecule\n"
                 f"(within-explainer SD {within:.2f}, between-explainer SD {between:.2f})",
                 fontsize=9.5)
    ax.set_ylim(-1.05, 1.05)
    _save(fig, out_dir, "fig1_explainer_distributions")


def fig2_risk_coverage(run_dir: str, out_dir: str) -> None:
    """Gating: what you actually get when you keep only the trusted fraction."""
    path = os.path.join(run_dir, "meta", "meta_report.json")
    if not os.path.exists(path):
        print("  skip fig2: no meta_report.json")
        return
    with open(path) as fh:
        report = json.load(fh)
    folds = report.get("scaffold_cv", {}).get("folds", [])
    if not folds:
        print("  skip fig2: no folds in meta_report.json")
        return

    curves = [pd.DataFrame(f["risk_coverage"]) for f in folds if f.get("risk_coverage")]
    coverage = curves[0]["coverage"].to_numpy()
    stacked = np.vstack([c["mean_faithfulness"].to_numpy() for c in curves])
    mean, sd = stacked.mean(axis=0), stacked.std(axis=0)

    fig, ax = plt.subplots(figsize=(5.0, 3.6))
    ax.plot(coverage, mean, color="#1f4e79", linewidth=1.8, label="gated by predicted $F$")
    ax.fill_between(coverage, mean - sd, mean + sd, color="#1f4e79", alpha=0.18)

    ungated = float(np.mean([f["ungated_mean"] for f in folds]))
    oracle = float(np.mean([f["oracle_mean"] for f in folds]))
    ax.axhline(ungated, color="#7f7f7f", linestyle="--", linewidth=1.1,
               label="ungated mean (all kept)")

    # The oracle is a single achievable value at one coverage, not a curve: drawing
    # it across the axis would claim a ceiling at coverages it was never computed for.
    half = int(np.argmin(np.abs(coverage - 0.5)))
    ax.plot([0.5], [oracle], marker="*", markersize=11, color="#c55a11",
            linestyle="none", label="best possible at 50% coverage", zorder=4)
    ax.plot([0.5, 0.5], [mean[half], oracle], color="#c55a11", linewidth=0.8,
            linestyle=":", zorder=1)
    ax.annotate(f"{mean[half]:+.3f}", xy=(0.5, mean[half]), xytext=(-8, -14),
                textcoords="offset points", fontsize=8, ha="right")
    ax.annotate(f"{oracle:+.3f}", xy=(0.5, oracle), xytext=(-8, 4),
                textcoords="offset points", fontsize=8, ha="right", color="#c55a11")
    ax.set_xlabel("coverage (fraction of explanations kept)")
    ax.set_ylabel("realised mean faithfulness")
    ax.set_title("Discarding low-confidence explanations\nraises what is left")
    ax.invert_xaxis()
    ax.margins(y=0.14)
    ax.legend(loc="upper left", fontsize=8)
    _save(fig, out_dir, "fig2_risk_coverage")


def fig3_cell_heatmap(run_dir: str, out_dir: str) -> None:
    """Within (explainer, architecture) Spearman: no identity confound survives."""
    path = os.path.join(run_dir, "meta", "robustness.json")
    if not os.path.exists(path):
        print("  skip fig3: no robustness.json")
        return
    with open(path) as fh:
        cells = json.load(fh).get("cells", [])
    if not cells:
        print("  skip fig3: no cells in robustness.json")
        return

    frame = pd.DataFrame(cells).pivot(index="explainer", columns="architecture", values="spearman")
    order = [e for e in PALETTE if e in frame.index]
    frame = frame.loc[order]

    fig, ax = plt.subplots(figsize=(5.0, 3.4))
    image = ax.imshow(frame.to_numpy(), cmap="YlGnBu", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(frame.columns)))
    ax.set_xticklabels(frame.columns)
    ax.set_yticks(range(len(frame.index)))
    ax.set_yticklabels([SHORT.get(e, e) for e in frame.index])
    ax.grid(False)

    values = frame.to_numpy()
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            if np.isnan(values[i, j]):
                ax.text(j, i, "n/a", ha="center", va="center", fontsize=7.5,
                        color="#999999", style="italic")
                continue
            ax.text(j, i, f"{values[i, j]:.2f}", ha="center", va="center", fontsize=8,
                    color="white" if values[i, j] > 0.55 else "#222222")
    fig.colorbar(image, ax=ax, label="Spearman $\\rho$", fraction=0.04, pad=0.03)
    ax.set_title("Prediction holds inside every cell\n(both identities removed)")
    _save(fig, out_dir, "fig3_cell_heatmap")


def _bxaic_task_accuracy(bxaic_dir: str) -> Dict[str, float]:
    accuracy: Dict[str, List[float]] = {}
    for path in glob.glob(os.path.join(bxaic_dir, "models", "*", "*", "metrics.json")):
        task = os.path.basename(os.path.dirname(os.path.dirname(path)))
        with open(path) as fh:
            score = json.load(fh).get("test_score")
        if score is not None and not np.isnan(score):
            accuracy.setdefault(task, []).append(float(score))
    return {task: float(np.mean(scores)) for task, scores in accuracy.items()}


def fig4_agreement_vs_accuracy(bxaic_dir: str, out_dir: str) -> None:
    """Where the two metrics diverge: agreement against how well the model learned."""
    files = glob.glob(os.path.join(bxaic_dir, "rows", "*.parquet"))
    if not files:
        print("  skip fig4: no B-XAIC rows")
        return
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    positives = df[(df.label > 0.5) & df.se_auroc.notna() & df.faithfulness.notna()]
    accuracy = _bxaic_task_accuracy(bxaic_dir)

    points = []
    for task, group in positives.groupby("task"):
        means = group.groupby("explainer")[["faithfulness", "se_auroc"]].mean()
        if len(means) < 3:
            continue
        points.append({
            "task": task,
            "rank_rho": float(spearmanr(means.faithfulness, means.se_auroc).statistic),
            "accuracy": accuracy.get(task, np.nan),
            "truth_atoms": float(group.n_truth_atoms.mean()),
        })
    frame = pd.DataFrame(points).dropna(subset=["accuracy"])
    if frame.empty:
        print("  skip fig4: no task accuracies found")
        return

    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.5))
    for ax, xcol, xlabel, title in (
        (axes[0], "accuracy", "task test AUROC",
         "Metric agreement against how well\nthe model learned the task"),
        (axes[1], "truth_atoms", "mean ground-truth atoms",
         "Metric agreement against\nground-truth pattern size"),
    ):
        ax.scatter(frame[xcol], frame.rank_rho, s=46, color="#1f4e79", zorder=3)
        # Several tasks share an x value (four sit at AUROC 1.000), so their
        # labels would print on top of each other; stagger them vertically.
        ax.margins(x=0.18, y=0.14)
        _annotate_spread(ax, [(float(r[xcol]), float(r.rank_rho), r.task)
                              for _, r in frame.iterrows()], fontsize=7)
        ax.axhline(0, color="#cc0000", linestyle="--", linewidth=1, zorder=0)
        rho = spearmanr(frame[xcol], frame.rank_rho).statistic
        ax.set_xlabel(xlabel)
        ax.set_title(f"{title}\n(Spearman {rho:+.2f})", fontsize=9)
    axes[0].set_ylabel("rank correlation between\nfaithfulness and plausibility")
    _save(fig, out_dir, "fig4_agreement_vs_accuracy")


def fig5_ne_vs_faithfulness(bxaic_dir: str, out_dir: str) -> None:
    """The null-explanation metric rewards attributions that say nothing."""
    files = glob.glob(os.path.join(bxaic_dir, "rows", "*.parquet"))
    if not files:
        print("  skip fig5: no B-XAIC rows")
        return
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)

    faith = df[df.faithfulness.notna()].groupby("explainer")["faithfulness"].mean()
    ne = df[(df.label < 0.5) & df.ne_score.notna()].groupby("explainer")["ne_score"].mean()
    # NB: do not name this column "ne" - DataFrame.ne is the not-equal method,
    # and attribute access would silently return it instead of the column.
    frame = pd.concat([faith.rename("faithfulness"), ne.rename("ne_score")], axis=1).dropna()
    if len(frame) < 3:
        print("  skip fig5: not enough explainers")
        return

    real = frame.drop(index="random", errors="ignore")
    rho = spearmanr(real["faithfulness"], real["ne_score"]).statistic

    fig, ax = plt.subplots(figsize=(5.2, 3.8))
    for explainer, row in frame.iterrows():
        ax.scatter(row["faithfulness"], row["ne_score"], s=70, zorder=3,
                   color=PALETTE.get(explainer, "#999999"),
                   edgecolor="black" if explainer == "random" else "none", linewidth=0.8)
    _annotate_spread(ax, [(float(r["faithfulness"]), float(r["ne_score"]),
                           SHORT.get(e, e)) for e, r in frame.iterrows()])
    ax.margins(x=0.18, y=0.12)
    if "random" in frame.index:
        level = frame.loc["random", "ne_score"]
        ax.axhline(level, color="#cc0000", linestyle="--", linewidth=1)
        ax.text(ax.get_xlim()[1], level + 0.012, "random attribution scores here ",
                fontsize=7, color="#cc0000", ha="right", va="bottom")

    ax.set_xlabel("mean faithfulness $F$")
    ax.set_ylabel("B-XAIC null-explanation score")
    ax.set_title("The null-explanation metric runs backwards\n"
                 f"(Spearman {rho:+.2f} across real explainers)")
    _save(fig, out_dir, "fig5_ne_vs_faithfulness")


# --------------------------------------------------------------------------- #

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="runs/v1")
    parser.add_argument("--bxaic", default="runs/bxaic")
    parser.add_argument("--seeds", nargs="*", default=None,
                        help="replication run directories; default: runs/seed*")
    parser.add_argument("--figures-dir", default=None)
    args = parser.parse_args()

    figures_dir = args.figures_dir or os.path.join(args.out, "figures")
    seed_dirs = args.seeds if args.seeds is not None else sorted(glob.glob("runs/seed*"))
    seeds = [d for d in (_load_rows(s) for s in seed_dirs) if d is not None]
    print(f"main run: {args.out} | B-XAIC: {args.bxaic} | "
          f"replication runs loaded: {len(seeds)}")

    main_rows = _load_rows(args.out)
    if main_rows is not None:
        main_rows = main_rows[main_rows.get("is_circular_explainer", 0) < 0.5]
        fig1_explainer_distributions(main_rows, figures_dir, seeds)
    else:
        print("  skip fig1: no all_rows.parquet")
    fig2_risk_coverage(args.out, figures_dir)
    fig3_cell_heatmap(args.out, figures_dir)
    fig4_agreement_vs_accuracy(args.bxaic, figures_dir)
    fig5_ne_vs_faithfulness(args.bxaic, figures_dir)
    print(f"\nfigures in {figures_dir}")


if __name__ == "__main__":
    main()
