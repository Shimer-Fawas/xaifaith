"""Faithfulness against plausibility, on the same molecules.

The project's premise is that an attribution can be faithful to the model and
wrong about the chemistry, or right about the chemistry and unfaithful. So far
that is argued across two papers with different setups. This measures both on
the same molecule, the same model and the same attribution, using B-XAIC's
ground-truth atoms for plausibility and our deletion metric for faithfulness.

    # 1. download B-XAIC into data/bxaic (see xaifaith.bxaic)
    python run_bxaic.py --inspect                 # check the reconstruction first
    python run_bxaic.py --tasks indole pains x    # train, attribute, compare

Output per (task, architecture, explainer, molecule):
    faithfulness   our metric, model-relative
    se_auroc       B-XAIC subgraph AUROC, chemistry-relative (positives)
    ne_score       B-XAIC null-explanation score (negatives)
    topk_recovery  fraction of ground-truth atoms in the top-k

The headline is the per-explainer Spearman between faithfulness and se_auroc.
Near zero means the two measure different things, which is the claim.
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from scipy.stats import spearmanr
from tqdm import tqdm

from xaifaith.bxaic import ATOM_LEVEL_TASKS, HF_URL, TASK_SOURCES, load_bxaic, plausibility, summarise_tasks
from xaifaith.data import subset
from xaifaith.explain import CIRCULAR_EXPLAINERS, compute_attribution
from xaifaith.faithfulness import FaithfulnessConfig, compute_references, faithfulness_for_attribution
from xaifaith.train import TrainConfig, load_model, train_model

DEFAULT_EXPLAINERS = [
    "saliency", "gradient_input", "integrated_gradients",
    "smoothgrad", "grad_cam", "gnn_explainer", "random",
]


def stage_inspect(args) -> None:
    print(f"inspecting {args.data}\n")
    table = summarise_tasks(args.data, max_molecules=args.max_molecules,
                            split_index=args.split_index)
    print(table.to_string(index=False))
    print(
        "\ncsv_sdf_alignment must be ~1.0: it checks that CSV row i really is SDF record i,\n"
        "which is what makes the released atom indices meaningful. Anything lower and the\n"
        "masks would be applied to the wrong atoms.\n"
        "recon_agreement cross-checks our RDKit rules against the released labels, and is\n"
        "only computed for the tasks where those rules are known to match (B, P, X, rings).\n"
        "positives_wo_mask counts positives with an empty explanation - expected to be 0."
    )


def stage_run(args) -> None:
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    fcfg = FaithfulnessConfig(
        n_random_orders=args.random_orders,
        max_curve_points=args.curve_points,
        greedy_max_atoms=args.greedy_max_atoms,
    )
    os.makedirs(os.path.join(args.out, "rows"), exist_ok=True)

    for task in args.tasks:
        loaded = load_bxaic(args.data, task, split_index=args.split_index,
                            max_molecules=args.max_molecules, seed=args.seed)
        dataset, split = loaded.dataset, loaded.split
        masks = {id(d): m for d, m in zip(dataset, loaded.masks)}

        if loaded.alignment < 0.99:
            print(f"[{task}] CSV/SDF alignment {loaded.alignment:.3f} - refusing to run; "
                  "the released atom indices would land on the wrong atoms")
            continue
        if loaded.labels.sum() < 50 or (1 - loaded.labels).sum() < 50:
            print(f"[{task}] too few of one class ({loaded.labels.sum():.0f} positive) - skipping")
            continue

        print(f"[{task}] {len(dataset)} molecules, {loaded.positive_rate:.1%} positive, "
              f"split_{args.split_index} test {len(split.test)}, "
              f"alignment {loaded.alignment:.3f}")

        for architecture in args.architectures:
            out_path = os.path.join(args.out, "rows", f"{task}__{architecture}.parquet")
            if os.path.exists(out_path) and not args.overwrite:
                print(f"  {architecture}: cached")
                continue

            ckpt = os.path.join(args.out, "models", task, architecture)
            if not os.path.exists(os.path.join(ckpt, "model.pt")) or args.overwrite:
                cfg = TrainConfig(architecture=architecture, epochs=args.epochs, seed=args.seed)
                metrics = train_model(dataset, split, "classification", cfg, ckpt, device=str(device))
                print(f"  {architecture}: test AUROC {metrics['test_score']:.3f}")
            model, _ = load_model(ckpt, device=str(device))

            rows: List[dict] = []
            for data in tqdm(subset(dataset, split.test), desc=f"{task}/{architecture}"):
                mask = masks[id(data)]
                label = float(data.y.view(-1)[0])
                if mask.size != int(data.num_nodes):
                    continue  # should not happen: both come from the same SDF mol

                refs = compute_references(model, data, device, fcfg)
                if refs.degenerate:
                    continue

                for explainer in args.explainers:
                    if explainer in CIRCULAR_EXPLAINERS:
                        continue
                    attribution = compute_attribution(explainer, model, data, device)
                    if not np.all(np.isfinite(attribution)):
                        continue
                    row = {
                        "task": task,
                        "architecture": architecture,
                        "explainer": explainer,
                        "smiles": data.smiles,
                        "label": float(label),
                        "n_atoms": int(data.num_nodes),
                        "atom_level_task": float(task in ATOM_LEVEL_TASKS),
                    }
                    row.update(faithfulness_for_attribution(model, data, attribution, refs, device, fcfg))
                    row.update(plausibility(attribution, mask, float(label)))
                    rows.append(row)

            df = pd.DataFrame(rows)
            if not df.empty:
                df["scaffold"] = [
                    MurckoScaffold.MurckoScaffoldSmiles(smiles=s, includeChirality=False)
                    if Chem.MolFromSmiles(s) is not None else ""
                    for s in df["smiles"]
                ]
                df.to_parquet(out_path, index=False)
            print(f"  {architecture}: {len(df)} rows -> {out_path}")


def stage_analyse(args) -> None:
    rows_dir = os.path.join(args.out, "rows")
    files = sorted(glob_parquet(rows_dir))
    if not files:
        raise SystemExit(f"no rows in {rows_dir}; run without --analyse-only first")
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    print(f"loaded {len(df)} rows / {df['smiles'].nunique()} molecules / "
          f"{df['task'].nunique()} tasks\n")

    positives = df[(df["label"] > 0.5) & df["se_auroc"].notna() & df["faithfulness"].notna()]

    print("FAITHFULNESS vs PLAUSIBILITY (positives; Spearman across molecules)")
    print(f"   {'explainer':>22}  {'mean F':>8}  {'mean SE':>8}  {'rho(F,SE)':>10}  {'n':>6}")
    summary: List[dict] = []
    for explainer in sorted(positives["explainer"].unique()):
        sub = positives[positives["explainer"] == explainer]
        if len(sub) < args.min_positives:
            print(f"   {explainer:>22}  only {len(sub)} positives "
                  f"(need {args.min_positives}) - skipped")
            continue
        rho = spearmanr(sub["faithfulness"], sub["se_auroc"]).statistic
        record = {
            "explainer": explainer,
            "mean_faithfulness": float(sub["faithfulness"].mean()),
            "mean_se_auroc": float(sub["se_auroc"].mean()),
            "rho_faith_vs_plaus": float(rho),
            "n": int(len(sub)),
        }
        summary.append(record)
        print(f"   {explainer:>22}  {record['mean_faithfulness']:>+8.3f}  "
              f"{record['mean_se_auroc']:>8.3f}  {rho:>10.3f}  {len(sub):>6d}")

    print("\n   rho near 0 => the two measure different things (the dissociation).")
    print("   rho near 1 => on these tasks they agree, and the distinction is academic.")

    print("\nRANK COMPARISON (do the two metrics order the explainers the same way?)")
    if len(summary) <= 2:
        print(f"   only {len(summary)} explainer(s) had enough positives - "
              "raise --max-molecules or lower --min-positives")
    else:
        frame = pd.DataFrame(summary)
        rank_rho = spearmanr(frame["mean_faithfulness"], frame["mean_se_auroc"]).statistic
        by_f = frame.sort_values("mean_faithfulness", ascending=False)["explainer"].tolist()
        by_p = frame.sort_values("mean_se_auroc", ascending=False)["explainer"].tolist()
        print(f"   by faithfulness: {' > '.join(by_f)}")
        print(f"   by plausibility: {' > '.join(by_p)}")
        print(f"   rank correlation: {rank_rho:.3f}")
        print("   Different orderings are the strongest form of the result: the method")
        print("   you would pick depends on which question you asked.")

    negatives = df[(df["label"] < 0.5) & df["ne_score"].notna()]
    if not negatives.empty:
        print("\nNULL EXPLANATIONS (negatives; 1.0 = nothing stood out)")
        ne = negatives.groupby("explainer")["ne_score"].agg(["mean", "count"])
        for name, row in ne.iterrows():
            print(f"   {name:>22}  {row['mean']:>6.3f}  n={int(row['count'])}")
        print("   A high score here with low faithfulness is the metric being gamed by")
        print("   saying nothing - exactly what B-XAIC flagged for GNNExplainer.")

    dest = os.path.join(args.out, "bxaic_summary.json")
    with open(dest, "w") as fh:
        json.dump({"per_explainer": summary}, fh, indent=2, default=float)
    print(f"\nsaved {dest}")


def glob_parquet(directory: str) -> List[str]:
    if not os.path.isdir(directory):
        return []
    return [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(".parquet")]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-index", type=int, default=0,
                        help="which predefined B-XAIC split (split_0 .. split_4)")
    parser.add_argument("--data", default="data/bxaic", help=f"unpacked B-XAIC ({HF_URL})")
    parser.add_argument("--out", default="runs/bxaic")
    parser.add_argument("--tasks", nargs="+", default=["indole", "pains", "x", "rings_max"],
                        choices=sorted(TASK_SOURCES))
    parser.add_argument("--architectures", nargs="+", default=["gin", "gat"])
    parser.add_argument("--explainers", nargs="+", default=DEFAULT_EXPLAINERS)
    parser.add_argument("--max-molecules", type=int, default=3000)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--random-orders", type=int, default=8)
    parser.add_argument("--curve-points", type=int, default=20)
    parser.add_argument("--greedy-max-atoms", type=int, default=60)
    parser.add_argument("--min-positives", type=int, default=30,
                        help="minimum positive molecules per explainer for a reported rho")
    parser.add_argument("--inspect", action="store_true", help="check the reconstruction and stop")
    parser.add_argument("--analyse-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.inspect:
        stage_inspect(args)
        return
    if not args.analyse_only:
        stage_run(args)
    stage_analyse(args)


if __name__ == "__main__":
    main()
