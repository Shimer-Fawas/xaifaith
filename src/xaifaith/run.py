"""End-to-end pipeline.

    python -m xaifaith.run train     --datasets bace bbbp --architectures gin gat
    python -m xaifaith.run attribute --datasets bace bbbp --architectures gin gat
    python -m xaifaith.run meta
    python -m xaifaith.run all       --datasets bace bbbp --architectures gin gat

Stage 2 (attribute) is the expensive one and is checkpointed per
dataset/architecture, so it can be interrupted and resumed, and sharded across
GCP workers with --shard i --num-shards n.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from typing import List, Optional, Sequence

import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from tqdm import tqdm

from .data import DATASETS, get_split, load_dataset, subset
from .explain import CIRCULAR_EXPLAINERS, EXPLAINERS, compute_attribution
from .faithfulness import FaithfulnessConfig, compute_references, faithfulness_for_attribution
from .features import assemble_feature_row, embed_dataset, model_features
from .meta import MetaConfig, run_all_protocols
from .train import TrainConfig, load_model, train_model

DEFAULT_EXPLAINERS = [
    "saliency",
    "gradient_input",
    "integrated_gradients",
    "smoothgrad",
    "grad_cam",
    "gnn_explainer",
    "occlusion",
    "random",
]


def _device(name: str) -> torch.device:
    return torch.device(name if (name == "cpu" or torch.cuda.is_available()) else "cpu")


def _checkpoint_dir(root: str, dataset: str, architecture: str) -> str:
    return os.path.join(root, "models", dataset, architecture)


# --------------------------------------------------------------------------- #
# stage 1: train backbones
# --------------------------------------------------------------------------- #

def stage_train(args) -> None:
    for dataset_name in args.datasets:
        dataset, task_type = load_dataset(dataset_name, root=args.data_root)
        split = get_split(dataset, os.path.join(args.out, "splits", f"{dataset_name}.json"), seed=args.seed)
        print(f"[{dataset_name}] {len(dataset)} molecules "
              f"(train {len(split.train)} / val {len(split.val)} / test {len(split.test)})")
        for architecture in args.architectures:
            out_dir = _checkpoint_dir(args.out, dataset_name, architecture)
            if os.path.exists(os.path.join(out_dir, "model.pt")) and not args.overwrite:
                print(f"  {architecture}: cached")
                continue
            cfg = TrainConfig(architecture=architecture, epochs=args.epochs, seed=args.seed)
            metrics = train_model(dataset, split, task_type, cfg, out_dir, device=args.device)
            print(f"  {architecture}: val {metrics['val_score']:.3f} test {metrics['test_score']:.3f}")


# --------------------------------------------------------------------------- #
# stage 2: attributions + faithfulness labels + features
# --------------------------------------------------------------------------- #

def stage_attribute(args) -> None:
    device = _device(args.device)
    fcfg = FaithfulnessConfig(
        n_random_orders=args.random_orders,
        max_curve_points=args.curve_points,
        greedy_max_atoms=args.greedy_max_atoms,
    )
    explainers = args.explainers or DEFAULT_EXPLAINERS

    for dataset_name in args.datasets:
        dataset, task_type = load_dataset(dataset_name, root=args.data_root)
        split = get_split(dataset, os.path.join(args.out, "splits", f"{dataset_name}.json"), seed=args.seed)

        # Faithfulness is measured on held-out chemistry only. Measuring it on
        # the training set would conflate "the explanation is faithful" with
        # "the model memorised this molecule".
        eval_pool = subset(dataset, split.test)
        if args.max_molecules and len(eval_pool) > args.max_molecules:
            rng = np.random.default_rng(args.seed)
            keep = rng.choice(len(eval_pool), args.max_molecules, replace=False)
            eval_pool = [eval_pool[i] for i in sorted(keep)]
        if args.num_shards > 1:
            eval_pool = eval_pool[args.shard :: args.num_shards]

        for architecture in args.architectures:
            ckpt = _checkpoint_dir(args.out, dataset_name, architecture)
            if not os.path.exists(os.path.join(ckpt, "model.pt")):
                print(f"[{dataset_name}/{architecture}] no checkpoint, skipping")
                continue

            shard_tag = f"_shard{args.shard}" if args.num_shards > 1 else ""
            out_path = os.path.join(
                args.out, "rows", f"{dataset_name}__{architecture}{shard_tag}.parquet"
            )
            if os.path.exists(out_path) and not args.overwrite:
                print(f"[{dataset_name}/{architecture}] cached")
                continue
            os.makedirs(os.path.dirname(out_path), exist_ok=True)

            model, task_type = load_model(ckpt, device=args.device)
            train_embeddings = embed_dataset(model, subset(dataset, split.train), device)

            rows: List[dict] = []
            n_degenerate = 0
            started = time.time()
            for data in tqdm(eval_pool, desc=f"{dataset_name}/{architecture}"):
                refs = compute_references(model, data, device, fcfg)
                if refs.degenerate:
                    n_degenerate += 1
                    continue
                mfeats = model_features(model, data, device, task_type, train_embeddings)
                for explainer in explainers:
                    attribution = compute_attribution(explainer, model, data, device)
                    if not np.all(np.isfinite(attribution)):
                        continue
                    scores = faithfulness_for_attribution(model, data, attribution, refs, device, fcfg)
                    row = assemble_feature_row(
                        data.smiles, attribution, mfeats, explainer, architecture, dataset_name
                    )
                    row.update(scores)
                    row["greedy_exact"] = float(refs.greedy_exact)
                    row["auc_random"] = refs.auc_random
                    row["auc_greedy"] = refs.auc_greedy
                    row["is_circular_explainer"] = float(explainer in CIRCULAR_EXPLAINERS)
                    rows.append(row)

            df = pd.DataFrame(rows)
            if not df.empty:
                df["scaffold"] = [
                    MurckoScaffold.MurckoScaffoldSmiles(smiles=s, includeChirality=False)
                    if Chem.MolFromSmiles(s) is not None
                    else ""
                    for s in df["smiles"]
                ]
                df.to_parquet(out_path, index=False)
            print(
                f"[{dataset_name}/{architecture}] {len(df)} rows, "
                f"{n_degenerate} degenerate molecules dropped, "
                f"{time.time() - started:.0f}s -> {out_path}"
            )


# --------------------------------------------------------------------------- #
# stage 3: the meta-model
# --------------------------------------------------------------------------- #

def stage_meta(args) -> None:
    rows_dir = os.path.join(args.out, "rows")
    files = [os.path.join(rows_dir, f) for f in sorted(os.listdir(rows_dir)) if f.endswith(".parquet")]
    if not files:
        raise SystemExit(f"no attribution rows in {rows_dir}; run the attribute stage first")

    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    print(f"loaded {len(df)} rows / {df['smiles'].nunique()} molecules / "
          f"{df['dataset'].nunique()} datasets / {df['architecture'].nunique()} architectures")

    if not args.include_circular:
        before = len(df)
        df = df[df["is_circular_explainer"] < 0.5].reset_index(drop=True)
        print(f"dropped {before - len(df)} rows from circular explainers (occlusion)")

    report = run_all_protocols(df, os.path.join(args.out, "meta"), MetaConfig(seed=args.seed))
    df.to_parquet(os.path.join(args.out, "meta", "all_rows.parquet"), index=False)

    print("\n--- headline numbers -------------------------------------------")
    for record in report["by_explainer"]:
        print(f"  {record['explainer']:>22}  mean F {record['mean']:+.3f}  "
              f"beats random {record['frac_beats_random']:.1%}  n={int(record['count'])}")
    print(f"\n  scaffold CV spearman        {report['scaffold_cv']['spearman_mean']:.3f} "
          f"± {report['scaffold_cv']['spearman_std']:.3f}")
    for key, label in (
        ("leave_one_architecture_out", "leave-one-architecture-out"),
        ("leave_one_dataset_out", "leave-one-dataset-out"),
    ):
        block = report[key]
        note = ""
        if block["skipped_folds"]:
            note = f"  ({len(block['skipped_folds'])} fold(s) too small, skipped)"
        print(f"  {label:<26}  {block['spearman_mean']:.3f}{note}")

    fold = report["scaffold_cv"]["folds"][0]
    print(f"\n  gating at 50% coverage: {fold['ungated_mean']:+.3f} -> {fold['gated_mean']:+.3f} "
          f"(oracle {fold['oracle_mean']:+.3f}, captured "
          f"{fold['fraction_of_oracle_gain']:.1%} of the available gain)")
    print(f"\nfull report: {os.path.join(args.out, 'meta', 'meta_report.json')}")


# --------------------------------------------------------------------------- #

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="xaifaith")
    parser.add_argument("stage", choices=["train", "attribute", "meta", "all"])
    parser.add_argument("--datasets", nargs="+", default=["bace", "bbbp", "clintox", "tox21_nr_ar"],
                        choices=sorted(DATASETS))
    parser.add_argument("--architectures", nargs="+", default=["gin", "gat", "gcn", "attentivefp"])
    parser.add_argument("--explainers", nargs="+", default=None, choices=sorted(EXPLAINERS))
    parser.add_argument("--out", default="runs/v1")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-molecules", type=int, default=600,
                        help="cap on held-out molecules per dataset (0 = all)")
    parser.add_argument("--random-orders", type=int, default=8)
    parser.add_argument("--curve-points", type=int, default=20)
    parser.add_argument("--greedy-max-atoms", type=int, default=60)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--include-circular", action="store_true",
                        help="keep occlusion rows in the meta-model (default: drop)")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "args.json"), "w") as fh:
        json.dump(vars(args), fh, indent=2)

    if args.stage in ("train", "all"):
        stage_train(args)
    if args.stage in ("attribute", "all"):
        stage_attribute(args)
    if args.stage in ("meta", "all"):
        stage_meta(args)


if __name__ == "__main__":
    main()
