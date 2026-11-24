"""Is the predictor just a molecule-size proxy?

The ablation showed every feature family reaching ~0.70 alone, including plain
RDKit descriptors. Those families share one obvious correlate: molecule size.
And the faithfulness metric is size-dependent by construction - on a small
molecule a random deletion order often hits the important atoms early, so the
random-to-greedy spread is narrow and F is compressed, while on a large molecule
the spread is wide and a good attribution scores high. So F should trend upward
with atom count, and a predictor could capture much of that by learning
"big molecule, trustworthy attribution" without any per-molecule insight.

Inside each (explainer, architecture) cell, with both identities removed:

  size_rho    Spearman of heavy-atom count against F. The raw size effect.
  size_only   a model given ONLY size-proxy features.
  no_mol      the full model with every mol_* feature removed.
  full        the full model (the headline number for this cell).
  stratified  the full model's Spearman computed WITHIN size tertiles and
              averaged. Size is near-constant inside a tertile, so this is a
              nonparametric control: whatever survives here is not size.

Read `stratified` against `full`. Holding up means size is a contributor, not
the mechanism. Collapsing toward zero means the result is mostly size.

    python size_check.py
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.model_selection import GroupKFold

from xaifaith.features import feature_columns
from xaifaith.meta import TARGET, MetaConfig, evaluate_split

# Descriptors that are essentially restatements of "how big is this molecule".
SIZE_PROXIES = ("mol_n_atoms", "mol_n_bonds", "mol_mw", "mol_bertz", "mol_n_rings")


def cv_spearman(df: pd.DataFrame, cfg: MetaConfig, columns: Sequence[str], n_folds: int = 5) -> float:
    if not columns or len(df) < 100:
        return float("nan")
    groups = df["scaffold"].astype(str).to_numpy()
    folds = min(n_folds, len(set(groups)))
    if folds < 2:
        return float("nan")
    scores = []
    for train_idx, test_idx in GroupKFold(n_splits=folds).split(df, groups=groups):
        res = evaluate_split(df.iloc[train_idx], df.iloc[test_idx], cfg, columns)
        scores.append(res["spearman"])
    return float(np.nanmean(scores))


def stratified_spearman(df: pd.DataFrame, cfg: MetaConfig, columns: Sequence[str]) -> float:
    """Full-model Spearman computed within size tertiles, then averaged.

    Size barely varies inside a tertile, so it cannot be what the model ranks on.
    """
    if "mol_n_atoms" not in df.columns:
        return float("nan")
    try:
        bands = pd.qcut(df["mol_n_atoms"], 3, labels=False, duplicates="drop")
    except ValueError:
        return float("nan")

    scores = []
    for band in sorted(pd.unique(bands.dropna())):
        sub = df[bands == band].reset_index(drop=True)
        if len(sub) < 100:
            continue
        # A band holds a third of the cell, so the leaf-size floor that suits the
        # full cell can leave the booster unable to split at all - it then
        # predicts a constant and Spearman is undefined. Relax it only when the
        # band is genuinely small; for cells of a few thousand rows this leaves
        # the config identical to the full-model fit, keeping the two comparable.
        band_cfg = MetaConfig(
            n_estimators=cfg.n_estimators,
            learning_rate=cfg.learning_rate,
            num_leaves=cfg.num_leaves,
            min_child_samples=max(5, min(cfg.min_child_samples, len(sub) // 15)),
            subsample=cfg.subsample,
            colsample=cfg.colsample,
            seed=cfg.seed,
        )
        value = cv_spearman(sub, band_cfg, columns, n_folds=3)
        if not np.isnan(value):
            scores.append(value)
    return float(np.nanmean(scores)) if scores else float("nan")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="runs/v1")
    parser.add_argument("--min-rows", type=int, default=400)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    path = os.path.join(args.out, "meta", "all_rows.parquet")
    if not os.path.exists(path):
        raise SystemExit(f"{path} not found - run `python -m xaifaith.run meta` first")
    df = pd.read_parquet(path).dropna(subset=[TARGET]).reset_index(drop=True)
    cfg = MetaConfig(seed=args.seed)
    print(f"loaded {len(df)} rows / {df['smiles'].nunique()} molecules")
    print(f"heavy-atom count: median {df['mol_n_atoms'].median():.0f}, "
          f"range {df['mol_n_atoms'].min():.0f}-{df['mol_n_atoms'].max():.0f}\n")

    print("IS IT JUST SIZE? (within each cell, both identities removed)")
    print(f"   {'explainer':>22}  {'arch':>12}  {'size_rho':>8}  {'size_only':>9}  "
          f"{'no_mol':>7}  {'full':>7}  {'stratified':>10}")

    results: List[Dict[str, object]] = []
    for explainer in sorted(df["explainer"].unique()):
        for arch in sorted(df["architecture"].unique()):
            mask = (df["explainer"] == explainer) & (df["architecture"] == arch)
            cell = df[mask]
            if len(cell) < args.min_rows:
                continue
            cell = cell.drop(columns=["explainer", "architecture"]).reset_index(drop=True)
            columns = feature_columns(cell.columns)
            size_cols = [c for c in columns if c in SIZE_PROXIES]
            no_mol = [c for c in columns if not c.startswith("mol_")]

            row = {
                "explainer": explainer,
                "architecture": arch,
                "size_rho": float(spearmanr(cell["mol_n_atoms"], cell[TARGET]).statistic),
                "size_only": cv_spearman(cell, cfg, size_cols),
                "no_mol": cv_spearman(cell, cfg, no_mol),
                "full": cv_spearman(cell, cfg, columns),
                "stratified": stratified_spearman(cell, cfg, columns),
                "n_rows": int(len(cell)),
            }
            results.append(row)
            print(f"   {explainer:>22}  {arch:>12}  {row['size_rho']:>+8.3f}  "
                  f"{row['size_only']:>9.3f}  {row['no_mol']:>7.3f}  "
                  f"{row['full']:>7.3f}  {row['stratified']:>10.3f}")

    real = [r for r in results if r["explainer"] != "random"]
    if real:
        print("\n   mean over real explainers:")
        for key in ("size_rho", "size_only", "no_mol", "full", "stratified"):
            print(f"      {key:>12}  {np.nanmean([r[key] for r in real]):+.3f}")
        retained = np.nanmean([r["stratified"] for r in real]) / max(
            np.nanmean([r["full"] for r in real]), 1e-9
        )
        print(f"\n   stratified retains {retained:.0%} of the full-model correlation.")
        print("   High: size is a contributor, not the mechanism - the claim stands.")
        print("   Low: most of the signal was size, and the claim needs rewording.")

    dest = os.path.join(args.out, "meta", "size_check.json")
    with open(dest, "w") as fh:
        json.dump(results, fh, indent=2, default=float)
    print(f"\nsaved {dest}")


if __name__ == "__main__":
    main()
