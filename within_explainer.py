"""Is the predictor learning per-molecule faithfulness, or just which explainer ran?

Explainer identity is a one-hot feature in the main meta-model, and the explainer
means span a wide range. So a high pooled Spearman is consistent with two very
different stories:

  (a) the predictor estimates faithfulness for THIS molecule         <- the claim
  (b) the predictor has learned "integrated_gradients is usually good" <- not the claim

This script separates them. Three analyses:

  1. WITHIN-EXPLAINER. Fit and evaluate separately inside each explainer, with
     explainer identity removed. Any signal here cannot come from ranking
     methods, because every row shares a method. This is the decisive test.

  2. POOLED WITHOUT IDENTITY. Fit on all rows with the explainer column dropped
     entirely. Compare to the pooled-with-identity number from run.py.

  3. GATING COMPOSITION. For the pooled model, what fraction of the kept rows at
     50% coverage is each explainer? If gating is just "keep IG", this shows it.

    python within_explainer.py                      # defaults to runs/v1
    python within_explainer.py --out runs/v1 --min-rows 500
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from xaifaith.features import feature_columns
from xaifaith.meta import TARGET, MetaConfig, evaluate_split, oracle_gap, _align_encode, _fit


def grouped_cv(df: pd.DataFrame, cfg: MetaConfig, n_folds: int = 5) -> Dict[str, float]:
    """Scaffold-grouped CV on a frame whose explainer column has been removed."""
    columns = feature_columns(df.columns)
    groups = df["scaffold"].astype(str).to_numpy()
    n_groups = len(set(groups))
    folds = min(n_folds, n_groups)
    if folds < 2 or len(df) < 100:
        return {}

    spearmans, gated, ungated, oracle, captured = [], [], [], [], []
    for train_idx, test_idx in GroupKFold(n_splits=folds).split(df, groups=groups):
        res = evaluate_split(df.iloc[train_idx], df.iloc[test_idx], cfg, columns)
        spearmans.append(res["spearman"])
        gated.append(res["gated_mean"])
        ungated.append(res["ungated_mean"])
        oracle.append(res["oracle_mean"])
        captured.append(res["fraction_of_oracle_gain"])
    return {
        "spearman": float(np.nanmean(spearmans)),
        "spearman_std": float(np.nanstd(spearmans)),
        "ungated_mean": float(np.nanmean(ungated)),
        "gated_mean": float(np.nanmean(gated)),
        "oracle_mean": float(np.nanmean(oracle)),
        "fraction_of_oracle_gain": float(np.nanmean(captured)),
        "n_rows": int(len(df)),
    }


def gating_composition(df: pd.DataFrame, cfg: MetaConfig, coverage: float = 0.5) -> pd.DataFrame:
    """Which explainers survive gating, when identity IS available to the model."""
    columns = feature_columns(df.columns)
    groups = df["scaffold"].astype(str).to_numpy()
    train_idx, test_idx = next(iter(GroupKFold(n_splits=5).split(df, groups=groups)))
    train_df, test_df = df.iloc[train_idx], df.iloc[test_idx]

    X_tr, X_te = _align_encode(train_df, test_df, columns, ["explainer", "architecture"])
    model = _fit(X_tr, train_df[TARGET].to_numpy(), cfg)
    pred = np.asarray(model.predict(X_te), dtype=np.float64)

    keep = max(1, int(round(coverage * len(test_df))))
    kept = test_df.iloc[np.argsort(-pred)[:keep]]

    before = test_df["explainer"].value_counts(normalize=True)
    after = kept["explainer"].value_counts(normalize=True)
    out = pd.DataFrame({"share_before": before, "share_after": after}).fillna(0.0)
    out["mean_F_in_pool"] = test_df.groupby("explainer")[TARGET].mean()
    return out.sort_values("share_after", ascending=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="runs/v1")
    parser.add_argument("--min-rows", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    path = os.path.join(args.out, "meta", "all_rows.parquet")
    if not os.path.exists(path):
        raise SystemExit(f"{path} not found - run `python -m xaifaith.run meta` first")
    df = pd.read_parquet(path).dropna(subset=[TARGET]).reset_index(drop=True)
    cfg = MetaConfig(seed=args.seed)
    print(f"loaded {len(df)} rows / {df['smiles'].nunique()} molecules / "
          f"{df['explainer'].nunique()} explainers\n")

    # ---- 1. within-explainer -------------------------------------------- #
    print("1. WITHIN EXPLAINER (identity removed; signal here is per-molecule)")
    print(f"   {'explainer':>22}  {'spearman':>16}  {'ungated':>8}  {'gated':>7}  "
          f"{'oracle':>7}  {'captured':>8}")
    rows: List[dict] = []
    for name in sorted(df["explainer"].unique()):
        sub = df[df["explainer"] == name].drop(columns=["explainer"]).reset_index(drop=True)
        if len(sub) < args.min_rows:
            print(f"   {name:>22}  {'too few rows':>16}")
            continue
        res = grouped_cv(sub, cfg)
        if not res:
            print(f"   {name:>22}  {'not evaluable':>16}")
            continue
        res["explainer"] = name
        rows.append(res)
        print(f"   {name:>22}  {res['spearman']:>7.3f} ± {res['spearman_std']:.3f}  "
              f"{res['ungated_mean']:>+8.3f}  {res['gated_mean']:>+7.3f}  "
              f"{res['oracle_mean']:>+7.3f}  {res['fraction_of_oracle_gain']:>7.1%}")

    if rows:
        print(f"\n   mean within-explainer spearman: "
              f"{np.nanmean([r['spearman'] for r in rows]):.3f}")
        print("   Above ~0.3 across explainers => the predictor sees per-molecule")
        print("   structure, not just which method produced the row.")

    # ---- 2. pooled without identity ------------------------------------- #
    print("\n2. POOLED, explainer identity removed")
    pooled = grouped_cv(df.drop(columns=["explainer"]).reset_index(drop=True), cfg)
    if pooled:
        print(f"   spearman {pooled['spearman']:.3f} ± {pooled['spearman_std']:.3f}   "
              f"gating {pooled['ungated_mean']:+.3f} -> {pooled['gated_mean']:+.3f} "
              f"(captured {pooled['fraction_of_oracle_gain']:.1%})")
        print("   Compare with the identity-aware number printed by `run.py meta`.")
        print("   A large drop means identity was carrying most of the signal.")

    # ---- 3. gating composition ------------------------------------------ #
    print("\n3. WHAT GATING ACTUALLY KEEPS (identity available, 50% coverage)")
    comp = gating_composition(df, cfg)
    print(f"   {'explainer':>22}  {'before':>8}  {'after':>8}  {'mean F':>8}")
    for name, row in comp.iterrows():
        print(f"   {name:>22}  {row['share_before']:>7.1%}  {row['share_after']:>7.1%}  "
              f"{row['mean_F_in_pool']:>+8.3f}")
    print("   If 'after' collapses onto one explainer, gating is method selection.")
    print("   A spread of explainers means it is selecting molecules.")

    report = {
        "within_explainer": rows,
        "pooled_without_identity": pooled,
        "gating_composition": comp.reset_index().to_dict("records"),
    }
    dest = os.path.join(args.out, "meta", "within_explainer.json")
    with open(dest, "w") as fh:
        json.dump(report, fh, indent=2, default=float)
    print(f"\nsaved {dest}")


if __name__ == "__main__":
    main()
