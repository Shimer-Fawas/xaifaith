"""Robustness checks for the faithfulness predictor.

Two things the main results cannot establish on their own.

A. GROUP-IDENTITY CONFOUND. within_explainer.py removed explainer identity, but
   architecture identity is still a feature there, so some signal could be
   per-architecture means rather than per-molecule structure. Here we evaluate
   inside each (explainer, architecture) cell with BOTH identities removed. Every
   row in a cell shares its method and its backbone, so anything left is
   per-molecule by construction. This is the last identity confound.

B. WHICH FEATURES ACTUALLY MATTER. LightGBM gain importance is biased toward
   high-cardinality continuous features and splits gain arbitrarily among
   correlated ones, so it cannot support a claim about feature families. Refit
   instead: once without each family (what does removing it cost?) and once with
   only that family (how far does it get alone?).

    python robustness.py
    python robustness.py --out runs/v1 --min-rows 400 --skip-cells
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from xaifaith.features import feature_columns
from xaifaith.meta import TARGET, MetaConfig, evaluate_split

# Families are prefix-matched against the feature columns actually present, so a
# renamed or added feature lands in the right family without editing this.
FAMILIES: Dict[str, Sequence[str]] = {
    "molecule": ("mol_",),
    "model_confidence": ("model_output", "model_prob", "model_margin", "model_entropy"),
    "model_sensitivity": ("model_grad_norm",),
    "model_uncertainty": ("model_mc_std",),
    "applicability_domain": ("model_knn_dist", "model_embedding_norm"),
    "attribution_shape": (
        "attr_entropy", "attr_gini", "attr_top1_share", "attr_topk_share",
        "attr_positive_frac", "attr_sign_balance", "attr_range_ratio",
    ),
    "attribution_magnitude": ("attr_mean_abs", "attr_max_abs", "attr_std"),
}


def _members(columns: Sequence[str], prefixes: Sequence[str]) -> List[str]:
    return [c for c in columns if any(c.startswith(p) for p in prefixes)]


def cv_spearman(
    df: pd.DataFrame, cfg: MetaConfig, columns: Sequence[str], n_folds: int = 5
) -> Dict[str, float]:
    """Scaffold-grouped CV restricted to `columns`."""
    if not columns or len(df) < 100:
        return {}
    groups = df["scaffold"].astype(str).to_numpy()
    folds = min(n_folds, len(set(groups)))
    if folds < 2:
        return {}

    scores, gated, ungated, captured = [], [], [], []
    for train_idx, test_idx in GroupKFold(n_splits=folds).split(df, groups=groups):
        res = evaluate_split(df.iloc[train_idx], df.iloc[test_idx], cfg, columns)
        scores.append(res["spearman"])
        gated.append(res["gated_mean"])
        ungated.append(res["ungated_mean"])
        captured.append(res["fraction_of_oracle_gain"])
    return {
        "spearman": float(np.nanmean(scores)),
        "spearman_std": float(np.nanstd(scores)),
        "ungated_mean": float(np.nanmean(ungated)),
        "gated_mean": float(np.nanmean(gated)),
        "fraction_of_oracle_gain": float(np.nanmean(captured)),
        "n_rows": int(len(df)),
        "n_features": len(columns),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="runs/v1")
    parser.add_argument("--min-rows", type=int, default=400)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--skip-cells", action="store_true", help="skip analysis A")
    parser.add_argument("--skip-ablation", action="store_true", help="skip analysis B")
    args = parser.parse_args()

    path = os.path.join(args.out, "meta", "all_rows.parquet")
    if not os.path.exists(path):
        raise SystemExit(f"{path} not found - run `python -m xaifaith.run meta` first")
    df = pd.read_parquet(path).dropna(subset=[TARGET]).reset_index(drop=True)
    cfg = MetaConfig(seed=args.seed)
    all_features = feature_columns(df.columns)
    print(f"loaded {len(df)} rows / {df['smiles'].nunique()} molecules / "
          f"{len(all_features)} features\n")

    report: Dict[str, object] = {}

    # ---- A. within (explainer, architecture) ---------------------------- #
    if not args.skip_cells:
        print("A. WITHIN (EXPLAINER, ARCHITECTURE) - both identities removed")
        print(f"   {'explainer':>22}  {'arch':>12}  {'spearman':>16}  "
              f"{'ungated':>8}  {'gated':>7}  {'rows':>6}")
        cells: List[dict] = []
        for explainer in sorted(df["explainer"].unique()):
            for arch in sorted(df["architecture"].unique()):
                sub = df[(df["explainer"] == explainer) & (df["architecture"] == arch)]
                if len(sub) < args.min_rows:
                    continue
                sub = sub.drop(columns=["explainer", "architecture"]).reset_index(drop=True)
                res = cv_spearman(sub, cfg, feature_columns(sub.columns))
                if not res:
                    continue
                res.update({"explainer": explainer, "architecture": arch})
                cells.append(res)
                print(f"   {explainer:>22}  {arch:>12}  "
                      f"{res['spearman']:>7.3f} ± {res['spearman_std']:.3f}  "
                      f"{res['ungated_mean']:>+8.3f}  {res['gated_mean']:>+7.3f}  "
                      f"{res['n_rows']:>6d}")
        report["cells"] = cells

        real = [c for c in cells if c["explainer"] != "random"]
        rand = [c for c in cells if c["explainer"] == "random"]
        if real:
            print(f"\n   mean over real explainers:  {np.nanmean([c['spearman'] for c in real]):.3f}")
        if rand:
            print(f"   mean over random control:   {np.nanmean([c['spearman'] for c in rand]):.3f}")
        print("   Real cells well above zero with the control at zero means no")
        print("   group-identity explanation survives: the signal is per-molecule.")

    # ---- B. feature-family ablation ------------------------------------- #
    if not args.skip_ablation:
        print("\nB. FEATURE FAMILY ABLATION (all explainers pooled, identities kept)")
        baseline = cv_spearman(df, cfg, all_features)
        print(f"   {'family':>24}  {'n':>4}  {'without':>16}  {'delta':>7}  {'alone':>16}")
        print(f"   {'(all features)':>24}  {len(all_features):>4}  "
              f"{baseline['spearman']:>7.3f} ± {baseline['spearman_std']:.3f}  "
              f"{'':>7}  {'':>16}")

        ablation: List[dict] = []
        for family, prefixes in FAMILIES.items():
            members = _members(all_features, prefixes)
            if not members:
                continue
            without = cv_spearman(df, cfg, [c for c in all_features if c not in members])
            alone = cv_spearman(df, cfg, members)
            row = {
                "family": family,
                "n_features": len(members),
                "without": without.get("spearman", float("nan")),
                "without_std": without.get("spearman_std", float("nan")),
                "alone": alone.get("spearman", float("nan")),
                "alone_std": alone.get("spearman_std", float("nan")),
                "delta": without.get("spearman", np.nan) - baseline["spearman"],
                "members": members,
            }
            ablation.append(row)
            print(f"   {family:>24}  {len(members):>4}  "
                  f"{row['without']:>7.3f} ± {row['without_std']:.3f}  "
                  f"{row['delta']:>+7.3f}  "
                  f"{row['alone']:>7.3f} ± {row['alone_std']:.3f}")

        report["baseline"] = baseline
        report["ablation"] = ablation
        print("\n   'without' = refit with that family removed; delta is what it cost.")
        print("   'alone'   = refit on that family only.")
        print("   A family near zero in BOTH columns is genuinely uninformative;")
        print("   a small delta with a high 'alone' just means it is redundant.")

    dest = os.path.join(args.out, "meta", "robustness.json")
    with open(dest, "w") as fh:
        json.dump(report, fh, indent=2, default=float)
    print(f"\nsaved {dest}")


if __name__ == "__main__":
    main()
