"""The faithfulness predictor and the gating evaluation.

Three evaluation protocols, in increasing order of how much they would convince
a reviewer:

  random      - grouped by scaffold, same datasets and architectures in train/test
  loao        - leave-one-architecture-out: can it predict faithfulness for a
                backbone it has never seen?
  lodo        - leave-one-dataset-out: can it transfer to new chemistry?

And the claim that matters: gating. If you keep only the explanations the model
says are faithful, does mean realised faithfulness rise? That is the risk-coverage
curve, and it is the thing a practitioner would actually use.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, roc_auc_score

from .features import feature_columns

try:
    import lightgbm as lgb

    HAVE_LIGHTGBM = True
except ImportError:  # pragma: no cover
    from sklearn.ensemble import HistGradientBoostingRegressor

    HAVE_LIGHTGBM = False

TARGET = "faithfulness"


@dataclass
class MetaConfig:
    n_estimators: int = 600
    learning_rate: float = 0.03
    num_leaves: int = 31
    min_child_samples: int = 40
    subsample: float = 0.8
    colsample: float = 0.8
    seed: int = 0


def _fit(X: np.ndarray, y: np.ndarray, cfg: MetaConfig):
    if HAVE_LIGHTGBM:
        model = lgb.LGBMRegressor(
            n_estimators=cfg.n_estimators,
            learning_rate=cfg.learning_rate,
            num_leaves=cfg.num_leaves,
            min_child_samples=cfg.min_child_samples,
            subsample=cfg.subsample,
            subsample_freq=1,
            colsample_bytree=cfg.colsample,
            random_state=cfg.seed,
            verbose=-1,
        )
    else:  # pragma: no cover
        model = HistGradientBoostingRegressor(
            max_iter=cfg.n_estimators, learning_rate=cfg.learning_rate, random_state=cfg.seed
        )
    model.fit(X, y)
    return model


def _encode(df: pd.DataFrame, columns: Sequence[str], categorical: Sequence[str]) -> np.ndarray:
    blocks = [df[list(columns)].to_numpy(dtype=np.float64)]
    for col in categorical:
        dummies = pd.get_dummies(df[col], prefix=col)
        blocks.append(dummies.to_numpy(dtype=np.float64))
    return np.nan_to_num(np.concatenate(blocks, axis=1), nan=0.0, posinf=0.0, neginf=0.0)


def _align_encode(
    train: pd.DataFrame, test: pd.DataFrame, columns: Sequence[str], categorical: Sequence[str]
) -> Tuple[np.ndarray, np.ndarray]:
    combined = pd.concat([train.assign(_split="tr"), test.assign(_split="te")], ignore_index=True)
    encoded = _encode(combined, columns, categorical)
    is_train = (combined["_split"] == "tr").to_numpy()
    return encoded[is_train], encoded[~is_train]


def risk_coverage(y_true: np.ndarray, y_pred: np.ndarray, n_points: int = 20) -> pd.DataFrame:
    """Mean realised faithfulness as a function of how much we keep.

    Sort by predicted faithfulness, keep the top fraction, report what you
    actually got. A useful predictor makes this curve rise as coverage falls.
    """
    order = np.argsort(-y_pred)
    y_sorted = y_true[order]
    rows = []
    for coverage in np.linspace(1.0 / n_points, 1.0, n_points):
        keep = max(1, int(round(coverage * len(y_sorted))))
        rows.append(
            {
                "coverage": float(coverage),
                "mean_faithfulness": float(y_sorted[:keep].mean()),
                "frac_beating_random": float((y_sorted[:keep] > 0).mean()),
                "n": keep,
            }
        )
    return pd.DataFrame(rows)


def oracle_gap(y_true: np.ndarray, y_pred: np.ndarray, coverage: float = 0.5) -> Dict[str, float]:
    """How much of the achievable gating gain the predictor captures."""
    keep = max(1, int(round(coverage * len(y_true))))
    actual = float(np.sort(y_true)[::-1][:keep].mean())
    realised = float(y_true[np.argsort(-y_pred)][:keep].mean())
    baseline = float(y_true.mean())
    denom = actual - baseline
    return {
        "gated_mean": realised,
        "ungated_mean": baseline,
        "oracle_mean": actual,
        "fraction_of_oracle_gain": float((realised - baseline) / denom) if abs(denom) > 1e-9 else float("nan"),
    }


def evaluate_split(
    train_df: pd.DataFrame, test_df: pd.DataFrame, cfg: MetaConfig, columns: Sequence[str]
) -> Dict[str, object]:
    categorical = [c for c in ("explainer", "architecture") if c in train_df.columns]
    X_tr, X_te = _align_encode(train_df, test_df, columns, categorical)
    y_tr = train_df[TARGET].to_numpy()
    y_te = test_df[TARGET].to_numpy()

    model = _fit(X_tr, y_tr, cfg)
    pred = np.asarray(model.predict(X_te), dtype=np.float64)

    rho = spearmanr(y_te, pred).statistic if len(np.unique(y_te)) > 1 else float("nan")
    try:
        auroc = roc_auc_score((y_te > 0).astype(int), pred)
    except ValueError:
        auroc = float("nan")

    result: Dict[str, object] = {
        "n_train": int(len(train_df)),
        "n_test": int(len(test_df)),
        "spearman": float(rho),
        "mae": float(mean_absolute_error(y_te, pred)),
        "auroc_beats_random": float(auroc),
        "risk_coverage": risk_coverage(y_te, pred).to_dict("records"),
    }
    result.update(oracle_gap(y_te, pred))

    if HAVE_LIGHTGBM:
        importance = model.feature_importances_
        names = list(columns) + [
            f"{c}={v}" for c in categorical for v in sorted(pd.concat([train_df[c], test_df[c]]).unique())
        ]
        pairs = sorted(zip(names[: len(importance)], importance), key=lambda t: -t[1])[:25]
        result["top_features"] = [{"feature": n, "gain": float(g)} for n, g in pairs]
    return result


def scaffold_group_cv(
    df: pd.DataFrame, cfg: MetaConfig, n_folds: int = 5, group_col: str = "scaffold"
) -> Dict[str, object]:
    from sklearn.model_selection import GroupKFold

    columns = feature_columns(df.columns)
    groups = df[group_col].astype(str).to_numpy()
    folds = GroupKFold(n_splits=n_folds)
    per_fold = []
    for train_idx, test_idx in folds.split(df, groups=groups):
        per_fold.append(evaluate_split(df.iloc[train_idx], df.iloc[test_idx], cfg, columns))
    return {
        "protocol": "scaffold_group_cv",
        "folds": per_fold,
        "spearman_mean": float(np.nanmean([f["spearman"] for f in per_fold])),
        "spearman_std": float(np.nanstd([f["spearman"] for f in per_fold])),
    }


def leave_one_out(df: pd.DataFrame, cfg: MetaConfig, column: str) -> Dict[str, object]:
    columns = feature_columns(df.columns)
    results, skipped = [], []
    for held in sorted(df[column].unique()):
        train_df = df[df[column] != held]
        test_df = df[df[column] == held]
        if len(test_df) < 50 or len(train_df) < 200:
            skipped.append({"held_out": held, "n_train": int(len(train_df)), "n_test": int(len(test_df))})
            continue
        res = evaluate_split(train_df, test_df, cfg, columns)
        res["held_out"] = held
        results.append(res)
    return {
        "protocol": f"leave_one_{column}_out",
        "folds": results,
        # folds too small to be meaningful are skipped, not silently averaged in
        "skipped_folds": skipped,
        "spearman_mean": float(np.nanmean([r["spearman"] for r in results])) if results else float("nan"),
    }


def run_all_protocols(df: pd.DataFrame, out_dir: str, cfg: Optional[MetaConfig] = None) -> Dict[str, object]:
    cfg = cfg or MetaConfig()
    df = df.dropna(subset=[TARGET]).reset_index(drop=True)

    report: Dict[str, object] = {
        "n_rows": int(len(df)),
        "n_molecules": int(df["smiles"].nunique()),
        "by_explainer": df.groupby("explainer")[TARGET]
        .agg(["mean", "std", "count", lambda s: float((s > 0).mean())])
        .rename(columns={"<lambda_0>": "frac_beats_random"})
        .reset_index()
        .to_dict("records"),
        "scaffold_cv": scaffold_group_cv(df, cfg),
        "leave_one_architecture_out": leave_one_out(df, cfg, "architecture"),
        "leave_one_dataset_out": leave_one_out(df, cfg, "dataset"),
    }

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "meta_report.json"), "w") as fh:
        json.dump(report, fh, indent=2, default=float)
    return report
