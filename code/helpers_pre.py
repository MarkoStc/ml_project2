"""
preprocessing.py

Utility functions for the TCGA BRCA multi-omics experiments:
- per-feature ANOVA (F, p, eta^2)
- PAM50 label extraction from meta
- small feature-selection blocks for ANOVA / variance-based selection
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.feature_selection import f_classif, SelectKBest
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


# -------------------------------------------------------------------
# 1. Per-feature ANOVA + eta^2
# -------------------------------------------------------------------

def per_feature_anova_np(
    X: np.ndarray,               # shape (N, P)
    y: np.ndarray,               # shape (N,), integer classes
    view_name: str,              # e.g. "mRNA", "DNAm", "RPPA"
    topN: int = 30,              # kept for backwards compatibility (not used)
    topK_scatter: int = 3000,    # kept for backwards compatibility (not used)
    save_prefix: str | None = None,
    feature_names: list[str] | None = None,
    plot: bool = True,
) -> pd.DataFrame:
    """
    Compute ANOVA F, p and eta^2 per feature for one view.

    Returns
    -------
    res : DataFrame
        Columns: ["feature", "F", "p", "eta2", "rank", "view"].
        Sorted in descending eta2.
    """
    N, P = X.shape
    K = len(np.unique(y))

    if feature_names is None:
        width = len(str(P))
        feature_names = [f"{view_name.lower()}_{i:0{width}d}" for i in range(P)]

    # 1) ANOVA F and p across classes for every feature
    F, p = f_classif(X, y)  # length P
    F = np.nan_to_num(F, nan=0.0, posinf=0.0, neginf=0.0)
    p = np.nan_to_num(p, nan=1.0, posinf=1.0, neginf=1.0)

    # 2) Effect size eta^2 (0..1)
    eta2 = ((K - 1) * F) / (((K - 1) * F) + (N - K) + 1e-12)

    # 3) Build table
    res = pd.DataFrame(
        {
            "feature": feature_names,
            "F": F,
            "p": p,
            "eta2": eta2,
        }
    ).sort_values("eta2", ascending=False).reset_index(drop=True)

    res["rank"] = np.arange(1, P + 1)
    res["view"] = view_name

    # Optional plot: histogram of eta^2
    if plot:
        plt.figure()
        plt.hist(res["eta2"].values, bins=50)
        plt.title(f"{view_name}: η² distribution (N={N}, K={K}, P={P})")
        plt.xlabel("eta²")
        plt.ylabel("count")
        plt.tight_layout()
        plt.show()

    # Optional: save table
    if save_prefix is not None:
        res.to_csv(f"{save_prefix}_{view_name}_anova.csv", index=False)

    return res


# -------------------------------------------------------------------
# 2. PAM50 label extraction
# -------------------------------------------------------------------

def get_pam50(data: dict, view_name: str) -> pd.Series:
    """
    Extract PAM50 labels from the meta table of a given view.

    Parameters
    ----------
    data : dict
        data["view_name"]["meta"] must exist.
    view_name : {"mRNA", "DNAm", "RPPA", ...}

    Returns
    -------
    s : pd.Series
        Index = patient IDs, values = PAM50 labels (str or NaN).
    """
    meta = data[view_name].get("meta", None)
    if meta is None or "paper_BRCA_Subtype_PAM50" not in meta.columns:
        return pd.Series(index=[], dtype="object")

    s = meta["paper_BRCA_Subtype_PAM50"].astype(str).str.strip()
    s = s.replace({"nan": np.nan, "None": np.nan, "": np.nan})
    s.name = view_name
    return s


# -------------------------------------------------------------------
# 3. Feature-selection score + view blocks
# -------------------------------------------------------------------

def variance_score(X: np.ndarray, y: np.ndarray | None = None):
    """
    Score function for SelectKBest: feature variance (higher = better).

    Parameters
    ----------
    X : array-like, shape (N, P)
    y : ignored (for compatibility with SelectKBest)

    Returns
    -------
    scores : np.ndarray, shape (P,)
        Variance of each feature.
    pvalues : np.ndarray, shape (P,)
        Dummy NaNs (not used by SelectKBest).
    """
    scores = np.var(X, axis=0)
    pvalues = np.full_like(scores, np.nan, dtype=float)
    return scores, pvalues


def _make_view_block(
    score_func,
    k: int | None = None,
    scale_before: bool = True,
    step_name: str = "kbest",
) -> Pipeline:
    """
    Generic helper: scaling + SelectKBest(score_func, k).
    """
    steps: list[tuple[str, object]] = []

    if scale_before:
        steps.append(("scale", StandardScaler()))

    if k is not None:
        steps.append((step_name, SelectKBest(score_func=score_func, k=k)))

    if not scale_before:
        steps.append(("scale", StandardScaler()))

    return Pipeline(steps)


def view_block_anova(k: int | None = None) -> Pipeline:
    """
    View block using ANOVA F-score (f_classif) for feature ranking.

    We scale first (standard practice with ANOVA-based scores).
    """
    return _make_view_block(
        score_func=f_classif,
        k=k,
        scale_before=True,
        step_name="anova",
    )


def view_block_mostVar(k: int | None = None) -> Pipeline:
    """
    View block selecting the k most variable features.

    We keep the raw variance for ranking, then scale afterwards.
    """
    return _make_view_block(
        score_func=variance_score,
        k=k,
        scale_before=False,
        step_name="var",
    )

def per_feature_variance_np(
    X: np.ndarray,
    view_name: str,
    plot: bool = True,
) -> pd.DataFrame:
    """
    Compute per-feature variance for one view and (optionally) plot its
    distribution.

    Returns
    -------
    res : DataFrame with columns ["feature", "var", "rank", "view"].
    """
    N, P = X.shape

    var = np.var(X, axis=0)
    res = pd.DataFrame(
        {
            "feature": [f"{view_name.lower()}_{i:0{len(str(P))}d}" for i in range(P)],
            "var": var,
        }
    ).sort_values("var", ascending=False).reset_index(drop=True)

    res["rank"] = np.arange(1, P + 1)
    res["view"] = view_name

    if plot:
        plt.figure()
        plt.hist(res["var"].values, bins=50)
        plt.title(f"{view_name}: variance distribution (N={N}, P={P})")
        plt.xlabel("variance")
        plt.ylabel("count")
        plt.tight_layout()
        plt.show()

    return res
