"""
pca_functions.py

Utility functions for multi-omics PCA / MOFA comparison.

Assumes `data` is a dict like:
    data[view_name]["expr"]: pandas DataFrame (samples x features)

Typical view_name values: "mRNA", "DNAm", "RPPA".
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.feature_selection import f_classif  # NEW


from numpy.linalg import lstsq
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score


from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, GridSearchCV, train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    balanced_accuracy_score,
    accuracy_score,
    roc_auc_score,
    ConfusionMatrixDisplay,
)

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import seaborn as sns


# ---------------------------------------------------------------------
# Basic helpers
# ---------------------------------------------------------------------

def zscore_df(df: pd.DataFrame,
              with_mean: bool = True,
              with_std: bool = True) -> pd.DataFrame:
    """
    Z-score a DataFrame column-wise using sklearn's StandardScaler.

    Parameters
    ----------
    df : DataFrame (N x P)
    with_mean : bool
    with_std : bool

    Returns
    -------
    DataFrame (N x P) z-scored per feature.
    """
    scaler = StandardScaler(with_mean=with_mean, with_std=with_std)
    vals = scaler.fit_transform(df.values)
    return pd.DataFrame(vals, index=df.index, columns=df.columns)


# ---------------------------------------------------------------------
# PCA helpers
# ---------------------------------------------------------------------

def run_pca_view(data: dict,
                 view_name: str,
                 patients=None,
                 n_components: int = 10):
    """
    Run PCA on one view (e.g. "mRNA", "DNAm", "RPPA").

    Parameters
    ----------
    data : dict
        multi-omics dict, data[view_name]["expr"] is a DataFrame.
    view_name : str
    patients : list-like or index, optional
        If given, restrict rows to these patient IDs.
    n_components : int

    Returns
    -------
    pca : sklearn.decomposition.PCA
    scores_df : DataFrame (N x K)
        Sample scores (embedding).
    load_df : DataFrame (P x K)
        Feature loadings.
    """
    X = data[view_name]["expr"]
    if patients is not None:
        X = X.loc[patients]

    X_z = zscore_df(X)

    pca = PCA(n_components=n_components, random_state=0)
    scores = pca.fit_transform(X_z.values)   # N x K
    loadings = pca.components_.T             # P x K

    pc_names = [f"PC{i+1}" for i in range(n_components)]
    scores_df = pd.DataFrame(scores, index=X.index, columns=pc_names)
    load_df = pd.DataFrame(loadings, index=X.columns, columns=pc_names)
    return pca, scores_df, load_df


def get_concat_matrix(data: dict,
                      patients,
                      block_scale: bool = True) -> pd.DataFrame:
    """
    Concatenate mRNA, DNAm, RPPA matrices for the same patients.

    Each view is z-scored per feature; if block_scale=True, each block
    is divided by sqrt(#features) so that large views don't dominate.

    Parameters
    ----------
    data : dict
    patients : list-like
    block_scale : bool

    Returns
    -------
    X_concat : DataFrame (N x sum(P_view))
    """
    X_rna = data["mRNA"]["expr"].loc[patients]
    X_meth = data["DNAm"]["expr"].loc[patients]
    X_prot = data["RPPA"]["expr"].loc[patients]

    X_rna_z = zscore_df(X_rna)
    X_meth_z = zscore_df(X_meth)
    X_prot_z = zscore_df(X_prot)

    if block_scale:
        X_rna_z = X_rna_z / np.sqrt(X_rna_z.shape[1])
        X_meth_z = X_meth_z / np.sqrt(X_meth_z.shape[1])
        X_prot_z = X_prot_z / np.sqrt(X_prot_z.shape[1])

    X_concat = pd.concat([X_rna_z, X_meth_z, X_prot_z], axis=1)
    return X_concat


# ---------------------------------------------------------------------
# Numerical comparison helpers
# ---------------------------------------------------------------------

def variance_explained_view(X_df: pd.DataFrame,
                            Z_df: pd.DataFrame) -> float:
    """
    Fraction of total variance in X (view) that can be reconstructed
    from embedding Z using linear least squares.

    Parameters
    ----------
    X_df : DataFrame (N x P)
    Z_df : DataFrame (N x K)

    Returns
    -------
    R2 : float in [0, 1]
    """
    common = X_df.index.intersection(Z_df.index)
    X = X_df.loc[common].values      # N x P
    Z = Z_df.loc[common].values      # N x K

    X_centered = X - X.mean(axis=0, keepdims=True)

    W, *_ = lstsq(Z, X_centered, rcond=None)  # K x P
    X_hat = Z @ W                             # N x P

    resid = X_centered - X_hat
    sse = np.sum(resid ** 2)
    sst = np.sum(X_centered ** 2)
    return 1.0 - sse / sst


def silhouette_in_embedding(Z_df: pd.DataFrame,
                            labels: pd.Series,
                            n_dims: int = 2) -> float:
    """
    Silhouette score of `labels` in first n_dims of embedding Z.

    Parameters
    ----------
    Z_df : DataFrame (N x K)
    labels : Series indexed by patient ID
    n_dims : int

    Returns
    -------
    float silhouette score.
    """
    labels = labels.dropna()
    common = Z_df.index.intersection(labels.index)
    Z = Z_df.loc[common].iloc[:, :n_dims].values
    y = labels.loc[common].values
    return silhouette_score(Z, y, metric="euclidean")


def corr_matrix(Z1_df: pd.DataFrame,
                Z2_df: pd.DataFrame,
                n1: int | None = None,
                n2: int | None = None) -> pd.DataFrame:
    """
    Correlation between columns of two embeddings (e.g. PCs vs MOFA).

    Returns a DataFrame (K1_used x K2_used).
    """
    common = Z1_df.index.intersection(Z2_df.index)
    A = Z1_df.loc[common]
    B = Z2_df.loc[common]

    if n1 is not None:
        A = A.iloc[:, :n1]
    if n2 is not None:
        B = B.iloc[:, :n2]

    C = np.corrcoef(A.values.T, B.values.T)
    nA = A.shape[1]
    corr_AB = C[:nA, nA:]
    return pd.DataFrame(corr_AB, index=A.columns, columns=B.columns)


def pairwise_silhouette_views(embeddings: dict,
                              labels: pd.Series,
                              n_dims: int = 2) -> pd.DataFrame:
    """
    Pairwise silhouette for every unordered label pair, for each embedding.

    Parameters
    ----------
    embeddings : dict {name: scores_df}
    labels : Series (e.g. PAM50)
    n_dims : int

    Returns
    -------
    DataFrame:
        index = embedding names
        columns = 'class1 vs class2'
    """
    from itertools import combinations

    lbl = labels.dropna()
    classes = sorted(lbl.unique())
    pairs = list(combinations(classes, 2))

    col_names = [f"{a} vs {b}" for (a, b) in pairs]
    res = pd.DataFrame(index=embeddings.keys(), columns=col_names, dtype=float)

    for emb_name, Z_df in embeddings.items():
        common = Z_df.index.intersection(lbl.index)
        Z_all = Z_df.loc[common].iloc[:, :n_dims]
        y_all = lbl.loc[common]

        for (a, b), col in zip(pairs, col_names):
            mask = y_all.isin([a, b])
            Z_pair = Z_all[mask]
            y_pair = y_all[mask]

            counts = y_pair.value_counts()
            if len(counts) < 2 or (counts < 2).any():
                res.loc[emb_name, col] = np.nan
                continue

            try:
                score = silhouette_score(Z_pair.values, y_pair.values,
                                         metric="euclidean")
            except Exception:
                score = np.nan
            res.loc[emb_name, col] = score

    return res


# ---------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------

def plot_pca_2d(scores_df: pd.DataFrame,
                labels: pd.Series | None = None,
                x_pc: int = 1,
                y_pc: int = 2,
                title: str | None = None,
                hue_name: str | None = None,
                figsize=(6, 5)):
    """
    2D scatter of two PCs/factors with optional coloring by labels.
    """
    x_col = f"PC{x_pc}"
    y_col = f"PC{y_pc}"

    if labels is not None:
        if hue_name is None:
            hue_name = labels.name if labels.name is not None else "label"
        df = scores_df.join(labels.rename(hue_name)).dropna()
    else:
        df = scores_df.copy()
        hue_name = None

    plt.figure(figsize=figsize)
    if hue_name is not None:
        sns.scatterplot(data=df, x=x_col, y=y_col, hue=hue_name,
                        s=40, alpha=0.8)
    else:
        sns.scatterplot(data=df, x=x_col, y=y_col, s=40, alpha=0.8)

    plt.xlabel(x_col)
    plt.ylabel(y_col)
    if title is not None:
        plt.title(title)
    plt.tight_layout()
    plt.show()


def plot_pca_3d(scores_df: pd.DataFrame,
                labels: pd.Series | None = None,
                pcs=(1, 2, 3),
                title: str | None = None,
                hue_name: str | None = None,
                figsize=(7, 6)):
    """
    3D scatter of three PCs/factors with optional coloring by labels.
    """
    x_pc, y_pc, z_pc = pcs
    x_col = f"PC{x_pc}"
    y_col = f"PC{y_pc}"
    z_col = f"PC{z_pc}"

    if labels is not None:
        if hue_name is None:
            hue_name = labels.name if labels.name is not None else "label"
        df = scores_df.join(labels.rename(hue_name)).dropna()
    else:
        df = scores_df.copy()
        hue_name = None

    x = df[x_col]
    y = df[y_col]
    z = df[z_col]

    if hue_name is not None:
        lab = df[hue_name]
        classes = lab.unique()
        colors = plt.cm.tab10(np.linspace(0, 1, len(classes)))
    else:
        lab = None
        classes = [None]
        colors = [plt.cm.tab10(0.0)]

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")

    if hue_name is not None:
        for c, col in zip(classes, colors):
            mask = (lab == c)
            ax.scatter(x[mask], y[mask], z[mask],
                       label=c, s=40, alpha=0.8, color=col)
        ax.legend()
    else:
        ax.scatter(x, y, z, s=40, alpha=0.8)

    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_zlabel(z_col)
    if title is not None:
        ax.set_title(title)
    plt.tight_layout()
    plt.show()


def select_top_variable_features(data: dict,
                                 n_keep: int = 2000,
                                 views: list[str] | None = None,
                                 use: str = "var"):
    """
    Select top `n_keep` most variable features per view.

    Parameters
    ----------
    data : dict
        data[view]["expr"] must be a DataFrame (samples x features).
    n_keep : int
        Maximum number of features to keep per view.
    views : list of str or None
        Which views to process. If None, use all keys in `data`.
    use : {"var", "mad"}
        Measure of variability to rank features.
        - "var": variance
        - "mad": median absolute deviation

    Returns
    -------
    filtered_data : dict
        Same structure as `data`, but expr matrices reduced to top features.
    feature_indices : dict
        For each view, the Index of selected feature names.
    """
    if views is None:
        views = list(data.keys())

    filtered_data = {}
    feature_indices = {}

    for v in views:
        X = data[v]["expr"]  # samples x features

        # compute variability per feature
        if use == "var":
            # variance across samples (ddof=1 for sample variance)
            var = X.var(axis=0, ddof=1)
        elif use == "mad":
            med = X.median(axis=0)
            var = (X - med).abs().median(axis=0)
        else:
            raise ValueError("use must be 'var' or 'mad'")

        # drop features that are all-NaN (just in case)
        var = var.dropna()

        k = min(n_keep, var.shape[0])  # in RPPA you will just keep all 464
        top_feats = var.sort_values(ascending=False).head(k).index

        feature_indices[v] = top_feats

        # copy original view entry and subset expr
        new_view = dict(data[v])               # shallow copy
        new_view["expr"] = data[v]["expr"][top_feats]
        filtered_data[v] = new_view

        print(f"{v}: kept {k} / {X.shape[1]} features")

    return filtered_data, feature_indices


def per_factor_r2_matrix(views: dict[str, pd.DataFrame],
                         Z_df: pd.DataFrame,
                         n_factors: int = 15) -> pd.DataFrame:
    """
    Compute R^2 per factor (column of Z_df) and per view.

    Parameters
    ----------
    views : dict
        {view_name: X_df} with X_df (N x P).
    Z_df : DataFrame
        Embedding (N x K), e.g. PCA scores.
    n_factors : int
        Max number of factors/PCs to use (starting from PC1).

    Returns
    -------
    DataFrame (n_factors_used x n_views)
        index  = Factor1, Factor2, ...
        columns = view names
        values = R^2 (0..1)
    """
    K = min(n_factors, Z_df.shape[1])
    factor_names = [f"Factor{i+1}" for i in range(K)]
    view_names = list(views.keys())
    R2 = pd.DataFrame(index=factor_names, columns=view_names, dtype=float)

    for vname, X_df in views.items():
        # align samples
        common = X_df.index.intersection(Z_df.index)
        X = X_df.loc[common].values             # N x P
        X_centered = X - X.mean(axis=0, keepdims=True)

        Z = Z_df.loc[common].iloc[:, :K].values # N x K

        for k in range(K):
            z_k = Z[:, [k]]                     # N x 1
            # regress X_centered on z_k
            W_k, *_ = lstsq(z_k, X_centered, rcond=None)  # (1 x P)
            X_hat_k = z_k @ W_k                              # N x P

            resid_k = X_centered - X_hat_k
            sse = np.sum(resid_k ** 2)
            sst = np.sum(X_centered ** 2)
            R2.iloc[k, R2.columns.get_loc(vname)] = 1.0 - sse / sst

    return R2


def get_pam50(data, view_name: str) -> pd.Series:
    meta = data[view_name].get("meta", None)
    if meta is None or "paper_BRCA_Subtype_PAM50" not in meta.columns:
        return pd.Series(index=[], dtype="object")
    s = meta["paper_BRCA_Subtype_PAM50"].astype(str).str.strip()
    s = s.replace({"nan": np.nan, "None": np.nan, "": np.nan})
    s.name = view_name
    return s


def select_top_anova_features(data: dict,
                              n_keep: int = 2000,
                              views: list[str] | None = None,
                              label_view: str = "mRNA"):
    """
    Select top `n_keep` features per view using one-way ANOVA F-score
    w.r.t. PAM50 labels from `label_view`.

    Assumptions
    -----------
    - `data` has the same structure as in your MOFA notebook:
          data[view]["expr"] : DataFrame (samples x features)
          data[view]["meta"] : DataFrame with PAM50 in
              'paper_BRCA_Subtype_PAM50'
    - You already have `get_pam50(data, view_name)` defined in helpers.

    Parameters
    ----------
    data : dict
        Multi-omics dict.
    n_keep : int
        Max number of features to keep per view.
    views : list[str] or None
        Views to process (e.g. ["mRNA", "DNAm", "RPPA"]).
        If None, use all keys in `data`.
    label_view : str
        View from which to take PAM50 labels (usually "mRNA").

    Returns
    -------
    filtered_data : dict
        Same structure as `data`, but expr matrices reduced to top features.
    feature_indices : dict
        For each view, the Index of selected feature names.
    """
    if views is None:
        views = list(data.keys())

    # Canonical PAM50 labels from label_view
    y_all = get_pam50(data, label_view).dropna()
    if y_all.empty:
        raise ValueError(
            f"No PAM50 labels found for view '{label_view}' "
            "in ANOVA feature selection."
        )

    filtered_data: dict = {}
    feature_indices: dict = {}

    for v in views:
        X = data[v]["expr"]  # samples x features

        # Align samples between expression and labels
        common = X.index.intersection(y_all.index)
        if len(common) == 0:
            raise ValueError(
                f"No overlapping samples between '{v}' and label view "
                f"'{label_view}'."
            )

        X_sub = X.loc[common]
        y_sub = y_all.loc[common]

        # Remove zero-variance features (ANOVA cannot handle them)
        var = X_sub.var(axis=0, ddof=1)
        good_cols = var[var > 0].index
        X_sub = X_sub[good_cols]

        # Impute any remaining NaNs with column means
        X_sub = X_sub.fillna(X_sub.mean(axis=0))

        # ANOVA F-scores per feature
        f_vals, p_vals = f_classif(X_sub.values, y_sub.values)
        scores = pd.Series(f_vals, index=good_cols)

        # Clean up any inf/NaN
        scores = scores.replace([np.inf, -np.inf], np.nan).fillna(0.0)

        k = min(n_keep, scores.shape[0])
        top_feats = scores.sort_values(ascending=False).head(k).index

        feature_indices[v] = top_feats

        new_view = dict(data[v])               # shallow copy
        new_view["expr"] = data[v]["expr"][top_feats]
        filtered_data[v] = new_view

        print(f"{v}: kept {k} / {X.shape[1]} features (ANOVA)")

    return filtered_data, feature_indices


def select_top_variable_features_per_view(
    data: dict,
    n_keep_per_view: dict,
    views: list[str] | None = None,
    use: str = "var",
):
    """
    Like `select_top_variable_features`, but lets you specify a different
    number of kept features per view.

    Parameters
    ----------
    data : dict
        data[view]["expr"] must be a DataFrame (samples x features).
    n_keep_per_view : dict
        Mapping view_name -> number of features to keep
        (e.g. {"mRNA": 2000, "DNAm": 2000, "RPPA": 464}).
    views : list of str or None
        Which views to process. If None, use keys of `n_keep_per_view`.
    use : {"var", "mad"}
        How to measure variability.

    Returns
    -------
    filtered_data : dict
        Same structure as `data`, but `expr` reduced to selected columns.
    feature_indices : dict
        view_name -> Index of selected feature names.
    """
    if views is None:
        views = list(n_keep_per_view.keys())

    filtered_data = {}
    feature_indices = {}

    for v in views:
        if v not in data:
            raise KeyError(f"View '{v}' not found in data.")
        if v not in n_keep_per_view:
            raise KeyError(f"n_keep_per_view has no entry for view '{v}'.")

        X = data[v]["expr"]

        if use == "var":
            var = X.var(axis=0, ddof=1)
        elif use == "mad":
            med = X.median(axis=0)
            var = (X - med).abs().median(axis=0)
        else:
            raise ValueError("use must be 'var' or 'mad'")

        var = var.dropna()
        k = min(int(n_keep_per_view[v]), var.shape[0])
        top_feats = var.sort_values(ascending=False).head(k).index

        feature_indices[v] = top_feats

        new_view = dict(data[v])
        new_view["expr"] = data[v]["expr"][top_feats]
        filtered_data[v] = new_view

        print(f"{v}: kept {k} / {X.shape[1]} features (most variable, {use})")

    return filtered_data, feature_indices


def select_top_anova_features_per_view(
    data: dict,
    n_keep_per_view: dict,
    views: list[str] | None = None,
    label_view: str = "mRNA",
):
    """
    Select top features per view using one-way ANOVA F-score w.r.t. PAM50 labels.

    Parameters
    ----------
    data : dict
        data[view]["expr"] must be a DataFrame (samples x features).
    n_keep_per_view : dict
        Mapping view_name -> number of features to keep.
    views : list of str or None
        Which views to process. If None, use keys of `n_keep_per_view`.
    label_view : str
        View from which to take PAM50 labels (via get_pam50).

    Returns
    -------
    filtered_data : dict
        Same structure as `data`, but `expr` reduced to selected columns.
    feature_indices : dict
        view_name -> Index of selected feature names.
    """
    if views is None:
        views = list(n_keep_per_view.keys())

    # PAM50 labels
    y_all = get_pam50(data, label_view).dropna()
    if y_all.empty:
        raise ValueError(
            f"No PAM50 labels found for view '{label_view}' in ANOVA selection."
        )

    filtered_data = {}
    feature_indices = {}

    for v in views:
        if v not in data:
            raise KeyError(f"View '{v}' not found in data.")
        if v not in n_keep_per_view:
            raise KeyError(f"n_keep_per_view has no entry for view '{v}'.")

        X = data[v]["expr"]

        # Align samples between expression and labels
        common = X.index.intersection(y_all.index)
        if len(common) == 0:
            raise ValueError(
                f"No overlapping samples between '{v}' and label view '{label_view}'."
            )

        X_sub = X.loc[common]
        y_sub = y_all.loc[common]

        # Drop zero-variance features
        var = X_sub.var(axis=0, ddof=1)
        good_cols = var[var > 0].index
        X_sub = X_sub[good_cols]

        # Impute NaNs
        X_sub = X_sub.fillna(X_sub.mean(axis=0))

        # ANOVA F-scores
        f_vals, p_vals = f_classif(X_sub.values, y_sub.values)
        scores = pd.Series(f_vals, index=good_cols)
        scores = scores.replace([np.inf, -np.inf], np.nan).fillna(0.0)

        k = min(int(n_keep_per_view[v]), scores.shape[0])
        top_feats = scores.sort_values(ascending=False).head(k).index

        feature_indices[v] = top_feats

        new_view = dict(data[v])
        new_view["expr"] = data[v]["expr"][top_feats]
        filtered_data[v] = new_view

        print(f"{v}: kept {k} / {X.shape[1]} features (ANOVA)")

    return filtered_data, feature_indices


def run_pca_view_topKvar(
    data,
    view_name,
    patients,
    K_feat,   # koliko najvariabilnijih feature-a zadržavaš
    K_pca,    # koliko PCA komponenti želiš
):
    """
    PCA na top K_feat najvariabilnijih feature-a, sa K_pca komponenti.
    """

    # 1) uzmi view + pacijente
    X_df = data[view_name].loc[patients]    # (N x P)

    # 2) top K_feat po varijansi
    var = X_df.var(axis=0)
    top_cols = var.sort_values(ascending=False).index[:K_feat]
    X_top = X_df[top_cols].values          # (N x K_feat)

    # 3) standardizacija
    X_scaled = StandardScaler().fit_transform(X_top)

    # 4) broj komponenti ne sme da pređe broj uzoraka ni broj feature-a
    n_components = min(K_pca, X_scaled.shape[0], X_scaled.shape[1])

    pca = PCA(n_components=n_components)
    scores = pca.fit_transform(X_scaled)   # N x n_components
    loadings = pca.components_.T           # K_feat x n_components

    return pca, scores, loadings, top_cols


def plot_mofa_2d(
    scores_df,
    labels=None,
    x_f=1,
    y_f=2,
    title=None,
    savepath: str | None = None
):
    x_col = f"Factor{x_f}"
    y_col = f"Factor{y_f}"

    if labels is not None:
        hue_name = labels.name or "label"
        df = scores_df.join(labels.rename(hue_name)).dropna()
    else:
        df = scores_df.copy()
        hue_name = None

    fig, ax = plt.subplots(figsize=(6, 5))

    if hue_name is not None:
        sns.scatterplot(data=df, x=x_col, y=y_col, hue=hue_name,
                        s=40, alpha=0.8, ax=ax)
    else:
        sns.scatterplot(data=df, x=x_col, y=y_col,
                        s=40, alpha=0.8, ax=ax)

    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    if title:
        ax.set_title(title)

    fig.tight_layout()

    if savepath is not None:
        fig.savefig(savepath, dpi=300, bbox_inches="tight")

    plt.show()



def plot_mofa_3d(scores_df: pd.DataFrame,
                 labels: pd.Series | None = None,
                 factors: tuple[int, int, int] = (1, 2, 3),
                 title: str | None = None,
                 figsize=(7, 6)):
    """3D scatter of three MOFA factors with optional coloring by labels."""
    f1, f2, f3 = factors
    x_col = f"Factor{f1}"
    y_col = f"Factor{f2}"
    z_col = f"Factor{f3}"

    if labels is not None:
        hue_name = labels.name or "label"
        df = scores_df.join(labels.rename(hue_name)).dropna()
    else:
        df = scores_df.copy()
        hue_name = None

    x = df[x_col]
    y = df[y_col]
    z = df[z_col]

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")

    if hue_name is not None:
        lab = df[hue_name]
        classes = lab.unique()
        colors = plt.cm.tab10(np.linspace(0, 1, len(classes)))
        for c, col in zip(classes, colors):
            mask = lab == c
            ax.scatter(x[mask], y[mask], z[mask],
                       label=c, s=40, alpha=0.8, color=col)
        ax.legend()
    else:
        ax.scatter(x, y, z, s=40, alpha=0.8)

    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_zlabel(z_col)
    if title:
        ax.set_title(title)
    plt.tight_layout()
    plt.show()


def get_pam50_any(data: dict,
                  label_views: tuple[str, ...] = ("mRNA", "DNAm", "RPPA")) -> pd.Series:
    """Combine PAM50 labels across multiple views into one Series.

    The first non-missing label across the views is kept for each sample.
    """
    combined: pd.Series | None = None
    for v in label_views:
        if v not in data:
            continue
        s = get_pam50(data, v)
        if combined is None:
            combined = s.copy()
        else:
            combined = combined.combine_first(s)

    if combined is None:
        return pd.Series(index=[], dtype="object")

    combined.name = "PAM50_any"
    return combined


def factor_class_correlation_matrix(
    Z_df: pd.DataFrame,
    labels: pd.Series,
    n_factors: int | None = None,
) -> pd.DataFrame:
    """Correlation between each factor/PC and each class indicator.

    For each class c we build a 0/1 indicator and compute the Pearson
    correlation with each factor column in Z_df.

    Returns
    -------
    DataFrame (n_factors_used x n_classes)
    """
    # Align samples
    labels = labels.dropna()
    common = Z_df.index.intersection(labels.index)
    Z = Z_df.loc[common]
    y = labels.loc[common]

    if n_factors is not None:
        Z = Z.iloc[:, :n_factors]

    classes = sorted(y.unique())
    corr = pd.DataFrame(index=Z.columns, columns=classes, dtype=float)

    for c in classes:
        m = (y == c).astype(float).values
        for factor in Z.columns:
            z = Z[factor].values
            if np.all(z == z[0]) or np.all(m == m[0]):
                r = np.nan
            else:
                r = np.corrcoef(z, m)[0, 1]
            corr.loc[factor, c] = r

    return corr


def run_logreg_on_factors(
    X_factors: pd.DataFrame,
    labels_any: pd.Series,
    title: str,
    verbose: bool = True,
):
    """Logistic regression with CV on factor embeddings.

    Parameters
    ----------
    X_factors : DataFrame (n_samples x n_factors)
        Embedding matrix (e.g. MOFA factors or PCA scores).
    labels_any : Series
        Class labels indexed by sample ID (e.g. PAM50_any).
    title : str
        Name used in printed output and confusion-matrix title.
    verbose : bool
        If True, print metrics and show the confusion matrix.

    Returns
    -------
    dict with keys:
        - title
        - best_params
        - mean_cv_bal_acc
        - std_cv_bal_acc
        - test_bal_acc
        - test_acc
        - test_roc_auc_ovr
    """
    # Align X and y by sample ID and drop missing labels
    y = labels_any.reindex(X_factors.index)
    mask = y.notna()
    X = X_factors.loc[mask]
    y = y.loc[mask]

    # Encode labels
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    # Train / test split
    Xtr, Xte, ytr, yte = train_test_split(
        X,
        y_enc,
        test_size=0.2,
        stratify=y_enc,
        random_state=42,
    )

    # Pipeline: scaler + multinomial logistic regression
    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    solver="saga",
                    multi_class="multinomial",
                    max_iter=5000,
                    class_weight="balanced",
                ),
            ),
        ]
    )

    # Hyperparameter grid and CV
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    param_grid = {
        "clf__C": [0.01, 0.1, 1.0, 10.0, 100.0],
        "clf__penalty": ["l1", "l2"],
    }

    gs = GridSearchCV(
        pipe,
        param_grid=param_grid,
        scoring="balanced_accuracy",
        cv=cv,
        n_jobs=-1,
        refit=True,
        verbose=0,
    )

    gs.fit(Xtr, ytr)

    best_params = gs.best_params_
    mean_cv = gs.best_score_
    std_cv = gs.cv_results_["std_test_score"][gs.best_index_]

    if verbose:
        print(f"{title}: best params {best_params}")
        print(f"CV balanced accuracy: {mean_cv:.3f} ± {std_cv:.3f}")

    # Test-set performance
    y_pred = gs.predict(Xte)
    proba = gs.predict_proba(Xte)

    bal_acc_test = balanced_accuracy_score(yte, y_pred)
    acc_test = accuracy_score(yte, y_pred)
    roc_auc_ovr = roc_auc_score(yte, proba, multi_class="ovr", average="weighted")

    if verbose:
        print(f"Test balanced accuracy: {bal_acc_test:.3f}")
        print(f"Test accuracy:        {acc_test:.3f}")
        print(f"Test ROC-AUC (OvR):   {roc_auc_ovr:.3f}")
        print("\nClassification report:")
        print(classification_report(yte, y_pred, target_names=le.classes_))

        cm = confusion_matrix(yte, y_pred)
        fig, ax = plt.subplots()
        disp = ConfusionMatrixDisplay(cm, display_labels=le.classes_)
        disp.plot(ax=ax, cmap="viridis", colorbar=True)
        ax.grid(False)
        #plt.title(f"Confusion matrix MOFA")
        plt.tight_layout()

        # --- save for LaTeX ---
        safe_title = title.replace(" ", "_")
        fig.savefig(f"confmat_{safe_title}.pdf", dpi=300, bbox_inches="tight")

        plt.show()


    return {
        "title": title,
        "best_params": best_params,
        "mean_cv_bal_acc": mean_cv,
        "std_cv_bal_acc": std_cv,
        "test_bal_acc": bal_acc_test,
        "test_acc": acc_test,
        "test_roc_auc_ovr": roc_auc_ovr,
    }

def run_logreg_with_pca(
    X_concat: pd.DataFrame,
    labels_any: pd.Series,
    title: str,
    n_components: int = 15,
    verbose: bool = True,
):
    """Logistic regression with PCA (no data leakage) on concatenated features.

    Steps:
    - Align X and y by sample ID and drop missing labels.
    - Train/test split.
    - Pipeline: StandardScaler -> PCA(n_components) -> multinomial logistic regression.
    - PCA and scaling are fitted *inside* CV folds (no leakage).

    Parameters
    ----------
    X_concat : DataFrame (n_samples x n_features)
        Original feature matrix (e.g. concatenated omics or raw features).
    labels_any : Series
        Class labels indexed by sample ID (e.g. PAM50_any).
    title : str
        Name used in printed output and confusion-matrix title.
    n_components : int
        Number of principal components to keep in PCA.
    verbose : bool
        If True, print metrics and show the confusion matrix.

    Returns
    -------
    dict with keys:
        - title
        - best_params
        - mean_cv_bal_acc
        - std_cv_bal_acc
        - test_bal_acc
        - test_acc
        - test_roc_auc_ovr
    """
    # Align X and y by sample ID and drop missing labels
    y = labels_any.reindex(X_concat.index)
    mask = y.notna()
    X = X_concat.loc[mask]
    y = y.loc[mask]

    # Encode labels
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    # Train / test split
    Xtr, Xte, ytr, yte = train_test_split(
        X,
        y_enc,
        test_size=0.2,
        stratify=y_enc,
        random_state=42,
    )

    # Pipeline: scaler -> PCA -> multinomial logistic regression
    # IMPORTANT: PCA is *inside* the pipeline, so in CV:
    # - scaler is fit on the train fold only
    # - PCA is fit on the train fold only
    # ==> no data leakage
    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=n_components, random_state=42)),
            (
                "clf",
                LogisticRegression(
                    solver="saga",
                    multi_class="multinomial",
                    max_iter=5000,
                    class_weight="balanced",
                ),
            ),
        ]
    )

    # Hyperparameter grid and CV (only on the classifier)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    param_grid = {
        "clf__C": [0.01, 0.1, 1.0, 10.0, 100.0],
        "clf__penalty": ["l1", "l2"],
    }

    gs = GridSearchCV(
        pipe,
        param_grid=param_grid,
        scoring="balanced_accuracy",
        cv=cv,
        n_jobs=-1,
        refit=True,
        verbose=0,
    )

    # Fit on training part: this will internally
    # - split Xtr into folds
    # - fit scaler + PCA + clf on each fold's training data
    gs.fit(Xtr, ytr)

    best_params = gs.best_params_
    mean_cv = gs.best_score_
    std_cv = gs.cv_results_["std_test_score"][gs.best_index_]

    if verbose:
        print(f"{title} (PCA {n_components}): best params {best_params}")
        print(f"CV balanced accuracy: {mean_cv:.3f} ± {std_cv:.3f}")

    # Test-set performance (using the refitted best pipeline)
    y_pred = gs.predict(Xte)
    proba = gs.predict_proba(Xte)

    bal_acc_test = balanced_accuracy_score(yte, y_pred)
    acc_test = accuracy_score(yte, y_pred)
    roc_auc_ovr = roc_auc_score(yte, proba, multi_class="ovr", average="weighted")

    if verbose:
        print(f"Test balanced accuracy: {bal_acc_test:.3f}")
        print(f"Test accuracy:        {acc_test:.3f}")
        print(f"Test ROC-AUC (OvR):   {roc_auc_ovr:.3f}")
        print("\nClassification report:")
        print(classification_report(yte, y_pred, target_names=le.classes_))

        cm = confusion_matrix(yte, y_pred)
        fig, ax = plt.subplots()
        disp = ConfusionMatrixDisplay(cm, display_labels=le.classes_)
        disp.plot(ax=ax, cmap="viridis", colorbar=True)
        ax.grid(False)
        plt.title(f"Confusion matrix – {title} (PCA {n_components})")
        plt.tight_layout()

        safe_title = title.replace(" ", "_")
        fig.savefig(f"confmat_{safe_title}_PCA{n_components}.pdf",
                    dpi=300, bbox_inches="tight")

        plt.show()

    return {
        "title": f"{title} (PCA {n_components})",
        "best_params": best_params,
        "mean_cv_bal_acc": mean_cv,
        "std_cv_bal_acc": std_cv,
        "test_bal_acc": bal_acc_test,
        "test_acc": acc_test,
        "test_roc_auc_ovr": roc_auc_ovr,
    }
