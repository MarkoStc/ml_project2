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

from numpy.linalg import lstsq
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

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
    
    
import numpy as np
import pandas as pd
from numpy.linalg import lstsq
import seaborn as sns
import matplotlib.pyplot as plt

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

