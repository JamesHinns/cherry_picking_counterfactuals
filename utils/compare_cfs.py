# Contains different methods to form comparison points between counterfactuals

from typing import Dict, Tuple, List, Any
import random
import numpy as np
from numpy.typing import NDArray

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.neighbors import KernelDensity
from scipy.spatial.distance import cdist

from sklearn.preprocessing import StandardScaler
from scipy import stats

import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf
from tensorflow.keras import layers, models

from utils.hashing import _sample_fp, canonical_fingerprint_df

# Plot instances
# Simply plot data in 2D; if >2 dims, first reduce via PCA
def plot_instances(data: pd.DataFrame) -> None:
    X = data.values
    if X.shape[1] > 2:
        pca = PCA(n_components=2)
        X_plot = pca.fit_transform(X)
        xlabel, ylabel = 'PC1', 'PC2'
    else:
        X_plot = X
        # if exactly 2 columns, use their names; if 1, plot against zeros
        xlabel = data.columns[0]
        if X_plot.shape[1] == 2:
            ylabel = data.columns[1]
        else:
            X_plot = np.column_stack([X_plot.flatten(), np.zeros_like(X_plot).flatten()])
            ylabel = ''
    plt.figure()
    plt.scatter(X_plot[:, 0], X_plot[:, 1], edgecolor='k', alpha=0.7)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title('Instance Scatter Plot')
    plt.show()

# Count changed features

def calc_change_mask(df_cfs: pd.DataFrame, x_features: pd.DataFrame, tol: float = 0.0) -> Tuple[np.ndarray, List[str]]:
    """
    Boolean mask [n_instances, n_features] where True means the feature changed.
    Assumes rows in df_cfs align with x_features.
    """
    cols  = list(x_features.columns)
    cfs   = df_cfs[cols].to_numpy()
    facts = x_features[cols].to_numpy()
    if tol <= 0:
        mask = cfs != facts
    else:
        mask = np.abs(cfs - facts) > tol
    return mask.astype(bool), cols

def _feature_change_counts(mask: np.ndarray, feature_names: List[str]) -> Dict[str, int]:
    # sum over instances -> per-feature counts
    counts = mask.sum(axis=0).astype(int).tolist()
    return {f: c for f, c in zip(feature_names, counts)}

def _jaccard_rowwise(mask_a: np.ndarray, mask_b: np.ndarray) -> np.ndarray:
    # per-instance Jaccard similarity of change sets
    inter = np.logical_and(mask_a, mask_b).sum(axis=1)
    union = np.logical_or(mask_a, mask_b).sum(axis=1)
    sim = np.divide(inter, union, out=np.ones_like(inter, dtype=float), where=union != 0)
    return sim

def _ci90_from_t(mean_diff: float, se: float, df: float, alpha: float = 0.05):
    if se == 0 or df <= 0:
        return float(mean_diff), float(mean_diff)
    tcrit = stats.t.ppf(1 - alpha, df)
    return float(mean_diff - tcrit * se), float(mean_diff + tcrit * se)

def tost_paired(
    diffs,
    baseline_vals,
    delta_frac: float = 0.05,
    delta_abs_floor: float = 1e-3,
    alpha: float = 0.05,
):
    diffs = np.asarray(diffs, dtype=float)
    if diffs.size == 0:
        return {"equivalent": False, "p1": None, "p2": None, "delta": None,
                "mean_diff": None, "n": 0, "ci90": (None, None)}

    mean_d = float(diffs.mean())
    sd_d = float(diffs.std(ddof=1)) if diffs.size > 1 else 0.0
    se = sd_d / np.sqrt(diffs.size) if sd_d > 0 else 0.0

    bsd = float(np.asarray(baseline_vals, dtype=float).std(ddof=1)) if baseline_vals is not None else 0.0
    delta = max(delta_frac * bsd, delta_abs_floor)

    if se == 0:
        equiv = abs(mean_d) <= delta
        return {"equivalent": bool(equiv), "p1": 0.0 if equiv else 1.0, "p2": 0.0 if equiv else 1.0,
                "delta": float(delta), "mean_diff": mean_d, "n": int(diffs.size),
                "ci90": (mean_d, mean_d)}

    df = diffs.size - 1
    t1 = (mean_d + delta) / se
    t2 = (mean_d - delta) / se
    p1 = 1 - stats.t.cdf(t1, df)
    p2 = stats.t.cdf(t2, df)
    lo, hi = _ci90_from_t(mean_d, se, df, alpha=alpha)
    equiv = (p1 < alpha) and (p2 < alpha)
    return {"equivalent": bool(equiv), "p1": float(p1), "p2": float(p2),
            "delta": float(delta), "mean_diff": mean_d, "n": int(diffs.size),
            "ci90": (float(lo), float(hi))}

def tost_independent(
    x, y,
    delta_frac: float = 0.05,
    delta_abs_floor: float = 1e-3,
    alpha: float = 0.05,
):
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
    nx, ny = x.size, y.size
    if nx == 0 or ny == 0:
        return {"equivalent": False, "p1": None, "p2": None, "delta": None,
                "mean_diff": None, "n": nx + ny, "ci90": (None, None)}

    mean_diff = float(x.mean() - y.mean())
    sx2, sy2 = float(x.var(ddof=1)), float(y.var(ddof=1))
    se = np.sqrt(sx2/nx + sy2/ny)
    # margin relative to baseline (y) SD + absolute floor
    delta = max(delta_frac * np.sqrt(sy2), delta_abs_floor)

    if se == 0:
        equiv = abs(mean_diff) <= delta
        return {"equivalent": bool(equiv), "p1": 0.0 if equiv else 1.0, "p2": 0.0 if equiv else 1.0,
                "delta": float(delta), "mean_diff": mean_diff, "n": nx + ny,
                "ci90": (mean_diff, mean_diff)}

    # Welch-Satterthwaite df
    df = (sx2/nx + sy2/ny)**2 / (((sx2/nx)**2)/(nx-1) + ((sy2/ny)**2)/(ny-1))
    t1 = (mean_diff + delta) / se
    t2 = (mean_diff - delta) / se
    p1 = 1 - stats.t.cdf(t1, df)
    p2 = stats.t.cdf(t2, df)
    lo, hi = _ci90_from_t(mean_diff, se, df, alpha=alpha)
    equiv = (p1 < alpha) and (p2 < alpha)
    return {"equivalent": bool(equiv), "p1": float(p1), "p2": float(p2),
            "delta": float(delta), "mean_diff": mean_diff, "n": nx + ny,
            "ci90": (float(lo), float(hi))}

# ------------------ Counterfactual Quality Metrics ------------------
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

def _heom_diag(cf_df, orig_df, num_cols, cat_cols, eps=1e-12, agg='l2', scale_mode='range'):
    # Numeric part
    Xc_num = cf_df[num_cols].to_numpy(float) if num_cols else np.empty((len(cf_df), 0), dtype=float)
    Xo_num = orig_df[num_cols].to_numpy(float) if num_cols else np.empty((len(orig_df), 0), dtype=float)

    if Xo_num.shape[1] > 0:
        if scale_mode == 'std':
            scale = Xo_num.std(axis=0, ddof=0)
        else:  # 'range' (HEOM default)
            scale = Xo_num.max(axis=0) - Xo_num.min(axis=0)
        scale = scale.astype(float)
        scale[scale < eps] = eps

        diff = np.abs(Xc_num - Xo_num) / scale
        if agg == 'l2':
            num_term = (diff ** 2).sum(axis=1)
        else:  # 'l1'
            num_term = diff.sum(axis=1)
    else:
        num_term = np.zeros(len(cf_df), dtype=float)

    # Categorical part (overlap distance: 0 if equal, 1 if different)
    if cat_cols:
        Xc_cat = cf_df[cat_cols].to_numpy()
        Xo_cat = orig_df[cat_cols].to_numpy()
        mismatches = (Xc_cat != Xo_cat)  # NaNs count as mismatches too, which aligns with HEOM's "missing -> 1"
        cat_counts = mismatches.sum(axis=1).astype(float)
        if agg == 'l2':
            # Each categorical feature contributes 0 or 1; squared is the same
            dist = np.sqrt(num_term + cat_counts)
        else:
            dist = num_term + cat_counts
    else:
        dist = np.sqrt(num_term) if agg == 'l2' else num_term

    return dist

def calc_proximity(cf_df, orig_df, num_cols=None, cat_cols=None, methods=None):
    if num_cols is None:
        num_cols = cf_df.select_dtypes(include=[np.number]).columns.tolist()
    if cat_cols is None:
        cat_cols = []
    if methods is None:
        methods = ['l1','l2','cosine','hamming']

    Xc = cf_df[num_cols].to_numpy(float)
    Xo = orig_df[num_cols].to_numpy(float)
    res = {}

    if 'l1' in methods:
        res['l1'] = cdist(Xc, Xo, 'cityblock').diagonal()
    if 'l2' in methods:
        res['l2'] = cdist(Xc, Xo, 'euclidean').diagonal()
    if 'cosine' in methods:
        res['cosine'] = cdist(Xc, Xo, 'cosine').diagonal()
    if 'mahalanobis' in methods:
        cov = np.cov(Xo, rowvar=False)
        VI  = np.linalg.pinv(cov + np.eye(cov.shape[0])*1e-6)
        res['mahalanobis'] = cdist(Xc, Xo, 'mahalanobis', VI=VI).diagonal()

    if 'hamming' in methods and cat_cols:
        Xc_cat = cf_df[cat_cols].to_numpy()
        Xo_cat = orig_df[cat_cols].to_numpy()
        res['hamming'] = (Xc_cat != Xo_cat).sum(axis=1) / len(cat_cols)

    # HEOM with Manhattan aggregation - from NICE.
    if 'heom_l1' in methods:
        res['heom_l1'] = _heom_diag(cf_df, orig_df, num_cols, cat_cols, eps=1e-12, agg='l1', scale_mode='range')

    return pd.DataFrame(res, index=cf_df.index)


def calc_sparsity(cf_df, orig_df, num_cols=None, cat_cols=None, tol=1e-8):
    if num_cols is None:
        num_cols = cf_df.select_dtypes(include=[np.number]).columns.tolist()
    if cat_cols is None:
        cat_cols = [c for c in cf_df.columns if c not in num_cols]

    if not num_cols and not cat_cols:
        raise ValueError('no features to compare')

    num_changed = (np.abs(cf_df[num_cols] - orig_df[num_cols]) > tol).sum(axis=1)
    cat_changed = ((cf_df[cat_cols] != orig_df[cat_cols]).sum(axis=1)
                   if cat_cols else 0)

    return num_changed + cat_changed

# ----- Plausibility funcs -----
def build_autoencoder(input_dim, encoding_dim=8):
    inp = layers.Input(shape=(input_dim,))
    encoded = layers.Dense(encoding_dim, activation='relu')(inp)
    decoded = layers.Dense(input_dim, activation='sigmoid')(encoded)
    ae = models.Model(inp, decoded)
    ae.compile(optimizer='adam', loss='mse')
    return ae

def train_autoencoders(
    df, label_col='Label', encoding_dim=8, epochs=50, batch_size=32, random_seed=None
):

    if random_seed is not None:
        tf.random.set_seed(random_seed)
        tf.config.experimental.enable_op_determinism()

    X = df.drop(columns=[label_col]).to_numpy()
    y = df[label_col].to_numpy()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    input_dim = X_scaled.shape[1]

    # General AE
    ae_all = build_autoencoder(input_dim, encoding_dim)
    ae_all.fit(X_scaled, X_scaled, epochs=epochs, batch_size=batch_size,
               shuffle=False, verbose=0)

    # Per-class AEs
    ae_by_class = {}
    for cls in np.unique(y):
        X_cls = X_scaled[y == cls]
        ae_cls = build_autoencoder(input_dim, encoding_dim)
        ae_cls.fit(X_cls, X_cls, epochs=epochs, batch_size=batch_size,
                   shuffle=False, verbose=0)
        ae_by_class[int(cls)] = ae_cls

    return scaler, ae_all, ae_by_class

def calc_im1_rowwise(cf_array, orig_classes, target_classes, ae_by_class):
    scores = []
    for i in range(len(cf_array)):
        ae_target = ae_by_class[target_classes[i]]
        ae_orig = ae_by_class[orig_classes[i]]

        rec_target = np.mean((cf_array[i] - ae_target.predict(cf_array[i:i+1], verbose=0))**2)
        rec_orig = np.mean((cf_array[i] - ae_orig.predict(cf_array[i:i+1], verbose=0))**2)

        scores.append(rec_target / (rec_orig + 1e-8))
    return np.array(scores)

def calc_im2_rowwise(cf_array, target_classes, ae_by_class, ae_all):
    scores = []
    for i in range(len(cf_array)):
        ae_target = ae_by_class[target_classes[i]]

        rec_targ = ae_target.predict(cf_array[i:i+1], verbose=0)
        rec_all  = ae_all.predict(cf_array[i:i+1], verbose=0)

        num = np.sum((rec_targ - rec_all)**2)
        denom = np.sum(np.abs(cf_array[i])) + 1e-8

        scores.append(num / denom)
    return np.array(scores)

def calc_plausibility(
    train_df, cf_df,
    orig_classes=None, target_classes=None,
    label_col="Label",
    metrics=("im1", "im2"),
    encoding_dim=8, epochs=50, batch_size=32,
    models=None,            # (scaler, ae_all, ae_by_class) to reuse
    random_seed=None        # Only used if models is None (training path)
):
    """
    Calculate plausibility metrics using either freshly trained AEs (default)
    or a pre-trained bundle passed via `models=(scaler, ae_all, ae_by_class)`.

    Returns:
        dict: {'im1': np.ndarray, 'im2': np.ndarray}
    """
    if isinstance(metrics, str):
        metrics = {metrics.lower()}
    else:
        metrics = {m.lower() for m in metrics}

    # Train or reuse models
    if models is None:
        scaler, ae_all, ae_by_class = train_autoencoders(
            train_df, label_col=label_col, encoding_dim=encoding_dim,
            epochs=epochs, batch_size=batch_size, random_seed=random_seed
        )
    else:
        # Expecting a tuple: (scaler, ae_all, ae_by_class)
        scaler, ae_all, ae_by_class = models

    # IMPORTANT: assumes cf_df columns align with train_df minus Label (as in your pipeline)
    cf_array = scaler.transform(cf_df.to_numpy())

    if target_classes is None and orig_classes is not None:
        uniq = np.unique(orig_classes)
        if set(uniq.tolist()) == {0, 1}:
            target_classes = 1 - np.asarray(orig_classes)

    out = {}
    if "im1" in metrics and orig_classes is not None and target_classes is not None:
        out["im1"] = calc_im1_rowwise(cf_array, orig_classes, target_classes, ae_by_class)
    if "im2" in metrics and target_classes is not None:
        out["im2"] = calc_im2_rowwise(cf_array, target_classes, ae_by_class, ae_all)
    return out

def compute_metrics_from_records(df: pd.DataFrame,
                                 records: List[Dict[str, Any]],
                                 to_explain: pd.DataFrame,
                                 exp_dir: os.path) -> pd.DataFrame:
    """Metrics without re-splitting: use each exp.train_df and the shared to_explain."""
    rows = []
    plaus_models_cache = {}
    baseline_mask = None
    baseline_feats = None
    baseline_row = None

    # Precompute x_full once (to_explain + labels)
    x_full = df.loc[to_explain.index]
    x_features = to_explain

    for rec in records:
        if not rec:  # skip failed runs
            continue
        exp = rec["exp"]
        df_cfs = pd.read_csv(rec["cf_csv"], index_col=0)

        # Instance-level metrics
        prox_arr  = np.asarray(calc_proximity(df_cfs, x_full, methods=["heom_l1"]))
        spars_arr = np.asarray(calc_sparsity(df_cfs, x_full))

        # Plausibility: train AEs on this experiment's training split (no re-split)
        ae_key = (exp.data_seed, exp.model_seed, exp.ae_seed)
        models = plaus_models_cache.get(ae_key)
        if models is None:
            models = train_autoencoders(
                exp.train_df, label_col="Label",
                encoding_dim=8, epochs=10, batch_size=32,
                random_seed=exp.ae_seed
            )
            plaus_models_cache[ae_key] = models

        plaus = calc_plausibility(
            exp.train_df, df_cfs,
            orig_classes=x_full["Label"].to_numpy(),
            models=models
        )
        im1_arr = plaus.get("im1")
        im2_arr = plaus.get("im2")

        # Change mask + counts
        mask, feat_names = calc_change_mask(df_cfs, x_features, tol=1e-6)
        counts = {f: int(c) for f, c in zip(feat_names, mask.sum(axis=0).astype(int))}

        # Baseline Jaccard reference (shared to_explain -> always comparable)
        jacc_mean = None
        if rec["label"] == "baseline":
            baseline_mask, baseline_feats = mask, feat_names
            jacc_mean = 1.0  # identical to itself
        elif baseline_mask is not None:
            base_mask = baseline_mask
            if baseline_feats != feat_names:
                order = [baseline_feats.index(f) for f in feat_names]
                base_mask = base_mask[:, order]
            jacc_mean = float(np.mean(_jaccard_rowwise(base_mask, mask)))

        row = {
            "label": rec["label"],
            "data_seed": exp.data_seed,
            "model_seed": exp.model_seed,
            "ae_seed": exp.ae_seed,
            "cf_seed": exp.cf_seed,
            "rep": 1,
            "n_instances": len(to_explain),
            "sample_fp": _sample_fp(to_explain.index.tolist()),
            "sample_index": to_explain.index.tolist(),
            "proximity": prox_arr.tolist(),
            "sparsity": spars_arr.tolist(),
            "im1": im1_arr.tolist() if im1_arr is not None else None,
            "im2": im2_arr.tolist() if im2_arr is not None else None,
            "proximity_mean": float(prox_arr.mean()),
            "sparsity_mean": float(spars_arr.mean()),
            "im1_mean": float(im1_arr.mean()) if im1_arr is not None else None,
            "im2_mean": float(im2_arr.mean()) if im2_arr is not None else None,
            "feature_names": feat_names,
            "feature_change_counts": counts,
            "jaccard_mean_to_baseline": jacc_mean,
            "cf_fingerprint": canonical_fingerprint_df(df_cfs),
            "stamp": rec["stamp"],
        }
        rows.append(row)
        if rec["label"] == "baseline":
            baseline_row = row

    out = pd.DataFrame(rows)
    os.makedirs(exp_dir, exist_ok=True)
    out.to_json(os.path.join(exp_dir, "metrics.json"), orient="records", indent=2)
    return out

#Temp density code, right now not used
# def calc_density(data: pd.DataFrame) -> NDArray[np.float64]:
#     """
#     Estimate point-wise density via a Gaussian kernel.
#     Returns an array of density values (floats) for each row of data.
#     """
#     X = data.values
#     kde = KernelDensity(bandwidth=1.0)  # you can tune bandwidth
#     kde.fit(X)
#     log_dens = kde.score_samples(X)
#     return np.exp(log_dens)

# # Use plot_instances, but also highlight the density
# def plot_density(data: pd.DataFrame) -> None:
#     dens = calc_density(data)
#     X = data.values
#     if X.shape[1] > 2:
#         pca = PCA(n_components=2)
#         X_plot = pca.fit_transform(X)
#         xlabel, ylabel = 'PC1', 'PC2'
#     else:
#         X_plot = X
#         xlabel = data.columns[0]
#         if X_plot.shape[1] == 2:
#             ylabel = data.columns[1]
#         else:
#             X_plot = np.column_stack([X_plot.flatten(), np.zeros_like(X_plot).flatten()])
#             ylabel = ''
#     plt.figure()
#     sc = plt.scatter(X_plot[:, 0], X_plot[:, 1], c=dens, cmap='viridis', edgecolor='k')
#     plt.colorbar(sc, label='Estimated Density')
#     plt.xlabel(xlabel)
#     plt.ylabel(ylabel)
#     plt.title('Density‐Colored Scatter Plot')
#     plt.show()


