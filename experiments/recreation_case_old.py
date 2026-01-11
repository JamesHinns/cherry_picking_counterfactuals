import os
import numpy as np
import pandas as pd
import json
import hashlib

import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # silence TF logs below FATAL

import os, json, hashlib, inspect, random, glob
from typing import Dict, Tuple, List, Optional, Any

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import dice_ml
from dice_ml import Data, Model

from utils.compare_cfs import calc_proximity, calc_sparsity, calc_plausibility, train_autoencoders
from utils.compare_cfs import _feature_change_counts, _jaccard_rowwise, tost_paired, calc_change_mask, tost_independent
from utils.load_data import load_adult

import tensorflow as tf

# Right now, we train the autoencoders and models from scratch with the same seed
# should just reuse to save time

# Directory to save per-experiment counterfactuals
RESULTS_ROOT = "results"
EXPERIMENT_NAME = "partial_information"
EXP_DIR = os.path.join(RESULTS_ROOT, EXPERIMENT_NAME)
SEED_KEYS = {"data_seed", "model_seed", "ae_seed", "cf_seed"}

def train_random_forest(train_df, test_df=None, n_estimators=100, seed=42):
    X_train = train_df.drop(columns=["Label"])
    y_train = train_df["Label"]
    model = RandomForestClassifier(n_estimators=n_estimators, random_state=seed).fit(X_train, y_train)

    acc_train = model.score(X_train, y_train)
    if test_df is not None:
        X_test = test_df.drop(columns=["Label"])
        y_test = test_df["Label"]
        acc_test = model.score(X_test, y_test)
        print(f"Trained RF | Train Acc: {acc_train:.3f} | Test Acc: {acc_test:.3f}")
    else:
        print(f"Trained RF | Train Acc: {acc_train:.3f}")
    return model

def generate_counterfactuals(model, df, query_instances, **cf_kwargs):
    features = [c for c in df.columns if c != "Label"]
    data_interface = Data(
        dataframe=df,
        continuous_features=[c for c in features if df[c].dtype != 'object'],
        outcome_name="Label"
    )
    model_interface = Model(model=model, backend="sklearn")
    method = cf_kwargs.pop('method', 'random')
    explainer = dice_ml.Dice(data_interface, model_interface, method=method)

    cf_result = explainer.generate_counterfactuals(
            query_instances,
            total_CFs=1,
            **{k: v for k, v in cf_kwargs.items() if v is not None}
    )

    if any(example.final_cfs_df is None for example in cf_result.cf_examples_list):
        return None

    cfs = [example.final_cfs_df.copy() for example in cf_result.cf_examples_list]
    df_cfs = pd.concat(cfs, ignore_index=True)
    df_cfs.index = query_instances.index
    df_cfs = df_cfs[query_instances.columns]
    return df_cfs

def canonical_fingerprint_df(df: pd.DataFrame, float_decimals: int = 6) -> str:
    """
    Stable fingerprint: sort columns and index, round floats, hash bytes.
    """
    if df is None or len(df) == 0:
        return "EMPTY"
    # sort
    df2 = df.copy()
    df2 = df2.sort_index()
    df2 = df2.reindex(sorted(df2.columns), axis=1)
    # round floats
    for c in df2.columns:
        if pd.api.types.is_float_dtype(df2[c]):
            df2[c] = df2[c].round(float_decimals)
    # to bytes
    payload = df2.to_csv(index=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()

def _cf_kwargs_from_params(p: Dict[str, Any]) -> Dict[str, Any]:
    """Pick non-seed, non-label CF params; attach cf random seed if present."""
    out = {k: v for k, v in p.items() if k not in SEED_KEYS and k != "label" and v is not None}
    if "method" in p and p["method"] is not None:
        out["method"] = p["method"]
    if p.get("cf_seed") is not None:
        out["random_seed"] = p["cf_seed"]
    return out

def _sample_fp(index_list: List[int]) -> str:
    return hashlib.sha256(",".join(map(str, index_list)).encode("utf-8")).hexdigest()

def _set_global_seeds(seed: int | None):
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


# ---------- stage 1: generate ----------

def _generate_and_save(
    df: pd.DataFrame,
    params: Dict[str, Any],
    rep: int,
    sample_size: int,
    test_size: float,
):
    os.makedirs(EXP_DIR, exist_ok=True)

    label      = params.get("label", "param")
    data_seed  = params.get("data_seed")
    model_seed = params.get("model_seed")
    ae_seed    = params.get("ae_seed")
    cf_seed    = params.get("cf_seed")

    train_df, test_df = train_test_split(df, test_size=test_size, random_state=data_seed)
    n = min(sample_size, len(train_df))
    x_full = train_df.sample(n=n, random_state=data_seed)
    x_features = x_full.drop(columns=["Label"])

    model = train_random_forest(train_df, test_df, seed=model_seed)

    _set_global_seeds(params.get("cf_seed"))
    df_cfs = generate_counterfactuals(model, train_df, x_features, **_cf_kwargs_from_params(params))

    stamp = f"{label}__rep{rep}__d{data_seed}__m{model_seed}__ae{ae_seed}__cf{cf_seed}"
    cf_path = os.path.join(EXP_DIR, f"{stamp}.csv")
    meta_path = os.path.join(EXP_DIR, f"{stamp}.meta.json")

    df_cfs.to_csv(cf_path, index=True)
    meta = {
        "label": label,
        "params": {k: v for k, v in params.items() if k not in SEED_KEYS and k != "label" and v is not None},
        "data_seed": data_seed,
        "model_seed": model_seed,
        "ae_seed": ae_seed,
        "cf_seed": cf_seed,
        "test_size": test_size,
        "sample_size": int(n),
        "sample_index": x_full.index.tolist(),
        "sample_fp": _sample_fp(x_full.index.tolist()),
        "cf_csv": os.path.basename(cf_path),
        "cf_fingerprint": canonical_fingerprint_df(df_cfs),
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

def generate_runs(
    df: pd.DataFrame,
    runs: List[Dict[str, Any]],
    reps: int = 1,
    sample_size: int = 500,
    test_size: float = 0.30,
    offset_cf_seed_by_rep: bool = False,
):
    """
    Generate CFs for exactly the runs provided (baseline included in `runs` once).
    If reps>1 and offset_cf_seed_by_rep=True, increment cf_seed by rep-1 when present.
    """
    for p in runs:
        for rep in range(1, reps + 1):
            p_rep = dict(p)
            if offset_cf_seed_by_rep and p_rep.get("cf_seed") is not None:
                p_rep["cf_seed"] = int(p_rep["cf_seed"]) + (rep - 1)
            _generate_and_save(df, p_rep, rep, sample_size, test_size)

# ---------- stage 2: measure ----------

def compute_metrics_for_experiment(df: pd.DataFrame, experiment_dir: str = EXP_DIR) -> pd.DataFrame:
    rows = []
    # cache AEs per (data_seed, model_seed, ae_seed)
    plaus_models_cache = {}
    # keep the single baseline mask by sample_fp (handles the common case)
    baseline_masks = {}

    meta_files = sorted(glob.glob(os.path.join(experiment_dir, "*.meta.json")))
    for meta_path in meta_files:
        with open(meta_path, "r") as f:
            meta = json.load(f)

        cf_path = os.path.join(experiment_dir, meta["cf_csv"])
        df_cfs = pd.read_csv(cf_path, index_col=0)

        train_df, _ = train_test_split(df, test_size=meta["test_size"], random_state=meta["data_seed"])
        x_full = df.loc[meta["sample_index"]]
        x_features = x_full.drop(columns=["Label"])

        # metrics (instance-level)
        prox_arr  = np.asarray(calc_proximity(df_cfs, x_full, methods=["heom_l1"]))
        spars_arr = np.asarray(calc_sparsity(df_cfs, x_full))

        # reuse AE models within (data_seed, model_seed, ae_seed)
        ae_key = (meta["data_seed"], meta["model_seed"], meta["ae_seed"])
        models = plaus_models_cache.get(ae_key)
        if models is None:
            models = train_autoencoders(
                train_df, label_col="Label",
                encoding_dim=8, epochs=10, batch_size=32,
                random_seed=meta["ae_seed"]
            )
            plaus_models_cache[ae_key] = models

        plaus = calc_plausibility(
            train_df, df_cfs,
            orig_classes=x_full["Label"].to_numpy(),
            models=models
        )
        im1_arr = plaus.get("im1")
        im2_arr = plaus.get("im2")

        # change mask + counts (use small tol to avoid float speckle)
        mask, feat_names = calc_change_mask(df_cfs, x_features, tol=1e-6)
        counts = {f: int(c) for f, c in zip(feat_names, mask.sum(axis=0).astype(int))}

        # Jaccard vs the single baseline if we share the same sample
        jacc_mean = None
        if meta["label"] == "baseline":
            baseline_masks[meta["sample_fp"]] = (mask, feat_names)
        else:
            base = baseline_masks.get(meta["sample_fp"])
            if base is not None:
                base_mask, base_feats = base
                if base_feats != feat_names:
                    order = [base_feats.index(f) for f in feat_names]
                    base_mask = base_mask[:, order]
                jacc_mean = float(np.mean(_jaccard_rowwise(base_mask, mask)))

        rows.append({
            "label": meta["label"],
            "data_seed": meta["data_seed"],
            "model_seed": meta["model_seed"],
            "ae_seed": meta["ae_seed"],
            "cf_seed": meta["cf_seed"],
            "rep": int(str(os.path.basename(meta_path)).split("__rep")[1].split("__")[0]),
            "n_instances": meta["sample_size"],
            "sample_fp": meta["sample_fp"],
            "sample_index": meta["sample_index"],
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
            "jaccard_mean_to_baseline": jacc_mean,  # None if different sample
            "cf_fingerprint": meta["cf_fingerprint"],
            "stamp": os.path.splitext(os.path.basename(meta_path))[0],
        })

    df_out = pd.DataFrame(rows)
    os.makedirs(EXP_DIR, exist_ok=True)
    df_out.to_json(os.path.join(EXP_DIR, "metrics.json"), orient="records", indent=2)
    return df_out

# ---------- main / example ----------

def main():
    df = load_adult()
    features_all = [c for c in df.columns if c != "Label"]
    subset = [c for c in features_all if c not in ["sex","race","native-country","marital-status"]]

    # TODO replace these with objects
    # DICE with default params, then just inherit from it.

    cf_and_model_repeats = 20

    model_seeds = np.random.choice(range(cf_and_model_repeats))
    cf_seeds = np.random.choice(range(cf_and_model_repeats))

    runs = [
        {"data_seed":42, "model_seed":42, "ae_seed":42, "cf_seed":42, "label":"baseline"},
        {"data_seed":42, "model_seed":42, "ae_seed":42, "cf_seed":42, "label":"repeat"},
        {"data_seed":42, "model_seed":42, "ae_seed":42, "cf_seed":42,
         "features_to_vary": subset, "label":"vary_subset"},
        # {"data_seed":42, "model_seed":42, "ae_seed":42, "cf_seed":7, "label":"cf_seed_only"},
        # # Model vary
        # {"data_seed":42, "model_seed":None, "ae_seed":42, "cf_seed":42, "label":"model_seed"},
        # {"data_seed":42, "model_seed":42, "ae_seed":42, "cf_seed":None, "method":"genetic", "label":"method_genetic_no_seed"},
        # {"data_seed":42, "model_seed":42, "ae_seed":42, "cf_seed":None, "method":"genetic", "label":"method_genetic_no_seed_repeat"},
        # {"data_seed":42, "model_seed":42, "ae_seed":42, "cf_seed":None, "method":"kdtree", "label":"method_kdtree_no_seed"},
        # {"data_seed":42, "model_seed":42, "ae_seed":42, "cf_seed":None, "method":"kdtree", "label":"method_kdtree_no_seed_repeat"},
    ]
    reps = 1
    generate_runs(df, runs, reps=reps, sample_size=500, test_size=0.30)
    metrics_df = compute_metrics_for_experiment(df)
    metrics_df.to_csv(os.path.join(EXP_DIR, "metrics.csv"), index=False)

    # pick the single baseline row
    base_rows = metrics_df[metrics_df["label"] == "baseline"]
    if base_rows.empty:
        print("baseline: not found"); return
    base = base_rows.iloc[0]
    base_counts = base["feature_change_counts"]
    base_fp = base["sample_fp"]

    def pretty_feature_diff(counts):
        diffs = []
        for f in sorted(set(base_counts) | set(counts)):
            a = int(base_counts.get(f, 0)); p = int(counts.get(f, 0))
            if a != p:
                diffs.append((f, a, p))
        diffs.sort(key=lambda t: abs(t[2]-t[1]), reverse=True)
        if not diffs:
            return "Feature change totals identical to baseline", []
        return ", ".join([f"{f} {a} -> {p}" for f, a, p in diffs]), [{"feature": f, "baseline": a, "param": p} for f,a,p in diffs]

    summary = {}  # label -> dict we will save

    labels = metrics_df["label"].unique().tolist()
    for lbl in labels:
        rows_lbl = metrics_df[metrics_df["label"] == lbl]

        # Means (literal)
        prox_all  = np.concatenate([np.asarray(r) for r in rows_lbl["proximity"]])
        spars_all = np.concatenate([np.asarray(r) for r in rows_lbl["sparsity"]])
        im1_all   = np.concatenate([np.asarray(r) for r in rows_lbl["im1"] if r is not None]) if rows_lbl["im1"].notna().any() else None
        im2_all   = np.concatenate([np.asarray(r) for r in rows_lbl["im2"] if r is not None]) if rows_lbl["im2"].notna().any() else None

        print(f"{lbl}: mean proximity = {prox_all.mean():.6g}")
        print(f"{lbl}: mean sparsity  = {spars_all.mean():.6g}")
        if im1_all is not None: print(f"{lbl}: mean im1       = {im1_all.mean():.6g}")
        if im2_all is not None: print(f"{lbl}: mean im2       = {im2_all.mean():.6g}")

        entry = {
            "label": lbl,
            "means": {
                "proximity": float(prox_all.mean()),
                "sparsity": float(spars_all.mean()),
                "im1": float(im1_all.mean()) if im1_all is not None else None,
                "im2": float(im2_all.mean()) if im2_all is not None else None,
            },
            "tost": {},  # filled below
            "feature_change_diff": None,
            "jaccard_distance_mean": None,
            "paired_with_baseline": None,
        }

        if lbl != "baseline":
            # Build paired deltas per row (only when samples match)
            prox_diffs, base_prox = [], []
            spars_diffs, base_spars = [], []
            im1_diffs, base_im1 = [], []
            im2_diffs, base_im2 = [], []

            for _, r in rows_lbl.iterrows():
                if r["sample_fp"] != base_fp:
                    continue
                prox_diffs.extend(np.asarray(r["proximity"]) - np.asarray(base["proximity"]))
                base_prox.extend(base["proximity"])
                spars_diffs.extend(np.asarray(r["sparsity"]) - np.asarray(base["sparsity"]))
                base_spars.extend(base["sparsity"])
                if r["im1"] is not None and base["im1"] is not None:
                    im1_diffs.extend(np.asarray(r["im1"]) - np.asarray(base["im1"]))
                    base_im1.extend(base["im1"])
                if r["im2"] is not None and base["im2"] is not None:
                    im2_diffs.extend(np.asarray(r["im2"]) - np.asarray(base["im2"]))
                    base_im2.extend(base["im2"])

            paired = bool(base_prox)  # same sample as baseline?
            entry["paired_with_baseline"] = paired

            if paired:
                prox_t  = tost_paired(np.asarray(prox_diffs),  np.asarray(base_prox))
                spars_t = tost_paired(np.asarray(spars_diffs), np.asarray(base_spars))
                print(f"{lbl}: TOST proximity -> equivalent={prox_t['equivalent']} (Δ={prox_t['delta']:.6g}, meanΔ={prox_t['mean_diff']:.6g}, 90%CI=({prox_t['ci90'][0]:.6g},{prox_t['ci90'][1]:.6g}), n={prox_t['n']})")
                print(f"{lbl}: TOST sparsity  -> equivalent={spars_t['equivalent']} (Δ={spars_t['delta']:.6g}, meanΔ={spars_t['mean_diff']:.6g}, 90%CI=({spars_t['ci90'][0]:.6g},{spars_t['ci90'][1]:.6g}), n={spars_t['n']})")
                entry["tost"]["proximity"] = prox_t
                entry["tost"]["sparsity"]  = spars_t
                if im1_diffs:
                    im1_t = tost_paired(np.asarray(im1_diffs), np.asarray(base_im1))
                    print(f"{lbl}: TOST im1       -> equivalent={im1_t['equivalent']} (Δ={im1_t['delta']:.6g}, meanΔ={im1_t['mean_diff']:.6g}, 90%CI=({im1_t['ci90'][0]:.6g},{im1_t['ci90'][1]:.6g}), n={im1_t['n']})")
                    entry["tost"]["im1"] = im1_t
                if im2_diffs:
                    im2_t = tost_paired(np.asarray(im2_diffs), np.asarray(base_im2))
                    print(f"{lbl}: TOST im2       -> equivalent={im2_t['equivalent']} (Δ={im2_t['delta']:.6g}, meanΔ={im2_t['mean_diff']:.6g}, 90%CI=({im2_t['ci90'][0]:.6g},{im2_t['ci90'][1]:.6g}), n={im2_t['n']})")
                    entry["tost"]["im2"] = im2_t
            else:
                # independent (Welch) fallback; always give a result
                prox_t  = tost_independent(prox_all, np.asarray(base["proximity"]))
                spars_t = tost_independent(spars_all, np.asarray(base["sparsity"]))
                print(f"{lbl}: TOST proximity (indep.) -> equivalent={prox_t['equivalent']} (Δ={prox_t['delta']:.6g}, meanΔ={prox_t['mean_diff']:.6g}, 90%CI=({prox_t['ci90'][0]:.6g},{prox_t['ci90'][1]:.6g}), n={prox_t['n']})")
                print(f"{lbl}: TOST sparsity  (indep.) -> equivalent={spars_t['equivalent']} (Δ={spars_t['delta']:.6g}, meanΔ={spars_t['mean_diff']:.6g}, 90%CI=({spars_t['ci90'][0]:.6g},{spars_t['ci90'][1]:.6g}), n={spars_t['n']})")
                entry["tost"]["proximity"] = prox_t
                entry["tost"]["sparsity"]  = spars_t
                if im1_all is not None and base["im1"] is not None:
                    im1_t = tost_independent(im1_all, np.asarray(base["im1"]))
                    print(f"{lbl}: TOST im1       (indep.) -> equivalent={im1_t['equivalent']} (Δ={im1_t['delta']:.6g}, meanΔ={im1_t['mean_diff']:.6g}, 90%CI=({im1_t['ci90'][0]:.6g},{im1_t['ci90'][1]:.6g}), n={im1_t['n']})")
                    entry["tost"]["im1"] = im1_t
                if im2_all is not None and base["im2"] is not None:
                    im2_t = tost_independent(im2_all, np.asarray(base["im2"]))
                    print(f"{lbl}: TOST im2       (indep.) -> equivalent={im2_t['equivalent']} (Δ={im2_t['delta']:.6g}, meanΔ={im2_t['mean_diff']:.6g}, 90%CI=({im2_t['ci90'][0]:.6g},{im2_t['ci90'][1]:.6g}), n={im2_t['n']})")
                    entry["tost"]["im2"] = im2_t

            # Feature-change totals vs baseline
            agg_counts = {}
            for cdict in rows_lbl["feature_change_counts"]:
                for f, c in cdict.items():
                    agg_counts[f] = agg_counts.get(f, 0) + int(c)
            pretty_txt, diffs_struct = pretty_feature_diff(agg_counts)
            print(f"{lbl}: {pretty_txt}")
            entry["feature_change_diff"] = diffs_struct if diffs_struct else "identical"

            # Jaccard vs baseline (mean over rows with same sample)
            jacc_vals = [float(v) for v in rows_lbl["jaccard_mean_to_baseline"] if v is not None and np.isfinite(v)]
            if jacc_vals:
                jd = 1.0 - float(np.mean(jacc_vals))
                print(f"{lbl}: Jaccard distance vs baseline = {jd:.6g}")
                entry["jaccard_distance_mean"] = jd
            else:
                print(f"{lbl}: Jaccard distance vs baseline = N/A (different sample)")
                entry["jaccard_distance_mean"] = None

        summary[lbl] = entry

    # Baseline means at the end
    print(f"baseline: mean proximity = {np.mean(base['proximity']):.6g}")
    print(f"baseline: mean sparsity  = {np.mean(base['sparsity']):.6g}")
    if base['im1'] is not None: print(f"baseline: mean im1       = {np.mean(base['im1']):.6g}")
    if base['im2'] is not None: print(f"baseline: mean im2       = {np.mean(base['im2']):.6g}")

    # Save the structured summary (everything we printed)
    with open(os.path.join(EXP_DIR, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # Also save a tiny CSV with just the means
    import csv
    means_rows = []
    for lbl, ent in summary.items():
        m = ent["means"]
        means_rows.append({
            "label": lbl,
            "mean_proximity": m["proximity"],
            "mean_sparsity":  m["sparsity"],
            "mean_im1":       m["im1"],
            "mean_im2":       m["im2"],
        })
    with open(os.path.join(EXP_DIR, "summary_means.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(means_rows[0].keys()))
        writer.writeheader()
        writer.writerows(means_rows)

if __name__ == "__main__":
    main()
