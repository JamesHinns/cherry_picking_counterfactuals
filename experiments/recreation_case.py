# recreation_case.py
import os, json, glob, hashlib
from typing import List, Dict, Any
import numpy as np
import pandas as pd

from utils.counterfactuals import Recreation_Experiment
from utils.compare_cfs import (
    calc_proximity, calc_sparsity, calc_plausibility, train_autoencoders,
    _jaccard_rowwise, tost_paired, calc_change_mask, tost_independent
)
from utils.load_data import load_adult

RESULTS_ROOT = "results"
EXPERIMENT_NAME = "partial_information"
EXP_DIR = os.path.join(RESULTS_ROOT, EXPERIMENT_NAME)

def canonical_fingerprint_df(df: pd.DataFrame, float_decimals: int = 6) -> str:
    if df is None or len(df) == 0:
        return "EMPTY"
    df2 = df.copy().sort_index()
    df2 = df2.reindex(sorted(df2.columns), axis=1)
    for c in df2.columns:
        if pd.api.types.is_float_dtype(df2[c]):
            df2[c] = df2[c].round(float_decimals)
    return hashlib.sha256(df2.to_csv(index=True).encode("utf-8")).hexdigest()

def _sample_fp(index_list: List[int]) -> str:
    return hashlib.sha256(",".join(map(str, index_list)).encode("utf-8")).hexdigest()

def run_and_save(exp: Recreation_Experiment, to_explain: pd.DataFrame, rep: int = 1) -> Dict[str, Any]:
    """Generate CFs for provided to_explain (shared across experiments), save, and return a record for metrics."""
    os.makedirs(EXP_DIR, exist_ok=True)

    df_cfs = exp.generate_counterfactuals(to_explain)
    if df_cfs is None:
        print(f"[warn] {exp.label}: CF generation returned None; skipping save.")
        return {}

    stamp = f"{exp.label}__rep{rep}__d{exp.data_seed}__m{exp.model_seed}__ae{exp.ae_seed}__cf{exp.cf_seed}"
    cf_path = os.path.join(EXP_DIR, f"{stamp}.csv")
    meta_path = os.path.join(EXP_DIR, f"{stamp}.meta.json")

    df_cfs.to_csv(cf_path, index=True)
    meta = {
        "label": exp.label,
        "data_seed": exp.data_seed,
        "model_seed": exp.model_seed,
        "ae_seed": exp.ae_seed,
        "cf_seed": exp.cf_seed,
        "sample_size": int(len(to_explain)),
        "sample_index": to_explain.index.tolist(),
        "sample_fp": _sample_fp(to_explain.index.tolist()),
        "cf_csv": os.path.basename(cf_path),
        "cf_fingerprint": canonical_fingerprint_df(df_cfs),
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    return {"stamp": stamp, "cf_csv": cf_path, "label": exp.label, "exp": exp}

def compute_metrics_from_records(df: pd.DataFrame,
                                 records: List[Dict[str, Any]],
                                 to_explain: pd.DataFrame) -> pd.DataFrame:
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
    os.makedirs(EXP_DIR, exist_ok=True)
    out.to_json(os.path.join(EXP_DIR, "metrics.json"), orient="records", indent=2)
    return out

def main():
    df = load_adult()

    # Select a deterministic, shared set of rows to explain (independent of any seeds)
    sample_size = min(500, len(df))
    idx = df.index.sort_values()[:sample_size]
    to_explain = df.loc[idx].drop(columns=["Label"])  # <<< consistent name

    features_all = [c for c in df.columns if c != "Label"]
    subset = [c for c in features_all if c not in ["sex","race","native-country","marital-status"]]

    # Define experiments; any seeds that aren't passed are set to 42

    cont_features = ['age','fnlwgt','education-num','capital-gain','capital-loss','hours-per-week']
    disc_features = [c for c in df.columns if c not in cont_features + ['target']]
    exps = [
        # baselines
        Recreation_Experiment(data_df=df, cont_features=cont_features, disc_features=disc_features,
                              label="baseline"),
        Recreation_Experiment(data_df=df, cont_features=cont_features, disc_features=disc_features,
                              label="baseline_repeat"),
        # Cherry Picking Methods
        Recreation_Experiment(data_df=df, cont_features=cont_features, disc_features=disc_features,
                              label="vary_subset", features_to_vary=subset),
        Recreation_Experiment(data_df=df, cont_features=cont_features, disc_features=disc_features,
                              label="vary_subset_repeat", features_to_vary=subset),
        # Method variations
        Recreation_Experiment(data_df=df, cont_features=cont_features, disc_features=disc_features,
                              label="kdtree_1", method="kdtree"),
        Recreation_Experiment(data_df=df, cont_features=cont_features, disc_features=disc_features,
                              label="kdtree_2", method="kdtree"),
        Recreation_Experiment(data_df=df, cont_features=cont_features, disc_features=disc_features,
                              label="genetic_1", method="genetic"),
        Recreation_Experiment(data_df=df, cont_features=cont_features, disc_features=disc_features,
                              label="genetic_2", method="genetic")
    ]

    # Try many different random inits of models and cf gens
    rng = np.random.default_rng()
    model_seeds = rng.choice(100, size=10, replace=False)
    cf_seeds    = rng.choice(100, size=10, replace=False)

    random_model_runs = [
        Recreation_Experiment(
            data_df=df,
            cont_features=cont_features,
            disc_features=disc_features,
            label=f"random_model_run_{i}",
            model_seed=int(s)
        )
        for i, s in enumerate(model_seeds, start=1)
    ]
    exps += random_model_runs

    random_cf_runs = [
        Recreation_Experiment(
            data_df=df,
            cont_features=cont_features,
            disc_features=disc_features,
            label=f"random_cf_run_{i}",
            cf_seed=int(s)
        )
        for i, s in enumerate(cf_seeds, start=1)
    ]
    exps += random_cf_runs

    # Run + save, collect records for metrics
    records = [run_and_save(exp, to_explain=to_explain, rep=1) for exp in exps]

    # Metrics computed using exp.train_df (no re-split, no test_size parameter)
    metrics_df = compute_metrics_from_records(df, records, to_explain)
    metrics_df.to_csv(os.path.join(EXP_DIR, "metrics.csv"), index=False)

    # --- summaries (unchanged logic, paired against baseline since to_explain is shared) ---
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
            if a != p: diffs.append((f, a, p))
        diffs.sort(key=lambda t: abs(t[2]-t[1]), reverse=True)
        if not diffs: return "Feature change totals identical to baseline", []
        return ", ".join([f"{f} {a} -> {p}" for f,a,p in diffs]), \
               [{"feature": f, "baseline": a, "param": p} for f,a,p in diffs]
    
    def _print_metrics_with_tost(lbl: str, entry: dict) -> None:
        means = entry.get("means", {})
        t = entry.get("tost", {})
        mode = "paired" if entry.get("paired_with_baseline") else "indep."

        def line(metric_key: str, pretty: str):
            val = means.get(metric_key)
            if val is None:
                return None
            v = t.get(metric_key)
            if not v:
                # baseline (no TOST vs itself) or metric not computed
                return f"{lbl}: {pretty}={val:.6g}"
            eq = "True" if v.get("equivalent") else "False"
            md = v.get("mean_diff")
            ci = v.get("ci90")
            n  = v.get("n")
            return (f"{lbl}: {pretty}={val:.6g} | TOST {mode}: eq={eq}, "
                    f"meanΔ={md:.6g}, 90%CI=({ci[0]:.6g},{ci[1]:.6g}), n={n}")

        for k, name in (("proximity","proximity"),
                        ("sparsity","sparsity"),
                        ("im1","im1"),
                        ("im2","im2")):
            ln = line(k, name)
            if ln: print(ln)

    summary: Dict[str, Any] = {}
    for lbl in metrics_df["label"].unique().tolist():
        rows_lbl = metrics_df[metrics_df["label"] == lbl]
        prox_all  = np.concatenate([np.asarray(r) for r in rows_lbl["proximity"]])
        spars_all = np.concatenate([np.asarray(r) for r in rows_lbl["sparsity"]])
        im1_all   = np.concatenate([np.asarray(r) for r in rows_lbl["im1"] if r is not None]) if rows_lbl["im1"].notna().any() else None
        im2_all   = np.concatenate([np.asarray(r) for r in rows_lbl["im2"] if r is not None]) if rows_lbl["im2"].notna().any() else None

        entry = {
            "label": lbl,
            "means": {
                "proximity": float(prox_all.mean()),
                "sparsity": float(spars_all.mean()),
                "im1": float(im1_all.mean()) if im1_all is not None else None,
                "im2": float(im2_all.mean()) if im2_all is not None else None,
            },
            "tost": {},
            "feature_change_diff": None,
            "jaccard_distance_mean": None,
            "paired_with_baseline": None,
        }

        if lbl != "baseline":
            prox_diffs, base_prox = [], []
            spars_diffs, base_spars = [], []
            im1_diffs, base_im1 = [], []
            im2_diffs, base_im2 = [], []

            for _, r in rows_lbl.iterrows():
                if r["sample_fp"] != base_fp:  # should match since to_explain is shared
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

            entry["paired_with_baseline"] = bool(base_prox)
            if entry["paired_with_baseline"]:
                entry["tost"]["proximity"] = tost_paired(np.asarray(prox_diffs), np.asarray(base_prox))
                entry["tost"]["sparsity"]  = tost_paired(np.asarray(spars_diffs), np.asarray(base_spars))
                if im1_diffs: entry["tost"]["im1"] = tost_paired(np.asarray(im1_diffs), np.asarray(base_im1))
                if im2_diffs: entry["tost"]["im2"] = tost_paired(np.asarray(im2_diffs), np.asarray(base_im2))
            else:
                entry["tost"]["proximity"] = tost_independent(prox_all, np.asarray(base["proximity"]))
                entry["tost"]["sparsity"]  = tost_independent(spars_all, np.asarray(base["sparsity"]))
                if im1_all is not None and base["im1"] is not None:
                    entry["tost"]["im1"] = tost_independent(im1_all, np.asarray(base["im1"]))
                if im2_all is not None and base["im2"] is not None:
                    entry["tost"]["im2"] = tost_independent(im2_all, np.asarray(base["im2"]))

            # Feature-change totals vs baseline
            agg_counts: Dict[str, int] = {}
            for cdict in rows_lbl["feature_change_counts"]:
                for f, c in cdict.items():
                    agg_counts[f] = agg_counts.get(f, 0) + int(c)
            pretty_txt, diffs_struct = pretty_feature_diff(agg_counts)
            print(f"{lbl}: {pretty_txt}")
            entry["feature_change_diff"] = diffs_struct if diffs_struct else "identical"

            # Print means and tost for this run unless features are identical to baseline
            if entry["feature_change_diff"] != "identical":
                _print_metrics_with_tost(lbl, entry)

            # Jaccard distance (mean over rows)
            jacc_vals = [float(v) for v in rows_lbl["jaccard_mean_to_baseline"] if v is not None and np.isfinite(v)]
            entry["jaccard_distance_mean"] = (1.0 - float(np.mean(jacc_vals))) if jacc_vals else None

        summary[lbl] = entry

    print(f"baseline: mean proximity = {np.mean(base['proximity']):.6g}")
    print(f"baseline: mean sparsity  = {np.mean(base['sparsity']):.6g}")
    if base['im1'] is not None: print(f"baseline: mean im1 = {np.mean(base['im1']):.6g}")
    if base['im2'] is not None: print(f"baseline: mean im2 = {np.mean(base['im2']):.6g}")

    with open(os.path.join(EXP_DIR, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # save a tiny CSV of means
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
