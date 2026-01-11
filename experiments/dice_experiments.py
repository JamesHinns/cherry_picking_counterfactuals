"""
Orchestrates counterfactual experiments on the Adult dataset.
- Builds experiment grid in-place (dataset-specific)
- Runs CF generation and saves outputs (via utils.runner)
- Computes metrics (utils.compared_cfs.compute_metrics_from_records)
- Writes metrics.csv and summaries (utils.summary.summarise_against_baseline)

This file intentionally embeds the small config constants that used to live in config.py.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import List
import numpy as np
import pandas as pd
import re

from utils.load_data import load_adult
from utils.counterfactuals import Recreation_Experiment
from utils.runner import run_and_save
from utils.compare_cfs import compute_metrics_from_records
from utils.summary import summarise_against_baseline
from utils.hashing import _sample_fp as sample_fingerprint

# -----------------------------------------------------------------------------
# Embedded config (formerly config.py)
# -----------------------------------------------------------------------------
RESULTS_ROOT = "results"
EXPERIMENT_NAME = "test"
EXP_DIR = Path(RESULTS_ROOT) / EXPERIMENT_NAME

def main() -> None:
    # ---------------------------------------------------------------------
    # 1) Load dataset
    # ---------------------------------------------------------------------
    df = load_adult()

    # ---------------------------------------------------------------------
    # 2) Build a deterministic, shared evaluation subset (dataset‑specific)
    #    Keep features-only table for CF generation
    # ---------------------------------------------------------------------
    sample_size = min(500, len(df))
    idx = df.index.sort_values()[:sample_size]
    to_explain = df.loc[idx].drop(columns=["Label"])  # features-only

    # ---------------------------------------------------------------------
    # 3) Feature groups (dataset‑specific)
    # ---------------------------------------------------------------------
    features_all: List[str] = [c for c in df.columns if c != "Label"]
    subset: List[str] = [
        c for c in features_all
        if c not in ["sex", "race", "native-country", "marital-status"]
    ]

    # Continuous / discrete split (follow original logic)
    cont_features = [
        "age", "fnlwgt", "education-num", "capital-gain", "capital-loss", "hours-per-week"
    ]
    # Everything not continuous and not the 'target' helper name (if present)
    disc_features = [c for c in df.columns if c not in cont_features + ["target"]]

    # ---------------------------------------------------------------------
    # 4) Define experiment grid (dataset‑specific; no helper functions)
    # ---------------------------------------------------------------------
    # ---------------------------------------------------------------------
    # 4) Define experiment grid (dataset-specific; no helper functions)
    # ---------------------------------------------------------------------
    # rng = np.random.default_rng()
    # model_seeds = rng.choice(100, size=5, replace=False)
    # cf_seeds = rng.choice(100, size=5, replace=False)

    # Reuse (even partial) model and cf seeds
    rng = np.random.default_rng()
    MS_NUM = 5
    CS_NUM = 5

    # existing seeds (very light parsing)
    ms_exist = {int(m.group(1)) for p in Path("models_cache").glob("rf_ms*_ne*.joblib")
                if (m := re.search(r"rf_ms(\d+)_ne", p.name))}
    cs_exist = {int(m.group(1)) for p in EXP_DIR.glob("*__cf*.meta.json")
                if (m := re.search(r"__cf(\d+)\.meta\.json$", p.name))}

    # reuse existing, then top up from 0..99 (no replacement)
    def pick(existing, n):
        existing = sorted(existing)
        if len(existing) >= n:
            return np.array(existing[:n], dtype=int)
        need = n - len(existing)
        pool = [s for s in range(100) if s not in existing]
        return np.array(existing + rng.choice(pool, size=need, replace=False).tolist(), dtype=int)

    model_seeds = pick(ms_exist, MS_NUM)
    cf_seeds    = pick(cs_exist, CS_NUM)

    print(f"[seeds] model_seeds = {model_seeds.tolist()}")
    print(f"[seeds] cf_seeds    = {cf_seeds.tolist()}")

    exps: List[Recreation_Experiment] = [
        # Baselines
        Recreation_Experiment(data_df=df, cont_features=cont_features, disc_features=disc_features,
                            label="baseline"),
        Recreation_Experiment(data_df=df, cont_features=cont_features, disc_features=disc_features,
                            label="vary_subset", features_to_vary=subset),
    ]

    # 4a) Model-seed variations (baseline features)
    exps += [
        Recreation_Experiment(
            data_df=df, cont_features=cont_features, disc_features=disc_features,
            label=f"random_model_run_{i}", model_seed=int(s)
        )
        for i, s in enumerate(model_seeds, start=1)
    ]

    # 4b) CF-seed variations (baseline features)
    exps += [
        Recreation_Experiment(
            data_df=df, cont_features=cont_features, disc_features=disc_features,
            label=f"random_cf_run_{i}", cf_seed=int(s)
        )
        for i, s in enumerate(cf_seeds, start=1)
    ]

    # 4c) Cherry-picked runs with random model seeds
    exps += [
        Recreation_Experiment(
            data_df=df, cont_features=cont_features, disc_features=disc_features,
            label=f"cherry_model_run_{i}", model_seed=int(s), features_to_vary=subset
        )
        for i, s in enumerate(model_seeds, start=1)
    ]

    # 4d) Cherry-picked runs with random CF seeds
    exps += [
        Recreation_Experiment(
            data_df=df, cont_features=cont_features, disc_features=disc_features,
            label=f"cherry_cf_run_{i}", cf_seed=int(s), features_to_vary=subset
        )
        for i, s in enumerate(cf_seeds, start=1)
    ]

    # --- Method variants with seeds & repeats ---
    # For kdtree/genetic, vary model_seed and cf_seed separately.
    # Do 2 repeats with EXACT same settings for each (same seeds), labelled _repeat1/_repeat2.

    # 4e) kdtree: vary model_seed (one per seed) + two extra for the first seed
    for i, s in enumerate(model_seeds, start=1):
        common = dict(data_df=df, cont_features=cont_features, disc_features=disc_features,
                    method="kdtree", model_seed=int(s))
        exps.append(Recreation_Experiment(label=f"kdtree_model_seed_{i}", **common))

    # Cutting kdtree experiemnts as each run (500 explanations), takes around an hour)
    # 4f) kdtree: vary cf_seed (one per seed; no repeats)
    for i, s in enumerate(cf_seeds, start=1):
        common = dict(data_df=df, cont_features=cont_features, disc_features=disc_features,
                    method="kdtree", cf_seed=int(s))
        exps.append(Recreation_Experiment(label=f"kdtree_cf_seed_{i}", **common))

    # 4g) genetic: vary model_seed (one per seed) + two extra for the first seed
    for i, s in enumerate(model_seeds, start=1):
        common = dict(data_df=df, cont_features=cont_features, disc_features=disc_features,
                    method="genetic", model_seed=int(s))
        exps.append(Recreation_Experiment(label=f"genetic_model_seed_{i}", **common))

    # 4h) genetic: vary cf_seed (one per seed; no repeats)
    for i, s in enumerate(cf_seeds, start=1):
        common = dict(data_df=df, cont_features=cont_features, disc_features=disc_features,
                    method="genetic", cf_seed=int(s))
        exps.append(Recreation_Experiment(label=f"genetic_cf_seed_{i}", **common))

    # --- Parameter sweeps (no cf_seed fixed) ---
    # Do these with all model_seeds; repeat the first seed once to check stability.
    # Notes from your method docs:
    # - stopping_threshold: general
    # - proximity_weight: used by ['genetic','gradientdescent'] (ignored by 'kdtree')
    # - sparsity_weight: used by ['genetic','kdtree']
    # - diversity_weight: used by ['genetic','gradientdescent'] (ignored by 'kdtree')
    # - posthoc_sparsity_*: general post-hoc

    # Choose compact value sets (1-at-a-time sweeps)
    stopping_threshold_vals = [0.1, 0.9]
    proximity_weight_vals   = [0.2, 2.0]
    sparsity_weight_vals    = [0.5, 1.0, 2.0]
    diversity_weight_vals   = [0.0, 0.5, 1.0]
    posthoc_param_vals      = [0.1, 1.0]
    posthoc_algo_vals       = ["linear", "binary"]

    # 4i) stopping_threshold (applies to both methods)
    for i, s in enumerate(model_seeds, start=1):
        for v in stopping_threshold_vals:
            exps.append(Recreation_Experiment(
                data_df=df, cont_features=cont_features, disc_features=disc_features,
                label=f"genetic_stopthr_{v}_model_{i}", method="genetic",
                model_seed=int(s), stopping_threshold=float(v)
            ))
            # exps.append(Recreation_Experiment(
            #     data_df=df, cont_features=cont_features, disc_features=disc_features,
            #     label=f"kdtree_stopthr_{v}_model_{i}", method="kdtree",
            #     model_seed=int(s), stopping_threshold=float(v)
            # ))
        # one repeat of the first seed only
        if i == 1:
            for v in stopping_threshold_vals:
                exps.append(Recreation_Experiment(
                    data_df=df, cont_features=cont_features, disc_features=disc_features,
                    label=f"genetic_stopthr_{v}_model_{i}_repeat", method="genetic",
                    model_seed=int(s), stopping_threshold=float(v)
                ))
                # exps.append(Recreation_Experiment(
                #     data_df=df, cont_features=cont_features, disc_features=disc_features,
                #     label=f"kdtree_stopthr_{v}_model_{i}_repeat", method="kdtree",
                #     model_seed=int(s), stopping_threshold=float(v)
                # ))

    # 4j) proximity_weight (genetic only)
    for i, s in enumerate(model_seeds, start=1):
        for v in proximity_weight_vals:
            exps.append(Recreation_Experiment(
                data_df=df, cont_features=cont_features, disc_features=disc_features,
                label=f"genetic_proxw_{v}_model_{i}", method="genetic",
                model_seed=int(s), proximity_weight=float(v)
            ))
        if i == 1:
            for v in proximity_weight_vals:
                exps.append(Recreation_Experiment(
                    data_df=df, cont_features=cont_features, disc_features=disc_features,
                    label=f"genetic_proxw_{v}_model_{i}_repeat", method="genetic",
                    model_seed=int(s), proximity_weight=float(v)
                ))

    # 4k) sparsity_weight (both methods)
    # for i, s in enumerate(model_seeds, start=1):
    #     for v in sparsity_weight_vals:
    #         exps.append(Recreation_Experiment(
    #             data_df=df, cont_features=cont_features, disc_features=disc_features,
    #             label=f"genetic_sparsew_{v}_model_{i}", method="genetic",
    #             model_seed=int(s), sparsity_weight=float(v)
    #         ))
    #         exps.append(Recreation_Experiment(
    #             data_df=df, cont_features=cont_features, disc_features=disc_features,
    #             label=f"kdtree_sparsew_{v}_model_{i}", method="kdtree",
    #             model_seed=int(s), sparsity_weight=float(v)
    #         ))
    #     if i == 1:
    #         for v in sparsity_weight_vals:
    #             exps.append(Recreation_Experiment(
    #                 data_df=df, cont_features=cont_features, disc_features=disc_features,
    #                 label=f"genetic_sparsew_{v}_model_{i}_repeat", method="genetic",
    #                 model_seed=int(s), sparsity_weight=float(v)
    #             ))
    #             exps.append(Recreation_Experiment(
    #                 data_df=df, cont_features=cont_features, disc_features=disc_features,
    #                 label=f"kdtree_sparsew_{v}_model_{i}_repeat", method="kdtree",
    #                 model_seed=int(s), sparsity_weight=float(v)
    #             ))

    # 4l) diversity_weight (genetic only)
    # for i, s in enumerate(model_seeds, start=1):
    #     for v in diversity_weight_vals:
    #         exps.append(Recreation_Experiment(
    #             data_df=df, cont_features=cont_features, disc_features=disc_features,
    #             label=f"genetic_divw_{v}_model_{i}", method="genetic",
    #             model_seed=int(s), diversity_weight=float(v)
    #         ))
    #     if i == 1:
    #         for v in diversity_weight_vals:
    #             exps.append(Recreation_Experiment(
    #                 data_df=df, cont_features=cont_features, disc_features=disc_features,
    #                 label=f"genetic_divw_{v}_model_{i}_repeat", method="genetic",
    #                 model_seed=int(s), diversity_weight=float(v)
    #             ))

    # 4m) posthoc sparsity params (both methods)
    for i, s in enumerate(model_seeds, start=1):
        for v in posthoc_param_vals:
            exps.append(Recreation_Experiment(
                data_df=df, cont_features=cont_features, disc_features=disc_features,
                label=f"genetic_posthoc_{v}_model_{i}", method="genetic",
                model_seed=int(s), posthoc_sparsity_param=float(v)
            ))
            # exps.append(Recreation_Experiment(
            #     data_df=df, cont_features=cont_features, disc_features=disc_features,
            #     label=f"kdtree_posthoc_{v}_model_{i}", method="kdtree",
            #     model_seed=int(s), posthoc_sparsity_param=float(v)
            # ))
        if i == 1:
            for v in posthoc_param_vals:
                exps.append(Recreation_Experiment(
                    data_df=df, cont_features=cont_features, disc_features=disc_features,
                    label=f"genetic_posthoc_{v}_model_{i}_repeat", method="genetic",
                    model_seed=int(s), posthoc_sparsity_param=float(v)
                ))
                # exps.append(Recreation_Experiment(
                #     data_df=df, cont_features=cont_features, disc_features=disc_features,
                #     label=f"kdtree_posthoc_{v}_model_{i}_repeat", method="kdtree",
                #     model_seed=int(s), posthoc_sparsity_param=float(v)
                # ))

    # 4n) posthoc sparsity algorithm (both methods)
    for i, s in enumerate(model_seeds, start=1):
        for algo in posthoc_algo_vals:
            exps.append(Recreation_Experiment(
                data_df=df, cont_features=cont_features, disc_features=disc_features,
                label=f"genetic_posthoc_{algo}_model_{i}", method="genetic",
                model_seed=int(s), posthoc_sparsity_algorithm=str(algo)
            ))
            # exps.append(Recreation_Experiment(
            #     data_df=df, cont_features=cont_features, disc_features=disc_features,
            #     label=f"kdtree_posthoc_{algo}_model_{i}", method="kdtree",
            #     model_seed=int(s), posthoc_sparsity_algorithm=str(algo)
            # ))
        if i == 1:
            for algo in posthoc_algo_vals:
                exps.append(Recreation_Experiment(
                    data_df=df, cont_features=cont_features, disc_features=disc_features,
                    label=f"genetic_posthoc_{algo}_model_{i}_repeat", method="genetic",
                    model_seed=int(s), posthoc_sparsity_algorithm=str(algo)
                ))
                # exps.append(Recreation_Experiment(
                #     data_df=df, cont_features=cont_features, disc_features=disc_features,
                #     label=f"kdtree_posthoc_{algo}_model_{i}_repeat", method="kdtree",
                #     model_seed=int(s), posthoc_sparsity_algorithm=str(algo)
                # ))

    # ---------------------------------------------------------------------
    # 5) Run experiments, save CFs, collect records
    # ---------------------------------------------------------------------
    print(f"[1/4] Preparing output dir at {EXP_DIR}", flush=True)
    EXP_DIR.mkdir(parents=True, exist_ok=True)

    REUSE_EXISTING = True
    records = []

    sample_fp = sample_fingerprint(to_explain.index.tolist())

    total = len(exps)
    print(f"[2/4] Running {total} experiments (reuse={'on' if REUSE_EXISTING else 'off'})", flush=True)

    for i, exp in enumerate(exps, start=1):
        label = exp.label
        print(f"  • ({i}/{total}) {label}: starting", flush=True)
        reused = False
        if REUSE_EXISTING:
            # try an exact-match reuse by scanning meta files for this label & sample_fp
            for meta_path in EXP_DIR.glob(f"{label}__rep1__d*__m*__ae*__cf*.meta.json"):
                try:
                    import json
                    with open(meta_path, "r") as f:
                        meta = json.load(f)
                    if meta.get("sample_fp") == sample_fp:
                        cf_csv = meta.get("cf_csv")
                        cf_path = (EXP_DIR / cf_csv) if cf_csv else meta_path.with_suffix("").with_suffix(".csv")
                        if cf_path.exists():
                            records.append({"stamp": meta_path.stem, "cf_csv": str(cf_path), "label": label, "exp": exp})
                            print(f"    ↪ reused existing CFs: {cf_path.name}", flush=True)
                            reused = True
                            break
                except Exception as e:
                    print(f"    ! reuse scan error on {meta_path.name}: {e}", flush=True)
        if not reused:
            rec = run_and_save(exp, to_explain, out_dir=EXP_DIR, rep=1)
            if rec:
                records.append(rec)
                print(f"    ✓ generated CFs: {Path(rec['cf_csv']).name}", flush=True)
            else:
                print(f"    ✗ CF generation returned empty record; skipping", flush=True)

    # ---------------------------------------------------------------------
    # 6) Compute metrics and write metrics.csv
    # ---------------------------------------------------------------------
    print("[3/4] Computing metrics…", flush=True)
    metrics_df = compute_metrics_from_records(df, records, to_explain, str(EXP_DIR))
    out_csv = EXP_DIR / "metrics.csv"
    metrics_df.to_csv(out_csv, index=False)
    print(f"    ✓ wrote {out_csv}", flush=True)

    # ---------------------------------------------------------------------
    # 7) Summaries (paired vs baseline when possible) and file outputs
    # ---------------------------------------------------------------------
    print("[4/4] Summarising against baseline…", flush=True)
    summarise_against_baseline(metrics_df, EXP_DIR)
    print("    ✓ summary.json and summary_means.csv written", flush=True)


if __name__ == "__main__":
    main()