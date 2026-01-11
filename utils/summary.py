from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import json
import csv

import numpy as np
import pandas as pd

from utils.compare_cfs import tost_paired, tost_independent

def _pretty_feature_diff(
    base_counts: Dict[str, int],
    agg_counts: Dict[str, int],
) -> Tuple[str, List[Dict[str, int]] | str]:
    """
    Compare aggregated feature-change totals vs baseline.
    Returns (human_text, structured_diffs | 'identical').
    """
    diffs: List[Tuple[str, int, int]] = []
    for f in sorted(set(base_counts) | set(agg_counts)):
        a = int(base_counts.get(f, 0))
        p = int(agg_counts.get(f, 0))
        if a != p:
            diffs.append((f, a, p))
    diffs.sort(key=lambda t: abs(t[2] - t[1]), reverse=True)
    if not diffs:
        return "Feature change totals identical to baseline", "identical"
    text = ", ".join([f"{f} {a} -> {p}" for f, a, p in diffs])
    struct = [{"feature": f, "baseline": a, "param": p} for f, a, p in diffs]
    return text, struct


def _print_metrics_with_tost(lbl: str, entry: dict) -> None:
    """Console print helper for means + TOST results (paired or independent)."""
    means = entry.get("means", {})
    t = entry.get("tost", {})
    mode = "paired" if entry.get("paired_with_baseline") else "indep."

    def line(metric_key: str, pretty: str) -> Optional[str]:
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
        n = v.get("n")
        return (
            f"{lbl}: {pretty}={val:.6g} | TOST {mode}: "
            f"eq={eq}, meanΔ={md:.6g}, 90%CI=({ci[0]:.6g},{ci[1]:.6g}), n={n}"
        )

    for k, name in (("proximity", "proximity"),
                    ("sparsity", "sparsity"),
                    ("im1", "im1"),
                    ("im2", "im2")):
        ln = line(k, name)
        if ln:
            print(ln)


def summarise_against_baseline(metrics_df: pd.DataFrame, out_dir: Path) -> None:
    """
    Build per-label summary against the 'baseline' runs and write:
      - summary.json
      - summary_means.csv
    Also prints concise baseline means and per-label TOST lines for non-identical feature diffs.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- locate baseline ---
    base_rows = metrics_df[metrics_df["label"] == "baseline"]
    if base_rows.empty:
        print("baseline: not found")
        # Still write empty files for robustness
        (out_dir / "summary.json").write_text(json.dumps({}, indent=2))
        with open(out_dir / "summary_means.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["label", "mean_proximity", "mean_sparsity", "mean_im1", "mean_im2"])
            writer.writeheader()
        return

    base = base_rows.iloc[0]
    base_counts: Dict[str, int] = base["feature_change_counts"]
    base_fp: str = base["sample_fp"]

    summary: Dict[str, Any] = {}

    # --- per label aggregation ---
    for lbl in metrics_df["label"].unique().tolist():
        rows_lbl = metrics_df[metrics_df["label"] == lbl]

        # Flatten arrays across all rows for this label
        prox_all = np.concatenate([np.asarray(r) for r in rows_lbl["proximity"]])
        spars_all = np.concatenate([np.asarray(r) for r in rows_lbl["sparsity"]])

        im1_all = None
        if rows_lbl["im1"].notna().any():
            im1_vals = [np.asarray(r) for r in rows_lbl["im1"] if r is not None]
            if im1_vals:
                im1_all = np.concatenate(im1_vals)

        im2_all = None
        if rows_lbl["im2"].notna().any():
            im2_vals = [np.asarray(r) for r in rows_lbl["im2"] if r is not None]
            if im2_vals:
                im2_all = np.concatenate(im2_vals)

        entry: Dict[str, Any] = {
            "label": lbl,
            "means": {
                "proximity": float(np.mean(prox_all)) if prox_all.size else None,
                "sparsity": float(np.mean(spars_all)) if spars_all.size else None,
                "im1": float(np.mean(im1_all)) if im1_all is not None and im1_all.size else None,
                "im2": float(np.mean(im2_all)) if im2_all is not None and im2_all.size else None,
            },
            "tost": {},
            "feature_change_diff": None,
            "jaccard_distance_mean": None,
            "paired_with_baseline": None,
        }

        # For non-baseline labels, compute TOSTs and compare feature-change totals
        if lbl != "baseline":
            prox_diffs: List[float] = []
            base_prox: List[float] = []
            spars_diffs: List[float] = []
            base_spars: List[float] = []
            im1_diffs: List[float] = []
            base_im1: List[float] = []
            im2_diffs: List[float] = []
            base_im2: List[float] = []

            # Pair rows only when sample fingerprints match the baseline (shared to_explain)
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

            paired = bool(base_prox)
            entry["paired_with_baseline"] = paired

            if paired:
                entry["tost"]["proximity"] = tost_paired(np.asarray(prox_diffs), np.asarray(base_prox))
                entry["tost"]["sparsity"] = tost_paired(np.asarray(spars_diffs), np.asarray(base_spars))
                if im1_diffs:
                    entry["tost"]["im1"] = tost_paired(np.asarray(im1_diffs), np.asarray(base_im1))
                if im2_diffs:
                    entry["tost"]["im2"] = tost_paired(np.asarray(im2_diffs), np.asarray(base_im2))
            else:
                entry["tost"]["proximity"] = tost_independent(prox_all, np.asarray(base["proximity"]))
                entry["tost"]["sparsity"] = tost_independent(spars_all, np.asarray(base["sparsity"]))
                if im1_all is not None and base["im1"] is not None:
                    entry["tost"]["im1"] = tost_independent(im1_all, np.asarray(base["im1"]))
                if im2_all is not None and base["im2"] is not None:
                    entry["tost"]["im2"] = tost_independent(im2_all, np.asarray(base["im2"]))

            # Aggregate feature-change counts across all rows for this label
            agg_counts: Dict[str, int] = {}
            for cdict in rows_lbl["feature_change_counts"]:
                for f, c in cdict.items():
                    agg_counts[f] = agg_counts.get(f, 0) + int(c)

            pretty_txt, diffs_struct = _pretty_feature_diff(base_counts, agg_counts)
            print(f"{lbl}: {pretty_txt}")
            entry["feature_change_diff"] = diffs_struct

            # Jaccard distance (1 - mean Jaccard-to-baseline across rows for this label)
            jacc_vals = [
                float(v) for v in rows_lbl["jaccard_mean_to_baseline"]
                if v is not None and np.isfinite(v)
            ]
            entry["jaccard_distance_mean"] = (1.0 - float(np.mean(jacc_vals))) if jacc_vals else None

            # Only print detailed TOST lines when there is a difference vs baseline
            if diffs_struct != "identical":
                _print_metrics_with_tost(lbl, entry)

        summary[lbl] = entry

    # --- console prints for baseline means ---
    # Guard for None to avoid formatting errors if arrays were empty (unlikely).
    b_prox = np.asarray(base["proximity"])
    b_spars = np.asarray(base["sparsity"])
    print(f"baseline: mean proximity = {np.mean(b_prox):.6g}" if b_prox.size else "baseline: mean proximity = NA")
    print(f"baseline: mean sparsity  = {np.mean(b_spars):.6g}" if b_spars.size else "baseline: mean sparsity  = NA")
    if base["im1"] is not None:
        b_im1 = np.asarray(base["im1"])
        if b_im1.size:
            print(f"baseline: mean im1 = {np.mean(b_im1):.6g}")
    if base["im2"] is not None:
        b_im2 = np.asarray(base["im2"])
        if b_im2.size:
            print(f"baseline: mean im2 = {np.mean(b_im2):.6g}")

    # --- write summary.json ---
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # --- write compact summary_means.csv ---
    means_rows = []
    for lbl, ent in summary.items():
        m = ent["means"]
        means_rows.append({
            "label": lbl,
            "mean_proximity": m.get("proximity"),
            "mean_sparsity": m.get("sparsity"),
            "mean_im1": m.get("im1"),
            "mean_im2": m.get("im2"),
        })

    # Handle the case where there are no rows gracefully
    fieldnames = ["label", "mean_proximity", "mean_sparsity", "mean_im1", "mean_im2"]
    with open(out_dir / "summary_means.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in means_rows:
            writer.writerows([row])
