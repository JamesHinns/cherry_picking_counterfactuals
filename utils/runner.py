# runner.py
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Union

import pandas as pd
from utils.counterfactuals import Recreation_Experiment
from .hashing import canonical_fingerprint_df, _sample_fp

# Could do a data class to clean up?
# @dataclass(frozen=True)
# class RunRecord:
#     stamp: str
#     cf_csv: Path
#     label: str
#     exp: Recreation_Experiment

def run_and_save(
    exp: Recreation_Experiment,
    to_explain: pd.DataFrame,
    out_dir: Union[str, Path],
    rep: int = 1,
) -> Dict[str, Any]:
    """
    Generate CFs for the provided `to_explain` (shared across experiments), save to `out_dir`,
    and return a record for metrics. Returns {} if CF generation fails (None).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df_cfs = exp.generate_counterfactuals(to_explain)
    if df_cfs is None:
        print(f"[warn] {exp.label}: CF generation returned None; skipping save.")
        return {}

    stamp = (
        f"{exp.label}__rep{rep}"
        f"__d{exp.data_seed}__m{exp.model_seed}__ae{exp.ae_seed}__cf{exp.cf_seed}"
    )
    cf_path = out_dir / f"{stamp}.csv"
    meta_path = out_dir / f"{stamp}.meta.json"

    # Save CFs
    df_cfs.to_csv(cf_path, index=True)

    # Save metadata
    meta = {
        "label": exp.label,
        "data_seed": exp.data_seed,
        "model_seed": exp.model_seed,
        "ae_seed": exp.ae_seed,
        "cf_seed": exp.cf_seed,
        "sample_size": int(len(to_explain)),
        "sample_index": to_explain.index.tolist(),
        "sample_fp": _sample_fp(to_explain.index.tolist()),
        "cf_csv": cf_path.name,
        "cf_fingerprint": canonical_fingerprint_df(df_cfs),
    }
    with meta_path.open("w") as f:
        json.dump(meta, f, indent=2)

    # Return a dict compatible with existing downstream code
    return {
        "stamp": stamp,
        "cf_csv": str(cf_path),  # keep as str for robust pandas I/O
        "label": exp.label,
        "exp": exp,
    }