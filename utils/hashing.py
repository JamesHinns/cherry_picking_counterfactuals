# Putting hashing functions here for now, changed a few times
import pandas as pd
import hashlib
from typing import List

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