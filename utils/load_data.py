import numpy as np
import pandas as pd

from pmlb import fetch_data

def generate_dataset(n, seed=None, noise=0.0):
    if seed is not None:
        np.random.seed(seed)
    df = pd.DataFrame({
        "Income": np.random.randint(0, 100001, size=n),
        "Gender": np.random.choice([0, 1], size=n)
    })

    score = (df["Income"] / 25000) + (1.5 * df["Gender"] - 2.5)
    df["Label"] = (score > 0).astype(int)

    if noise > 0:
        flip_count = int(np.floor(noise * n))
        flip_idx = np.random.choice(df.index, size=flip_count, replace=False)
        df.loc[flip_idx, "Label"] = 1 - df.loc[flip_idx, "Label"]

    return df

def load_adult():
    df = fetch_data("adult", return_X_y=False)
    df = df.rename(columns={"target": "Label"})

    return df