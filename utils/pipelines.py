import copy
import datetime
import numbers
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from pmlb import fetch_data
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from torch.autograd import Variable

from utils.counterfactuals import DiCE

# Temp model caching - added for complete recreatability, however each
# model ends up being just under the github 100mb limit, so use with care
MODEL_DIR = Path("models_cache")
MODEL_DIR.mkdir(exist_ok=True)

class CounterfactualPipeline:

    def __init__(self, 
                data_df,  # full dataset
                cont_features=None, 
                disc_features=None,
                data_seed=None, 
                model_seed=None, 
                ae_seed=None, 
                cf_seed=None,
                default_seed=42, 
                label="default",
                cf_alg=None,
                model=None,
                train_df=None,
                test_df=None,
                split_test_size=0.2,
                **cf_kwargs):

        # ---- seeds / label -------------------------------------------------
        self.data_seed  = data_seed  if data_seed  is not None else default_seed
        self.model_seed = model_seed if model_seed is not None else default_seed
        self.ae_seed    = ae_seed    if ae_seed    is not None else default_seed
        self.cf_seed    = cf_seed    if cf_seed    is not None else default_seed
        self.label      = label
        self.kwargs     = cf_kwargs

        # ---- single split ---------------------------------------------------
        self.SPLIT_TEST_SIZE = split_test_size

        if train_df is None or test_df is None:
            self.train_df, self.test_df = train_test_split(
                data_df, test_size=self.SPLIT_TEST_SIZE, random_state=self.data_seed
            )
        else:
            self.train_df, self.test_df = train_df, test_df

        # ---- model (disk cache only) ----------------------------------------
        if model is None:
            # always use on-disk cache; trains & saves if missing
            self.model = get_or_train_rf_disk(
                data_df=data_df,
                data_seed=self.data_seed,
                model_seed=self.model_seed,
                n_estimators=100,
            )
        else:
            self.model = model

        # ---- DiCE explainer -------------------------------------------------
        if cf_alg is None:
            cf_method = self.kwargs.pop("method", "random")
            cf_alg = DiCE(
                data_df=data_df,
                model=self.model,
                method=cf_method,
                cont_features=cont_features,
                disc_features=disc_features,
            )
        self.cf_alg = cf_alg

    def train_random_forest(self, n_estimators=100):
        x_train = self.train_df.drop(columns=["Label"])
        y_train = self.train_df["Label"]
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=self.model_seed
        ).fit(x_train, y_train)

        # simple diagnostics
        train_acc = model.score(x_train, y_train)
        x_test = self.test_df.drop(columns=["Label"])
        y_test = self.test_df["Label"]
        test_acc = model.score(x_test, y_test)
        print(f"Trained RF | Train Acc: {train_acc:.3f} | Test Acc: {test_acc:.3f}")

        self.test_acc = test_acc
        return model
    
    def generate_counterfactuals(self, to_explain: pd.DataFrame):
        return self.cf_alg.generate_counterfactuals(to_explain, self.cf_seed, **self.kwargs)


# ----- Helpers ------

def _rf_path(model_seed: int, n_estimators: int = 100) -> Path:
    # one model per model_seed; include n_estimators to avoid accidents
    return MODEL_DIR / f"rf_ms{int(model_seed)}_ne{int(n_estimators)}.joblib"

def get_or_train_rf_disk(
    data_df: pd.DataFrame,
    data_seed: int,
    model_seed: int,
    n_estimators: int = 100,
) -> RandomForestClassifier:
    """
    Load RandomForest for model_seed from disk if present, else train and save.
    NOTE: This does NOT persist train/test splits. Recreation_Experiment keeps its own split.
    """
    path = _rf_path(model_seed, n_estimators)
    if path.exists():
        try:
            rf = joblib.load(path)
            print(f"[ModelCache] Loaded RF (model_seed={model_seed}) from {path}")
            return rf
        except Exception as e:
            print(f"[ModelCache] Failed to load {path} ({e}); retraining...")

    # Train a fresh RF on a split consistent with data_seed
    train_df, _ = train_test_split(data_df, test_size=0.2, random_state=data_seed)
    X = train_df.drop(columns=["Label"])
    y = train_df["Label"]

    rf = RandomForestClassifier(n_estimators=n_estimators, random_state=model_seed)
    rf.fit(X, y)

    joblib.dump(rf, path)
    print(f"[ModelCache] Trained & saved RF (model_seed={model_seed}) -> {path}")
    return rf