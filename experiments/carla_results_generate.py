#!/usr/bin/env python3
# experiments/temp.py — minimal, hardcoded run

import os
# Force every stack to CPU
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"     # covers TF & PyTorch
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"      # quiet TF logs
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"  # if jax/xla present

# Optional: if TensorFlow is importable, hide GPUs explicitly
try:
    import tensorflow as tf
    try:
        tf.config.set_visible_devices([], "GPU")
    except Exception:
        pass
except Exception:
    pass

import os, warnings, inspect
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from typing import Dict, Optional, List

import logging

# ---------- CARLA imports ----------
from carla.data.catalog import OnlineCatalog
from carla.models.catalog import MLModelCatalog
from carla.recourse_methods import Wachter, GrowingSpheres, ActionableRecourse

_HAS_DICE = _HAS_REVISE = _HAS_CCHVAE = _HAS_FACE = _HAS_CLUE = True
try:
    from carla.recourse_methods import Dice as CarlaDice
except Exception:
    _HAS_DICE = False
try:
    from carla.recourse_methods import Revise
except Exception:
    _HAS_REVISE = False
try:
    from carla.recourse_methods import CCHVAE
except Exception:
    _HAS_CCHVAE = False
try:
    from carla.recourse_methods import FACE
except Exception:
    _HAS_FACE = False
try:
    from carla.recourse_methods import CLUE
except Exception:
    _HAS_CLUE = False


# ---------------------- Hardcoded config ----------------------
DATASET_NAME = "adult"
MODEL_TYPE = "linear"      # PyTorch linear
BACKEND = "pytorch"
OUTPUT_DIR = "results/carla"
SEED = 42
LOG_PATH = os.path.join(OUTPUT_DIR, "runs.log")

METHODS = [
    # "growing_spheres",
    # "actionable_recourse",
    # "dice",
    "wachter",
    "cchvae",
    "revise",
    "face",
    "clue",
]

RUNS_PER_METHOD = 1
MAX_RETRIES = 3          # outer: whole set retries (if none found at all)
FILL_RETRIES = 1          # inner: per-row retries to fill missing CFs

SAMPLE_SIZE = 200
# Shared VAE defaults (used only when a method needs it)
VAE_LATENT_DIM = 10
VAE_HIDDEN = 64
VAE_EPOCHS = 50
VAE_BATCH_SIZE = 256
VAE_LR = 1e-3

# Wachter defaults
WACHTER_PARAMS = {"max_iter": 2000, "lr": 0.05, "lambda_": 0.05, "eps": 0.05}
GS_PARAMS = {"n_inits": 10}
AR_PARAMS = {}
DICE_PARAMS = {"total_CFs": 1}
CCHVAE_PARAMS = {}
REVISE_PARAMS = {}
FACE_PARAMS = {}
CLUE_PARAMS = {"train_ae": False, "retrain_ae": False}

# ---------------------- Version helpers ----------------------
def _get_df(dataset):
    return getattr(dataset, "df", getattr(dataset, "raw", None))

def _get_df_train(dataset):
    return getattr(dataset, "df_train", None)

def _get_target(dataset):
    return getattr(dataset, "target", "target")


# ---------------------- Simple shared VAE ----------------------
class TabularVAE(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int = 10, hidden: int = 64):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.mu = nn.Linear(hidden, latent_dim)
        self.logvar = nn.Linear(hidden, latent_dim)
        self.dec = nn.Sequential(
            nn.Linear(latent_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, input_dim),
        )

    def encode(self, x):
        h = self.enc(x);  mu, logvar = self.mu(h), self.logvar(h);  return mu, logvar
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar); eps = torch.randn_like(std); return mu + eps * std
    def decode(self, z): return self.dec(z)
    def forward(self, x):
        mu, logvar = self.encode(x); z = self.reparameterize(mu, logvar); recon = self.decode(z)
        return recon, mu, logvar

def _vae_loss(x, recon, mu, logvar):
    recon_loss = nn.functional.mse_loss(recon, x, reduction="mean")
    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kl

def train_tabular_vae(X: np.ndarray, latent_dim=10, hidden=64, epochs=50, batch_size=256, lr=1e-3, device="cpu"):
    model = TabularVAE(X.shape[1], latent_dim, hidden).to(device)
    opt = optim.Adam(model.parameters(), lr=lr)
    X_tensor = torch.tensor(X, dtype=torch.float32)
    n = X.shape[0]
    for _ in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, batch_size):
            idx = perm[i:i+batch_size]
            batch = X_tensor[idx].to(device)
            opt.zero_grad()
            recon, mu, logvar = model(batch)
            loss = _vae_loss(batch, recon, mu, logvar)
            loss.backward(); opt.step()
    return model

class SharedVAEAdapter:
    def __init__(self, vae: TabularVAE, device: str = "cpu"):
        self.vae = vae
        self.device = device

    # swallow any retrain attempts from methods (e.g., CLUE)
    def train(self, *args, **kwargs): return self
    def fit(self, *args, **kwargs): return self

    def _to_np(self, X):
        if isinstance(X, pd.DataFrame):
            return X.values.astype(np.float32), X.columns, X.index
        elif isinstance(X, np.ndarray):
            return X.astype(np.float32), None, None
        else:
            X = np.asarray(X, dtype=np.float32)
            return X, None, None

    def encode(self, X):
        # Always return NumPy to avoid pandas truth-value pitfalls inside methods
        X_np, _, _ = self._to_np(X)
        self.vae.eval()
        with torch.no_grad():
            t = torch.tensor(X_np, dtype=torch.float32, device=self.device)
            mu, logvar = self.vae.encode(t)
            z = self.vae.reparameterize(mu, logvar).cpu().numpy()
        return z


    def decode(self, Z):
        Z_np, _, idx = self._to_np(Z)
        self.vae.eval()
        with torch.no_grad():
            zt = torch.tensor(Z_np, dtype=torch.float32, device=self.device)
            x_hat = self.vae.decode(zt).cpu().numpy()
        # If Z was a DF and we previously encoded from X with known columns,
        # most CARLA methods will pass the same factuals back. We can’t always
        # know the original column names here, so we expose a helper:
        return x_hat

    # Optional helpers some CARLA versions look for:
    def encode_df(self, X_df: pd.DataFrame) -> pd.DataFrame:
        return self.encode(X_df)

    def decode_to_df(self, Z: np.ndarray, template_df: pd.DataFrame) -> pd.DataFrame:
        X_hat = self.decode(Z)
        return pd.DataFrame(X_hat, index=template_df.index, columns=template_df.columns)

    @property
    def model(self): 
        return self.vae



# ---------------------- Runner ----------------------
class CarlaCFRunner:
    def __init__(self):
        self.rng = np.random.RandomState(SEED)
        torch.manual_seed(SEED)
        self.dataset = OnlineCatalog(DATASET_NAME)
        self.mlmodel = MLModelCatalog(self.dataset, model_type=MODEL_TYPE, backend=BACKEND)
        self.device = "cpu"
        #self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.shared_vae_adapter: Optional[SharedVAEAdapter] = None
        self.registry = {
            "wachter": (Wachter, True,  False, True),
            "growing_spheres": (GrowingSpheres, False, False, True),
            "actionable_recourse": (ActionableRecourse, False, False, True),
            # "dice": (CarlaDice if _HAS_DICE else None, False, False, _HAS_DICE),
            "cchvae": (CCHVAE if _HAS_CCHVAE else None, True, True, _HAS_CCHVAE),
            "revise": (Revise if _HAS_REVISE else None, True, True, _HAS_REVISE),
            "face": (FACE if _HAS_FACE else None, False, True, _HAS_FACE),
            "clue": (CLUE if _HAS_CLUE else None, True, True, _HAS_CLUE),
        }

    def train_model_if_needed(self):
        # some models expose different signatures; we’ll not pass lr at all to avoid the crash
        train_fn = getattr(self.mlmodel, "train", None)
        if not callable(train_fn):
            return
        try:
            sig = inspect.signature(train_fn)
            params = set(sig.parameters.keys())
        except (TypeError, ValueError):
            params = set()
        # pass only epochs if present; otherwise call without args
        if "epochs" in params:
            train_fn(epochs=20)   # fixed
        else:
            train_fn()

    def _ensure_shared_vae(self):
        if self.shared_vae_adapter is not None:
            return
        df_train = getattr(self.dataset, "df_train", None) or _get_df(self.dataset)
        target = _get_target(self.dataset)
        X_train_df = df_train.drop(columns=[target], errors="ignore")
        X_train = self.dataset.transform(X_train_df)
        if isinstance(X_train, pd.DataFrame): X_train = X_train.values
        vae = train_tabular_vae(
            X_train,
            latent_dim=VAE_LATENT_DIM,
            hidden=VAE_HIDDEN,
            epochs=VAE_EPOCHS,
            batch_size=VAE_BATCH_SIZE,
            lr=VAE_LR,
            device=self.device,
        )
        self.shared_vae_adapter = SharedVAEAdapter(vae, device=self.device)

    def _build_method(self, method: str):
        method = method.lower()
        if method not in self.registry:
            raise ValueError(f"Unknown method '{method}'.")
        MethodClass, needs_grad, needs_vae, available = self.registry[method]
        if not available or MethodClass is None:
            raise RuntimeError(f"Method '{method}' not available in this environment.")
        if needs_grad and self.mlmodel.backend != "pytorch":
            raise RuntimeError(f"'{method}' requires backend='pytorch'.")

        params: Dict = {}
        if method == "wachter":
            params.update(WACHTER_PARAMS)
            pos = int(getattr(self.dataset, "positive", 1))
            # Make sure these are plain ints, not tensors
            params["y_target"] = pos
            params["target_class"] = pos
            params["desired_class"] = pos
        elif method == "growing_spheres":
            params.update(GS_PARAMS)
        elif method == "actionable_recourse":
            params.update(AR_PARAMS)
        # elif method == "dice":
        #     params.update(DICE_PARAMS)
        elif method == "cchvae":
            params.update(CCHVAE_PARAMS)
        elif method == "revise":
            params.update(REVISE_PARAMS)
        elif method == "face":
            params.update(FACE_PARAMS)
        elif method == "clue":
            params.update(CLUE_PARAMS)

        if needs_vae:
            self._ensure_shared_vae()
            ae = self.shared_vae_adapter
            params.setdefault("ae", ae)
            params.setdefault("vae", ae)
            params.setdefault("ae_model", ae)
            params.setdefault("autoencoder", ae)
            # some versions expect callables too
            params.setdefault("encode", ae.encode_df if hasattr(ae, "encode_df") else ae.encode)
            params.setdefault("decode", ae.decode)

        return MethodClass(self.mlmodel, params)



    def select_factuals(self, sample_size=SAMPLE_SIZE, seed: Optional[int] = None) -> pd.DataFrame:
        df = _get_df(self.dataset); target = _get_target(self.dataset)
        try:
            uniques = set(pd.unique(df[target]))
        except Exception:
            uniques = set()
        if uniques == {0, 1}:
            neg = df[df[target] == 0]
            pool = neg if len(neg) >= sample_size else df
        else:
            pool = df

        rng = np.random.RandomState(seed) if seed is not None else self.rng
        idx = rng.choice(pool.index, size=min(sample_size, len(pool)), replace=False)
        return pool.loc[idx].copy()


    def run_once(self, method: str, factuals: pd.DataFrame) -> pd.DataFrame:
        """
        Build the method and generate counterfactuals.
        - Wachter: ensure int/long target class (also passed via constructor).
        - CCHVAE/Revise: feed NumPy (avoids ambiguous DataFrame truth-value paths).
        """
        self.train_model_if_needed()

        target_col = _get_target(self.dataset)
        X_df = factuals.drop(columns=[target_col], errors="ignore")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            method_lc = method.lower()
            method_obj = self._build_method(method_lc)

            # Wachter: ensure class index set (keep your existing guard here if you like)
            if method_lc == "wachter":
                pos = int(getattr(self.dataset, "positive", 1))
                for attr in ("y_target", "target_class", "_target_class", "target", "desired_class"):
                    if hasattr(method_obj, attr):
                        try:
                            setattr(method_obj, attr, pos)
                        except Exception:
                            pass
                try:
                    import torch as _torch
                    if hasattr(method_obj, "y_target") and not isinstance(getattr(method_obj, "y_target"), (int, np.integer)):
                        setattr(method_obj, "y_target", _torch.tensor(pos, dtype=_torch.long))
                except Exception:
                    pass

            # --- Choose input type per method ---
            if method_lc == "wachter":
                pos = int(getattr(self.dataset, "positive", 1))
                # Overwrite any attribute versions with ints (not tensors)
                for attr in ("y_target", "target_class", "_target_class", "target", "desired_class"):
                    if hasattr(method_obj, attr):
                        try:
                            setattr(method_obj, attr, pos)
                        except Exception:
                            pass
            else:
                X_in = X_df.values.astype(np.float32, copy=False)  # NumPy for CCHVAE/Revise (and others)

            cf_out = method_obj.get_counterfactuals(X_in)

        # ---- Normalise output to a DataFrame aligned with original columns/index ----
        if cf_out is None:
            return pd.DataFrame(index=X_df.index)

        if isinstance(cf_out, pd.DataFrame):
            overlap = [c for c in X_df.columns if c in cf_out.columns]
            out = cf_out[overlap].copy() if overlap else cf_out.copy()
            out.index = X_df.index
            return out

        if isinstance(cf_out, np.ndarray):
            n_cols = min(cf_out.shape[1], X_df.shape[1]) if cf_out.ndim == 2 else X_df.shape[1]
            cols = list(X_df.columns[:n_cols])
            if cf_out.ndim == 1:
                cf_out = cf_out.reshape(-1, 1 if n_cols == 1 else n_cols)
            return pd.DataFrame(cf_out[:, :n_cols], index=X_df.index, columns=cols)

        try:
            return pd.DataFrame(cf_out, index=X_df.index)
        except Exception:
            return pd.DataFrame(index=X_df.index)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # set up logging to both file and console
    logger = logging.getLogger("carla_runs")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fh = logging.FileHandler(LOG_PATH, mode="a", encoding="utf-8")
    ch = logging.StreamHandler()
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    fh.setFormatter(fmt)
    ch.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(ch)
    logger.info(f"Logging to {LOG_PATH}")


    runner = CarlaCFRunner()

    for method in METHODS:
        for run in range(1, RUNS_PER_METHOD + 1):
            factuals = runner.select_factuals(seed=SEED)
            success = False
            last_error = None

            for attempt in range(1, MAX_RETRIES + 1):
                try:
                    df_cfs = runner.run_once(method, factuals)

                    # If empty DF (no columns) or shape is zero, try whole-run retry
                    if df_cfs is None or df_cfs.empty:
                        last_error = "no_results_df"
                        continue

                    # Determine which rows got CFs (row not all-NaN)
                    row_success = ~df_cfs.isna().all(axis=1)
                    if row_success.sum() == 0:
                        # No individual CFs found at all → treat as fail for this attempt
                        last_error = "no_individual_cfs"
                        continue

                    # Try to fill missing rows up to FILL_RETRIES
                    fill_used = 0
                    filled_count = 0
                    while (not row_success.all()) and fill_used < FILL_RETRIES:
                        missing_idx = df_cfs.index[~row_success]
                        # Re-run only on missing rows
                        df_retry = runner.run_once(method, factuals.loc[missing_idx])

                        if df_retry is None or df_retry.empty:
                            fill_used += 1
                            continue

                        retry_success = ~df_retry.isna().all(axis=1)
                        if retry_success.any():
                            good_idx = df_retry.index[retry_success]
                            # Update only the rows where we newly found CFs
                            for col in df_cfs.columns:
                                if col in df_retry.columns:
                                    df_cfs.loc[good_idx, col] = df_retry.loc[good_idx, col]
                            # Recompute success mask and accounting
                            prev_count = row_success.sum()
                            row_success = ~df_cfs.isna().all(axis=1)
                            filled_count += (row_success.sum() - prev_count)

                        fill_used += 1

                    out_path = os.path.join(OUTPUT_DIR, f"{method}_run{run}.csv")
                    df_cfs.to_csv(out_path, index=True)
                    remaining = int((~row_success).sum())
                    logger.info(
                        f"{method}_run{run}: saved | path: {out_path} | retries: {attempt - 1} "
                        f"| fill_attempts: {fill_used} | filled: {filled_count} | remaining_no_cf: {remaining}"
                    )
                    success = True
                    break

                except RuntimeError as e:
                    last_error = str(e)
                    logger.warning(f"{method}: skip (constructor failed: {last_error})")
                    success = False
                    break
                except Exception as e:
                    last_error = str(e)
                    continue

            if not success:
                if last_error:
                    logger.warning(f"{method}_run{run}: no_cfs_after_{MAX_RETRIES}_attempts | last_error: {last_error}")
                else:
                    logger.warning(f"{method}_run{run}: no_cfs_after_{MAX_RETRIES}_attempts")
    logger.info("Done.")


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        main()
