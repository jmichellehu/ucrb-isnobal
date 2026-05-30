"""Evaluation metrics. Implements INTERFACE_SPEC §5."""
from __future__ import annotations
import numpy as np
import pandas as pd


def compute_metrics(
    modeled: np.ndarray | pd.Series,
    observed: np.ndarray | pd.Series,
) -> dict:
    """Standard evaluation metrics with NaN-pair removal.

    Uses np.std(ddof=0) and np.mean per INTERFACE_SPEC §5 to match model_eval.ipynb.
    """
    m = np.asarray(modeled, dtype=float)
    o = np.asarray(observed, dtype=float)
    valid = np.isfinite(m) & np.isfinite(o)
    m, o = m[valid], o[valid]

    if len(m) < 2:
        return dict(n=int(len(m)), r=np.nan, r2=np.nan, rmse=np.nan,
                    nrmse_range=np.nan, nrmse_mean=np.nan, mae=np.nan, kge=np.nan)

    r = float(np.corrcoef(o, m)[0, 1])
    rmse = float(np.sqrt(np.mean((m - o) ** 2)))
    obs_range = float(o.max() - o.min())
    obs_mean = float(np.mean(o))
    mae = float(np.mean(np.abs(m - o)))
    alpha = float(np.std(m) / np.std(o)) if np.std(o) > 0 else np.nan
    beta  = float(np.mean(m) / np.mean(o)) if np.mean(o) != 0 else np.nan
    kge = float(1 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2))

    return dict(
        n=int(len(m)),
        r=r,
        r2=r ** 2,
        rmse=rmse,
        nrmse_range=rmse / obs_range if obs_range > 0 else np.nan,
        nrmse_mean=rmse / obs_mean if obs_mean != 0 else np.nan,
        mae=mae,
        kge=kge,
    )
