"""Validation against reference data. Implements INTERFACE_SPEC §4.4."""
from __future__ import annotations
import numpy as np
import pandas as pd
import xarray as xr

from isnobal_eval.metrics import compute_metrics  # re-exported per spec


def compare_snotel(
    snow_ds: xr.Dataset,
    snotel_df: pd.DataFrame,
    variable: str,
    site_ids: list | None = None,
) -> pd.DataFrame:
    """Pair modeled and observed values at SNOTEL locations.

    Returns DataFrame: columns=[site_id, date, modeled, observed].
    """
    if snotel_df.empty:
        return pd.DataFrame(columns=['site_id', 'date', 'modeled', 'observed'])

    sites = snotel_df['site_name'].unique()
    if site_ids is not None:
        sites = [s for s in sites if s in site_ids]

    records = []
    for site in sites:
        site_rows = snotel_df[snotel_df['site_name'] == site]
        x_utm = site_rows['x_utm'].iloc[0]
        y_utm = site_rows['y_utm'].iloc[0]

        if np.isnan(x_utm) or np.isnan(y_utm):
            continue

        modeled_ts = (
            snow_ds[variable]
            .sel(x=x_utm, y=y_utm, method='nearest')
            .compute()
            .to_series()
        )
        modeled_ts.index = pd.to_datetime(modeled_ts.index).normalize()

        if variable not in site_rows.columns:
            continue
        obs_ts = site_rows[variable].copy()
        obs_ts.index = pd.to_datetime(obs_ts.index).normalize()

        aligned = pd.DataFrame({'modeled': modeled_ts, 'observed': obs_ts}).dropna()
        aligned['site_id'] = site
        aligned['date'] = aligned.index
        records.append(aligned[['site_id', 'date', 'modeled', 'observed']])

    if not records:
        return pd.DataFrame(columns=['site_id', 'date', 'modeled', 'observed'])

    return pd.concat(records, ignore_index=True)


__all__ = ['compare_snotel', 'compute_metrics']
