"""Point temporal analysis. Implements INTERFACE_SPEC §4.1."""
from __future__ import annotations
import numpy as np
import pandas as pd
import xarray as xr


def extract_point_timeseries(
    snow_ds: xr.Dataset,
    em_ds: xr.Dataset,
    net_solar_ds: xr.Dataset,
    x: float,
    y: float,
    variables: list | None = None,
) -> pd.DataFrame:
    """Extract daily time series for a single nearest-neighbor pixel.

    Computes net_LW = net_rad - net_solar and appends as a column.
    Returns DataFrame with DatetimeIndex and one column per variable.
    """
    sel_kw = dict(x=x, y=y, method='nearest')
    drop_coords = ['x', 'y', 'spatial_ref']

    frames = []
    for ds in (snow_ds, em_ds, net_solar_ds):
        pt = ds.sel(**sel_kw).compute()
        df = pt.to_dataframe().drop(columns=drop_coords, errors='ignore')
        frames.append(df)

    df = pd.concat(frames, axis=1)
    df.index = pd.to_datetime(df.index)

    if 'net_rad' in df.columns and 'net_solar' in df.columns:
        df['net_LW'] = df['net_rad'] - df['net_solar']

    if variables is not None:
        df = df[[v for v in variables if v in df.columns]]

    return df
