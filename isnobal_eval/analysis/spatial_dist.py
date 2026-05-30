"""Spatial distribution analysis. Implements INTERFACE_SPEC §4.2."""
from __future__ import annotations
import numpy as np
import pandas as pd
import xarray as xr


def compute_terrain_distribution(
    ds: xr.Dataset,
    variable: str,
    topo_ds: xr.Dataset,
    stratify_by: str = 'elevation',
    bins: list | None = None,
) -> pd.DataFrame:
    """Variable mean per terrain bin at each time step.

    Returns DataFrame: index=DatetimeIndex, columns=bin labels, values=mean.
    """
    var_arr = ds[variable].values  # (time, y, x)
    basin_mask = (
        topo_ds['mask'].values.astype(bool)
        if 'mask' in topo_ds
        else np.ones(var_arr.shape[1:], dtype=bool)
    )

    topo_map = {
        'elevation': 'dem',
        'aspect':    'aspect',
        'slope':     'slope',
        'veg_class': 'veg_type',
    }
    strat_arr = topo_ds[topo_map[stratify_by]].values
    time_index = pd.to_datetime(ds.time.values)

    # Veg class: categorical bins
    if stratify_by == 'veg_class' and bins is None:
        codes = np.unique(strat_arr[basin_mask & np.isfinite(strat_arr)].astype(int))
        cols = {}
        for code in codes:
            px = basin_mask & (strat_arr.astype(int) == code)
            cols[str(code)] = np.nanmean(var_arr[:, px], axis=1) if px.sum() else np.full(len(time_index), np.nan)
        return pd.DataFrame(cols, index=time_index)

    # Aspect: fixed cardinal bins with circular wrap
    if stratify_by == 'aspect' and bins is None:
        bin_labels = ['North', 'East', 'South', 'West']
        bin_masks = [
            basin_mask & ((strat_arr >= 315) | (strat_arr < 45)),
            basin_mask & (strat_arr >= 45)  & (strat_arr < 135),
            basin_mask & (strat_arr >= 135) & (strat_arr < 225),
            basin_mask & (strat_arr >= 225) & (strat_arr < 315),
        ]
        cols = {}
        for label, px in zip(bin_labels, bin_masks):
            cols[label] = np.nanmean(var_arr[:, px], axis=1) if px.sum() else np.full(len(time_index), np.nan)
        return pd.DataFrame(cols, index=time_index)

    # Numeric bins (elevation, slope)
    if bins is None:
        if stratify_by == 'elevation':
            valid = strat_arr[basin_mask & np.isfinite(strat_arr)]
            edges = np.linspace(valid.min(), valid.max(), 11)
        else:  # slope
            edges = [0, 10, 20, 30, 40, 50, 60, 90]
    else:
        edges = bins

    cols = {}
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        label = f'{int(lo)}_{int(hi)}'
        if i == len(edges) - 2:
            px = basin_mask & (strat_arr >= lo) & (strat_arr <= hi)
        else:
            px = basin_mask & (strat_arr >= lo) & (strat_arr < hi)
        cols[label] = np.nanmean(var_arr[:, px], axis=1) if px.sum() else np.full(len(time_index), np.nan)

    return pd.DataFrame(cols, index=time_index)
