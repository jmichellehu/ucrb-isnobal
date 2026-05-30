"""Conditional time series analysis. Implements INTERFACE_SPEC §4.3."""
from __future__ import annotations
import numpy as np
import pandas as pd
import xarray as xr


def build_terrain_mask(
    topo_ds: xr.Dataset,
    elev_range: tuple | None = None,
    aspect_range: tuple | None = None,
    slope_max: float | None = None,
    veg_classes: list | None = None,
) -> xr.DataArray:
    """Build 2-D boolean mask over (y, x). All active filters are ANDed."""
    mask = np.ones(topo_ds['dem'].shape, dtype=bool)

    if 'mask' in topo_ds:
        mask &= topo_ds['mask'].values.astype(bool)

    if elev_range is not None:
        dem = topo_ds['dem'].values
        mask &= (dem >= elev_range[0]) & (dem <= elev_range[1])

    if aspect_range is not None and 'aspect' in topo_ds:
        asp = topo_ds['aspect'].values
        lo, hi = aspect_range
        # Circular wrap: lo > hi means the range crosses 0° (e.g. 315–45 = North)
        if lo > hi:
            mask &= (asp >= lo) | (asp <= hi)
        else:
            mask &= (asp >= lo) & (asp <= hi)

    if slope_max is not None and 'slope' in topo_ds:
        mask &= topo_ds['slope'].values <= slope_max

    if veg_classes is not None and 'veg_type' in topo_ds:
        veg = topo_ds['veg_type'].values.astype(int)
        veg_mask = np.zeros_like(mask)
        for vc in veg_classes:
            veg_mask |= (veg == vc)
        mask &= veg_mask

    return xr.DataArray(mask, dims=['y', 'x'], coords={'y': topo_ds.y, 'x': topo_ds.x})


def compute_masked_timeseries(
    ds: xr.Dataset,
    variable: str,
    mask: xr.DataArray,
) -> pd.DataFrame:
    """Spatial summary stats over masked pixels for each time step.

    Returns DataFrame: DatetimeIndex, columns=[mean, std, q25, q75].
    """
    mask_np = mask.values.astype(bool)
    var_arr = ds[variable].values  # (time, y, x)
    masked  = var_arr[:, mask_np]  # (time, n_pixels)

    return pd.DataFrame(
        {
            'mean': np.nanmean(masked, axis=1),
            'std':  np.nanstd(masked, axis=1),
            'q25':  np.nanpercentile(masked, 25, axis=1),
            'q75':  np.nanpercentile(masked, 75, axis=1),
        },
        index=pd.to_datetime(ds.time.values),
    )
