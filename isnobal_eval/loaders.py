"""Data loaders for isnobal_eval. Implements INTERFACE_SPEC §3."""
from __future__ import annotations
import importlib.util
import pathlib
import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd

# Prefer the lightweight standalone snotel_io; fall back to processing.py on CHPC.
# processing.py chains through helpers.py which has heavy module-level imports
# (rioxarray, rasterio, matplotlib-scalebar) unused by any eval function.
import sys as _sys
try:
    _eval_dir = str(pathlib.Path(__file__).parents[1] / 'eval')
    if _eval_dir not in _sys.path:
        _sys.path.insert(0, _eval_dir)
    from snotel_io import locate_snotel_in_poly, get_snotel as _get_snotel
    _USE_SNOTEL_IO = True
except ImportError:
    _env_path = '/uufs/chpc.utah.edu/common/home/u6058223/git_dirs/env/'
    if _env_path not in _sys.path:
        _sys.path.insert(0, _env_path)
    _proc_path = pathlib.Path(__file__).parents[1] / 'scripts' / 'processing.py'
    _spec = importlib.util.spec_from_file_location('processing', _proc_path)
    proc = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(proc)
    locate_snotel_in_poly = proc.locate_snotel_in_poly
    _get_snotel = proc.get_snotel
    _USE_SNOTEL_IO = False


def load_snow(config: dict) -> xr.Dataset:
    """Open snow zarr store lazily. Dims: (time, y, x)."""
    return xr.open_zarr(config['paths']['snow_zarr'], consolidated=True)


def load_em(config: dict) -> xr.Dataset:
    """Open energy model zarr store lazily. Dims: (time, y, x)."""
    return xr.open_zarr(config['paths']['em_zarr'], consolidated=True)


def load_net_solar(config: dict) -> xr.Dataset:
    """Open net solar zarr store lazily. Dims: (time, y, x)."""
    return xr.open_zarr(config['paths']['net_solar_zarr'], consolidated=True)


def load_topo(config: dict) -> xr.Dataset:
    """Open topo.nc. Dims: (y, x). Aspect absent — call compute_aspect first."""
    return xr.open_dataset(config['paths']['topo_nc'])


def compute_aspect(topo_ds: xr.Dataset) -> xr.Dataset:
    """Add 'aspect' variable (degrees, 0=North clockwise) derived from dem."""
    dem = topo_ds['dem'].values
    y = topo_ds['dem'].y.values
    x = topo_ds['dem'].x.values
    dz_dy, dz_dx = np.gradient(dem, y, x)
    aspect = np.degrees(np.arctan2(dz_dx, -dz_dy)) % 360
    return topo_ds.assign(aspect=(('y', 'x'), aspect.astype(np.float32)))


def load_basin_poly(config: dict) -> gpd.GeoDataFrame:
    """Load basin polygon reprojected to config EPSG."""
    return gpd.read_file(config['paths']['basin_poly']).to_crs(f"epsg:{config['epsg']}")


def load_snotel(config: dict, site_ids: list | None = None) -> pd.DataFrame:
    """Fetch daily SNOTEL data for sites inside the basin polygon.

    Returns flat DataFrame: DatetimeIndex, columns [<variable>, site_name, x_utm, y_utm].
    Unit conversion per INTERFACE_SPEC §2.5.
    """
    snotel_cfg = config['snotel']
    snowvar = snotel_cfg.get('snowvar', 'SNOWDEPTH')
    buffer_m = int(snotel_cfg.get('buffer_m', 200))
    wy = config['water_year']
    epsg = config['epsg']

    sites_gdf = locate_snotel_in_poly(
        config['paths']['basin_poly'],
        config['paths']['snotel_sites_geojson'],
        buffer=buffer_m,
        epsg=epsg,
    )

    if site_ids is not None:
        sites_gdf = sites_gdf[sites_gdf['site_name'].isin(site_ids)]

    if sites_gdf.empty:
        return pd.DataFrame()

    site_nums  = sites_gdf['site_num'].tolist()
    site_names = sites_gdf['site_name'].tolist()
    states     = sites_gdf['state'].tolist()

    coord_gdf, dfs = _get_snotel(
        sitenum=site_nums,
        sitename=site_names,
        ST=states,
        WY=wy,
        epsg=epsg,
        snowvar=snowvar,
    )

    name_to_xy = {
        name: (coord_gdf.iloc[i].geometry.x, coord_gdf.iloc[i].geometry.y)
        for i, name in enumerate(site_names)
        if i < len(coord_gdf)
    }

    frames = []
    for site_name, df in dfs.items():
        if snowvar == 'SNOWDEPTH' and 'SNOWDEPTH_m' in df.columns:
            out = df[['SNOWDEPTH_m']].rename(columns={'SNOWDEPTH_m': 'thickness'})
        elif snowvar == 'SWE' and 'SWE_m' in df.columns:
            out = df[['SWE_m']].copy()
            out['specific_mass'] = out['SWE_m'] * 1000.0
            out = out[['specific_mass']]
        else:
            continue
        out['site_name'] = site_name
        x, y = name_to_xy.get(site_name, (np.nan, np.nan))
        out['x_utm'] = x
        out['y_utm'] = y
        frames.append(out)

    if not frames:
        return pd.DataFrame()

    return pd.concat(frames).sort_index()
