"""
Standalone evaluation utilities for iSnobal model output.
Implements the interface defined in eval/INTERFACE_SPEC.md.
Migration path: function signatures are identical to isnobal_eval package;
swap `from eval_utils import X` for `from isnobal_eval.module import X`.
"""
import sys
import pathlib
import yaml
import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd

# processing.py requires its own env/ helpers; add both paths before importing
_REPO = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(_REPO / 'scripts'))
sys.path.insert(0, '/uufs/chpc.utah.edu/common/home/u6058223/git_dirs/env/')
import processing as proc


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def load_config(path: str) -> dict:
    """Load a YAML config file and return it as a dict."""
    with open(path) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_snow(config: dict) -> xr.Dataset:
    """Open the snow zarr store lazily. Returns Dataset with dims (time, y, x)."""
    return xr.open_zarr(config['paths']['snow_zarr'], consolidated=True)


def load_em(config: dict) -> xr.Dataset:
    """Open the energy model zarr store lazily. Returns Dataset with dims (time, y, x)."""
    return xr.open_zarr(config['paths']['em_zarr'], consolidated=True)


def load_net_solar(config: dict) -> xr.Dataset:
    """Open the net solar zarr store lazily. Returns Dataset with dims (time, y, x)."""
    return xr.open_zarr(config['paths']['net_solar_zarr'], consolidated=True)


def load_topo(config: dict) -> xr.Dataset:
    """Open topo.nc. Returns Dataset with dims (y, x). Aspect not present; call compute_aspect."""
    return xr.open_dataset(config['paths']['topo_nc'])


def compute_aspect(topo_ds: xr.Dataset) -> xr.Dataset:
    """Add 'aspect' variable (degrees, 0=North clockwise) derived from topo_ds['dem']."""
    dem = topo_ds['dem'].values
    y = topo_ds['dem'].y.values
    x = topo_ds['dem'].x.values
    # np.gradient with coordinate arrays gives true d/dy and d/dx
    dz_dy, dz_dx = np.gradient(dem, y, x)
    aspect = np.degrees(np.arctan2(dz_dx, -dz_dy)) % 360
    return topo_ds.assign(aspect=(('y', 'x'), aspect.astype(np.float32)))


def load_basin_poly(config: dict) -> gpd.GeoDataFrame:
    """Load basin polygon and reproject to config EPSG."""
    return gpd.read_file(config['paths']['basin_poly']).to_crs(f"epsg:{config['epsg']}")


def load_snotel(config: dict, site_ids: list | None = None) -> pd.DataFrame:
    """Fetch daily SNOTEL data for sites in the basin polygon.

    Returns a DataFrame with a DatetimeIndex and columns
    [site_name, <variable>, x_utm, y_utm]. site_name is repeated per row
    so the caller can groupby('site_name'). Unit conversion per INTERFACE_SPEC §2.5.
    """
    snotel_cfg = config['snotel']
    snowvar = snotel_cfg.get('snowvar', 'SNOWDEPTH')
    buffer_m = int(snotel_cfg.get('buffer_m', 200))
    wy = config['water_year']
    epsg = config['epsg']

    sites_gdf = proc.locate_snotel_in_poly(
        config['paths']['basin_poly'],
        config['paths']['snotel_sites_geojson'],
        buffer=buffer_m,
        epsg=epsg,
    )

    if site_ids is not None:
        sites_gdf = sites_gdf[sites_gdf['site_name'].isin(site_ids)]

    if sites_gdf.empty:
        return pd.DataFrame()

    site_nums = sites_gdf['site_num'].tolist()
    site_names = sites_gdf['site_name'].tolist()
    states = sites_gdf['state'].tolist()

    coord_gdf, dfs = proc.get_snotel(
        sitenum=site_nums,
        sitename=site_names,
        ST=states,
        WY=wy,
        epsg=epsg,
        snowvar=snowvar,
    )

    # Map site_name → UTM coords from coord_gdf (order matches site iteration)
    name_to_xy = {
        name: (coord_gdf.iloc[i].geometry.x, coord_gdf.iloc[i].geometry.y)
        for i, name in enumerate(site_names)
        if i < len(coord_gdf)
    }

    frames = []
    for site_name, df in dfs.items():
        if snowvar == 'SNOWDEPTH' and 'SNOWDEPTH_m' in df.columns:
            out = df[['SNOWDEPTH_m']].copy()
            out.columns = ['thickness']
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


# ---------------------------------------------------------------------------
# Analysis: point_temporal
# ---------------------------------------------------------------------------

def extract_point_timeseries(
    snow_ds: xr.Dataset,
    em_ds: xr.Dataset,
    net_solar_ds: xr.Dataset,
    x: float,
    y: float,
    variables: list | None = None,
) -> pd.DataFrame:
    """Extract daily time series for a single pixel from all three stores.

    Selects nearest pixel to (x, y). Computes net_LW = net_rad - net_solar.
    Returns DataFrame with DatetimeIndex and one column per variable.
    """
    sel_kw = dict(x=x, y=y, method='nearest')
    snow_pt = snow_ds.sel(**sel_kw).compute()
    em_pt   = em_ds.sel(**sel_kw).compute()
    ns_pt   = net_solar_ds.sel(**sel_kw).compute()

    frames = [
        snow_pt.to_dataframe().drop(columns=['x', 'y', 'spatial_ref'], errors='ignore'),
        em_pt.to_dataframe().drop(columns=['x', 'y', 'spatial_ref'], errors='ignore'),
        ns_pt.to_dataframe().drop(columns=['x', 'y', 'spatial_ref'], errors='ignore'),
    ]
    df = pd.concat(frames, axis=1)
    if 'net_rad' in df.columns and 'net_solar' in df.columns:
        df['net_LW'] = df['net_rad'] - df['net_solar']

    if variables is not None:
        df = df[[v for v in variables if v in df.columns]]

    df.index = pd.to_datetime(df.index)
    return df


# ---------------------------------------------------------------------------
# Analysis: spatial_distribution
# ---------------------------------------------------------------------------

def compute_terrain_distribution(
    ds: xr.Dataset,
    variable: str,
    topo_ds: xr.Dataset,
    stratify_by: str = 'elevation',
    bins: list | None = None,
) -> pd.DataFrame:
    """Variable mean per terrain bin at each time step.

    Returns DataFrame: index=time, columns=bin labels, values=mean of variable.
    """
    var_arr = ds[variable].values  # (time, y, x)
    basin_mask = topo_ds['mask'].values.astype(bool) if 'mask' in topo_ds else np.ones(var_arr.shape[1:], dtype=bool)

    topo_map = {'elevation': 'dem', 'aspect': 'aspect', 'slope': 'slope', 'veg_class': 'veg_type'}
    topo_var = topo_map[stratify_by]
    strat_arr = topo_ds[topo_var].values

    if bins is None:
        if stratify_by == 'elevation':
            valid = strat_arr[basin_mask & np.isfinite(strat_arr)]
            edges = np.linspace(valid.min(), valid.max(), 11)
        elif stratify_by == 'aspect':
            edges = [0, 45, 135, 225, 315, 360]
        elif stratify_by == 'slope':
            edges = [0, 10, 20, 30, 40, 50, 60, 90]
        elif stratify_by == 'veg_class':
            edges = None  # handled separately
    else:
        edges = bins

    time_index = pd.to_datetime(ds.time.values)

    if stratify_by == 'veg_class' and edges is None:
        codes = np.unique(strat_arr[basin_mask & np.isfinite(strat_arr)].astype(int))
        cols = {str(c): np.full(len(time_index), np.nan) for c in codes}
        for code in codes:
            px_mask = basin_mask & (strat_arr.astype(int) == code)
            if px_mask.sum() == 0:
                continue
            cols[str(code)] = np.nanmean(var_arr[:, px_mask], axis=1)
        return pd.DataFrame(cols, index=time_index)

    # Aspect: circular bins (N, E, S, W)
    if stratify_by == 'aspect':
        bin_labels = ['North', 'East', 'South', 'West']
        bin_masks = [
            basin_mask & ((strat_arr >= 315) | (strat_arr < 45)),
            basin_mask & (strat_arr >= 45) & (strat_arr < 135),
            basin_mask & (strat_arr >= 135) & (strat_arr < 225),
            basin_mask & (strat_arr >= 225) & (strat_arr < 315),
        ]
        cols = {}
        for label, px_mask in zip(bin_labels, bin_masks):
            if px_mask.sum() == 0:
                cols[label] = np.full(len(time_index), np.nan)
            else:
                cols[label] = np.nanmean(var_arr[:, px_mask], axis=1)
        return pd.DataFrame(cols, index=time_index)

    # Generic numeric bins (elevation, slope)
    cols = {}
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        label = f'{int(lo)}_{int(hi)}'
        px_mask = basin_mask & (strat_arr >= lo) & (strat_arr < hi)
        if i == len(edges) - 2:  # include upper edge in last bin
            px_mask = basin_mask & (strat_arr >= lo) & (strat_arr <= hi)
        if px_mask.sum() == 0:
            cols[label] = np.full(len(time_index), np.nan)
        else:
            cols[label] = np.nanmean(var_arr[:, px_mask], axis=1)
    return pd.DataFrame(cols, index=time_index)


# ---------------------------------------------------------------------------
# Analysis: conditional_timeseries
# ---------------------------------------------------------------------------

def build_terrain_mask(
    topo_ds: xr.Dataset,
    elev_range: tuple | None = None,
    aspect_range: tuple | None = None,
    slope_max: float | None = None,
    veg_classes: list | None = None,
) -> xr.DataArray:
    """Build 2-D boolean mask over (y, x). Pixels must satisfy all active filters."""
    mask = np.ones(topo_ds['dem'].shape, dtype=bool)

    if 'mask' in topo_ds:
        mask &= topo_ds['mask'].values.astype(bool)

    if elev_range is not None:
        dem = topo_ds['dem'].values
        mask &= (dem >= elev_range[0]) & (dem <= elev_range[1])

    if aspect_range is not None and 'aspect' in topo_ds:
        asp = topo_ds['aspect'].values
        lo, hi = aspect_range
        if lo > hi:  # circular wrap (e.g. 315–45)
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

    Returns DataFrame: index=DatetimeIndex, columns=[mean, std, q25, q75].
    """
    mask_np = mask.values.astype(bool)
    var_arr = ds[variable].values  # (time, y, x)
    masked = var_arr[:, mask_np]   # (time, n_pixels)

    return pd.DataFrame(
        {
            'mean': np.nanmean(masked, axis=1),
            'std':  np.nanstd(masked, axis=1),
            'q25':  np.nanpercentile(masked, 25, axis=1),
            'q75':  np.nanpercentile(masked, 75, axis=1),
        },
        index=pd.to_datetime(ds.time.values),
    )


# ---------------------------------------------------------------------------
# Analysis: validation
# ---------------------------------------------------------------------------

def compare_snotel(
    snow_ds: xr.Dataset,
    snotel_df: pd.DataFrame,
    variable: str,
    site_ids: list | None = None,
) -> pd.DataFrame:
    """Pair modeled and observed values at SNOTEL locations.

    Returns DataFrame with columns [site_id, date, modeled, observed].
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

        obs_col = variable  # e.g. 'thickness'
        if obs_col not in site_rows.columns:
            continue
        obs_ts = site_rows[obs_col].copy()
        obs_ts.index = pd.to_datetime(obs_ts.index).normalize()

        aligned = pd.DataFrame({'modeled': modeled_ts, 'observed': obs_ts}).dropna()
        aligned['site_id'] = site
        aligned['date'] = aligned.index
        records.append(aligned[['site_id', 'date', 'modeled', 'observed']])

    if not records:
        return pd.DataFrame(columns=['site_id', 'date', 'modeled', 'observed'])

    return pd.concat(records, ignore_index=True)


def compute_metrics(
    modeled: np.ndarray | pd.Series,
    observed: np.ndarray | pd.Series,
) -> dict:
    """Standard evaluation metrics. NaN pairs dropped before computation.

    Uses np.std(ddof=0) and np.mean per INTERFACE_SPEC §5.
    """
    m = np.asarray(modeled, dtype=float)
    o = np.asarray(observed, dtype=float)
    valid = np.isfinite(m) & np.isfinite(o)
    m, o = m[valid], o[valid]

    if len(m) < 2:
        return dict(r=np.nan, r2=np.nan, rmse=np.nan,
                    nrmse_range=np.nan, nrmse_mean=np.nan, mae=np.nan, kge=np.nan)

    r = np.corrcoef(o, m)[0, 1]
    rmse = np.sqrt(np.mean((m - o) ** 2))
    obs_range = o.max() - o.min()
    obs_mean = np.mean(o)
    mae = np.mean(np.abs(m - o))

    alpha = np.std(m) / np.std(o) if np.std(o) > 0 else np.nan
    beta  = np.mean(m) / np.mean(o) if np.mean(o) != 0 else np.nan
    kge = 1 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)

    return dict(
        n=int(len(m)),
        r=float(r),
        r2=float(r ** 2),
        rmse=float(rmse),
        nrmse_range=float(rmse / obs_range) if obs_range > 0 else np.nan,
        nrmse_mean=float(rmse / obs_mean) if obs_mean != 0 else np.nan,
        mae=float(mae),
        kge=float(kge),
    )
