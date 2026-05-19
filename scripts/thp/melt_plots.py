#!/usr/bin/env python
"""Melt analysis figures for Joint Western and Eastern Snow Conference 2026.

Plots of melt timing based on iSnobal and terrain data.
Basins and water year coverage: Animas, Yampa, Jordan from WY 2022-2024.

Usage
-----
    melt_plots.py animas 2022
    melt_plots.py jordan 2023 -o /path/to/figures --overwrite
"""
import os
import sys
import logging
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize, ListedColormap
import xarray as xr

import geopandas as gpd

sys.path.append('/uufs/chpc.utah.edu/common/home/u6058223/git_dirs/env/')
import helpers as h

LOGGER = logging.getLogger(__name__)

# Energy balance input terms — positive values contribute to melt
EB_TERMS = ['net_rad', 'sensible_heat', 'latent_heat', 'snow_soil', 'precip_advected']

# Aspect analysis constants
N_ASPECT_BINS  = 8
ASPECT_LABELS  = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']


def _compute_aspect_from_dem(dem_da: xr.DataArray) -> xr.DataArray:
    """Compute aspect in degrees [0, 360) clockwise from North using numpy.gradient.

    Cell size is inferred from the x/y coordinate spacing. If the y coordinate
    decreases going down (north at top of array, standard for geographic rasters),
    the row gradient is sign-flipped to obtain the northward slope component.
    """
    dem = dem_da.values.astype(float)

    x_c = dem_da.coords.get('x')
    y_c = dem_da.coords.get('y')
    cell_x = float(abs(x_c[1] - x_c[0])) if x_c is not None and len(x_c) > 1 else 100.0
    cell_y = float(abs(y_c[1] - y_c[0])) if y_c is not None and len(y_c) > 1 else 100.0

    dz_drow, dz_dcol = np.gradient(dem, cell_y, cell_x)

    # If y decreases going down (north at top), flip row gradient to get northward slope
    y_sign    = -1.0 if (y_c is not None and float(y_c[0]) > float(y_c[-1])) else 1.0
    dz_north  = y_sign * dz_drow
    dz_east   = dz_dcol

    aspect_deg = np.degrees(np.arctan2(dz_east, dz_north)) % 360.0
    return xr.DataArray(aspect_deg, dims=dem_da.dims, coords=dem_da.coords,
                        attrs={'units': 'degrees',
                               'long_name': 'Aspect (degrees clockwise from North)'})


_ABBREV_MAP: dict = {
    'net_solar':       'solar',
    'net_LW':          'longwave',
    'net_rad':         'net rad',
    'sensible_heat':   'sensible',
    'latent_heat':     'latent',
    'snow_soil':       'ground',
    'precip_advected': 'advected',
}


def _abbrev(term: str) -> str:
    """Short display label for an EB term variable name.

    Edit _ABBREV_MAP to customise labels without touching call sites.
    Falls back to splitting on '_' for any term not in the map.
    """
    if term in _ABBREV_MAP:
        return _ABBREV_MAP[term]
    parts = term.split('_')
    return parts[1] if (parts[0] == 'net' and len(parts) > 1) else parts[0]


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format='%(asctime)s %(levelname)s %(message)s')


def _figure_exists(out_fn, overwrite: bool) -> bool:
    """Return True if the figure already exists on disk and overwrite is False.

    Use this to skip expensive upstream computations when their output file
    already exists — not just to skip the file write itself.
    """
    if out_fn is None or overwrite:
        return False
    return os.path.exists(out_fn)


def _all_exist(paths, overwrite: bool) -> bool:
    """Return True if every path in `paths` exists on disk and overwrite is False.

    Use this to gate a computation that feeds multiple output files — only skip
    if every downstream output is already present.
    """
    if overwrite:
        return False
    return all(os.path.exists(p) for p in paths)


def _save_figure(fig, out_fn, overwrite=False, dpi=300, dry_run=False) -> None:
    """Save figure to out_fn. No-op if out_fn is None or dry_run is True."""
    if out_fn is None:
        return
    if dry_run:
        LOGGER.info('[dry-run] Would save: %s', out_fn)
        return
    if os.path.exists(out_fn) and not overwrite:
        LOGGER.info('Figure exists, skipping: %s  (pass --overwrite to replace)', out_fn)
        return
    parent = os.path.dirname(out_fn)
    if parent:
        os.makedirs(parent, exist_ok=True)
    fig.savefig(out_fn, dpi=dpi, bbox_inches='tight')
    LOGGER.info('Saved: %s', out_fn)


def time_to_dowy(time_values, wy: int):
    """Convert datetime values to Day of Water Year (DOWY 1 = Oct 1).

    Handles both numpy datetime64 arrays and xarray DataArrays.
    NaT values in DataArrays are propagated as NaN.

    Parameters
    ----------
    time_values : numpy datetime64 array or xr.DataArray of datetime64
    wy          : int — water year (WY 2022 starts Oct 1 2021)
    """
    wy_start = np.datetime64(f'{wy - 1}-10-01', 'ns')
    if isinstance(time_values, xr.DataArray):
        dowy = (time_values - wy_start) / np.timedelta64(1, 'D') + 1
        return dowy.where(time_values.notnull())
    return ((time_values.astype('datetime64[ns]') - wy_start)
            / np.timedelta64(1, 'D') + 1).astype(int)


def get_dowy_month_ticks(wy: int):
    """Return (ticks, labels) for the first of each month in the water year.

    Tick positions are DOWY values. Handles leap years correctly by computing
    from actual dates rather than fixed offsets.

    Parameters
    ----------
    wy : int — water year

    Returns
    -------
    ticks  : list of int DOWY values for each month start
    labels : list of str month abbreviations ('Oct', 'Nov', ...)
    """
    wy_start = pd.Timestamp(f'{wy - 1}-10-01')
    month_starts = pd.date_range(start=wy_start, end=f'{wy}-09-01', freq='MS')
    ticks  = [(d - wy_start).days + 1 for d in month_starts]
    labels = [d.strftime('%b') for d in month_starts]
    return ticks, labels


def clip_to_polygon(ds: xr.Dataset,
                    polygon_fn: str,
                    drop: bool = True) -> xr.Dataset:
    """
    Clip a dataset to a polygon before analysis
    """
    if not Path(polygon_fn).exists():
        raise FileNotFoundError(f"Polygon file not found: {polygon_fn}")

    poly = gpd.read_file(polygon_fn)
    poly = poly[poly.geometry.notna() & ~poly.geometry.is_empty]
    if poly.empty:
        raise ValueError(f"No valid polygon geometries: {polygon_fn}")
    if poly.crs is None:
        raise ValueError(f"Polygon has no CRS: {polygon_fn}")
    LOGGER.info('Polygon CRS: %s', poly.crs)

    ds = ds.rio.set_spatial_dims(x_dim="x", y_dim="y", inplace=False)
    if ds.rio.crs is None:
        LOGGER.warning("Dataset CRS missing; assuming matches polygon CRS %s", poly.crs)
        ds = ds.rio.write_crs(poly.crs, inplace=False)
    else:
        poly = poly.to_crs(ds.rio.crs)
    clipped = ds.rio.clip(poly.geometry, poly.crs, drop=drop)

    if clipped.sizes.get("x", 0) == 0 or clipped.sizes.get("y", 0) == 0:
        raise ValueError(f"Clip is empty for polygon: {polygon_fn}")

    return clipped


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data(basin: str, wy: int, workdir: str, script_dir: str):
    """Load em and snow Zarr stores and terrain dataset for a basin + water year.

    Returns
    -------
    em_ds, snow_ds, terrain_ds : xr.Dataset
    """
    zarr_dir = os.path.join(workdir, 'zarr_stores')
    em_ds   = xr.open_zarr(f'{zarr_dir}/{basin}_unified_em_wy{wy}.zarr')
    snow_ds = xr.open_zarr(f'{zarr_dir}/{basin}_unified_snow_wy{wy}.zarr')

    terrain_fn = h.fn_list(script_dir, f'{basin}_scripts/topo.nc')[0]
    _terrain_keep = {'dem', 'aspect'}
    with xr.open_dataset(terrain_fn) as _ds:
        drop_vars = [v for v in _ds.data_vars if v not in _terrain_keep]
    terrain_ds = xr.open_dataset(terrain_fn, drop_variables=drop_vars)

    if 'aspect' not in terrain_ds.data_vars:
        LOGGER.info('No aspect variable in topo.nc — computing from DEM')
        terrain_ds = terrain_ds.assign(aspect=_compute_aspect_from_dem(terrain_ds['dem']))

    LOGGER.info('Loaded data for %s WY%s', basin, wy)
    return em_ds, snow_ds, terrain_ds


# ---------------------------------------------------------------------------
# Melt analysis functions
# ---------------------------------------------------------------------------

def get_melt_init(ds: xr.Dataset) -> xr.DataArray:
    """Return (x, y) DataArray of melt initiation dates.

    Melt initiation: first date where SWI > 0, snowmelt > 0, and
    cold_content == 0 for 7 consecutive days. Pixels with no qualifying
    period return NaT.
    """
    init_cond = ((ds['SWI'] > 0) & (ds['snowmelt'] > 0) & (ds['cold_content'] == 0)).compute()
    init_cond = init_cond.rolling(time=7, min_periods=7).min().shift(time=-6, fill_value=0).astype(bool)
    first_idx = init_cond.argmax(dim='time')
    has_melt  = init_cond.any(dim='time')
    return ds['time'].isel(time=first_idx).where(has_melt)


def get_meltout(ds: xr.Dataset, melt_timing: xr.DataArray) -> xr.DataArray:
    """Return (x, y) DataArray of meltout dates.

    Meltout: first date after melt initiation where snowmelt == 0,
    sum_EB == 0, and cold_content == 0. Pixels with no qualifying date
    return NaT.
    """
    meltout_cond = (
        (ds['sum_EB'] == 0) & (ds['snowmelt'] == 0) & (ds['cold_content'] == 0)
    ).compute()
    meltout_cond = meltout_cond & (meltout_cond.time > melt_timing)
    first_idx   = meltout_cond.argmax(dim='time')
    has_meltout = meltout_cond.any(dim='time')
    return ds['time'].isel(time=first_idx).where(has_meltout)


# ---------------------------------------------------------------------------
# Elevation matrix processing
# ---------------------------------------------------------------------------

def compute_elevation_melt_matrix(melt_season_ds: xr.Dataset,
                                   terrain_ds: xr.Dataset,
                                   elev_bins: np.ndarray):
    """Build a (time, n_elev_bins) pixel-count matrix of melt season activity.

    Uses one-hot encoding of elevation bins and matrix multiplication to
    count melting pixels per bin per day without Python loops.

    Returns
    -------
    elev_flat         : (n_pixels,) per-pixel DEM values, including NaN/out-of-range
    melt_count_matrix : (n_time, n_bins) melting pixel counts per bin per day
    one_hot           : (n_valid_pixels, n_bins) routing matrix
    valid             : (n_pixels,) boolean mask — True for pixels with finite
                        elevation that fall within elev_bins range
    elev_bin_idx_valid: (n_valid_pixels,) elevation bin index per valid pixel
    """
    elev_flat   = terrain_ds['dem'].values.ravel()
    n_time      = len(melt_season_ds['time'])
    during_flat = melt_season_ds['during_melt_season'].values.reshape(n_time, -1)

    elev_bin_idx = np.digitize(elev_flat, elev_bins) - 1
    n_bins       = len(elev_bins) - 1
    valid        = np.isfinite(elev_flat) & (elev_bin_idx >= 0) & (elev_bin_idx < n_bins)

    one_hot = np.zeros((valid.sum(), n_bins))
    one_hot[np.arange(valid.sum()), elev_bin_idx[valid]] = 1
    melt_count_matrix = during_flat[:, valid].astype(float) @ one_hot  # (n_time, n_bins)

    LOGGER.debug('Elevation matrix shape: %s  n_valid_pixels: %d',
                 melt_count_matrix.shape, valid.sum())
    return elev_flat, melt_count_matrix, one_hot, valid, elev_bin_idx[valid]

def normalize_melt_matrix(count_matrix: np.ndarray, one_hot: np.ndarray) -> np.ndarray:
    """Normalize melt count matrix by elevation bin to remove band size bias.

    Returns fraction of pixels melting per bin per time step (0–1 range).
    """
    normalized = count_matrix.copy()
    normalized /= one_hot.sum(axis=0)  # divide each column by pixel count per bin
    return normalized


def compute_variable_elevation_matrix(em_ds: xr.Dataset,
                                       var_name: str,
                                       melt_cond: xr.DataArray,
                                       valid: np.ndarray,
                                       one_hot: np.ndarray,
                                       aggregation: str = 'sum',
                                       melt_count_matrix: np.ndarray = None
                                       ) -> np.ndarray:
    """Aggregate a variable per (time, elev_bin) on melt days.

    Non-melt pixels/days are zeroed before aggregation so they contribute
    nothing to the bin totals.

    Parameters
    ----------
    var_name           : em_ds variable to aggregate (e.g. 'SWI', 'snowmelt')
    melt_cond          : (time, y, x) boolean mask — True on melt days/pixels
    valid              : 1D boolean mask selecting pixels with valid elevation
    one_hot            : (n_valid_pixels, n_bins) routing matrix from
                         compute_elevation_melt_matrix
    aggregation        : 'sum'         — total per bin per day (default)
                         'mean_all'    — mean across all pixels in bin
                         'mean_melt'   — mean across actively melting pixels only
                         'median_all'  — median across all pixels (zeros included)
                         'median_melt' — median across actively melting pixels only
    melt_count_matrix  : (n_time, n_bins) required when aggregation='mean_melt'

    Returns
    -------
    (n_time, n_bins) float array
    """
    _valid_aggs = {'sum', 'mean_all', 'mean_melt', 'median_all', 'median_melt'}
    if aggregation not in _valid_aggs:
        raise ValueError(f"aggregation must be one of {_valid_aggs}, got '{aggregation}'")
    if aggregation == 'mean_melt' and melt_count_matrix is None:
        raise ValueError("aggregation='mean_melt' requires melt_count_matrix")

    # .where(melt_cond, 0) zeros non-melt pixels but keeps NaN where the source
    # variable itself is NaN (e.g. no-data edge pixels with melt_cond=True).
    # nan_to_num converts those to 0 so they contribute nothing to the sum —
    # consistent with how nanmedian silently excludes them in the median paths.
    var_flat  = np.nan_to_num(
        em_ds[var_name].where(melt_cond, 0).values.reshape(len(em_ds['time']), -1),
        nan=0.0,
    )                                                        # (n_time, n_pixels)
    var_valid = var_flat[:, valid]                           # (n_time, n_valid_pixels)

    if aggregation == 'sum':
        return var_valid @ one_hot

    if aggregation == 'mean_all':
        pixels_per_bin = one_hot.sum(axis=0)
        result = var_valid @ one_hot
        return np.where(pixels_per_bin > 0, result / pixels_per_bin, np.nan)

    if aggregation == 'mean_melt':
        result = var_valid @ one_hot
        return np.where(melt_count_matrix > 0, result / melt_count_matrix, np.nan)

    # Median options — bin loop; derive bin membership from one_hot
    elev_bin_idx_valid = np.argmax(one_hot, axis=1)          # (n_valid_pixels,)
    n_time, n_bins = var_valid.shape[0], one_hot.shape[1]
    result = np.full((n_time, n_bins), np.nan)

    for b in range(n_bins):
        mask = (elev_bin_idx_valid == b)
        if not mask.any():
            continue
        col = var_valid[:, mask]                             # (n_time, n_pixels_in_bin)
        if aggregation == 'median_melt':
            col = np.where(col > 0, col, np.nan)            # exclude non-melt pixels
        result[:, b] = np.nanmedian(col, axis=1)

    return result


def plot_variable_elevation_heatmap(var_matrix: np.ndarray,
                                     var_name: str,
                                     dowy: np.ndarray,
                                     elev_bins: np.ndarray,
                                     wy: int,
                                     basin: str,
                                     units: str = 'mm',
                                     pixel_res_m: float = 100,
                                     aggregation: str = 'sum',
                                     pixels_per_bin: np.ndarray = None,
                                     cmap='YlGnBu',
                                     vmin=None, vmax=None,
                                     xlim=(1, 365),
                                     figsize=(7, 3),
                                     tick_fontsize: int = 8,
                                     specify_ax=None, annotate=True,
                                     out_fn=None, overwrite=False,
                                     dry_run=False):
    """Elevation × DOWY heatmap for any aggregated em.nc variable.

    Parameters
    ----------
    var_matrix     : (n_time, n_bins) array from compute_variable_elevation_matrix
    units          : 'mm'  — raw (no conversion, assumes iSnobal mm output)
                     'm'   — convert mm → m (divide by 1000)
                     'taf' — total volume in thousands of acre-feet; meaningful
                             only with aggregation='sum'
    pixel_res_m    : pixel resolution in meters (default 100 m)
    aggregation    : used for the colorbar label only
    pixels_per_bin : (n_bins,) pixel counts from one_hot.sum(axis=0).
                     When provided: bins with 0 pixels → transparent,
                     bins with pixels but NaN value → black.
                     When None: all-NaN bins get grey (absent from basin),
                     individual NaN cells in active bins get black.
    tick_fontsize  : font size for x/y tick labels and axis labels
    specify_ax     : (fig, ax) tuple to plot into an existing axes instead of
                     creating a new figure — use this for megafigure subplots.
                     The colorbar is omitted when specify_ax is provided.
    """
    if dry_run:
        LOGGER.info('[dry-run] Skipping variable heatmap: %s  ->  %s', var_name, out_fn)
        return None, None

    # Unit conversion
    pixel_area_m2 = pixel_res_m ** 2
    # TAF conversion chain (input = mm summed over pixels in a bin):
    #   mm → m   : / 1000
    #   × area   : × pixel_area_m2  →  m³  (same area per pixel, factors out of sum)
    #   m³ → AF  : × 0.000810714   (1 AF = 1233.48 m³)
    #   AF → TAF : / 1000
    #   combined : × pixel_area_m2 × 0.000810714 / 1_000_000
    _unit_scale = {
        'mm':          (1.0,                                       'mm'),
        'm':           (1e-3,                                      'm'),
        'taf':         (pixel_area_m2 * 0.000810714 / 1_000_000,  'TAF'),
        'precomputed': (1.0,                                       'TAF'),  # data already in TAF
    }
    if units not in _unit_scale:
        raise ValueError(f"units must be one of {list(_unit_scale)}, got '{units}'")

    scale, unit_label = _unit_scale[units]
    plot_matrix = var_matrix * scale

    # Colorbar label reflects both units and aggregation mode
    _agg_labels = {
        'sum':         'total',
        'mean_all':    'mean per pixel',
        'mean_melt':   'mean per melting pixel',
        'median_all':  'median per pixel',
        'median_melt': 'median per melting pixel',
    }
    agg_label  = _agg_labels.get(aggregation, aggregation)
    cbar_label = f'{var_name} — {agg_label} ({unit_label})'

    bin_step = float(elev_bins[1] - elev_bins[0])
    ticks, labels = get_dowy_month_ticks(wy)

    # Build RGBA array so NaN cells can be coloured explicitly
    _vmin = vmin if vmin is not None else np.nanpercentile(plot_matrix, 2)
    _vmax = vmax if vmax is not None else np.nanpercentile(plot_matrix, 98)
    norm_obj = Normalize(vmin=_vmin, vmax=_vmax)
    rgba     = plt.get_cmap(cmap)(norm_obj(plot_matrix.T))    # (n_bins, n_time, 4)

    # NaN handling — mirrors plot_eb_term_heatmap:
    #   pixels_per_bin == 0  → elevation range absent from basin  → transparent
    #   pixels exist but NaN → no variable value that cell         → black
    nan_cells = np.isnan(plot_matrix.T)
    if pixels_per_bin is not None:
        empty_bins = (pixels_per_bin < 0.5)              # float-safe: 0 pixels
        rgba[nan_cells & (~empty_bins[:, np.newaxis])] = [0.0, 0.0, 0.0, 1.0]
        rgba[empty_bins] = [0.5, 0.5, 0.5, 0.8]         # grey across full row
    else:
        # Without pixels_per_bin, detect absent elevation ranges as bins that
        # are all-NaN or all-zero across every time step — render grey.
        absent_bins = (np.all(np.isnan(plot_matrix.T), axis=1) |
                       np.all(plot_matrix.T == 0, axis=1))        # (n_bins,)
        rgba[nan_cells & (~absent_bins[:, np.newaxis])] = [0.0, 0.0, 0.0, 1.0]
        rgba[absent_bins] = [0.5, 0.5, 0.5, 0.8]

    if specify_ax is not None:
        fig, ax = specify_ax
    else:
        fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(rgba, aspect='auto', origin='lower',
              extent=[dowy[0], dowy[-1], elev_bins[0], elev_bins[-1]])

    ax.set_xticks(ticks)
    ax.set_xticklabels(labels, fontsize=tick_fontsize)
    ax.set_xlim(*xlim)
    ax.set_ylim(elev_bins[0], elev_bins[-1] - bin_step / 2)
    ax.set_ylabel('Elevation (m)', fontsize=tick_fontsize)
    ax.tick_params(axis='both', labelsize=tick_fontsize)
    if annotate:
        ax.annotate(f'{basin.capitalize()} WY {wy}', xy=(0.02, 1.2), xycoords='axes fraction',
                    color='black', fontstyle='oblique',
                    ha='left', va='top', fontsize=max(tick_fontsize, 9))

    # Skip standalone colorbar when embedded in a megafigure (caller manages it)
    if specify_ax is None:
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm_obj)
        sm.set_array([])
        fig.colorbar(sm, ax=ax, orientation='horizontal', location='top',
                     fraction=0.05, pad=0.03, label=cbar_label)
        plt.tight_layout()
        _save_figure(fig, out_fn, overwrite)

    return fig, ax

# ---------------------------------------------------------------------------
# EB proportion functions
# ---------------------------------------------------------------------------

def compute_eb_proportions(em_ds: xr.Dataset, melt_cond: xr.DataArray,
                            eb_terms: list = None) -> xr.Dataset:
    """Compute fractional contribution of each positive EB term on melt days.

    For each pixel and melt day, the fraction for term k is:
        frac_k = term_k / sum_of_positive_terms   (where term_k > 0)

    Non-melt days and negative terms are NaN. Result is saved to Zarr for
    later use; the full spatial grid is preserved for flexibility.

    Parameters
    ----------
    eb_terms : list of variable names to include. Defaults to module-level
               EB_TERMS. Pass a custom list when net_rad has been split into
               net_solar and net_LW (both must already be present in em_ds).

    Returns
    -------
    xr.Dataset with one variable per EB term, shape (time, y, x)
    """
    terms   = eb_terms or EB_TERMS
    pos_sum = sum(em_ds[t].where(em_ds[t] > 0, 0) for t in terms)
    pos_sum = pos_sum.where(pos_sum > 0)  # NaN where no positive flux

    return xr.Dataset({
        term: (em_ds[term] / pos_sum)
              .where(em_ds[term] > 0)   # only positive contributors
              .where(melt_cond)          # only on melt days
        for term in terms
    })


def aggregate_term_median(frac_flat: np.ndarray,
                           elev_bin_idx_valid: np.ndarray,
                           n_bins: int) -> np.ndarray:
    """Median fractional contribution per (time, elev_bin).

    Loops over bins (not pixels) — fast for typical bin counts (~10-20).
    Zeros (non-melt pixels set by nan_to_num) are excluded before taking
    the median so they don't pull results toward zero on partially-melting days.

    Parameters
    ----------
    frac_flat          : (n_time, n_valid_pixels) array, 0 on non-melt pixels
    elev_bin_idx_valid : elevation bin index per valid pixel, shape (n_valid,)
    n_bins             : number of elevation bins

    Returns
    -------
    (n_time, n_bins) array, NaN where no melt occurred in a bin on a given day
    """
    n_time = frac_flat.shape[0]
    median_matrix = np.full((n_time, n_bins), np.nan)

    for b in range(n_bins):
        bin_mask = (elev_bin_idx_valid == b)
        if not bin_mask.any():
            continue
        bin_fracs = frac_flat[:, bin_mask]                   # (n_time, n_pixels_in_bin)
        # Replace zeros with NaN so non-melt pixels don't bias the median
        bin_fracs = np.where(bin_fracs > 0, bin_fracs, np.nan)
        median_matrix[:, b] = np.nanmedian(bin_fracs, axis=1)

    return median_matrix


def plot_eb_term_heatmap(median_matrix: np.ndarray, term_name: str,
                          dowy: np.ndarray, elev_bins: np.ndarray,
                          wy: int, basin: str,
                          pixels_per_bin: np.ndarray = None,
                          cmap='Purples', figsize=(7, 3),
                          tick_fontsize: int = 8,
                          specify_ax=None, annotate=True,
                          out_fn=None, overwrite=False, dry_run=False):
    """Heatmap of median fractional EB contribution by DOWY × elevation.

    Colour encodes median fractional contribution. Alpha per elevation band
    is the median of that band's values across all melt days, normalised so
    the band with the highest typical contribution is fully opaque.

    Parameters
    ----------
    median_matrix : (n_time, n_bins) array from aggregate_term_median
    term_name     : EB variable name, used in title and filename
    specify_ax    : (fig, ax) tuple to plot into an existing axes instead of
                    creating a new figure — use this for megafigure subplots.
                    The colorbar is omitted when specify_ax is provided.
    """
    if dry_run:
        LOGGER.info('[dry-run] Skipping EB heatmap: %s  ->  %s', term_name, out_fn)
        return None, None

    bin_step = float(elev_bins[1] - elev_bins[0])
    ticks, labels = get_dowy_month_ticks(wy)

    # Per-band alpha: median over time, normalised to [0, 1]
    band_median = np.nanmedian(median_matrix, axis=0)        # (n_bins,)
    max_val     = np.nanmax(band_median)
    alpha_per_band = (band_median / max_val
                      if max_val > 0 else np.ones_like(band_median))

    # RGBA: colour = median fraction on fixed [0, 1] scale so all 5 term plots
    # are directly comparable; alpha = per-band summary
    norm   = Normalize(vmin=0, vmax=1)
    rgba   = plt.get_cmap(cmap)(norm(median_matrix.T))        # (n_bins, n_time, 4)
    rgba[:, :, 3] = alpha_per_band[:, np.newaxis]            # alpha per row (elev band)

    # NaN cell treatment:
    #   pixels_per_bin == 0  → elevation range not in basin       → transparent
    #   pixels exist but NaN → no melt activity (that cell/band)  → black
    nan_cells = np.isnan(median_matrix.T)                     # (n_bins, n_time)
    if pixels_per_bin is not None:
        empty_bins = (pixels_per_bin < 0.5)                    # float-safe: 0 pixels
        rgba[nan_cells & (~empty_bins[:, np.newaxis])] = [0.0, 0.0, 0.0, 1.0]
        rgba[empty_bins] = [0.0, 0.0, 0.0, 0.0]
    else:
        # Without pixels_per_bin, detect absent elevation ranges as bins that
        # are all-NaN across every time step (elevation range not in this basin).
        # Those get grey; individual NaN cells within active bins get black.
        absent_bins = np.all(np.isnan(median_matrix), axis=0)    # (n_bins,)
        rgba[nan_cells & (~absent_bins[:, np.newaxis])] = [0.0, 0.0, 0.0, 1.0]
        rgba[absent_bins] = [0.5, 0.5, 0.5, 0.8]

    if specify_ax is not None:
        fig, ax = specify_ax
    else:
        fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(rgba, aspect='auto', origin='lower',
              extent=[dowy[0], dowy[-1], elev_bins[0], elev_bins[-1]])
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels, fontsize=tick_fontsize)
    ax.set_xlim(1, 365)
    ax.set_ylim(elev_bins[0], elev_bins[-1] - bin_step / 2)
    ax.set_ylabel('Elevation (m)', fontsize=tick_fontsize)
    ax.tick_params(axis='both', labelsize=tick_fontsize)
    if annotate:
        ax.annotate(f'{basin.capitalize()} WY {wy}', xy=(0.02, 1.2), xycoords='axes fraction',
                    color='black', fontstyle='oblique',
                    ha='left', va='top', fontsize=12)
    term_label = term_name.replace('_', ' ').capitalize()
    ax.annotate(f'{term_label}', xy=(0.98, 0.05), xycoords='axes fraction',
                color='white', fontweight='semibold',
                ha='right', va='bottom', fontsize=10)

    # Skip standalone colorbar when embedded in a megafigure (caller manages it)
    if specify_ax is None:
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        fig.colorbar(sm, ax=ax, orientation='horizontal', location='top',
                 fraction=0.05, pad=0.03,
                 label='Median fractional contribution')
        plt.tight_layout()
        _save_figure(fig, out_fn, overwrite)

    return fig, ax


def plot_dominant_eb_term(term_median_matrices: dict,
                           dowy: np.ndarray,
                           elev_bins: np.ndarray,
                           wy: int,
                           basin: str,
                           pixels_per_bin: np.ndarray = None,
                           xlim=(1, 365),
                           figsize=(7, 3),
                           annotate=True,
                           out_fn=None, overwrite=False, dry_run=False):
    """Categorical heatmap of the dominant EB term per DOWY × elevation band.

    Colour  = which term has the highest median fractional contribution.
    Opacity = margin of dominance: (1st - 2nd) / 1st, normalised to [0, 1].
    Transparent cells are closely contested; opaque cells have a clear winner.

    Parameters
    ----------
    term_median_matrices : dict mapping term name → (n_time, n_bins) median array
    """
    if dry_run:
        LOGGER.info('[dry-run] Skipping dominant EB term heatmap  ->  %s', out_fn)
        return None, None

    terms   = list(term_median_matrices.keys())
    n_terms = len(terms)

    # Stack to (n_terms, n_time, n_bins); replace NaN with -inf for argmax
    stacked      = np.stack([term_median_matrices[t] for t in terms], axis=0)
    all_nan      = np.all(np.isnan(stacked), axis=0)              # (n_time, n_bins)
    stacked_fill = np.where(np.isnan(stacked), -np.inf, stacked)

    dominant_idx         = np.argmax(stacked_fill, axis=0).astype(float)
    dominant_idx[all_nan] = np.nan                                 # mask no-data cells

    # Margin of dominance = (top1 - top2) / top1
    sorted_desc = np.sort(stacked_fill, axis=0)[::-1]             # descending
    top1 = sorted_desc[0]
    top2 = sorted_desc[1] if n_terms > 1 else np.zeros_like(top1)
    margin           = np.where(top1 > 0, (top1 - top2) / top1, 0.0)
    margin[all_nan]  = 0.0

    # Categorical colormap — one colour per term
    colors = plt.cm.Set2(np.linspace(0, 1, n_terms))
    cmap   = ListedColormap(colors)
    norm   = Normalize(vmin=-0.5, vmax=n_terms - 0.5)

    # Build RGBA: colour by dominant index, alpha by margin
    rgba            = cmap(norm(dominant_idx.T))                   # (n_bins, n_time, 4)
    rgba[:, :, 3]   = margin.T
    # NaN cell treatment (mirrors plot_eb_term_heatmap):
    #   pixels_per_bin == 0  → elevation range not in basin  → transparent
    #   pixels exist but NaN → no dominant term that cell     → black
    nan_cells = np.isnan(dominant_idx.T)                           # (n_bins, n_time)
    if pixels_per_bin is not None:
        empty_bins = (pixels_per_bin == 0)
        rgba[nan_cells & (~empty_bins[:, np.newaxis])] = [0.0, 0.0, 0.0, 1.0]
        rgba[empty_bins] = [0.0, 0.0, 0.0, 0.0]
    else:
        rgba[nan_cells] = [0.0, 0.0, 0.0, 1.0]

    bin_step   = float(elev_bins[1] - elev_bins[0])
    ticks, tick_labels = get_dowy_month_ticks(wy)

    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(rgba, aspect='auto', origin='lower',
              extent=[dowy[0], dowy[-1], elev_bins[0], elev_bins[-1]])
    ax.set_xticks(ticks)
    ax.set_xticklabels(tick_labels)
    ax.set_xlim(*xlim)
    ax.set_ylim(elev_bins[0], elev_bins[-1] - bin_step / 2)
    # ax.set_xlabel(f'WY {wy} (DOWY)')
    ax.set_ylabel('Elevation (m)')
    # Annotate basin and wy
    if annotate:
        ax.annotate(f'{basin.capitalize()} WY {wy}', xy=(0.02, 1.2), xycoords='axes fraction',
                    color='black', fontstyle='oblique',
                    ha='left', va='top', fontsize=12)

    patches = [mpatches.Patch(color=colors[i], label=_abbrev(terms[i]),
                               linewidth=0) for i in range(n_terms)]
    leg = ax.legend(handles=patches, title='Dominant EB term', title_fontsize=10,
                    loc='lower center', bbox_to_anchor=(0.5, .98),
                    ncol=int(n_terms/2), fontsize=8, frameon=False,
                    handlelength=0.6, handletextpad=0.3, columnspacing=0.8)
    leg._legend_box.sep = 4

    for text, color in zip(leg.get_texts(), colors):
        text.set_color(color)

    plt.tight_layout()
    _save_figure(fig, out_fn, overwrite)
    return fig, ax


# ---------------------------------------------------------------------------
# Plotting functions
# ---------------------------------------------------------------------------

def plot_basin_mean_timeseries(em_basin_mean: xr.Dataset, basin: str, wy: int,
                                colors=None, figsize=(12, 6),
                                out_fn=None, overwrite=False, dry_run=False):
    """Basin-mean snowmelt, SWI, and cold content timeseries with melt event markers.

    Parameters
    ----------
    colors : dict, optional
        Override line/marker colors for any variable.
        Keys: 'snowmelt', 'SWI', 'cold_content', 'melt_events'.
        Example: {'snowmelt': 'dodgerblue', 'melt_events': 'magenta'}
    dry_run : bool
        Log what would be plotted/saved without creating figures.
    """
    if dry_run:
        LOGGER.info('[dry-run] Skipping plot: basin mean timeseries WY%s  ->  %s', wy, out_fn)
        return None, None
    _colors = {'snowmelt': None, 'SWI': None, 'cold_content': 'orange', 'melt_events': 'red'}
    if colors:
        _colors.update(colors)

    fig, ax1 = plt.subplots(figsize=figsize)
    em_basin_mean['snowmelt'].plot(ax=ax1, label='Snowmelt', marker='.', lw=1,
                                   color=_colors['snowmelt'])
    em_basin_mean['SWI'].plot(ax=ax1, label='SWI', marker='.', markersize=1, lw=1,
                               color=_colors['SWI'])

    ax2 = ax1.twinx()
    em_basin_mean['cold_content'].plot(ax=ax2, label='Cold Content',
                                       alpha=0.2, lw=2, color=_colors['cold_content'])

    melt_events = em_basin_mean.where(
        (em_basin_mean['sum_EB'] == 0) & (em_basin_mean['snowmelt'] > 0), drop=True
    )
    for melt_time in melt_events['time'].values:
        ax1.axvline(melt_time, color=_colors['melt_events'], linestyle='--', alpha=0.5)
    ax1.set_title(f'{basin.capitalize()} WY {wy} — Basin-mean Snowmelt, SWI, and Cold Content')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Snowmelt (mm/day)')
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')

    _save_figure(fig, out_fn, overwrite)
    return fig, (ax1, ax2)


def plot_spatial_summary(da: xr.DataArray, wy: int=None,
                          map_title=None, hist_title=None,
                          cmap='magma',
                          vmin=None, vmax=None,
                          cbar_label='', bins=50, hist_xlabel='',
                          sum_dim=None, setfc='r', figsize=(8, 4),
                          show_stats=True, out_fn=None, overwrite=False,
                          dry_run=False):
    """Spatial map + histogram side-by-side for any 2D summary DataArray.

    Parameters
    ----------
    da         : xr.DataArray — 2D spatial or 3D if sum_dim provided
    wy         : int          — water year, used in default titles
    cmap       : colormap     — colormap for the spatial map
    sum_dim    : str          — dimension to reduce along before plotting
    out_fn     : str          — output path; None skips saving
    overwrite  : bool         — overwrite existing figure
    dry_run    : bool         — log what would be plotted/saved without creating figures
    """
    if dry_run:
        LOGGER.info('[dry-run] Skipping plot: spatial summary  title=%s  ->  %s',
                    map_title or f'WY{wy}', out_fn)
        return None, None
    if sum_dim is not None:
        da = da.sum(dim=sum_dim)
    da = da.compute()

    fig, axa = plt.subplots(1, 2, figsize=figsize)

    h.plot_one(da,
               title=map_title or f'WY {wy}',
               cmap=cmap,
               setfc=setfc,
               specify_ax=(fig, axa[0]),
               vmin=vmin, vmax=vmax,
               cbar_kwargs={'label': cbar_label},
               turnofflabels=True,
               turnoffaxes=True)

    h.plot_hist(da,
                bins=bins,
                specify_ax=(fig, axa[1]),
                xlabel=hist_xlabel,
                title=hist_title or f'WY {wy} distribution')

    if show_stats:
        stats_text = (
            f'n: {int((da > 0).sum().values)}\n'
            f'Mean: {float(da.mean().values):.1f}\n'
            f'Median: {float(da.median().values):.1f}\n'
            f'Std dev: {float(da.std().values):.1f}'
        )
        axa[1].text(0.95, 0.95, stats_text,
                    transform=axa[1].transAxes, fontsize=10,
                    verticalalignment='top', horizontalalignment='right',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.tight_layout()
    _save_figure(fig, out_fn, overwrite)
    return fig, axa


def plot_duration_elevation(duration_flat: np.ndarray, elev_flat: np.ndarray,
                             elev_bins: np.ndarray, basin: str, wy: int,
                             cmap='YlOrRd', vmin=None, vmax=None,
                             duration_bins=np.arange(0, 100, 5),
                             figsize=(6, 3), out_fn=None, overwrite=False,
                             dry_run=False):
    """2D histogram of melt season duration vs elevation."""
    if dry_run:
        LOGGER.info('[dry-run] Skipping plot: duration vs elevation  basin=%s WY%s  ->  %s', basin, wy, out_fn)
        return None, None
    valid = (duration_flat > 0) & np.isfinite(elev_flat)

    fig, ax = plt.subplots(figsize=figsize)
    ax.hist2d(duration_flat[valid], elev_flat[valid],
              bins=[duration_bins, elev_bins],
              cmap=cmap, vmin=vmin, vmax=vmax)
    plt.colorbar(ax.collections[0], ax=ax, label='Frequency')
    ax.set_xlabel(f'Melt season duration (WY {wy})')
    ax.set_ylabel(f'{basin.capitalize()} Elevation (m)')

    _save_figure(fig, out_fn, overwrite)
    return fig, ax


def plot_melt_frequency_heatmap(melt_count_matrix: np.ndarray, dowy: np.ndarray,
                                 elev_bins: np.ndarray, basin: str, wy: int,
                                 cmap='YlOrRd', vmin=None, vmax=None,
                                 xlim=(1, 365), figsize=(6, 3),
                                 out_fn=None, overwrite=False, dry_run=False):
    """pcolormesh of melting pixel count over time × elevation."""
    if dry_run:
        LOGGER.info('[dry-run] Skipping plot: melt frequency heatmap  basin=%s WY%s  ->  %s', basin, wy, out_fn)
        return None, None
    bin_step = float(elev_bins[1] - elev_bins[0])
    ticks, labels = get_dowy_month_ticks(wy)

    fig, ax = plt.subplots(figsize=figsize)
    c = ax.pcolormesh(dowy, elev_bins[:-1], melt_count_matrix.T,
                      cmap=cmap, vmin=vmin, vmax=vmax)
    plt.colorbar(c, ax=ax, label='Melting pixels')
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels)
    ax.set_xlim(*xlim)
    ax.set_ylim(elev_bins[0], elev_bins[-1] - bin_step / 2)
    ax.set_xlabel(f'WY {wy} (DOWY)')
    ax.set_ylabel(f'{basin.capitalize()} Elevation (m)')

    _save_figure(fig, out_fn, overwrite)
    return fig, ax


def plot_hyps_heatmap(melt_count_matrix: np.ndarray, dowy: np.ndarray,
                      elev_bins: np.ndarray, elev_flat: np.ndarray,
                      basin: str, wy: int,
                      cmap='cividis', vmin=None, vmax=None,
                      hyps_color='steelblue', cumul_color='dodgerblue',
                      xlim=(1, 365), figsize=(8, 4),
                      out_fn=None, overwrite=False, dry_run=False):
    """Hypsometric curve (left) + melt frequency heatmap (right) with shared y-axis.

    Parameters
    ----------
    cmap        : colormap for the heatmap panel
    hyps_color  : bar fill color for the area-fraction hypsometric bars
    cumul_color : line color for the cumulative fraction curve
    dry_run     : log what would be plotted/saved without creating figures
    """
    if dry_run:
        LOGGER.info('[dry-run] Skipping plot: hyps heatmap  basin=%s WY%s  ->  %s', basin, wy, out_fn)
        return None, None
    bin_step = float(elev_bins[1] - elev_bins[0])
    counts, _ = np.histogram(elev_flat[np.isfinite(elev_flat)], bins=elev_bins)

    fig, (ax_hyps, ax) = plt.subplots(1, 2, figsize=figsize,
                                       sharey=True,
                                       gridspec_kw={'width_ratios': [1, 4]})

    ax_hyps.barh(elev_bins[:-1], counts / counts.sum(),
                 height=bin_step * 0.9, align='edge',
                 alpha=0.3, color=hyps_color, edgecolor='w')
    ax_hyps.set_xlabel('Basin\narea fraction')
    ax_hyps.set_ylabel(f'{basin.capitalize()} Elevation (m)')

    ax_hyps2 = ax_hyps.twiny()
    ax_hyps2.plot(np.cumsum(counts / counts.sum()), elev_bins[:-1] + bin_step / 2,
                  color=cumul_color, lw=2.5)
    ax_hyps2.set_xlim(0, 1)
    ax_hyps2.set_xlabel('Cumulative fraction')

    ticks, labels = get_dowy_month_ticks(wy)
    c = ax.pcolormesh(dowy, elev_bins[:-1], melt_count_matrix.T,
                      cmap=cmap, vmin=vmin, vmax=vmax)
    plt.colorbar(c, ax=ax, label='Melting pixels', pad=0.01)
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels)
    ax.set_xlim(*xlim)
    ax.set_ylim(elev_bins[0], elev_bins[-1] - bin_step / 2)
    ax.set_xlabel(f'WY {wy} (DOWY)')
    ax.tick_params(labelleft=False)

    plt.tight_layout()
    _save_figure(fig, out_fn, overwrite)
    return fig, (ax_hyps, ax)


# ---------------------------------------------------------------------------
# SWI attribution by EB term
# ---------------------------------------------------------------------------

def compute_swi_by_eb_term(em_eb: xr.Dataset,
                            prop_ds: xr.Dataset,
                            eb_terms_active: list,
                            melt_cond: xr.DataArray,
                            valid: np.ndarray,
                            one_hot: np.ndarray) -> dict:
    """Seasonal SWI attributed to each EB term per elevation bin.

    For each term k and elevation bin b:
        swi_k[b] = Σ_t  Σ_{i∈b}  SWI[t,i] × frac_k[t,i]

    SWI is pre-computed once; each term loads its fractional contribution from
    prop_ds one at a time to limit peak memory.

    Returns
    -------
    dict mapping term name → (n_bins,) float array
        Values are in mm × pixels (same units as compute_variable_elevation_matrix
        with aggregation='sum').  Apply the TAF conversion factor for plotting.
    """
    # Precompute valid SWI once (zeroed on non-melt days/pixels, NaN→0)
    swi_valid = np.nan_to_num(
        em_eb['SWI'].where(melt_cond, 0).values.reshape(len(em_eb['time']), -1),
        nan=0.0,
    )[:, valid]                                      # (n_time, n_valid)

    swi_by_term = {}
    for term in eb_terms_active:
        LOGGER.debug('Computing SWI attribution for %s', term)
        frac_valid = np.nan_to_num(
            prop_ds[term].values.reshape(len(prop_ds['time']), -1),
            nan=0.0,
        )[:, valid]                                  # (n_time, n_valid)

        # Element-wise attribution, then bin and sum over time
        swi_by_term[term] = ((swi_valid * frac_valid) @ one_hot).sum(axis=0)
        del frac_valid

    return swi_by_term


def plot_swi_eb_total(swi_by_term: dict, eb_terms_active: list,
                      wy: int, basin: str,
                      pixel_res_m: float = 100,
                      precomputed: bool = False,
                      cmap: str = 'Set2',
                      xlim=None,
                      tick_fontsize: int = 9,
                      label_pct_threshold: float = 0.05,
                      specify_ax=None,
                      figsize=(5, 3.5), annotate=True,
                      out_fn=None, overwrite=False, dry_run=False):
    """Horizontal bar chart of cumulative seasonal SWI attributed to each EB term.

    Each bar represents the total water volume (TAF) produced under the influence
    of that energy balance term across the entire basin for the water year.

    Parameters
    ----------
    xlim          : (xmin, xmax) tuple to set a fixed x-axis range — use this
                    for consistent per-basin scale across water years in a megafigure.
                    Defaults to auto-scaling from the data.
    tick_fontsize : font size for tick labels and axis labels
    specify_ax    : (fig, ax) tuple to embed into an existing axes rather than
                    creating a new figure. Colorbar/tight_layout omitted when set.
    """
    if dry_run:
        LOGGER.info('[dry-run] Skipping SWI-EB total plot  ->  %s', out_fn)
        return None, None

    # When loading from the cached _swi_attributed.nc the values are already in TAF;
    # pass precomputed=True to skip the conversion applied to raw mm×pixels output.
    taf_scale = 1.0 if precomputed else (pixel_res_m ** 2) * 0.000810714 / 1_000_000
    colors = plt.get_cmap(cmap)(np.linspace(0, 1, len(eb_terms_active)))
    labels = [_abbrev(t) for t in eb_terms_active]
    values = [float(np.asarray(swi_by_term[t]).sum()) * taf_scale
              for t in eb_terms_active]

    if specify_ax is not None:
        fig, ax = specify_ax
    else:
        fig, ax = plt.subplots(figsize=figsize)

    bars = ax.barh(labels, values, color=colors, edgecolor='w')

    x_max   = xlim[1] if xlim is not None else (max(values) if values else 1.0)
    total   = sum(values) if sum(values) > 0 else 1.0
    for bar, val, color in zip(bars, values, colors):
        pct = 100.0 * val / total
        if pct / 100.0 < label_pct_threshold:
            continue
        ax.text(bar.get_width() + x_max * 0.02,
                bar.get_y() + bar.get_height() / 2,
                f'{int(val)} ({pct:.0f}%)',
                va='center',
                fontsize=tick_fontsize - 1,
                # c=color
                )

    ax.set_xlabel('Cumulative seasonal SWI (TAF)', fontsize=tick_fontsize)
    ax.tick_params(axis='both', labelsize=tick_fontsize)
    if annotate:
        ax.annotate(f'{basin.capitalize()} WY {wy}', xy=(0.98, 0.8),
                    xycoords='axes fraction', ha='right', va='bottom', c='gray', alpha=0.6,
                    fontsize=tick_fontsize, fontstyle='oblique', fontweight='bold')

    if xlim is not None:
        ax.set_xlim(*xlim)
    else:
        ax.set_xlim(0, x_max * 1.2)

    if specify_ax is None:
        ax.set_title(f'{basin.capitalize()} WY {wy} — SWI by EB term')
        plt.tight_layout()
        _save_figure(fig, out_fn, overwrite)

    return fig, ax


def plot_swi_eb_elevation(swi_by_term: dict, eb_terms_active: list,
                           elev_bins: np.ndarray, wy: int, basin: str,
                           pixel_res_m: float = 100,
                           pixels_per_bin: np.ndarray = None,
                           cmap: str = 'Set2',
                           frac_label_threshold: float = 0.05,
                           taf_label_threshold: float = None,
                           figsize=(10, 5),
                           out_fn=None, overwrite=False, dry_run=False):
    """Stacked horizontal bar of seasonal SWI by EB term and elevation band.

    Left panel  : absolute SWI per elevation band (TAF).
    Right panel : fractional contribution per elevation band (normalised to 1),
                  showing relative importance of each term independent of band size.

    Parameters
    ----------
    swi_by_term          : output of compute_swi_by_eb_term
    eb_terms_active      : ordered list of term names (controls stack order and legend)
    pixels_per_bin       : (n_bins,) pixel counts; used only to suppress labels
                           on bins absent from this basin
    frac_label_threshold : fractional value (0–1) above which a white % label is drawn
                           on the right edge of each segment in the fractional panel
    taf_label_threshold  : minimum TAF value above which a white value label is drawn
                           on the right edge of each segment in the absolute panel;
                           None disables labels on the absolute panel
    """
    if dry_run:
        LOGGER.info('[dry-run] Skipping SWI-EB elevation plot  ->  %s', out_fn)
        return None, None

    taf_scale   = (pixel_res_m ** 2) * 0.000810714 / 1_000_000
    bin_step    = float(elev_bins[1] - elev_bins[0])
    n_bins      = len(elev_bins) - 1
    colors      = plt.get_cmap(cmap)(np.linspace(0, 1, len(eb_terms_active)))
    labels      = [_abbrev(t) for t in eb_terms_active]
    absent_bins = (pixels_per_bin < 0.5) if pixels_per_bin is not None \
                  else np.zeros(n_bins, dtype=bool)   # used only for label suppression

    # Build (n_terms, n_bins) matrices
    taf_matrix  = np.stack([swi_by_term[t] * taf_scale for t in eb_terms_active], axis=0)
    bin_totals  = taf_matrix.sum(axis=0, keepdims=True)
    frac_matrix = np.where(bin_totals > 0, taf_matrix / bin_totals, 0.0)

    y      = elev_bins[:-1]
    height = bin_step * 0.85

    fig, (ax_abs, ax_frac) = plt.subplots(1, 2, figsize=figsize, sharey=True)

    for panel, matrix, xlabel in [
        (ax_abs,  taf_matrix,  'Cumulative seasonal SWI (TAF)'),
        (ax_frac, frac_matrix, 'Fractional contribution'),
    ]:
        left = np.zeros(n_bins)
        for i in range(len(eb_terms_active)):
            panel.barh(y, matrix[i], left=left, height=height,
                       align='edge', color=colors[i], edgecolor='none',
                       label=labels[i])
            left += matrix[i]

        panel.set_xlabel(xlabel)
        panel.set_ylim(elev_bins[0], elev_bins[-1])  # full range; no imshow padding needed

    # White labels on fractional panel for segments ≥ frac_label_threshold
    left_frac = np.zeros(n_bins)
    for i in range(len(eb_terms_active)):
        for b in range(n_bins):
            fv = frac_matrix[i, b]
            if fv >= frac_label_threshold and not absent_bins[b]:
                ax_frac.text(
                    left_frac[b] + fv - 0.01,
                    y[b] + height / 2,
                    f'{fv:.0%}',
                    ha='right', va='center',
                    fontsize=6, color='white', fontweight='bold',
                )
        left_frac += frac_matrix[i]

    # White TAF value labels on absolute panel for segments ≥ taf_label_threshold
    if taf_label_threshold is not None:
        left_taf = np.zeros(n_bins)
        for i in range(len(eb_terms_active)):
            for b in range(n_bins):
                tv = taf_matrix[i, b]
                if tv >= taf_label_threshold and not absent_bins[b]:
                    ax_abs.text(
                        left_taf[b] + tv -2,
                        y[b] + height / 2,
                        f'{tv:.0f}',
                        ha='right', va='center',
                        fontsize=6, color='white', fontweight='bold',
                    )
            left_taf += taf_matrix[i]

    ax_abs.set_ylabel(f'{basin.capitalize()} Elevation (m)')
    ax_frac.tick_params(labelleft=False)
    ax_frac.set_xlim(0, 1)

    handles = [mpatches.Patch(color=colors[i], label=labels[i], linewidth=0)
               for i in range(len(eb_terms_active))]
    fig.legend(handles=handles,
               loc='lower center', bbox_to_anchor=(0.5, 1.0),
               ncol=len(eb_terms_active),
               fontsize=8, handlelength=0.8, handletextpad=0.4,
               columnspacing=1.0, framealpha=0.8)

    fig.suptitle(f'{basin.capitalize()} WY {wy} — Seasonal SWI by EB term and elevation')
    plt.tight_layout(rect=[0, 0.06, 1, 1])
    _save_figure(fig, out_fn, overwrite)
    return fig, (ax_abs, ax_frac)


# ---------------------------------------------------------------------------
# Aspect-based SWI analysis
# ---------------------------------------------------------------------------

def _aspect_bin_idx(aspect_flat: np.ndarray) -> np.ndarray:
    """Map aspect in degrees [0, 360) to N_ASPECT_BINS bin indices (0=N, 1=NE, …).

    North-centred bins: N = [−22.5, 22.5), NE = [22.5, 67.5), etc.
    If the input looks like radians (max < 2π + ε), it is converted to degrees first.
    """
    if aspect_flat.size and np.nanmax(aspect_flat) < 2 * np.pi + 0.1:
        aspect_flat = np.degrees(aspect_flat)
    return ((aspect_flat + 22.5) % 360 / 45).astype(int)


def compute_swi_aspect_matrices(em_eb: xr.Dataset,
                                 prop_ds: xr.Dataset,
                                 eb_terms_active: list,
                                 melt_cond: xr.DataArray,
                                 valid: np.ndarray,
                                 one_hot: np.ndarray,
                                 aspect_flat: np.ndarray) -> tuple:
    """Compute aspect-binned SWI matrices for all three aspect plots.

    All outputs are in mm × pixels (apply TAF conversion in plotting functions).

    Parameters
    ----------
    aspect_flat : (n_all_pixels,) aspect values in degrees or radians

    Returns
    -------
    swi_aspect       : dict {term: (N_ASPECT_BINS,)} seasonal total attributed SWI
    swi_elev_aspect  : dict {term: (n_elev_bins, N_ASPECT_BINS)} seasonal totals
    swi_daily_aspect : (n_time, N_ASPECT_BINS) total SWI per day per aspect (all terms)
    """
    n_valid    = int(valid.sum())
    n_time     = len(em_eb['time'])
    n_elev     = one_hot.shape[1]

    # Aspect bin index per valid pixel
    aspect_valid  = aspect_flat[valid]
    asp_idx       = _aspect_bin_idx(aspect_valid)                  # (n_valid,)

    # Aspect one-hot  (n_valid, N_ASPECT_BINS)
    asp_oh = np.zeros((n_valid, N_ASPECT_BINS))
    asp_oh[np.arange(n_valid), asp_idx % N_ASPECT_BINS] = 1

    # Joint elevation × aspect one-hot  (n_valid, n_elev × N_ASPECT_BINS)
    elev_idx   = np.argmax(one_hot, axis=1)                        # (n_valid,)
    joint_idx  = elev_idx * N_ASPECT_BINS + asp_idx % N_ASPECT_BINS
    joint_oh   = np.zeros((n_valid, n_elev * N_ASPECT_BINS))
    joint_oh[np.arange(n_valid), joint_idx] = 1

    # Precompute SWI once
    swi_valid = np.nan_to_num(
        em_eb['SWI'].where(melt_cond, 0).values.reshape(n_time, -1),
        nan=0.0,
    )[:, valid]                                                     # (n_time, n_valid)

    # Total SWI seasonality per aspect (no EB attribution needed)
    swi_daily_aspect = swi_valid @ asp_oh                          # (n_time, N_ASPECT_BINS)

    swi_aspect      = {}
    swi_elev_aspect = {}
    for term in eb_terms_active:
        frac_valid = np.nan_to_num(
            prop_ds[term].values.reshape(n_time, -1),
            nan=0.0,
        )[:, valid]                                                 # (n_time, n_valid)

        # Seasonal attributed SWI per valid pixel (sum over time)
        seasonal = (swi_valid * frac_valid).sum(axis=0)            # (n_valid,)
        del frac_valid

        swi_aspect[term]      = seasonal @ asp_oh                  # (N_ASPECT_BINS,)
        swi_elev_aspect[term] = (seasonal @ joint_oh).reshape(
            n_elev, N_ASPECT_BINS)                                 # (n_elev, N_ASPECT_BINS)

    return swi_aspect, swi_elev_aspect, swi_daily_aspect


def plot_dominant_eb_aspect(swi_elev_aspect: dict, eb_terms_active: list,
                             elev_bins: np.ndarray, wy: int, basin: str,
                             cmap: str = 'Set2',
                             figsize=(6, 5),
                             out_fn=None, overwrite=False, dry_run=False):
    """2D heatmap of dominant EB term per elevation × aspect bin.

    Colour  = term with highest attributed SWI in that cell.
    Opacity = margin of dominance (same as plot_dominant_eb_term).
    """
    if dry_run:
        LOGGER.info('[dry-run] Skipping dominant EB aspect heatmap  ->  %s', out_fn)
        return None, None

    n_terms   = len(eb_terms_active)
    colors    = plt.get_cmap(cmap)(np.linspace(0, 1, n_terms))
    cmap_cat  = ListedColormap(colors)
    norm_cat  = Normalize(vmin=-0.5, vmax=n_terms - 0.5)
    bin_step  = float(elev_bins[1] - elev_bins[0])

    stacked   = np.stack([swi_elev_aspect[t] for t in eb_terms_active], axis=0)
    all_zero  = stacked.sum(axis=0) == 0                           # (n_elev, N_ASPECT_BINS)
    dom_idx   = np.argmax(stacked, axis=0).astype(float)
    dom_idx[all_zero] = np.nan

    sorted_d  = np.sort(stacked, axis=0)[::-1]
    margin    = np.where(sorted_d[0] > 0,
                         (sorted_d[0] - sorted_d[1]) / sorted_d[0], 0.0)
    margin[all_zero] = 0.0

    rgba      = cmap_cat(norm_cat(dom_idx))                        # (n_elev, N_ASPECT, 4)
    rgba[..., 3]          = margin
    rgba[np.isnan(dom_idx)] = [0, 0, 0, 0]

    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(rgba, aspect='auto', origin='lower',
              extent=[-0.5, N_ASPECT_BINS - 0.5, elev_bins[0], elev_bins[-1]])
    ax.set_xticks(range(N_ASPECT_BINS))
    ax.set_xticklabels(ASPECT_LABELS)
    ax.set_ylabel(f'{basin.capitalize()} Elevation (m)')
    ax.set_xlabel('Aspect')
    ax.set_ylim(elev_bins[0], elev_bins[-1])

    handles = [mpatches.Patch(color=colors[i], label=_abbrev(eb_terms_active[i]),
                               linewidth=0) for i in range(n_terms)]
    ax.legend(handles=handles, loc='lower center', bbox_to_anchor=(0.5, 1.01),
              ncol=n_terms, fontsize=7, handlelength=0.6, handletextpad=0.3)

    ax.set_title(f'{basin.capitalize()} WY {wy} — Dominant EB term\n'
                 f'(opacity = margin of dominance)', pad=30)
    plt.tight_layout()
    _save_figure(fig, out_fn, overwrite)
    return fig, ax


def plot_swi_radial(swi_aspect: dict, eb_terms_active: list,
                    wy: int, basin: str,
                    pixel_res_m: float = 100,
                    cmap: str = 'Set2',
                    grid_color: str = 'lightgrey',
                    grid_linewidth: float = 0.5,
                    figsize=(6, 6),
                    out_fn=None, overwrite=False, dry_run=False):
    """Polar stacked bar chart of seasonal SWI by aspect direction, coloured by EB term."""
    if dry_run:
        LOGGER.info('[dry-run] Skipping SWI radial plot  ->  %s', out_fn)
        return None, None

    # convert mm to meters (first /1000), convert to AF, and then TAF (second /1000)
    taf_scale  = (pixel_res_m ** 2) / 1000 * 0.000810714 / 1000 
    colors     = plt.get_cmap(cmap)(np.linspace(0, 1, len(eb_terms_active)))
    labels     = [_abbrev(t) for t in eb_terms_active]
    taf_matrix = np.stack([swi_aspect[t] * taf_scale for t in eb_terms_active], axis=0)

    angles = np.linspace(0, 2 * np.pi, N_ASPECT_BINS, endpoint=False)
    width  = 2 * np.pi / N_ASPECT_BINS

    fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(projection='polar'))
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)                                     # clockwise

    bottom = np.zeros(N_ASPECT_BINS)
    for i in range(len(eb_terms_active)):
        ax.bar(angles, taf_matrix[i], width=width, bottom=bottom,
               color=colors[i], edgecolor='white', linewidth=0.3,
               label=labels[i], alpha=0.88)
        bottom += taf_matrix[i]

    # Gridlines (concentric circles + radial spokes) and outer spine
    ax.grid(color=grid_color, linewidth=grid_linewidth, linestyle='--')
    ax.spines['polar'].set_color(grid_color)
    ax.spines['polar'].set_linewidth(grid_linewidth)

    ax.set_xticks(angles)
    ax.set_xticklabels(ASPECT_LABELS, fontsize=9)
    ax.set_title(f'{basin.capitalize()} WY {wy}\nSeasonal SWI by aspect (TAF)', pad=20)

    handles = [mpatches.Patch(color=colors[i], label=labels[i], linewidth=0)
               for i in range(len(eb_terms_active))]
    ax.legend(handles=handles, loc='lower center',
              bbox_to_anchor=(0.5, -0.12), ncol=len(eb_terms_active),
              fontsize=7, handlelength=0.7, handletextpad=0.3, framealpha=0.8)

    plt.tight_layout()
    _save_figure(fig, out_fn, overwrite)
    return fig, ax


def plot_swi_seasonality_aspect(swi_daily_aspect: np.ndarray,
                                 dowy: np.ndarray, wy: int, basin: str,
                                 pixel_res_m: float = 100,
                                 figsize=(8, 4),
                                 out_fn=None, overwrite=False, dry_run=False):
    """Cumulative SWI curves over DOWY for each aspect class.

    Each line represents total SWI (all EB terms combined) accumulated
    over the water year for pixels of a given aspect direction.
    """
    if dry_run:
        LOGGER.info('[dry-run] Skipping SWI seasonality by aspect  ->  %s', out_fn)
        return None, None

    taf_scale = (pixel_res_m ** 2) * 0.000810714 / 1_000_000
    cumul     = np.cumsum(swi_daily_aspect * taf_scale, axis=0)    # (n_time, N_ASPECT_BINS)

    # Circular colormap so N and NW are visually adjacent
    colors = plt.get_cmap('hsv')(np.linspace(0, 1, N_ASPECT_BINS, endpoint=False))
    ticks, tick_labels = get_dowy_month_ticks(wy)

    fig, ax = plt.subplots(figsize=figsize)
    for a in range(N_ASPECT_BINS):
        ax.plot(dowy, cumul[:, a], color=colors[a], label=ASPECT_LABELS[a], lw=1.8)

    ax.set_xticks(ticks)
    ax.set_xticklabels(tick_labels)
    ax.set_xlabel(f'WY {wy} (DOWY)')
    ax.set_ylabel('Cumulative SWI (TAF)')
    ax.set_title(f'{basin.capitalize()} WY {wy} — Cumulative SWI by aspect')
    ax.legend(ncol=N_ASPECT_BINS, fontsize=7, loc='upper left',
              handlelength=0.8, handletextpad=0.3, columnspacing=0.8)
    plt.tight_layout()
    _save_figure(fig, out_fn, overwrite)
    return fig, ax


def plot_swi_elev_aspect_radial(swi_elev_aspect: dict, eb_terms_active: list,
                                 elev_bins: np.ndarray, wy: int, basin: str,
                                 pixel_res_m: float = 100,
                                 cmap: str = 'YlGnBu',
                                 vmin=None, vmax=None,
                                 grid_color: str = 'lightgrey',
                                 grid_linewidth: float = 0.5,
                                 figsize=(6, 6),
                                 out_fn=None, overwrite=False, dry_run=False):
    """Radial (polar) heatmap of total seasonal SWI by elevation × aspect.

    Angle = aspect direction (N at top, clockwise).
    Radius = elevation (inner = low, outer = high).
    Colour = total seasonal SWI in TAF (all EB terms summed).

    Parameters
    ----------
    grid_color      : colour for polar gridlines and outer spine
    grid_linewidth  : linewidth for polar gridlines and outer spine
    """
    if dry_run:
        LOGGER.info('[dry-run] Skipping SWI elev×aspect radial  ->  %s', out_fn)
        return None, None

    taf_scale    = (pixel_res_m ** 2) * 0.000810714 / 1_000_000
    total_matrix = sum(swi_elev_aspect[t] * taf_scale for t in eb_terms_active)
                                                                   # (n_elev, N_ASPECT)

    # Mask cells with no data
    total_masked = np.where(total_matrix > 0, total_matrix, np.nan)

    _vmin = vmin if vmin is not None else 0
    _vmax = vmax if vmax is not None else float(np.nanpercentile(total_masked[np.isfinite(total_masked)], 95)) \
            if np.isfinite(total_masked).any() else 1.0

    # Centre bin 0 (N) at 0° (North): shift edges by half a bin width
    #   without this, bin 0 spans [0°, 45°] centred at NE, not N
    bin_width    = 2 * np.pi / N_ASPECT_BINS
    theta_offset = -bin_width / 2

    # Subdivide each angular bin into N_sub steps so pcolormesh draws
    # smooth curved arcs rather than straight-line triangular boundaries
    N_sub       = 20
    theta_fine  = np.linspace(theta_offset,
                              theta_offset + 2 * np.pi,
                              N_ASPECT_BINS * N_sub + 1)
    matrix_fine = np.repeat(total_masked, N_sub, axis=1)          # (n_elev, N_ASPECT*N_sub)

    r_edges  = elev_bins
    norm_obj = Normalize(vmin=_vmin, vmax=_vmax)

    fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(projection='polar'))
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)                                     # clockwise

    c = ax.pcolormesh(theta_fine, r_edges, matrix_fine,
                      cmap=cmap, norm=norm_obj, shading='flat')

    # Elevation r-ticks at every other bin edge for readability
    r_ticks = elev_bins[::2]
    ax.set_rticks(r_ticks)
    ax.set_yticklabels([f'{int(r)}' for r in r_ticks], fontsize=7)
    ax.set_rmin(elev_bins[0])
    ax.set_rmax(elev_bins[-1])

    # Compass direction labels at bin centres (centred on the shifted bins)
    theta_centres = theta_offset + bin_width * (np.arange(N_ASPECT_BINS) + 0.5)
    ax.set_xticks(theta_centres)
    ax.set_xticklabels(ASPECT_LABELS, fontsize=9)

    ax.grid(color=grid_color, linewidth=grid_linewidth, linestyle='--')
    ax.spines['polar'].set_color(grid_color)
    ax.spines['polar'].set_linewidth(grid_linewidth)

    ax.set_title(f'{basin.capitalize()} WY {wy}\nSeasonal SWI by elevation × aspect (TAF)',
                 pad=20)

    plt.colorbar(c, ax=ax, orientation='horizontal',
                 fraction=0.04, pad=0.08, label='Seasonal SWI (TAF)')
    plt.tight_layout()
    _save_figure(fig, out_fn, overwrite)
    return fig, ax


def plot_swi_elev_aspect(swi_elev_aspect: dict, eb_terms_active: list,
                          elev_bins: np.ndarray, wy: int, basin: str,
                          pixel_res_m: float = 100,
                          cmap: str = 'YlGnBu',
                          vmin=None, vmax=None,
                          figsize=(6, 5),
                          out_fn=None, overwrite=False, dry_run=False):
    """Heatmap of total seasonal SWI (all EB terms combined) by elevation × aspect.

    Colour encodes how much water (TAF) each elevation-aspect cell produces over
    the water year, regardless of which energy balance term drives the melt.
    Complements plot_dominant_eb_aspect which shows *which* term dominates.

    Parameters
    ----------
    swi_elev_aspect : dict output of compute_swi_aspect_matrices
    eb_terms_active : list of term names — used only to sum across terms
    """
    if dry_run:
        LOGGER.info('[dry-run] Skipping SWI elev×aspect heatmap  ->  %s', out_fn)
        return None, None

    taf_scale    = (pixel_res_m ** 2) * 0.000810714 / 1_000_000
    bin_step     = float(elev_bins[1] - elev_bins[0])

    # Sum attributed SWI across all EB terms  →  (n_elev_bins, N_ASPECT_BINS)
    total_matrix = sum(swi_elev_aspect[t] * taf_scale for t in eb_terms_active)

    _vmin = vmin if vmin is not None else 0
    _vmax = vmax if vmax is not None else np.nanpercentile(total_matrix[total_matrix > 0], 95) \
            if (total_matrix > 0).any() else 1.0
    norm_obj = Normalize(vmin=_vmin, vmax=_vmax)
    rgba     = plt.get_cmap(cmap)(norm_obj(total_matrix))       # (n_elev, N_ASPECT, 4)

    # Transparent cells where no SWI (no pixels or no melt)
    rgba[total_matrix == 0, 3] = 0.0

    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(rgba, aspect='auto', origin='lower',
              extent=[-0.5, N_ASPECT_BINS - 0.5, elev_bins[0], elev_bins[-1]])
    ax.set_xticks(range(N_ASPECT_BINS))
    ax.set_xticklabels(ASPECT_LABELS)
    ax.set_ylabel(f'{basin.capitalize()} Elevation (m)')
    ax.set_xlabel('Aspect')
    ax.set_ylim(elev_bins[0], elev_bins[-1])
    ax.set_title(f'{basin.capitalize()} WY {wy} — Seasonal SWI by elevation × aspect')

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm_obj)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, orientation='horizontal', location='top',
                 fraction=0.04, pad=0.03, label='Seasonal SWI (TAF)')

    plt.tight_layout()
    _save_figure(fig, out_fn, overwrite)
    return fig, ax


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Melt timing and season analysis plots for iSnobal model output.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('basin', type=str, help='Basin name')
    parser.add_argument('wy',    type=int, help='Water year')
    parser.add_argument(
        '-w', '--workdir',
        default='/uufs/chpc.utah.edu/common/home/skiles-group2/model_runs_jmh/thp',
        help='Root directory containing Zarr stores subdirectory',
    )
    parser.add_argument(
        '-s', '--script_dir',
        default='/uufs/chpc.utah.edu/common/home/skiles-group3/jmhu/isnobal_scripts',
        help='Directory containing basin setup folders (for topo.nc)',
    )
    parser.add_argument(
        '-o', '--outdir',
        default='/uufs/chpc.utah.edu/common/home/u6058223/public_html/thp_update/melt_figures',
        help='Root output directory for saved figures',
    )
    parser.add_argument(
        '-b', '--bin_step',
        type=int,
        default=200,
        help='Elevation bin step size',
    )
    parser.add_argument(
        '--polygon',
        default=None,
        help='Optional polygon vector file (GeoJSON/shapefile) used to clip all datasets before analysis',
    )
    parser.add_argument(
        '--overwrite', action='store_true',
        help='Overwrite existing figures',
    )
    parser.add_argument(
        '--dry-run', action='store_true',
        help='Run analysis but skip figure creation and saving',
    )
    parser.add_argument(
        '--no-net-solar-split', action='store_true',
        help='Use net_rad as a single EB term instead of splitting into '
             'net_solar + net_LW (default: split if net_solar zarr exists)',
    )
    parser.add_argument(
        '-v', '--verbose', action='store_true',
        help='Enable DEBUG-level logging',
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def __main__() -> None:
    args = parse_arguments()
    _configure_logging(args.verbose)

    basin = args.basin
    wy = args.wy
    bin_step = args.bin_step
    out_dir = args.outdir
    ow = args.overwrite
    dr = args.dry_run

    # fig_dir = os.path.join(out_dir, 'melt_plots', basin, str(wy))
    # for shared elevation limit plotting
    fig_dir = os.path.join(out_dir, 'melt_plots', 'shared_lims', basin, str(wy))

    # --- Load data ---
    em_ds, snow_ds, terrain_ds = load_data(basin, wy, args.workdir, args.script_dir)

    if args.polygon:
        LOGGER.info('Clipping datasets to polygon: %s', args.polygon)
        em_ds      = clip_to_polygon(em_ds,      args.polygon, drop=True)
        snow_ds    = clip_to_polygon(snow_ds,    args.polygon, drop=True)
        terrain_ds = clip_to_polygon(terrain_ds, args.polygon, drop=True)

        # Datasets on the same nominal grid can produce slightly different
        # bounding boxes after polygon clipping (sub-pixel coordinate offsets
        # cause different edge pixels to be included).  Snap everything to
        # em_ds as the authoritative grid so all spatial dimensions agree.
        terrain_ds = terrain_ds.reindex_like(em_ds, method='nearest')
        snow_ds    = snow_ds.reindex_like(em_ds,    method='nearest')
        LOGGER.debug('Post-clip grid: x=%d  y=%d',
                     em_ds.sizes['x'], em_ds.sizes['y'])

    # Define all output paths up front so skip checks and plot calls share the same strings
    _out = {
        'basin_mean':    f'{fig_dir}/{basin}_basin_mean_timeseries_wy{wy}.png',
        'melt_days':     f'{fig_dir}/{basin}_melt_days_wy{wy}.png',
        'melt_init':     f'{fig_dir}/{basin}_melt_initiation_wy{wy}.png',
        'meltout':       f'{fig_dir}/{basin}_meltout_wy{wy}.png',
        'melt_season':   f'{fig_dir}/{basin}_melt_season_duration_wy{wy}.png',
        'duration_elev': f'{fig_dir}/{basin}_duration_elevation_wy{wy}.png',
        'freq':          f'{fig_dir}/{basin}_melt_season_heatmap_freq_wy{wy}.png',
        'freq_norm':     f'{fig_dir}/{basin}_melt_season_heatmap_norm_wy{wy}.png',
        'hyps':          f'{fig_dir}/{basin}_hyps_heatmap_wy{wy}.png',
        'terrain':       os.path.join(args.outdir, 'terrain', f'{basin}_elevation.png'),
    }

    # --- Basin mean timeseries ---
    _out_bm = _out['basin_mean']
    if not _figure_exists(_out_bm, ow):
        em_basin_mean = em_ds.mean(dim=['x', 'y']).compute()
        plot_basin_mean_timeseries(
            em_basin_mean, basin, wy,
            out_fn=_out_bm, overwrite=ow, dry_run=dr,
        )
    else:
        LOGGER.info('Figure exists, skipping computation: %s', _out_bm)

    # melt_cond is a lazy dask expression — defining it is free; .compute() only
    # happens inside plot_spatial_summary or compute_variable_elevation_matrix
    melt_cond = (em_ds['sum_EB'] > 0) & (em_ds['snowmelt'] > 0)

    # --- Melt days ---
    # melt_cond is lazy so no upstream computation to skip; _save_figure handles the file check
    plot_spatial_summary(
        melt_cond, wy,
        sum_dim='time',
        map_title=f'Melt days WY {wy}', hist_title=f'Melt days distribution WY {wy}',
        vmin=0, vmax=120, cbar_label='Melt days',
        bins=np.arange(0, 121, 5), hist_xlabel='Melt days',
        out_fn=_out['melt_days'], overwrite=ow, dry_run=dr,
    )

    # --- Melt timing chain ---
    # get_melt_init and get_meltout are expensive (full zarr load + rolling window).
    # Skip the entire chain only when ALL chain outputs already exist.
    _chain = [_out['melt_init'], _out['meltout'], _out['melt_season'],
              _out['duration_elev'], _out['freq'], _out['freq_norm'], _out['hyps']]

    if _all_exist(_chain, ow):
        # All chain plots exist — skip get_melt_init and get_meltout entirely.
        # Use melt_cond (sum_EB > 0 & snowmelt > 0) as a proxy for during_melt_season
        # to build the elevation matrix for SWI / EB normalization downstream.
        # melt_cond is a cheaper proxy (no rolling window) and is semantically
        # appropriate for "how many pixels are actively melting" normalization.
        LOGGER.info('All melt-chain outputs exist — using melt_cond proxy, skipping melt timing')
        melt_season_ds = melt_cond.rename('during_melt_season').to_dataset()
    else:
        # --- Melt initiation ---
        melt_timing = get_melt_init(em_ds)
        plot_spatial_summary(
            time_to_dowy(melt_timing, wy).where(melt_timing.notnull()), wy,
            map_title=f'Melt initiation WY{wy}',
            cbar_label='Melt initiation DOWY',
            bins=np.arange(1, 275, 5), hist_xlabel='Melt initiation DOWY',
            out_fn=_out['melt_init'], overwrite=ow, dry_run=dr,
        )

        # --- Meltout ---
        meltout_timing = get_meltout(em_ds, melt_timing)
        # NaT pixels: melt initiated but no meltout detected — shown as fill colour.
        # melt_season_ds uses meltout_filled (WY-end fallback) for duration calculations.
        plot_spatial_summary(
            time_to_dowy(meltout_timing, wy).where(meltout_timing.notnull()), wy,
            map_title=f'Meltout WY{wy}',
            cbar_label='Meltout DOWY',
            bins=np.arange(1, 275, 5), hist_xlabel='Meltout DOWY',
            out_fn=_out['meltout'], overwrite=ow, dry_run=dr,
        )

        last_day       = em_ds['time'].values[-1]
        meltout_filled = meltout_timing.fillna(last_day).where(melt_timing.notnull())
        during_melt_season = (em_ds['time'] >= melt_timing) & (em_ds['time'] <= meltout_filled)
        melt_season_ds = during_melt_season.to_dataset(name='during_melt_season')

        plot_spatial_summary(
            melt_season_ds['during_melt_season'], wy,
            sum_dim='time',
            map_title=f'Melt season duration WY {wy}',
            vmin=0, vmax=100, cbar_label='days',
            bins=np.arange(1, 366, 5), hist_xlabel='Melt season duration (days)',
            out_fn=_out['melt_season'], overwrite=ow, dry_run=dr,
        )

    # --- Terrain ---
    # Terrain is static per basin — saved without a WY suffix, generated once
    plot_spatial_summary(
        terrain_ds['dem'],
        figsize=(10, 4),
        map_title=f'{basin.capitalize()} elevation',
        hist_title=f'{basin.capitalize()} elevation distribution',
        cbar_label='masl', hist_xlabel='Elevation (masl)',
        out_fn=_out['terrain'], overwrite=ow, dry_run=dr,
    )

    # --- Elevation-based heatmaps --
    # Default approach
    # elev_bins = np.arange(
    #     np.round(terrain_ds['dem'].min().values, -2),
    #     np.round(terrain_ds['dem'].max().values, -2) + bin_step,
    #     bin_step,
    # )

    # shared elev bins across basins
    elev_bins = np.arange(1200, 4400 + bin_step, bin_step)

    duration_flat = melt_season_ds['during_melt_season'].sum(dim='time').values.ravel()
    melt_bits = compute_elevation_melt_matrix(melt_season_ds, terrain_ds, elev_bins)
    elev_flat, melt_count_matrix, one_hot, valid, elev_bin_idx_valid = melt_bits
    n_bins         = len(elev_bins) - 1
    pixels_per_bin = one_hot.sum(axis=0)   # (n_bins,) — 0 for elevation ranges absent from basin
    dowy           = time_to_dowy(melt_season_ds['time'].values, wy)

    if not _figure_exists(_out['duration_elev'], ow):
        plot_duration_elevation(
            duration_flat, elev_flat, elev_bins, basin, wy,
            out_fn=_out['duration_elev'], overwrite=ow, dry_run=dr,
        )
    if not _figure_exists(_out['freq'], ow):
        plot_melt_frequency_heatmap(
            melt_count_matrix, dowy, elev_bins, basin, wy,
            out_fn=_out['freq'], overwrite=ow, dry_run=dr,
        )
    if not _figure_exists(_out['freq_norm'], ow):
        norm_matrix = normalize_melt_matrix(melt_count_matrix, one_hot)
        plot_melt_frequency_heatmap(
            norm_matrix, dowy, elev_bins, basin, wy,
            vmin=0, vmax=1,
            out_fn=_out['freq_norm'], overwrite=ow, dry_run=dr,
        )
    if not _figure_exists(_out['hyps'], ow):
        plot_hyps_heatmap(
            melt_count_matrix, dowy, elev_bins, elev_flat, basin, wy,
            out_fn=_out['hyps'], overwrite=ow, dry_run=dr,
        )

    # --- SWI contribution heatmaps ---
    # Define output filenames up front so existence checks share the same paths
    _swi = {
        'sum_taf':    f'{fig_dir}/{basin}_swi_sum_taf_wy{wy}.png',
        'sum_mm':     f'{fig_dir}/{basin}_swi_sum_mm_wy{wy}.png',
        'mean_all':   f'{fig_dir}/{basin}_swi_mean_all_wy{wy}.png',
        'mean_melt':  f'{fig_dir}/{basin}_swi_mean_melt_wy{wy}.png',
        'median_all': f'{fig_dir}/{basin}_swi_median_all_wy{wy}.png',
        'median_melt':f'{fig_dir}/{basin}_swi_median_melt_wy{wy}.png',
    }

    # swi_sum NetCDF cache — saves (dowy, elev_bin) in TAF alongside other caches
    # so megafigures can be assembled without re-reading from Zarr:
    #   ds = xr.open_dataset(fn).expand_dims(basin=[basin], wy=[wy])
    #   combined = xr.combine_by_coords(ds_list)   # → (basin, wy, dowy, elev_bin)
    _taf_scale  = (100 ** 2) * 0.000810714 / 1_000_000      # 100 m pixel resolution
    _swi_sum_nc = os.path.join(args.workdir, 'zarr_stores',
                               f'{basin}_swi_sum_taf_wy{wy}.nc')

    # Compute if either figure or the cache NetCDF is missing
    _need_sum = not (
        _figure_exists(_swi['sum_taf'], ow)
        and _figure_exists(_swi['sum_mm'], ow)
        and (os.path.exists(_swi_sum_nc) and not ow)
    )

    if _need_sum:
        swi_sum = compute_variable_elevation_matrix(
            em_ds, 'SWI', melt_cond, valid, one_hot, aggregation='sum',
        )
        if not os.path.exists(_swi_sum_nc) or ow:
            xr.Dataset({
                'swi_taf': xr.DataArray(
                    swi_sum * _taf_scale,
                    dims=['dowy', 'elev_bin'],
                    coords={'dowy': dowy, 'elev_bin': elev_bins[:-1]},
                    attrs={'units': 'TAF',
                           'long_name': 'Total SWI per elevation bin per day'},
                ),
                # pixels_per_bin lets megafigure distinguish truly absent bins
                # (0 pixels → grey) from present-but-no-SWI bins (0 value → no tint)
                'pixels_per_bin': xr.DataArray(
                    pixels_per_bin.astype(float),
                    dims=['elev_bin'],
                    coords={'elev_bin': elev_bins[:-1]},
                    attrs={'long_name': 'Number of valid pixels per elevation bin'},
                ),
            }).assign_attrs(basin=basin, wy=int(wy)).to_netcdf(_swi_sum_nc)
            LOGGER.info('Saved SWI sum cache: %s', _swi_sum_nc)
        plot_variable_elevation_heatmap(
            swi_sum, 'SWI', dowy, elev_bins, wy, basin, vmin=0,
            units='taf', aggregation='sum', pixels_per_bin=pixels_per_bin,
            out_fn=_swi['sum_taf'], overwrite=ow, dry_run=dr,
        )
        plot_variable_elevation_heatmap(
            swi_sum, 'SWI', dowy, elev_bins, wy, basin, vmin=0,
            units='mm', aggregation='sum', pixels_per_bin=pixels_per_bin,
            out_fn=_swi['sum_mm'], overwrite=ow, dry_run=dr,
        )
    else:
        LOGGER.info('SWI sum figures and cache exist, skipping computation')

    # 3–6: each aggregation is independent — skip compute if output exists
    for agg, out_fn in [
        ('mean_all',    _swi['mean_all']),
        ('mean_melt',   _swi['mean_melt']),
        ('median_all',  _swi['median_all']),
        ('median_melt', _swi['median_melt']),
    ]:
        if _figure_exists(out_fn, ow):
            LOGGER.info('Figure exists, skipping computation: %s', out_fn)
            continue
        plot_variable_elevation_heatmap(
            compute_variable_elevation_matrix(
                em_ds, 'SWI', melt_cond, valid, one_hot,
                aggregation=agg,
                melt_count_matrix=(melt_count_matrix if 'melt' in agg else None),
            ),
            'SWI', dowy, elev_bins, wy, basin, vmin=0,
            units='mm', aggregation=agg, pixels_per_bin=pixels_per_bin,
            out_fn=out_fn, overwrite=ow, dry_run=dr,
        )

    # --- EB proportions ---
    zarr_dir = os.path.join(args.workdir, 'zarr_stores')

    # Optionally split net_rad → net_solar + net_LW for finer EB attribution.
    # Requires a pre-built net_solar zarr (run build_zarr_datacube.py -var net_solar).
    # Falls back to net_rad as a single term if the zarr is absent or the flag is set.
    net_solar_fn = f'{zarr_dir}/{basin}_unified_net_solar_wy{wy}.zarr'
    split_net_rad = (not args.no_net_solar_split
                     and os.path.exists(net_solar_fn))

    if split_net_rad:
        LOGGER.info('net_solar zarr found — splitting net_rad into net_solar + net_LW')
        net_solar_ds = xr.open_zarr(net_solar_fn)
        if args.polygon:
            net_solar_ds = clip_to_polygon(net_solar_ds, args.polygon, drop=True)
            net_solar_ds = net_solar_ds.reindex_like(em_ds, method='nearest')
        em_eb = (
            em_ds
            .assign(
                net_solar = net_solar_ds['net_solar'],
                net_LW    = em_ds['net_rad'] - net_solar_ds['net_solar'],
            )
            .drop_vars('net_rad')   # net_rad = net_solar + net_LW; drop to
                                    # prevent accidental double-counting
        )
        eb_terms_active = ['net_solar', 'net_LW', 'sensible_heat',
                           'latent_heat', 'snow_soil', 'precip_advected']
        prop_suffix = 'split'
    else:
        if args.no_net_solar_split:
            LOGGER.info('net_rad splitting disabled by --no-net-solar-split')
        else:
            LOGGER.info('net_solar zarr not found — using net_rad as a single term')
        em_eb           = em_ds
        eb_terms_active = EB_TERMS
        prop_suffix     = 'netrad'

    prop_fn = f'{zarr_dir}/{basin}_eb_proportions_{prop_suffix}_wy{wy}.zarr'

    if not os.path.exists(prop_fn) or ow:
        LOGGER.info('Computing EB proportions (%s)...', prop_suffix)
        prop_ds = compute_eb_proportions(em_eb, melt_cond, eb_terms=eb_terms_active)
        prop_ds.to_zarr(prop_fn, mode='w-' if not ow else 'w', consolidated=True)
        LOGGER.info('Saved EB proportions: %s', prop_fn)
    else:
        LOGGER.info('EB proportions already exist, loading: %s', prop_fn)

    prop_ds = xr.open_zarr(prop_fn)

    # Guard: cached zarr spatial dims must match the current em_eb grid.
    # A mismatch means the zarr was built under different clipping settings
    # (e.g. previously without --polygon). Re-run with --overwrite to fix.
    _prop_nx = prop_ds.sizes.get('x')
    _prop_ny = prop_ds.sizes.get('y')
    _em_nx   = em_eb.sizes.get('x')
    _em_ny   = em_eb.sizes.get('y')
    if _prop_nx != _em_nx or _prop_ny != _em_ny:
        raise ValueError(
            f'EB proportions zarr spatial dims ({_prop_nx}×{_prop_ny}) do not match '
            f'em_ds ({_em_nx}×{_em_ny}). The zarr was likely built with different '
            f'clipping settings. Re-run with --overwrite to regenerate it.'
        )

    # Rank all terms globally by mean fractional contribution.
    # Mean is used here (not median) because it requires only a fast dask tree
    # reduction rather than a full sort; ranking only needs relative order.
    # Median is used for all per-bin aggregation below.
    term_global_means = {
        term: float(prop_ds[term].mean().compute()) for term in eb_terms_active
    }
    ranked_terms = sorted(eb_terms_active, key=lambda t: term_global_means[t], reverse=True)
    term_rank    = {term: i + 1 for i, term in enumerate(ranked_terms)}
    LOGGER.info('EB term global ranking: %s',
                ', '.join(f'{t} (#{r})' for t, r in
                          sorted(term_rank.items(), key=lambda x: x[1])))

    # --- EB median matrices (cached as NetCDF) ---
    # The (n_terms, n_dowy, n_elev_bin) median matrices are small (~30 KB) but
    # expensive to compute (one full proportions zarr load per term). Cache them
    # as a self-describing NetCDF with DOWY and elevation bin coordinates so
    # subsequent runs skip the aggregation entirely.
    median_fn = prop_fn.replace('.zarr', '_medians.nc')

    if os.path.exists(median_fn) and not ow:
        LOGGER.info('Loading cached EB median matrices: %s', median_fn)
        median_ds = xr.open_dataset(median_fn)
        term_median_matrices = {
            t: median_ds['median_fraction'].sel(term=t).values
            for t in eb_terms_active
        }
    else:
        # Compute one term at a time to limit peak memory; each requires a full
        # spatial load from the proportions zarr (one read per term).
        term_median_matrices = {}
        for term in eb_terms_active:
            rank = term_rank[term]
            LOGGER.info('Aggregating EB term: %s (global rank %d)', term, rank)

            frac_flat = np.nan_to_num(
                prop_ds[term].values.reshape(len(prop_ds['time']), -1),
                nan=0.0,
            )   # (n_time, n_all_pixels)

            # Filter to valid pixels before passing to aggregate_term_median,
            # which expects (n_time, n_valid_pixels) matching elev_bin_idx_valid
            term_median_matrices[term] = aggregate_term_median(
                frac_flat[:, valid], elev_bin_idx_valid, n_bins
            )
            del frac_flat

        # Save to NetCDF with meaningful coordinates
        median_ds = xr.Dataset({
            'median_fraction': xr.DataArray(
                np.stack([term_median_matrices[t] for t in eb_terms_active], axis=0),
                dims=['term', 'dowy', 'elev_bin'],
                coords={
                    'term':     eb_terms_active,
                    'dowy':     dowy,
                    'elev_bin': elev_bins[:-1],
                },
                attrs={'long_name': 'Median fractional EB contribution',
                       'units':     'fraction (0-1)'},
            )
        })
        median_ds.to_netcdf(median_fn)
        LOGGER.info('Saved EB median matrices: %s', median_fn)


    # Plot individual term heatmaps
    for term in eb_terms_active:
        rank = term_rank[term]
        plot_eb_term_heatmap(
            term_median_matrices[term], term, dowy, elev_bins, wy, basin,
            pixels_per_bin=pixels_per_bin,
            out_fn=f'{fig_dir}/{basin}_eb_rank{rank}_{term}_wy{wy}.png',
            overwrite=ow, dry_run=dr,
        )

    # Dominant term plot — requires all term matrices simultaneously
    plot_dominant_eb_term(
        term_median_matrices, dowy, elev_bins, wy, basin,
        pixels_per_bin=pixels_per_bin,
        out_fn=f'{fig_dir}/{basin}_eb_dominant_wy{wy}.png', overwrite=ow, dry_run=dr,
    )

    # --- SWI attributed to EB terms ---
    _out_swi_eb_total = f'{fig_dir}/{basin}_swi_eb_total_wy{wy}.png'
    _out_swi_eb_elev  = f'{fig_dir}/{basin}_swi_eb_elevation_wy{wy}.png'
    _swi_eb_nc        = prop_fn.replace('.zarr', '_swi_attributed.nc')

    if not (_figure_exists(_out_swi_eb_total, ow) and _figure_exists(_out_swi_eb_elev, ow)):
        LOGGER.info('Computing SWI attribution by EB term...')
        swi_by_term = compute_swi_by_eb_term(
            em_eb, prop_ds, eb_terms_active, melt_cond, valid, one_hot,
        )

        # Cache attributed values as NetCDF — enables multi-basin/multi-year
        # retrieval and shared-axes comparison plots:
        #   ds = xr.open_dataset(fn).expand_dims(basin=[basin], wy=[wy])
        #   combined = xr.concat(ds_list, dim='run')
        if not os.path.exists(_swi_eb_nc) or ow:
            _pixel_res_m = 100  # iSnobal 100 m resolution
            _taf = (_pixel_res_m ** 2) * 0.000810714 / 1_000_000
            xr.Dataset({
                'swi_attributed_taf': xr.DataArray(
                    np.stack([swi_by_term[t] * _taf for t in eb_terms_active], axis=0),
                    dims=['term', 'elev_bin'],
                    coords={'term': eb_terms_active, 'elev_bin': elev_bins[:-1]},
                    attrs={'units': 'TAF',
                           'long_name': 'Seasonal SWI attributed to each EB term per elevation bin'},
                )
            }).assign_attrs(basin=basin, wy=int(wy)).to_netcdf(_swi_eb_nc)
            LOGGER.info('Saved SWI-EB attributed values: %s', _swi_eb_nc)

        plot_swi_eb_total(
            swi_by_term, eb_terms_active, wy, basin,
            out_fn=_out_swi_eb_total, overwrite=ow, dry_run=dr,
        )
        plot_swi_eb_elevation(
            swi_by_term, eb_terms_active, elev_bins, wy, basin,
            pixels_per_bin=pixels_per_bin,
            taf_label_threshold=5,
            out_fn=_out_swi_eb_elev, overwrite=ow, dry_run=dr,
        )
    else:
        LOGGER.info('SWI-EB figures exist, skipping computation')

    # --- Aspect-based SWI analysis ---
    if 'aspect' not in terrain_ds:
        LOGGER.warning('No aspect variable in terrain_ds; skipping aspect plots. '
                       'Check that topo.nc contains an aspect layer.')
    else:
        aspect_flat = terrain_ds['aspect'].values.ravel()

        _out_dom_asp      = f'{fig_dir}/{basin}_eb_dominant_aspect_wy{wy}.png'
        _out_radial       = f'{fig_dir}/{basin}_swi_radial_wy{wy}.png'
        _out_seasonal     = f'{fig_dir}/{basin}_swi_seasonality_aspect_wy{wy}.png'
        _out_elev_asp     = f'{fig_dir}/{basin}_swi_elev_aspect_wy{wy}.png'
        _out_elev_asp_rad = f'{fig_dir}/{basin}_swi_elev_aspect_radial_wy{wy}.png'

        if not _all_exist([_out_dom_asp, _out_radial, _out_seasonal,
                           _out_elev_asp, _out_elev_asp_rad], ow):
            LOGGER.info('Computing aspect-binned SWI matrices...')
            swi_aspect, swi_elev_aspect, swi_daily_aspect = compute_swi_aspect_matrices(
                em_eb, prop_ds, eb_terms_active, melt_cond,
                valid, one_hot, aspect_flat,
            )
            plot_dominant_eb_aspect(
                swi_elev_aspect, eb_terms_active, elev_bins, wy, basin,
                out_fn=_out_dom_asp, overwrite=ow, dry_run=dr,
            )
            plot_swi_radial(
                swi_aspect, eb_terms_active, wy, basin,
                out_fn=_out_radial, overwrite=ow, dry_run=dr,
            )
            plot_swi_elev_aspect(
                swi_elev_aspect, eb_terms_active, elev_bins, wy, basin,
                out_fn=_out_elev_asp, overwrite=ow, dry_run=dr,
            )
            plot_swi_elev_aspect_radial(
                swi_elev_aspect, eb_terms_active, elev_bins, wy, basin,
                out_fn=_out_elev_asp_rad, overwrite=ow, dry_run=dr,
            )
            plot_swi_seasonality_aspect(
                swi_daily_aspect, dowy, wy, basin,
                out_fn=_out_seasonal, overwrite=ow, dry_run=dr,
            )
        else:
            LOGGER.info('Aspect figures exist, skipping computation')

    LOGGER.info('Done.')


if __name__ == '__main__':
    __main__()
