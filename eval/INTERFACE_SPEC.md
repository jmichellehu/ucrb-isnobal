# iSnobal Evaluation Framework — Interface Specification

**Version:** 1.0  
**Date:** 2026-05-29  
**Status:** LOCKED CONTRACT — both Plan 1 (notebook) and Plan 2 (package) must implement these signatures exactly.

---

## 0. Scope and Conventions

This document defines the shared interface between:
- **Plan 1**: `eval/eval_utils.py` — standalone functions imported by a report notebook.
- **Plan 2**: `isnobal_eval/` Python package with the same public API.

Constraints:
- All function signatures are identical across both plans.
- The config dict/YAML schema is the single point of truth for file paths and parameters.
- Variable names in the registry below are the only accepted names for zarr variables, DataFrame columns, and function arguments.
- Coordinate names in all xarray objects follow the zarr stores: `x`, `y`, `time`.
- CRS for all spatial data is **EPSG:32613** (UTM zone 13N) for the animas basin. Other basins may differ; the EPSG is always read from the config.

---

## 1. Config Schema

The config is a YAML file loaded into a plain Python `dict`. Both plans call `load_config(path)` to read it.

### 1.1 Complete example — Animas WY2024

```yaml
basin: animas
water_year: 2024

paths:
  snow_zarr: /uufs/chpc.utah.edu/common/home/skiles-group2/model_runs_jmh/thp/zarr_stores/animas_unified_snow_wy2024.zarr
  em_zarr:   /uufs/chpc.utah.edu/common/home/skiles-group2/model_runs_jmh/thp/zarr_stores/animas_unified_em_wy2024.zarr
  net_solar_zarr: /uufs/chpc.utah.edu/common/home/skiles-group2/model_runs_jmh/thp/zarr_stores/animas_unified_net_solar_wy2024.zarr
  topo_nc:   /uufs/chpc.utah.edu/common/home/skiles-group3/jmhu/isnobal_scripts/animas_setup/output_100m/topo.nc
  snotel_sites_geojson: /uufs/chpc.utah.edu/common/home/skiles-group3/ancillary_sdswe_products/SNOTEL/snotel_sites_32613.json
  basin_poly: /uufs/chpc.utah.edu/common/home/skiles-group3/jmhu/ancillary/polys/animas.shp
  output_dir: /uufs/chpc.utah.edu/common/home/u6058223/git_dirs/ucrb-isnobal/eval/output/animas_wy2024

epsg: 32613
state: CO

terrain_filter:
  elev_range: [2500, 4000]       # meters, inclusive on both ends; null means no filter
  aspect_range: [315, 45]        # degrees, interpreted as a circular range (North-facing); null means no filter
  slope_max: 45                  # degrees; null means no filter
  veg_classes: null              # list of integer veg_type codes, or null for all classes

snotel:
  variable_mapping:
    SNOWDEPTH: thickness         # metloom column SNOWDEPTH (inches) → iSnobal variable name; unit conversion to meters applied in loader
    SWE: specific_mass           # metloom column SWE (inches) → iSnobal variable name; convert inches→m then m→kg/m2 (*1000)
  snowvar: SNOWDEPTH             # which variable to pull by default: SNOWDEPTH or SWE
  buffer_m: 200                  # buffer in meters applied to basin polygon when finding sites
```

### 1.2 Config field reference

| Field | Type | Required | Notes |
|---|---|---|---|
| `basin` | str | yes | Lowercase basin name, matches zarr store filename stem |
| `water_year` | int | yes | Four-digit water year, e.g. 2024 |
| `paths.snow_zarr` | str | yes | Absolute path to snow zarr store |
| `paths.em_zarr` | str | yes | Absolute path to em zarr store |
| `paths.net_solar_zarr` | str | yes | Absolute path to net_solar zarr store |
| `paths.topo_nc` | str | yes | Absolute path to topo.nc (basin setup output) |
| `paths.snotel_sites_geojson` | str | yes | Absolute path to snotel_sites_{epsg}.json |
| `paths.basin_poly` | str | yes | Absolute path to basin polygon shapefile (.shp) |
| `paths.output_dir` | str | yes | Directory for all output figures and CSVs |
| `epsg` | int | yes | UTM zone EPSG for the basin, e.g. 32613 |
| `state` | str | yes | Two-letter state abbreviation, e.g. CO |
| `terrain_filter.elev_range` | [float, float] or null | no | Min/max elevation in meters |
| `terrain_filter.aspect_range` | [float, float] or null | no | Min/max aspect in degrees; supports wrap-around (e.g. [315, 45] = North) |
| `terrain_filter.slope_max` | float or null | no | Maximum slope in degrees |
| `terrain_filter.veg_classes` | list[int] or null | no | LANDFIRE veg_type integer codes |
| `snotel.variable_mapping` | dict | yes | Maps metloom column names to iSnobal zarr variable names |
| `snotel.snowvar` | str | yes | `SNOWDEPTH` or `SWE` |
| `snotel.buffer_m` | int | no | Default 200 |

---

## 2. Variable Name Registry

This is the single source of truth. All code must use these exact names as zarr keys, DataFrame column names, and function `variable` arguments.

### 2.1 Snow store (`animas_unified_snow_wy2024.zarr`)

Dimensions: `time` (366), `y` (820), `x` (492). Grid: 100 m UTM 13N.

| zarr variable | Human label | Units | Description |
|---|---|---|---|
| `thickness` | Snow depth | m | Predicted thickness of the snowcover |
| `snow_density` | Snow density | kg m-3 | Predicted average snow density |
| `specific_mass` | SWE | kg m-2 | Predicted specific mass of the snowcover |
| `liquid_water` | Liquid water | kg m-2 | Predicted mass of liquid water in the snowcover |
| `temp_surf` | Surface temperature | C | Predicted temperature of the surface layer |
| `temp_lower` | Lower layer temperature | C | Predicted temperature of the lower layer |
| `temp_snowcover` | Snowcover temperature | C | Predicted temperature of the snowcover |
| `thickness_lower` | Lower layer depth | m | Predicted thickness of the lower layer |
| `water_saturation` | Water saturation | percent | Predicted percentage of liquid water saturation |

### 2.2 Energy model store (`animas_unified_em_wy2024.zarr`)

Dimensions: `time` (366), `y` (820), `x` (492). Daily values; flux variables (`snowmelt`, `evaporation`, `SWI`) are sums of h00 + h23; all others are h23 end-of-day values.

| zarr variable | Human label | Units | Description |
|---|---|---|---|
| `net_rad` | Net all-wave radiation | W m-2 | Average net all-wave radiation |
| `sensible_heat` | Sensible heat | W m-2 | Average sensible heat transfer |
| `latent_heat` | Latent heat | W m-2 | Average latent heat exchange |
| `snow_soil` | Snow-soil heat flux | W m-2 | Average snow/soil heat exchange |
| `precip_advected` | Advected heat from precipitation | W m-2 | Average advected heat from precipitation |
| `sum_EB` | Sum of energy balance terms | W m-2 | Average sum of EB terms for snowcover |
| `evaporation` | Evaporation | kg m-2 | Total evaporation (h00+h23 sum) |
| `snowmelt` | Snowmelt | kg m-2 | Total snowmelt (h00+h23 sum) |
| `SWI` | Surface water input (runoff) | kg m-2 | Total runoff / surface water input (h00+h23 sum) |
| `cold_content` | Cold content | J m-2 | Snowcover cold content |

**Derived variable (not in zarr; computed on the fly):**

| variable name | Human label | Units | Derivation |
|---|---|---|---|
| `net_LW` | Net longwave radiation | W m-2 | `net_rad - net_solar` |

### 2.3 Net solar store (`animas_unified_net_solar_wy2024.zarr`)

Dimensions: `time` (366), `y` (820), `x` (492).

| zarr variable | Human label | Units | Description |
|---|---|---|---|
| `net_solar` | Net solar radiation | W m-2 | Net solar radiation |

### 2.4 Topo file (`topo.nc`) — static, no time dimension

Source: SMRF/AWSM basin_setup output. Dimensions: `y` (820), `x` (492) for animas.

| topo variable | Human label | Units | Notes |
|---|---|---|---|
| `dem` | Elevation | m | Digital elevation model |
| `slope` | Slope | degrees | Slope angle |
| `mask` | Basin mask | unitless (uint8) | 1 = in basin |
| `veg_height` | Vegetation height | m | LANDFIRE canopy height |
| `veg_type` | Vegetation type | unitless (uint16) | LANDFIRE EVT integer code |
| `veg_tau` | Vegetation transmissivity | unitless [0–1] | Canopy transmissivity to shortwave |
| `veg_k` | Vegetation extinction coefficient | unitless | Canopy extinction coefficient |
| `sky_view_factor` | Sky view factor | unitless | For 16 azimuth angles |
| `terrain_config_factor` | Terrain configuration factor | unitless | For 16 azimuth angles |

**Note:** `aspect` is not stored in `topo.nc`. It must be computed from `dem` using `numpy.gradient` or `richdem`/`xrspatial.aspect` prior to terrain filtering. The computed aspect array should be registered under the name `aspect` (units: degrees, 0=North, clockwise).

### 2.5 SNOTEL variable mapping

metloom returns columns in imperial units. The loaders apply conversions before storing.

| metloom column | iSnobal zarr variable | Conversion |
|---|---|---|
| `SNOWDEPTH` (inches) | `thickness` | `* 0.0254` → meters |
| `SWE` (inches) | `specific_mass` | `* 0.0254 * 1000` → kg m-2 |

---

## 3. Loader API

All loaders accept a `config` dict as the first argument and return lazily-loaded xarray/geopandas objects where possible. Loaders do not apply terrain filters; that is the responsibility of `build_terrain_mask`.

```python
def load_config(path: str) -> dict:
    """Load a YAML config file and return it as a dict."""

def load_snow(config: dict) -> xr.Dataset:
    """Open the snow zarr store for the basin/WY in config.
    Returns a lazy xarray Dataset with dims (time, y, x).
    Wraps xr.open_zarr on config['paths']['snow_zarr'].
    """

def load_em(config: dict) -> xr.Dataset:
    """Open the energy model zarr store for the basin/WY in config.
    Returns a lazy xarray Dataset with dims (time, y, x).
    Wraps xr.open_zarr on config['paths']['em_zarr'].
    """

def load_net_solar(config: dict) -> xr.Dataset:
    """Open the net solar zarr store for the basin/WY in config.
    Returns a lazy xarray Dataset with dims (time, y, x).
    Wraps xr.open_zarr on config['paths']['net_solar_zarr'].
    """

def load_topo(config: dict) -> xr.Dataset:
    """Open the topo.nc file for the basin in config.
    Returns an xarray Dataset with dims (y, x) containing:
    dem, slope, mask, veg_height, veg_type, veg_tau, veg_k,
    sky_view_factor, terrain_config_factor.
    Aspect is NOT present in topo.nc; call compute_aspect(topo_ds)
    to add it.
    Wraps xr.open_dataset on config['paths']['topo_nc'].
    """

def compute_aspect(topo_ds: xr.Dataset) -> xr.Dataset:
    """Compute aspect (degrees, 0=North, clockwise) from topo_ds['dem']
    and return topo_ds with a new 'aspect' variable added.
    Uses numpy gradient-based computation consistent with processing.bin_aspect().
    """

def load_snotel(config: dict, site_ids: list[str] | None = None) -> pd.DataFrame:
    """Fetch daily SNOTEL data for sites within the basin polygon.

    Uses proc.locate_snotel_in_poly to find sites, then proc.get_snotel
    (metloom) to retrieve data. Applies unit conversion per Section 2.5.

    Parameters
    ----------
    config : dict
        Config dict; reads paths.snotel_sites_geojson, paths.basin_poly,
        epsg, state, water_year, snotel.snowvar, snotel.buffer_m.
    site_ids : list[str] or None
        If provided, restrict to these site_name values (from GeoJSON 'site_name'
        property). If None, returns all sites within the polygon.

    Returns
    -------
    pd.DataFrame
        MultiIndex columns: (site_name, variable_name).
        Index: DatetimeIndex (daily).
        Variable columns include the iSnobal-mapped name (e.g. 'thickness')
        alongside any original metloom columns retained for reference.
        Wraps proc.locate_snotel_in_poly and proc.get_snotel.
    """

def load_basin_poly(config: dict) -> gpd.GeoDataFrame:
    """Load the basin polygon shapefile.
    Returns a GeoDataFrame in the CRS specified by config['epsg'].
    Wraps gpd.read_file on config['paths']['basin_poly'].
    """
```

---

## 4. Analysis API

### 4.1 point_temporal — Extract time series at a single pixel

```python
def extract_point_timeseries(
    snow_ds: xr.Dataset,
    em_ds: xr.Dataset,
    net_solar_ds: xr.Dataset,
    x: float,
    y: float,
    variables: list[str] | None = None,
) -> pd.DataFrame:
    """Extract daily time series for a single pixel from all three stores.

    Selects the nearest pixel to (x, y) using xr.Dataset.sel(method='nearest').
    Computes net_LW = net_rad - net_solar and appends it as a column.

    Parameters
    ----------
    snow_ds, em_ds, net_solar_ds : xr.Dataset
        Lazy datasets from load_snow/load_em/load_net_solar.
    x, y : float
        UTM coordinates in the CRS of the datasets.
    variables : list[str] or None
        Subset of variable names from the registry (Section 2). If None,
        returns all variables from all three stores plus net_LW.

    Returns
    -------
    pd.DataFrame
        Index: DatetimeIndex (daily).
        Columns: variable names (strings from registry).
    """
```

### 4.2 spatial_distribution — Variable distribution stratified by terrain/veg

```python
def compute_terrain_distribution(
    ds: xr.Dataset,
    variable: str,
    topo_ds: xr.Dataset,
    stratify_by: str = 'elevation',
    bins: list | None = None,
) -> pd.DataFrame:
    """Compute the spatial distribution of a variable stratified by a terrain attribute.

    Parameters
    ----------
    ds : xr.Dataset
        Any of the zarr stores (snow, em, net_solar). Must contain `variable`.
    variable : str
        Variable name from Section 2 registry.
    topo_ds : xr.Dataset
        Output of load_topo (plus compute_aspect if stratify_by='aspect').
        Must contain the field required by stratify_by.
    stratify_by : str
        One of: 'elevation', 'aspect', 'slope', 'veg_class'.
        Maps to topo variables: dem, aspect, slope, veg_type.
    bins : list or None
        Bin edges. If None, uses defaults:
          elevation: 10 equally-spaced bins across dem range (proc.bin_elev logic)
          aspect:    4 cardinal bins [N, E, S, W] (proc.bin_aspect logic)
          slope:     7 class bins [0-10, 10-20, 20-30, 30-40, 40-50, 50-60, 60+]
                     (proc.bin_slope logic)
          veg_class: unique veg_type integer codes present in the domain

    Returns
    -------
    pd.DataFrame
        One row per time step, one column per bin.
        Column names encode the bin range or class label as a string,
        e.g. '2500_2750' for elevation, 'North' for aspect, '1_10' for slope,
        or the integer code string for veg_class.
        Values are the mean of `variable` over all valid (non-NaN) pixels in that bin.
    """
```

### 4.3 conditional_timeseries — Time series for a terrain-filtered pixel subset

```python
def build_terrain_mask(
    topo_ds: xr.Dataset,
    elev_range: tuple[float, float] | None = None,
    aspect_range: tuple[float, float] | None = None,
    slope_max: float | None = None,
    veg_classes: list[int] | None = None,
) -> xr.DataArray:
    """Build a 2-D boolean mask over the (y, x) grid.

    Each filter is ANDed together; pixels must satisfy all active filters.
    aspect_range is circular: (315, 45) selects 315–360 OR 0–45 degrees.
    topo_ds must contain 'dem', 'slope', 'aspect' (see compute_aspect), 'veg_type'.

    Parameters
    ----------
    topo_ds : xr.Dataset
        Output of load_topo + compute_aspect.
    elev_range : (min_m, max_m) or None
        Inclusive elevation range in meters.
    aspect_range : (min_deg, max_deg) or None
        Inclusive aspect range in degrees [0, 360). Supports circular wrap.
    slope_max : float or None
        Maximum slope in degrees (inclusive).
    veg_classes : list[int] or None
        List of veg_type integer codes to include.

    Returns
    -------
    xr.DataArray
        Boolean DataArray with dims (y, x). True where pixel is in the filtered set.
    """

def compute_masked_timeseries(
    ds: xr.Dataset,
    variable: str,
    mask: xr.DataArray,
) -> pd.DataFrame:
    """Compute summary statistics over time for pixels selected by mask.

    Parameters
    ----------
    ds : xr.Dataset
        Any zarr store. Must contain `variable` with dims (time, y, x).
    variable : str
        Variable name from Section 2 registry.
    mask : xr.DataArray
        Boolean (y, x) DataArray from build_terrain_mask.

    Returns
    -------
    pd.DataFrame
        Index: DatetimeIndex (daily).
        Columns: ['mean', 'std', 'q25', 'q75'].
        Each row is the spatial statistic over the masked pixels for that time step.
    """
```

### 4.4 validation — Model vs. SNOTEL comparison

```python
def compare_snotel(
    snow_ds: xr.Dataset,
    snotel_df: pd.DataFrame,
    variable: str,
    site_ids: list[str] | None = None,
) -> pd.DataFrame:
    """Pair modeled and observed values at SNOTEL point locations.

    Samples snow_ds[variable] at the UTM coordinates of each SNOTEL site using
    nearest-neighbor selection (method='nearest'), then aligns with snotel_df
    on a common daily DatetimeIndex.

    Parameters
    ----------
    snow_ds : xr.Dataset
        Output of load_snow. Must contain `variable`.
    snotel_df : pd.DataFrame
        Output of load_snotel. MultiIndex columns (site_name, variable_name).
    variable : str
        Must be a snow store variable from Section 2.1 (e.g. 'thickness',
        'specific_mass'). The same name is used to look up the modeled and
        observed columns.
    site_ids : list[str] or None
        Subset of site_name values. If None, uses all sites in snotel_df.

    Returns
    -------
    pd.DataFrame
        Columns: ['site_id', 'date', 'modeled', 'observed'].
        'modeled' and 'observed' values are in the units of the zarr variable
        (meters for thickness; kg m-2 for specific_mass).
        One row per (site, date) pair where both modeled and observed are non-NaN.
    """

def compute_metrics(
    modeled: np.ndarray | pd.Series,
    observed: np.ndarray | pd.Series,
) -> dict:
    """Compute a standard set of evaluation metrics.

    NaN pairs are dropped before computation.

    Parameters
    ----------
    modeled : array-like
        Modeled values (1-D).
    observed : array-like
        Observed values (1-D), same length as modeled.

    Returns
    -------
    dict with keys:
        r          : Pearson correlation coefficient
        r2         : Coefficient of determination (r^2)
        rmse       : Root mean squared error (same units as input)
        nrmse_range: RMSE normalized by (max(obs) - min(obs))
        nrmse_mean : RMSE normalized by mean(obs)
        mae        : Mean absolute error (same units as input)
        kge        : Kling-Gupta Efficiency
    """
```

---

## 5. Metrics Definitions

All formulas use clean (NaN-dropped) paired arrays. Let `o` = observed, `m` = modeled, `n` = number of valid pairs.

### Pearson correlation coefficient
```
r = corrcoef(o, m)[0, 1]
  = sum((o - mean(o)) * (m - mean(m))) / (n * std(o) * std(m))
```

### Coefficient of determination
```
r2 = r ** 2
```

### Root mean squared error
```
rmse = sqrt(nanmean((m - o) ** 2))
```

### Normalized RMSE — range
```
nrmse_range = rmse / (nanmax(o) - nanmin(o))
```

### Normalized RMSE — mean
```
nrmse_mean = rmse / nanmean(o)
```

### Mean absolute error
```
mae = nanmean(abs(m - o))
```

### Kling-Gupta Efficiency
```
alpha = std(m) / std(o)          # variability ratio
beta  = mean(m) / mean(o)        # bias ratio
kge   = 1 - sqrt((r - 1)**2 + (alpha - 1)**2 + (beta - 1)**2)
```

Source: `model_eval.ipynb` cell `calculate_kge`. The formula in that cell uses `np.std` (population std, ddof=0) and `np.mean`. Both plans must use `np.std` (ddof=0), not `pd.Series.std` (ddof=1), to match the existing notebook results.

**Interpretation:** KGE = 1 is perfect; KGE > -0.41 is considered better than the mean flow benchmark (Knoben et al. 2019 threshold, referenced in `model_eval.ipynb` plot annotations).

---

## 6. Migration Path Note

**Plan 1** places all functions in `eval/eval_utils.py` as module-level functions. The report notebook imports them as:
```python
from eval_utils import (
    load_config, load_snow, load_em, load_net_solar,
    load_topo, compute_aspect, load_snotel, load_basin_poly,
    extract_point_timeseries, compute_terrain_distribution,
    build_terrain_mask, compute_masked_timeseries,
    compare_snotel, compute_metrics,
)
```

**Plan 2** places the same functions in `isnobal_eval/loaders.py`, `isnobal_eval/analysis.py`, and `isnobal_eval/metrics.py` within an installable package. The notebook switches imports with a one-line change per import group:
```python
# Plan 1:
from eval_utils import load_snow, load_em, ...
# Plan 2:
from isnobal_eval.loaders import load_snow, load_em, ...
```

Function signatures, return types, and argument names must be byte-for-byte identical. The only difference is the module path. No wrapper or adapter layer should be needed.

---

## 7. Ancillary Path Reference

These are the canonical paths on CHPC for the test case (animas WY2024). Hard-coding these in implementations is discouraged; always read from config.

| Asset | Canonical path |
|---|---|
| SNOTEL sites (EPSG 32613) | `/uufs/chpc.utah.edu/common/home/skiles-group3/ancillary_sdswe_products/SNOTEL/snotel_sites_32613.json` |
| SNOTEL sites (EPSG 4326) | `/uufs/chpc.utah.edu/common/home/skiles-group3/ancillary_sdswe_products/SNOTEL/snotel_sites.json` |
| Animas basin polygon | `/uufs/chpc.utah.edu/common/home/skiles-group3/jmhu/ancillary/polys/animas.shp` |
| Animas topo.nc (100 m) | `/uufs/chpc.utah.edu/common/home/skiles-group3/jmhu/isnobal_scripts/animas_setup/output_100m/topo.nc` |
| Snow zarr (animas WY2024) | `/uufs/chpc.utah.edu/common/home/skiles-group2/model_runs_jmh/thp/zarr_stores/animas_unified_snow_wy2024.zarr` |
| EM zarr (animas WY2024) | `/uufs/chpc.utah.edu/common/home/skiles-group2/model_runs_jmh/thp/zarr_stores/animas_unified_em_wy2024.zarr` |
| Net solar zarr (animas WY2024) | `/uufs/chpc.utah.edu/common/home/skiles-group2/model_runs_jmh/thp/zarr_stores/animas_unified_net_solar_wy2024.zarr` |
| SNOTEL basin extracts | `/uufs/chpc.utah.edu/common/home/skiles-group3/ancillary_sdswe_products/SNOTEL/basin_extracts/` |
| Model run workdir | `/uufs/chpc.utah.edu/common/home/skiles-group2/model_runs_jmh/thp/` |

The `snotel_sites_geojson` GeoJSON has features with properties: `ntwk`, `state`, `site_name`, `ts`, `start`, `lat`, `lon`, `elev`, `county`, `huc`, `site_num`. The `site_name` property is the stable key used as `site_id` throughout this framework.

---

## 8. Data Dimensions Reference

All three zarr stores for animas WY2024 share the same spatial grid. This is the expected shape that loaders must produce:

| Dimension | Size | Notes |
|---|---|---|
| `time` | 366 | Daily; WY2024 is a leap water year (Oct 2023 – Sep 2024) |
| `y` | 820 | Northing axis, UTM 13N meters |
| `x` | 492 | Easting axis, UTM 13N meters |

Pixel resolution: 100 m. Grid origin and transform: `229889.3, 100.0, 0.0, 4205258.0, 0.0, -100.0` (GeoTransform from topo.nc).

Time encoding: `hours since 2023-10-01`, calendar=standard. After `xr.open_zarr`, the time coordinate decodes to `numpy.datetime64` values at 23:00 UTC each day (the h23 timestep convention from iSnobal).

---

*End of specification.*
