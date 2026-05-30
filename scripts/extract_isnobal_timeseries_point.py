#!/usr/bin/env python
'''
Extracts modeled outputs timeseries data from snow.nc or em.nc files at basin SNOTEL sites
and saves extracted data as csv files.
'''
import sys
import glob
import os
import logging
import time

import argparse
from pathlib import Path, PurePath
from typing import List, Tuple, Optional, Dict, Sequence

import pandas as pd
import geopandas as gpd
import xarray as xr

try:
    import processing as proc
except ModuleNotFoundError:
    repo_root = Path(__file__).resolve().parent.parent
    scripts_path = repo_root / 'scripts'
    if str(scripts_path) not in sys.path:
        sys.path.append(str(scripts_path))
    import processing as proc

LOGGER = logging.getLogger(__name__)


def _configure_logging(verbose: bool) -> None:
    """Ensure module-wide logging is configured once."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format='%(asctime)s %(levelname)s %(message)s')

def _get_var_settings(dataset: str, varname: str) -> Tuple[List[str], Optional[List[str]]]:
    """Return drop-variable list plus default vars for `varname` in `dataset`."""
    if dataset == 'snow':
        if varname == 'depth':
            drop_var_list = ['snow_density', 'specific_mass', 'liquid_water', 'temp_surf',
                             'temp_lower', 'temp_snowcover', 'thickness_lower',
                             'water_saturation', 'projection']
            return drop_var_list, ['thickness']
        if varname == 'density':
            drop_var_list = ['thickness', 'specific_mass', 'liquid_water', 'temp_surf',
                             'temp_lower', 'temp_snowcover', 'thickness_lower',
                             'water_saturation', 'projection']
            return drop_var_list, ['snow_density']
        if varname == 'swe':
            drop_var_list = ['thickness', 'snow_density', 'liquid_water', 'temp_surf',
                             'temp_lower', 'temp_snowcover', 'thickness_lower',
                             'water_saturation', 'projection']
            return drop_var_list, ['specific_mass']
        if varname == 'all':
            drop_var_list = ['liquid_water', 'temp_surf', 'temp_lower', 'temp_snowcover',
                             'thickness_lower', 'water_saturation', 'projection']
            return drop_var_list, ['thickness', 'snow_density', 'specific_mass']
        raise SystemExit("Variable not recognized, exiting...")
    return ['projection'], None

def _resolve_vars_to_extract(dataset: str, vars_to_extract: Optional[List[str]],
                             default_vars: Optional[List[str]],
                             reference_ds: xr.Dataset) -> List[str]:
    """Choose the list of variables to export based on arguments and dataset."""
    if vars_to_extract:
        return vars_to_extract
    if dataset == 'snow':
        return default_vars or []
    return list(reference_ds.data_vars)

def _write_site_csv(basin: str, label: str, dataset: str, variable: str, wy: str,
                    outdir: str, sitenames: List[str],
                    ds_list: List[xr.Dataset], method: str, overwrite: bool,
                    x_coords: Sequence[float], y_coords: Sequence[float]):
    """Sample `dataset` variable for each site and save the resulting csv."""
    # Simplify if the dataset and variable are identical
    if dataset == variable:
        model_ts_fn = f'{outdir}/{basin}_{label}_{dataset}_snotelmetloom_wy{wy}.csv'
    else:
        model_ts_fn = f'{outdir}/{basin}_{label}_{dataset}_{variable}_snotelmetloom_wy{wy}.csv'
    LOGGER.debug('writing %s', model_ts_fn)
    file_exists = os.path.exists(model_ts_fn)
    if file_exists and not overwrite:
        LOGGER.info('%s exists, skipping', model_ts_fn)
        return
    if file_exists and overwrite:
        LOGGER.info('%s exists, overwrite requested', model_ts_fn)

    t_total = time.perf_counter()
    LOGGER.info('Sampling start variable=%s label=%s files=%d sites=%d',
                variable, label, len(ds_list), len(sitenames))

    site_index = xr.DataArray(range(len(sitenames)), dims='site')

    t_sel = time.perf_counter()
    site_samples = [
        ds[variable]
        .sel(x=list(x_coords), y=list(y_coords), method=method)
        .isel(x=site_index, y=site_index)
        for ds in ds_list
    ]
    LOGGER.info('Selection list built variable=%s elapsed=%.2fs',
                variable, time.perf_counter() - t_sel)

    t_concat = time.perf_counter()
    site_samples = xr.concat(site_samples, dim='time')
    LOGGER.info('Concat complete variable=%s elapsed=%.2fs dims=%s',
                variable, time.perf_counter() - t_concat, dict(site_samples.sizes))

    t_df = time.perf_counter()
    site_samples = site_samples.compute()
    basin_df = pd.DataFrame(site_samples.values,
                            index=site_samples['time'].values,
                            columns=sitenames)
    LOGGER.info('DataFrame build complete variable=%s elapsed=%.2fs shape=%s',
                variable, time.perf_counter() - t_df, basin_df.shape)
    if dataset == 'snow' and variable == 'specific_mass':
        basin_df = basin_df / 1000

    LOGGER.info('saving %s', model_ts_fn)
    t_csv = time.perf_counter()
    basin_df.to_csv(model_ts_fn)
    LOGGER.info('CSV write complete variable=%s elapsed=%.2fs total_elapsed=%.2fs',
                variable, time.perf_counter() - t_csv, time.perf_counter() - t_total)

def fn_list(this_dir: str, fn_pattern: str) -> List[str]:
    """Match and sort filenames based on a regex pattern in specified directory

    Parameters
    -------------
    this_dir: directory path to search
    fn_pattern: regex pattern to match files

    Returns
    -------------
    fns: list of filenames matched and sorted
    """
    fns = []
    pattern = os.path.join(this_dir, fn_pattern)
    for f in glob.glob(pattern):
        fns.append(f)
    fns.sort()
    LOGGER.debug('matched files: %s', fns)
    return fns

def prep_basin_data(basin: str, wy: List, poly_fn: str, st: str, workdir: str, site_locs_fn: str,
                    epsg: int = 32613, filter_on: Optional[int] = None) -> Tuple[str, str, gpd.GeoDataFrame, str]:
    """Prepare basin snotel data and directories for processing"""
    # Locate SNOTEL sites within basin
    found_sites = proc.locate_snotel_in_poly(poly_fn=poly_fn, site_locs_fn=site_locs_fn, epsg=epsg, buffer=200)
    LOGGER.info('Found %s sites within %s basin polygon', len(found_sites), basin)
    # Add some quick backchecks here, we appear to be pulling from CO
    LOGGER.debug('Found sites: %s', found_sites)
    # Get site names and site numbers
    sitenames = found_sites['site_name']
    sitenums = found_sites['site_num']
    LOGGER.info('Sitenames: %s', list(sitenames))

    st_arr = [st] * len(sitenums)
    LOGGER.info('Using metloom to extract SNOTEL data...')
    gdf_metloom, _ = proc.get_snotel(sitenums, sitenames, st_arr, WY=wy, epsg=epsg)

    # Get the basin directories based on input wy
    LOGGER.info('Finding basin directories...')
    basindirs = fn_list(workdir, f'{basin}*/wy{wy}/{basin}*/')

    # Based on the basindirs, generate a dict
    label_dict = dict()
    stem_lookup: Dict[str, str] = {}

    # Updated for unified runs with veg corrections ~3.25.26
    for basindir in basindirs:
        LOGGER.debug('considering basindir %s', basindir)
        stem = PurePath(basindir).stem
        ending = stem.split('_')[-1]
        LOGGER.debug('detected ending %s', ending)
        #if ending == 'unified':
        label_dict[stem] = 'unified'
        stem_lookup[basindir] = stem
    # for basindir in basindirs:
    #     ending = PurePath(basindir).stem.split('basin_')[-1]
    #     if ending == '100m':
    #         label_dict[str(PurePath(basindir).stem)] = 'iSnobal-HRRR'
    #     elif ending =='100m_solar_albedo':
    #          label_dict[str(PurePath(basindir).stem)] = 'HRRR-SPIReS'
    #     else:
    #         label_dict[str(PurePath(basindir).stem)] = 'unified'

    basindirs = [basindir for basindir in basindirs
                 if not filter_on or len(fn_list(basindir, 'run20*/snow.nc')) >= filter_on]
    LOGGER.debug('%d basindirs remain after filter', len(basindirs))

    # Now generate the labels based on the basindirs and the dict you've created
    labels = [label_dict[stem_lookup[basindir]] for basindir in basindirs]
    LOGGER.debug('using labels %s', labels)

    return labels, basindirs, gdf_metloom, sitenames

def modify_site_locs(poly_fn, site_locs_fn, epsg='32613', outepsg=None):
    """Modify site locations to match the basin projection"""
    if outepsg is None:
        try:
            # From the shp file, get the correct epsg
            poly_gdf = gpd.read_file(poly_fn)
            outepsg = poly_gdf.crs.to_epsg()
            LOGGER.debug('Detected output EPSG %s', outepsg)
        except FileNotFoundError:
            LOGGER.error("No output EPSG provided or found in %s, exiting...", poly_fn)
            sys.exit(1)
    outname = f'{site_locs_fn.split(epsg)[0]}{outepsg}.json'

    # Check to see if this file exists
    if not os.path.exists(outname):
        # Load the SNOTEL sites file
        all_sites_gdf = gpd.read_file(site_locs_fn)

        # Reproject to the basin projection
        all_sites_gdf_out = all_sites_gdf.to_crs(f'EPSG:{outepsg}')

        # Save to file if this does not exist
        LOGGER.info('Writing %s to file...', outname)
        all_sites_gdf_out.to_file(outname)
    else:
        LOGGER.debug('Using cached site locs file %s', outname)

    return outname, outepsg

def extract_timeseries(basindirs: list, labels: list, basin: str,
                       wy: str, gdf_metloom: gpd.GeoDataFrame,
                       sitenames: list, varname: str = 'all',
                       method: str = 'nearest',
                       chunks: str = 'auto', month: str = 'run20',
                       dataset: str = 'snow',
                       vars_to_extract: Optional[List[str]] = None,
                       overwrite: bool = False
                       ):
    """Extract timeseries data from snow.nc or em.nc files and write to csv.

    Parameters
    -------------
    basindirs: list of model output directories.
    labels: short names for each run.
    gdf_metloom: points to sample.
    sitenames: full list of SNOTEL site names.
    varname: shorthand for snow variable sets (depth/density/swe/all).
    method: resampling method for nearest neighbor selection.
    chunks: xarray chunks.
    month: run directory pattern.
    dataset: which netcdf to read ('snow' or 'em').
    vars_to_extract: explicit variables to export; for em default is all data_vars.
    overwrite: whether to overwrite existing csvs.

    Returns
    -------------
    None
    """
    dataset = dataset.lower()
    if dataset not in {'snow', 'em', 'net_solar', 'thermal', 'air_temp', 'hrrr_solar'}:
        raise SystemExit(f"Unsupported dataset '{dataset}'. \
                         Choose 'snow', 'em', 'net_solar', 'thermal', 'air_temp', or 'hrrr_solar'.")

    LOGGER.debug('selected varname %s', varname)
    drop_var_list, default_vars = _get_var_settings(dataset, varname)

    x_coords = list(gdf_metloom.geometry.x.values)
    y_coords = list(gdf_metloom.geometry.y.values)
    LOGGER.debug('sampling x_coords=%s y_coords=%s', x_coords, y_coords)

    for kdx, (label, basindir) in enumerate(zip(labels, basindirs)):
        t_run = time.perf_counter()
        LOGGER.info('processing run %d/%d label=%s', kdx + 1, len(labels), label)
        outdir = PurePath(basindir).parents[0].as_posix()
        pattern = f"{month}*/{dataset}.nc"
        basin_days = fn_list(basindir, pattern)
        if not basin_days:
            LOGGER.warning('No %s.nc files found in %s, skipping', dataset, basindir)
            continue
        LOGGER.info('found %s files for %s', len(basin_days), label)
        t_open = time.perf_counter()
        ds_list = [xr.open_dataset(day_fn,
                                   chunks=chunks,
                                   engine="netcdf4",
                                   drop_variables=drop_var_list) for day_fn in basin_days]
        LOGGER.info('Opened %d datasets for %s elapsed=%.2fs',
                    len(ds_list), label, time.perf_counter() - t_open)
        LOGGER.info('Resolving variables to extract...')
        vars_from_dataset = _resolve_vars_to_extract(dataset,
                                                     vars_to_extract,
                                                     default_vars,
                                                     ds_list[0])
        LOGGER.info('Variables resolved: %s', vars_from_dataset)
        for v in vars_from_dataset:
            if v not in ds_list[0].data_vars:
                LOGGER.info('%s not present in %s.nc, skipping', v, dataset)
                continue
            LOGGER.info('Writing data for variable: %s', v)
            t_var = time.perf_counter()
            _write_site_csv(basin=basin, label=label, dataset=dataset, variable=v,
                            wy=wy, outdir=outdir, sitenames=sitenames,
                            ds_list=ds_list, method=method,
                            overwrite=overwrite,
                            x_coords=x_coords,
                            y_coords=y_coords)
            LOGGER.info('Variable complete variable=%s label=%s elapsed=%.2fs',
                        v, label, time.perf_counter() - t_var)
        LOGGER.info('Run complete label=%s elapsed=%.2fs',
                    label, time.perf_counter() - t_run)

def parse_arguments():
    """Parse command line arguments and return the parsed namespace.

    Returns
    -------------
    argparse.Namespace: parsed command line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Extract iSnobal model output variables from snow.nc or em.nc "
                    "at point sites [SNOTEL] within a basin for a given water year."
    )
    parser.add_argument(
        'basin',
        type=str,
        help='Basin name',
    )
    parser.add_argument(
        'wy',
        type=int,
        help='Water year of interest',
    )
    parser.add_argument(
        '-workdir',
        type=str,
        help='Directory containing model output basindirs',
        default='/uufs/chpc.utah.edu/common/home/skiles-group2/model_runs_jmh/thp/',
    )
    parser.add_argument(
        '-poly',
        '--shapefile',
        type=str,
        help='Shapefile of basin polygon',
        default=None,
    )
    parser.add_argument(
        '-st',
        '--state',
        type=str,
        help='State abbreviation',
        default='CO',
    )
    parser.add_argument(
        '-loc',
        '--epsg',
        type=str,
        help='EPSG code for site locations and shapefile; default is 32613 (UTM zone 13N)',
        default='32613',
    )
    parser.add_argument(
        '-var',
        '--variable',
        type=str,
        choices=['depth', 'density', 'swe', 'all'],
        default='depth',
        help='iSnobal snow variable',
    )
    parser.add_argument(
        '-f',
        '--filter',
        type=int,
        help='Filter on number of snow.nc files',
        default=None,
    )
    parser.add_argument(
        '--dataset',
        choices=['snow', 'em', 'net_solar', 'thermal', 'air_temp', 'hrrr_solar'],
        default='snow',
        help='Name of the model output netcdf file to read from each run directory',
    )
    parser.add_argument(
        '--vars',
        nargs='+',
        help='List of variables to output; defaults to the varname selection for snow and all data_vars for all others',
    )
    parser.add_argument(
        '-o',
        '--overwrite',
        help='Overwrite existing files',
        action='store_true',
    )
    parser.add_argument(
        '-v',
        '--verbose',
        help='Enable verbose output',
        action='store_true',
    )
    return parser.parse_args()

def __main__():
    args = parse_arguments()
    basin = args.basin
    wy = args.wy
    workdir = args.workdir
    poly_fn = args.shapefile
    state_abbrev = args.state
    epsg = args.epsg
    varname = args.variable
    filter_on = args.filter
    overwrite = args.overwrite
    verbose = args.verbose
    dataset = args.dataset
    vars_to_extract = args.vars

    _configure_logging(verbose)
    # Set up directories
    ancillary_dir = '/uufs/chpc.utah.edu/common/home/skiles-group3/ancillary_sdswe_products/'

    # Set up filename for all active SNOTEL sites
    site_locs_fn = f'{ancillary_dir}/SNOTEL/snotel_sites_{epsg}.json'

    if poly_fn is None:
        LOGGER.info('No shapefile provided, using default detection...')
        poly_dir = '/uufs/chpc.utah.edu/common/home/skiles-group3/jmhu/ancillary/polys'
        poly_fn = fn_list(poly_dir, f'*{basin}*shp')[0]
        poly_gdf = gpd.read_file(poly_fn)
        poly_epsg = poly_gdf.crs.to_epsg()
        # Check if the detected EPSG matches the site locations EPSG
        if str(poly_epsg) != epsg:
            # if it doesn't match, log an error, the filename, epsg codes, and exit
            # the basin polygon should take precedence, but may not be appropriately projected
            LOGGER.error('EPSG mismatch: detected %s from %s but expected %s from site locations file %s. \
                Please provide a shapefile with matching EPSG or reproject the shapefile to match the site locations EPSG.',
                poly_epsg, poly_fn, epsg, site_locs_fn)
            sys.exit(1)
    else:
        LOGGER.info('Detected input shapefile: %s', poly_fn)
        LOGGER.info('Detected EPSG is %s', epsg)

     ### SNOTEL extraction and point specification
    LOGGER.info('Preparing basin data for %s...', basin)

    labels, basindirs, gdf_metloom, sitenames = prep_basin_data(basin=basin, wy=wy,
                                                                poly_fn=poly_fn, st=state_abbrev,
                                                                epsg=epsg,
                                                                workdir=workdir,
                                                                site_locs_fn=site_locs_fn,
                                                                filter_on=filter_on)
    LOGGER.info('Extracting timeseries data...')

    extract_timeseries(basindirs, labels, basin,
                       wy, gdf_metloom, sitenames,
                       overwrite=overwrite, varname=varname,
                       dataset=dataset, vars_to_extract=vars_to_extract)

if __name__ == "__main__":
    __main__()
