#!/usr/bin/env python
'''
This script extracts modeled snow depth, density, or water equivalent timeseries data at identified SNOTEL sites within a basin
and outputs as csv files.
'''
import sys
import glob
import os

import argparse
import pandas as pd
import geopandas as gpd
from pathlib import PurePath
from typing import List, Tuple, Optional
import xarray as xr

sys.path.append('/uufs/chpc.utah.edu/common/home/u6058223/git_dirs/ucrb-isnobal/scripts/')
import processing as proc

def fn_list(thisDir: str, fn_pattern: str, verbose: bool = False) -> List[str]:
    """Match and sort filenames based on a regex pattern in specified directory

    Parameters
    -------------
    thisDir: directory path to search
    fn_pattern: regex pattern to match files
    verbose: print filenames

    Returns
    -------------
    fns: list of filenames matched and sorted
    """
    fns = []
    for f in glob.glob(thisDir + "/" + fn_pattern):
        fns.append(f)
    fns.sort()
    if verbose:
         print(fns)
    return fns

def prep_basin_data(basin: str, WY: List, poly_fn: str, ST: str, workdir: str, site_locs_fn: str,
                    epsg: int = 32613, filter_on: Optional[int] = None, verbose: bool = False) -> Tuple[str, str, gpd.GeoDataFrame, str]:
    """Prepare basin snotel data and directories for processing"""
    # Locate SNOTEL sites within basin
    found_sites = proc.locate_snotel_in_poly(poly_fn=poly_fn, site_locs_fn=site_locs_fn, buffer=200)

    # Get site names and site numbers
    sitenames = found_sites['site_name']
    sitenums = found_sites['site_num']
    if verbose:
        print(sitenames)

    ST_arr = [ST] * len(sitenums)
    if verbose:
        print('Using metloom to extract SNOTEL data...')
    gdf_metloom, _ = proc.get_snotel(sitenums, sitenames, ST_arr, WY=WY, epsg=epsg)

    # Get the basin directories based on input wy
    if verbose:
        print('Finding basin directories...')
    basindirs = fn_list(workdir, f'{basin}*/wy{WY}/{basin}*/')

    # Based on the basindirs, generate a dict
    label_dict = dict()
    for basindir in basindirs:
        ending = PurePath(basindir).stem.split('basin_')[-1]
        if ending == '100m':
            label_dict[str(PurePath(basindir).stem)] = 'iSnobal-HRRR'
        elif ending =='100m_solar_albedo':
             label_dict[str(PurePath(basindir).stem)] = 'HRRR-MODIS'

    if filter_on:
        # Filter out basindirs that don't have enough snow.nc files, make sure most of WY is run (at least 270 files)
        basindirs = [basindir for basindir in basindirs if len(fn_list(basindir, 'run20*/snow.nc'))>=filter_on]
    else:
        basindirs = [basindir for basindir in basindirs]

    # Now generate the labels based on the basindirs and the dict you've created
    labels = [label_dict[str(PurePath(basindir).stem)] for basindir in basindirs]
    if verbose:
         print(labels)

    return labels, basindirs, gdf_metloom, sitenames

def modify_site_locs(poly_fn, site_locs_fn, epsg='32613', outepsg=None, verbose=False):
    """Modify site locations to match the basin projection"""
    if outepsg is None:
        try:
             # From the shp file, get the correct epsg
             poly_gdf = gpd.read_file(poly_fn)
             outepsg = poly_gdf.crs.to_epsg()
             if verbose:
                print(outepsg)
        except FileNotFoundError:
            print(f"No output EPSG provided or found in {poly_fn}, exiting...")
            sys.exit(1)
    outname = f'{site_locs_fn.split(epsg)[0]}{outepsg}.json'

    # Check to see if this file exists
    if not os.path.exists(outname):
        # Load the SNOTEL sites file
        all_sites_gdf = gpd.read_file(site_locs_fn)

        # Reproject to the basin projection
        all_sites_gdf_out = all_sites_gdf.to_crs(f'EPSG:{outepsg}')

        # Save to file if this does not exist
        if verbose:
            print('Writing {outname} to file...')
        all_sites_gdf_out.to_file(outname)

    return outname, outepsg

def extract_timeseries(basindirs: list, labels: list, basin: str,
                       wy: str, gdf_metloom: gpd.GeoDataFrame,
                       sitenames: list, varname: str = 'all',
                       method: str = 'nearest',
                       chunks: str = 'auto', month: str = 'run20',
                       overwrite: bool = False, verbose: bool = True,
                       ):
            """
            Extracts timeseries data for a given variable and sites based on input
            geodataframe from a list of netcdf files and writes to csv
            Parameters
            -------------
            basindirs: list of basin directories
            labels: of short names for model runs
            gdf_metloom: geodataframe of SNOTEL sites
            sitenames: list of SNOTEL site names
            varname: variable name to extract ['depth', 'density', 'swe', or 'all'], defaults to all
            method: method for extracting/resampling data, defaults to 'nearest'
            chunks: chunk size for xarray, defaults to 'auto'
            month: dir pattern to search for snow.nc files, defaults to 'run20'
            overwrite: flag to overwrite existing files, defaults to False
            verbose: flag to print filenames

            Returns
            -------------
            None
            """
            if verbose:
                print(varname)
            if varname == 'depth':
                drop_var_list=['snow_density', 'specific_mass',
                               'liquid_water', 'temp_surf', 'temp_lower',
                                'temp_snowcover', 'thickness_lower',
                                'water_saturation', 'projection']
                thisvar = 'thickness'
            elif varname == 'density':
                 drop_var_list=['thickness', 'specific_mass',
                               'liquid_water', 'temp_surf', 'temp_lower',
                                'temp_snowcover', 'thickness_lower',
                                'water_saturation', 'projection']
                 thisvar = 'snow_density'
            elif varname == 'swe':
                drop_var_list=['thickness', 'snow_density',
                               'liquid_water', 'temp_surf', 'temp_lower',
                                'temp_snowcover', 'thickness_lower',
                                'water_saturation', 'projection']
                thisvar = 'specific_mass'
            elif varname == 'all':
                drop_var_list = ['liquid_water', 'temp_surf', 'temp_lower',
                                'temp_snowcover', 'thickness_lower',
                                'water_saturation', 'projection']
                thisvar = ['thickness', 'snow_density', 'specific_mass']
            else:
                print("Variable not recognized, exiting...")
                sys.exit(1)

            for kdx, (label, basindir) in enumerate(zip(labels, basindirs)):
                print(kdx, label)
                outdir = PurePath(basindir).parents[0].as_posix()
                if type(thisvar) is not list:
                     thisvar = [thisvar]
                for v in thisvar:
                    print(v)
                    outfn = f'{basin}_{label}_{v}_snotelmetloom_wy{wy}.csv'
                    model_ts_fn = f'{outdir}/{outfn}'
                    if verbose:
                        print(model_ts_fn)

                    # Default - do not overwrite files
                    if os.path.exists(model_ts_fn) and not overwrite:
                        print(f"{model_ts_fn} exists, skipping!")
                        continue
                    else:
                        if os.path.exists(model_ts_fn) and overwrite:
                            print(f"{model_ts_fn} exists, but overwrite flag on, re-calculating...")
                        else:
                            print("^^DNE, calculating...")
                        days = dict()
                        # this is file name, it contains depth and density
                        basin_days = fn_list(basindir, f"{month}*/snow.nc")
                        days[label] = basin_days
                        if verbose:
                            print(len(basin_days))

                        ds_dict = dict()

                        # extract the snow state variables for the selected sites
                        ds_list = [xr.open_dataset(day_fn, chunks=chunks, drop_variables=drop_var_list) for day_fn in days[label]]

                        # Nearest neighbor selection, may build in different options in the future
                        snow_var_data = [ds[v].sel(x=list(gdf_metloom.geometry.x.values),
                                                        y=list(gdf_metloom.geometry.y.values), method=method) for ds in ds_list]
                        snow_var_data = xr.concat(snow_var_data, dim='time')
                        ds_dict[f'{label}_{v}'] = snow_var_data

                        # Turn these into dataframes and write to csvs
                        basin_dict = dict()
                        for jdx, sitename in enumerate(sitenames):
                            ds = ds_dict[f'{label}_{v}'][:, jdx, jdx]
                            basin_dict[sitename] = ds.values

                        # Turn it into a dataframe
                        basin_df = pd.DataFrame(basin_dict, index=ds['time'].values)

                        # Adjust units for SWE, need to divide by 1000 to convert from mm to m
                        if v == 'specific_mass':
                            basin_df = basin_df / 1000

                        if verbose:
                            print(f'Saving to {model_ts_fn}')

                        # Save the the dataframe as csv for easy access later
                        basin_df.to_csv(model_ts_fn)

def parse_arguments():
        """Parse command line arguments.

        Returns:
        argparse.Namespace: Parsed command line arguments.
        """
        parser = argparse.ArgumentParser(description='Extract iSnobal model output snow variable\
                                         ["depth" (thickness), "density", or "swe"] at point sites [SNOTEL]\
                                        within a basin for a given water year.')
        parser.add_argument('basin', type=str, help='Basin name')
        parser.add_argument('wy', type=int, help='Water year of interest')
        parser.add_argument('-shp', '--shapefile', type=str, help='Shapefile of basin polygon', default=None)
        parser.add_argument('-st', '--state', type=str, help='State abbreviation', default='CO')
        parser.add_argument('-loc', '--sitelocs', type=str, help='json file of point locations',
                            default='SNOTEL/snotel_sites_32613.json')
        parser.add_argument('-var', '--variable', type=str, help='iSnobal snow variable',
                            choices=['depth', 'density', 'swe', 'all'], default='depth')
        parser.add_argument('-f', '--filter', type=int, help='Filter on number of snow.nc files', default=None)
        parser.add_argument('-o', '--overwrite', help='Overwrite existing files', default=False)
        parser.add_argument('-v', '--verbose', help='Print filenames', default=True)
        return parser.parse_args()

def __main__():
    args = parse_arguments()
    basin = args.basin
    wy = args.wy
    poly_fn = args.shapefile
    state_abbrev = args.state
    sitelocs = args.sitelocs
    varname = args.variable
    filter_on = args.filter
    overwrite = args.overwrite
    verbose = args.verbose

    # Set up directories
    workdir = '/uufs/chpc.utah.edu/common/home/skiles-group3/model_runs/'
    ancillary_dir = '/uufs/chpc.utah.edu/common/home/skiles-group3/ancillary_sdswe_products/'

    # Set up filename for all active SNOTEL sites
    site_locs_fn = f'{ancillary_dir}/{sitelocs}'

    if poly_fn is None:
        if verbose:
            print('No shapefile provided, using default detection...')
        poly_dir = '/uufs/chpc.utah.edu/common/home/skiles-group3/jmhu/ancillary/polys'
        poly_fn = fn_list(poly_dir, f'*{basin}*shp')[0]
        poly_gdf = gpd.read_file(poly_fn)
        epsg = poly_gdf.crs.to_epsg()
    else:
        print(f'Detected input shapefile: {poly_fn}')
        site_locs_fn, epsg = modify_site_locs(poly_fn, site_locs_fn=site_locs_fn, epsg='32613')
        print(f'Detected EPSG is {epsg}')

     ### SNOTEL extraction and point specification
    if verbose:
        print(f'Preparing basin data for {basin}...')
    labels, basindirs, gdf_metloom, sitenames = prep_basin_data(basin=basin, WY=wy, poly_fn=poly_fn, ST=state_abbrev, epsg=epsg,
                                                                workdir=workdir, site_locs_fn=site_locs_fn, filter_on=filter_on, verbose=verbose)
    if verbose:
        print('Extracting timeseries data...')
    extract_timeseries(basindirs, labels, basin, wy, gdf_metloom, sitenames, overwrite=overwrite, varname=varname, verbose=verbose)

if __name__ == "__main__":
    __main__()
