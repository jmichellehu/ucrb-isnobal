#!/usr/bin/env python
'''
This script extracts modeled snow depth or snow density timeseries data at identified SNOTEL sites within a basin
and outputs as csv files.
'''
import sys
import glob
import os

import argparse
import pandas as pd
import geopandas as gpd
from pathlib import PurePath
from typing import List
import xarray as xr

sys.path.append('/uufs/chpc.utah.edu/common/home/u6058223/git_dirs/ucrb-isnobal/scripts/')
import processing as proc

def fn_list(thisDir: str, fn_pattern: str, verbose: bool = False) -> List[str]:
    """Match and sort filenames based on a regex pattern in specified directory

    Parameters
    -------------
    thisDir: str
        directory path to search
    fn_pattern: str
        regex pattern to match files
    verbose: boolean
        print filenames

    Returns
    -------------
    fns: list
        list of filenames matched and sorted
    """
    fns = []
    for f in glob.glob(thisDir + "/" + fn_pattern):
        fns.append(f)
    fns.sort()
    if verbose:
         print(fns)
    return fns

def prep_basin_data(basin, WY, poly_fn, ST, workdir, site_locs_fn, verbose):
    """Prepare basin snotel data and directories for processing"""
    # Locate SNOTEL sites within basin
    found_sites = proc.locate_snotel_in_poly(poly_fn=poly_fn, site_locs_fn=site_locs_fn)

    # Get site names and site numbers
    sitenames = found_sites['site_name']
    sitenums = found_sites['site_num']
    if verbose:
        print(sitenames)

    ST_arr = [ST] * len(sitenums)
    gdf_metloom, _ = proc.get_snotel(sitenums, sitenames, ST_arr, WY=WY)

    # Get the basin directories based on input wy
    basindirs = fn_list(workdir, f'{basin}*/wy{WY}/{basin}*/')

    # Based on the basindirs, generate a dict
    label_dict = dict()
    for basindir in basindirs:
        ending = PurePath(basindir).stem.split('basin_')[-1]
        if ending == '100m':
            label_dict[str(PurePath(basindir).stem)] = 'iSnobal-HRRR'
        elif ending =='100m_solar_albedo':
             label_dict[str(PurePath(basindir).stem)] = 'HRRR-MODIS'

    # Filter out basindirs that don't have enough snow.nc files, make sure most of WY is run (at least 270 files)
    basindirs = [basindir for basindir in basindirs if len(fn_list(basindir, 'run20*/snow.nc'))>=270]

    # Now generate the labels based on the basindirs and the dict you've created
    labels = [label_dict[str(PurePath(basindir).stem)] for basindir in basindirs]
    if verbose:
         print(labels)

    return labels, basindirs, gdf_metloom, sitenames

def extract_timeseries(basindirs: list, labels: list, basin: str,
                       wy: str, gdf_metloom: gpd.GeoDataFrame,
                       sitenames: list, overwrite: bool = False, verbose: bool = True,
                       chunks='auto', month='run20',
                       varname='depth'
                       ):
            """
            Extracts timeseries data for a given variable and sites based on input
            geodataframe from a list of netcdf files and writes to csv
            Parameters
            -------------
            basindirs: list
                list of basin directories
            labels: list
                list of short names for model runs
            gdf_metloom: geodataframe
                geodataframe of SNOTEL sites
            sitenames: list
                list of SNOTEL site names
            model_ts_fn: str
                output csv filename
            verbose: boolean
                print filenames

            Returns
            -------------
            None
            """
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
                mult = 1000
            else:
                print("Variable not recognized, exiting...")
                sys.exit(1)

            for kdx, (label, basindir) in enumerate(zip(labels, basindirs)):
                print(kdx, label)
                outdir = PurePath(basindir).parents[0].as_posix()
                outfn = f'{basin}_{label}_{thisvar}_snotelmetloom_wy{wy}.csv'
                model_ts_fn = f'{outdir}/{outfn}'
                if verbose:
                    print(model_ts_fn)

                # Default - do not overwrite files
                if os.path.exists(model_ts_fn) and not overwrite:
                    print(f"{model_ts_fn} exists, skipping!")
                elif os.path.exists(model_ts_fn) and overwrite:
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

                # Nearest neighbor selection, may build in buffered radius option in the future
                snow_var_data = [ds[thisvar].sel(x=list(gdf_metloom.geometry.x.values),
                                                y=list(gdf_metloom.geometry.y.values), method='nearest') for ds in ds_list]
                snow_var_data = xr.concat(snow_var_data, dim='time')
                ds_dict[f'{label}_{thisvar}'] = snow_var_data

                # Turn these into dataframes and write to csvs
                basin_dict = dict()
                for jdx, sitename in enumerate(sitenames):
                    ds = ds_dict[f'{label}_{thisvar}'][:, jdx, jdx]
                    basin_dict[sitename] = ds.values

                # Turn it into a dataframe
                basin_df = pd.DataFrame(basin_dict, index=ds['time'].values)

                # Adjust units for SWE, need to divide by 1000 to convert from mm to m
                if varname == 'swe':
                    basin_df = basin_df / mult

                if verbose:
                    print(f'Saving to {model_ts_fn}')

                # Save the the dataframe as csv for easy access later
                basin_df.to_csv(model_ts_fn)

def parse_arguments():
        """Parse command line arguments.

        Returns:
        ----------
        argparse.Namespace
            Parsed command line arguments.
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
                            choices=['depth', 'density', 'swe'], default='depth')
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
    overwrite = args.overwrite
    verbose = args.verbose

    # Set up directories
    workdir = '/uufs/chpc.utah.edu/common/home/skiles-group3/model_runs/'
    ancillary_dir = '/uufs/chpc.utah.edu/common/home/skiles-group3/ancillary_sdswe_products/'

    # Set up filename for all active SNOTEL sites
    allsites_fn = f'{ancillary_dir}/{sitelocs}'

    if poly_fn is None:
        poly_dir = '/uufs/chpc.utah.edu/common/home/skiles-group3/jmhu/ancillary/polys'
        poly_fn = fn_list(poly_dir, f'*{basin}*shp')[0]

     ### SNOTEL extraction and point specification
    labels, basindirs, gdf_metloom, sitenames = prep_basin_data(basin=basin, WY=wy, poly_fn=poly_fn, ST=state_abbrev,
                                                                workdir=workdir, site_locs_fn=allsites_fn, verbose=verbose)

    extract_timeseries(basindirs, labels, basin, wy, gdf_metloom, sitenames, overwrite=overwrite, varname=varname, verbose=verbose)

if __name__ == "__main__":
    __main__()
