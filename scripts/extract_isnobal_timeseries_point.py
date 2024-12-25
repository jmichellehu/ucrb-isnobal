#!/usr/bin/env python
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

def extract_timeseries(basindirs: list, labels: list, basin: str,
                       wy: str, gdf_metloom: gpd.GeoDataFrame,
                       sitenames: list, verbose: bool = True,
                       chunks='auto', month='run20',
                       varname='snow'
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
            if varname == 'snow':
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

            for kdx, (label, basindir) in enumerate(zip(labels, basindirs)):
                outdir = PurePath(basindir).parents[0].as_posix()
                outfn = f'{basin}_{label}_{thisvar}_snotelmetloom_wy{wy}.csv'
                model_ts_fn = f'{outdir}/{outfn}'
                if verbose:
                    print(model_ts_fn)

                # Default - do not overwrite files
                if os.path.exists(model_ts_fn):
                    print(f"{model_ts_fn} exists, skipping!")
                else:
                    print("^^DNE, calculating...")
                days = dict()
                basin_days = fn_list(basindir, f"{month}*/{varname}.nc")
                days[label] = basin_days

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
        parser = argparse.ArgumentParser(description='Extract iSnobal model output snow depth ["thickness"] \
                                         at point sites [SNOTEL] within a basin for a given water year.')
        parser.add_argument('basin', type=str, help='Basin name')
        parser.add_argument('wy', type=int, help='Water year of interest')
        parser.add_argument('-shp', '--shapefile', type=str, help='Shapefile of basin polygon', default=None)
        parser.add_argument('-st', '--state', type=str, help='State abbreviation', default='CO')
        parser.add_argument('-loc', '--sitelocs', type=str, help='json file of point locations',
                            default='SNOTEL/snotel_sites_32613.json')
        parser.add_argument('-v', '--verbose', help='Print filenames', default=True)
        return parser.parse_args()

def __main__():
    args = parse_arguments()
    basin = args.basin
    wy = args.wy
    poly_fn = args.shapefile
    state_abbrev = args.state
    sitelocs = args.sitelocs
    verbose = args.verbose

    # Set up directories
    workdir = '/uufs/chpc.utah.edu/common/home/skiles-group3/model_runs/'
    ancillary_dir = '/uufs/chpc.utah.edu/common/home/skiles-group3/ancillary_sdswe_products/'

    # Set up filename for all active SNOTEL sites
    allsites_fn = f'{ancillary_dir}/{sitelocs}'

    if poly_fn is None:
        poly_dir = '/uufs/chpc.utah.edu/common/home/skiles-group1/jmhu/ancillary/polys'
        poly_fn = fn_list(poly_dir, f'*{basin}*shp')[0]

    # Locate SNOTEL sites within basin using metloom
    found_sites = proc.locate_snotel_in_poly(poly_fn=poly_fn, site_locs_fn=allsites_fn, buffer=200)

    # Get site names and site numbers
    sitenames = found_sites['site_name']
    sitenums = found_sites['site_num']
    print(sitenames)

    # Use the site numbers to get the metloom snotel site coords and data
    ST_arr = [state_abbrev] * len(sitenums)
    gdf_metloom, _ = proc.get_snotel(sitenums, sitenames, ST_arr, WY=wy)

    # Get the basin directories based on input wy
    basindirs = fn_list(workdir, f'{basin}*/wy{wy}/{basin}*/')

    # these should be modified based on the basindirs, add dict
    labels = ['iSnobal-HRRR', 'HRRR-MODIS']

    extract_timeseries(basindirs, labels, basin, wy, gdf_metloom, sitenames, verbose=verbose)

if __name__ == "__main__":
    __main__()
