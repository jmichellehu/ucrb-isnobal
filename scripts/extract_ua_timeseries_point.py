#!/usr/bin/env python
# coding: utf-8
# Script to extract University of Arizona snow data ["DEPTH" or "SWE"] at point sites [SNOTEL] within a basin for a given water year.

import os
import sys
import glob

import argparse
import pandas as pd
import xarray as xr
from typing import List

import fsspec

sys.path.append('/uufs/chpc.utah.edu/common/home/u6058223/git_dirs/ucrb-isnobal/scripts/')
import processing as proc


ancillarydir = '/uufs/chpc.utah.edu/common/home/skiles-group3/ancillary_sdswe_products'
poly_dir = '/uufs/chpc.utah.edu/common/home/skiles-group3/jmhu/ancillary/polys'

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

def parse_arguments():
        """Parse command line arguments.

        Returns:
        ----------
        argparse.Namespace
            Parsed command line arguments.
        """
        parser = argparse.ArgumentParser(description='Extract University of Arizona snow data ["DEPTH" or "SWE"] \
                                         at point sites [SNOTEL] within a basin for a given water year.')
        parser.add_argument('basin', type=str, help='Basin name')
        parser.add_argument('wy', type=int, help='Water year of interest')
        parser.add_argument('-shp', '--shapefile', type=str, help='Shapefile of basin polygon', default=None)
        parser.add_argument('-var', '--variable', type=str, help='UA snow data variable', 
                            choices=['DEPTH', 'SWE'], default='DEPTH')
        parser.add_argument('-loc', '--sitelocs', type=str, help='json file of point locations',
                            default='SNOTEL/snotel_sites_32613.json')
        parser.add_argument('-o', '--out_path', type=str, help='Output path', default=None)
        parser.add_argument('-v', '--verbose', action='store_true', help='Print filenames')
        return parser.parse_args()

def __main__():
    args = parse_arguments()
    verbose = args.verbose
    basin = args.basin
    wy = args.wy
    var = args.variable
    poly_fn = args.shapefile
    allsites_fn = f'{ancillarydir}/{args.sitelocs}'
    outname = args.out_path

    if poly_fn is None:
        try:
            poly_fn = fn_list(poly_dir, f'*{basin}*shp')[0]
        except IndexError:
            sys.exit(f'No shapefile found for {basin}')

    # Locate SNOTEL sites within basin using metloom
    found_sites = proc.locate_snotel_in_poly(poly_fn=poly_fn, site_locs_fn=allsites_fn, buffer=200)

    # Get site names and site numbers
    sitenames = found_sites['site_name']
    sitenums = found_sites['site_num']
    if verbose:
        print(sitenames)

    ST_arr = ['CO'] * len(sitenums)
    gdf_metloom, _ = proc.get_snotel(sitenums, sitenames, ST_arr, WY=wy)

    # Get the UA data
    # Modified from https://tutorial.xarray.dev/intermediate/remote_data/remote-data.html ! works!
    uris = f'https://climate.arizona.edu/data/UA_SWE/DailyData_800m/WY{wy}/UA_SWE_Depth_800m_v1_*_stable.nc'

    # Prepend the cache type to the URIs, this is called protocol chaining in fsspec-speak
    file = fsspec.open_local(f"simplecache::{uris}", filecache={'cache_storage': '/tmp/fsspec_cache'})

    ua_ds = xr.open_mfdataset(file, engine="netcdf4") #h5netcdf not currently installed

    # Reassign coordinate names
    ua_ds = ua_ds.rename({'lat': 'y', 'lon': 'x'})

    # Explicitly write out the crs
    ua_ds.rio.write_crs(input_crs=ua_ds.crs.attrs['spatial_ref'], inplace=True)

    # Transform  the x and y coords of snotel sites into this SWANN dst_crs
    gdf_metloom_reproj = gdf_metloom.to_crs(ua_ds.crs.attrs['spatial_ref'])

    # Extract values at snotel coordinates
    ua_data = ua_ds[var].sel(x=list(gdf_metloom_reproj.geometry.x.values),
                                y=list(gdf_metloom_reproj.geometry.y.values),
                                method='nearest')

    # Convert to a dict and convert depth from millimeters to meters
    # Both DEPTH and SWE are in mm
    ua_dict = dict()
    for jdx, sitename in enumerate(sitenames):
        ds = ua_data[:, jdx, jdx]
        ua_dict[sitename] = ds.values / 1000

    # Turn it into a dataframe
    ua_datadf = pd.DataFrame(ua_dict, index=ds['time'].values)

    # Save the the dataframe as csv for easy access later
    if outname is None:
        ua_basin_dir = f'{ancillarydir}/UA_SWE_depth/{basin}'
        if not os.path.exists(ua_basin_dir):
            os.makedirs(ua_basin_dir)
        outname = f'{ua_basin_dir}/{basin}_ua_800m_snotelmetloom_{var}_wy{wy}.csv'

    ua_datadf.to_csv(outname)
    if verbose:
        print(outname)

if __name__ == "__main__":
    __main__()