#!/usr/bin/env python
'''
This script extracts DENSITY timeseries data at identified SNOTEL sites within a basin
from SNOTEL, NWM, and UA snow data and outputs as csv files.
'''
import os
import sys

import argparse
import pandas as pd
import geopandas as gpd
import xarray as xr

sys.path.append('/uufs/chpc.utah.edu/common/home/u6058223/git_dirs/env/')
import helpers as h

sys.path.append('/uufs/chpc.utah.edu/common/home/u6058223/git_dirs/ucrb-isnobal/scripts/')
import processing as proc

# Global variables
workdir = '/uufs/chpc.utah.edu/common/home/skiles-group3/model_runs/'
script_dir = '/uufs/chpc.utah.edu/common/home/skiles-group3/jmhu/isnobal_scripts'
poly_dir = '/uufs/chpc.utah.edu/common/home/skiles-group3/jmhu/ancillary/polys'

# SNOTEL all sites geojson fn - snotel site json
allsites_fn = '/uufs/chpc.utah.edu/common/home/skiles-group3/ancillary_sdswe_products/SNOTEL/snotel_sites_32613.json'

def calc_snotel_density(poly_fn: str, site_locs_fn: str, basin: str, WY: int, snowvar: str,
                        ST: str = 'CO', verbose: bool = True):
    """Calculate density from SNOTEL depth and SWE as accessed via metloom
    Parameters
    ----------
    poly_fn: str
        Shapefile of basin polygon
    site_locs_fn: str
        json file of point locations
    basin: str
        Basin name
    WY: int
        Water year of interest
    snowvar: str
        SNOTEL snow data variable, also somewhat unnecessary as the code is only for density calculations, should be 'both'
    ST: str
        State abbreviation, defaults to 'CO'
    verbose: bool
        Defaults to True

    Returns
    ----------
    gdf_metloom: geopandas dataframe
        Dataframe of SNOTEL data
    sitenames: list
        List of SNOTEL site names
    """
    # - identify SNOTEL sites within the specified basin
    # - extract site metadata (site name, site number, and coordinates)
    # - extract snow depth values for WY of interest
    # Locate SNOTEL sites within basin
    found_sites = proc.locate_snotel_in_poly(poly_fn=poly_fn, site_locs_fn=site_locs_fn)

    # Get site names and site numbers
    sitenames = found_sites['site_name']
    sitenums = found_sites['site_num']
    if verbose:
        print(sitenames)

    # Create array of state abbreviations for metloom
    ST_arr = [ST] * len(sitenums)

    # Use metloom to get SNOTEL data for depth, swe, and density
    gdf_metloom, snotel_dfs, _ = proc.get_snotel(sitenums, sitenames, ST_arr, WY=int(WY), return_meta=True, snowvar=snowvar)

    # Concatenate the snotel_dfs into one dataframe
    big_snotel_df = pd.concat(snotel_dfs, axis=1)
    # Write out the SNOTEL data to csv
    snotel_dir = f'/uufs/chpc.utah.edu/common/home/skiles-group3/ancillary_sdswe_products/SNOTEL/basin_extracts/{basin}'
    snotel_ts_fn = f'{snotel_dir}_{WY}_snotelmetloom.csv'
    big_snotel_df.to_csv(snotel_ts_fn)
    if verbose:
        print(snotel_ts_fn)

    return gdf_metloom, sitenames

def calc_nwm_density(basin: str, WY: int, NWM_var: str = 'DENSITY', verbose: bool = True,
                     nwm_dir: str = '/uufs/chpc.utah.edu/common/home/skiles-group3/ancillary_sdswe_products/NWM/'):
    """Calculate density from NWM-extracted depth and SWE outputs produced with extract_nwm_timeseries_point.py
    Parameters
    ----------
    basin: str
        Basin name
    NWM_var: str
        NWM snow data variable, this is somewhat unnecessary as the code is only for density calculations
    WY: int
        Water year of interest
    verbose: bool
        Defaults to True

    Returns
    ----------
    nwm_ts_fn: str
        Filename of NWM density timeseries csv
    """
    # # Check if the csv already exists
    # if len(h.fn_list(nwm_dir, f'{basin}/*{NWM_var}*{WY}.csv')) > 0:
    #     nwm_ts_fn = h.fn_list(nwm_dir, f'{basin}/*{NWM_var}*{WY}.csv')[0]
    # else:
    # Pull depth and SWE from NWM and calculate density
    nwm_depth_csv = h.fn_list(nwm_dir, f'{basin}/*SNOWH*{WY}.csv')[0]
    nwm_SWE_csv = h.fn_list(nwm_dir, f'{basin}/*SNEQV*{WY}.csv')[0]
    nwm_df_list = []
    for nwm_var_csv in [nwm_depth_csv, nwm_SWE_csv]:
        nwm_df = pd.read_csv(nwm_var_csv, index_col=0)
        nwm_df.index.name = 'Date'

        # Set as DatetimeIndex
        nwm_df.index = pd.to_datetime(nwm_df.index)
        nwm_daily_df = nwm_df.resample('1D').mean()

        # NWM SWE is in mm, need to convert
        if nwm_var_csv == nwm_SWE_csv:
            nwm_daily_df = nwm_daily_df * 0.001
        nwm_df_list.append(nwm_daily_df)

    # density calc is SWE / depth
    nwm_daily_df = nwm_df_list[-1] / nwm_df_list[0]

    # save this to file (because it doesn't otherwise exist)
    # in the same format as the extracted NWM files output from extract_nwm_timeseries_point.py
    nwm_ts_fn = f'{nwm_dir}/{basin}/{basin}_nwm_snotelmetloom_{NWM_var}_wy{WY}.csv'
    nwm_daily_df.to_csv(nwm_ts_fn)

    if verbose:
            print(nwm_ts_fn)
    return nwm_ts_fn

def calc_ua_density(basin: str, UA_var: str, WY: int, gdf_metloom: gpd.GeoDataFrame,
                    use4k: bool = False, verbose: bool = True,
                    ua_dir: str = '/uufs/chpc.utah.edu/common/home/skiles-group3/ancillary_sdswe_products/UA_SWE_depth'):
    """Calculate density from UA-extracted depth and SWE outputs produced with extract_ua_timeseries_point.py
    Parameters
    ----------
    basin: str
        Basin name
    UA_var: str
        UA snow data variable, this is somewhat unnecessary as the code is only for density calculations
    use4k: bool
        Use 4 km UA data, defaults to False, using 800 m data
    WY: int
        Water year of interest
    verbose: bool
        Defaults to True
    gdf_metloom: geopandas dataframe
    ua_dir: str
        Directory where UA data is stored

    Returns
    ----------
    ua_ts_fn: str
        Filename of UA density timeseries
    """
    # Establish filename of time series based on this WY (netcdf much smaller than csv)
    if use4k:
        ua_ts_fn = f'{ua_dir}/{basin}/{basin}_ua_snotelmetloom_{UA_var}_wy{WY}.csv'
    else:
        ua_ts_fn = f'{ua_dir}/{basin}/{basin}_ua_800m_snotelmetloom_{UA_var}_wy{WY}.csv'
    if verbose:
        print(ua_ts_fn)

    if use4k:
        # Get this water year's file
        ua_wy_fn = h.fn_list(ua_dir, f'*{WY}*')
        ua_ds = xr.open_mfdataset(ua_wy_fn, drop_variables=["time_str"])
        # compute density, drop the depth and SWE variables
        ua_ds[UA_var] = ua_ds['SWE'] / ua_ds['DEPTH']

        # Reassign coordinate names
        ua_ds = ua_ds.rename({'lat': 'y', 'lon': 'x'})

        # Explicitly write out crs
        ua_ds.rio.write_crs(input_crs=ua_ds.crs.attrs['spatial_ref'],
                            inplace=True)

        # Transform  the x and y coords of snotel sites into this ua dst_crs
        gdf_metloom_reproj = gdf_metloom.to_crs(ua_ds.crs.attrs['spatial_ref'])

        # Extract values at snotel coordinates
        ua_data = ua_ds[UA_var].sel(x=list(gdf_metloom_reproj.geometry.x.values),
                                        y=list(gdf_metloom_reproj.geometry.y.values),
                                        method='nearest')

        # Convert to a dict and convert snow var from millimeters to meters
        ua_dict = dict()
        mult = 1000
        for jdx, sitename in enumerate(sitenames):
            ds = ua_data[:, jdx, jdx]
            ua_dict[sitename] = ds.values * mult

        # Turn it into a dataframe
        ua_datadf = pd.DataFrame(ua_dict, index=ds['time'].values)

        # Save the the dataframe as csv for easy access later
        if not os.path.exists(ua_ts_fn):
            ua_basin_dir = f'{ua_dir}/{basin}'
            if not os.path.exists(ua_basin_dir):
                os.makedirs(ua_basin_dir)

        # Write out for next time
        ua_datadf.to_csv(ua_ts_fn)
    else:
        # this is for 800 m
        # Get the SNOTEL extracted points' depth and swe filenames
        ua_depth_fn = h.fn_list(ua_dir, f'{basin}/{basin}_ua_800m_snotelmetloom_DEPTH_wy{WY}.csv')[0]
        ua_swe_fn = h.fn_list(ua_dir, f'{basin}/{basin}_ua_800m_snotelmetloom_SWE_wy{WY}.csv')[0]
        depth = pd.read_csv(ua_depth_fn, index_col=0)
        swe = pd.read_csv(ua_swe_fn, index_col=0)
        ua_datadf = swe / depth * 1000 # density in kg/m^3
        # Save the the dataframe as csv for easy access later
        if not os.path.exists(ua_ts_fn):
            ua_basin_dir = f'{ua_dir}/{basin}'
            if not os.path.exists(ua_basin_dir):
                os.makedirs(ua_basin_dir)

        # Write out for next time
        ua_datadf.to_csv(ua_ts_fn)
    if verbose:
        print(ua_ts_fn)
    return ua_ts_fn

def parse_arguments():
    """Parse command line arguments.

    Returns:
    ----------
    argparse.Namespace
        Parsed command line arguments.
    """
    parser = argparse.ArgumentParser(description='Calculate density from depth and SWE outputs\
                                        for NWM, UA, SNOTEL data at point sites [SNOTEL] within a basin \
                                        for a given water year.')
    parser.add_argument('basin', type=str, help='Basin name')
    parser.add_argument('wy', type=int, help='Water year of interest')
    parser.add_argument('-v', '--verbose', action='store_true', help='Print filenames')
    return parser.parse_args()

def __main__():
    args = parse_arguments()
    basin = args.basin
    WY = args.wy
    verbose = args.verbose

    # Select just for all variable output of time decay
    basindirs = h.fn_list(workdir, f'{basin}*/wy{WY}/{basin}*/')
    if verbose:
        print(basindirs)

    # Figure out filenames
    poly_fn = h.fn_list(script_dir, f'*{basin}*setup/polys/*shp')[0]
    if verbose:
        print(poly_fn)

    # Calculate density for
    # NWM, UA, SNOTEL
    var = 'density'
    snowvar = 'both' # change this to both to pull depth and SWE from get_snotel()
    NWM_var = var.upper()
    UA_var = var.upper()

    ### SNOTEL extraction and point specification
    gdf_metloom, sitenames = calc_snotel_density(poly_fn=poly_fn, site_locs_fn=allsites_fn,
                                                 ST='CO', basin=basin, WY=WY,
                                                 snowvar=snowvar, verbose=verbose)
    ### NWM
    _ = calc_nwm_density(basin=basin, NWM_var=NWM_var, WY=WY, verbose=verbose)

    ### UA
    _ = calc_ua_density(basin=basin, UA_var=UA_var, use4k=False, WY=WY,
                        gdf_metloom=gdf_metloom, verbose=verbose)

if __name__ == "__main__":
    __main__()