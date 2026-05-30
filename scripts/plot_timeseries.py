#!/usr/bin/env python
'''Script to plot timeseries of snow depth.'''

# Import modules here
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pyproj
import geopandas as gpd
import xarray as xr
import seaborn as sns

from pathlib import PurePath
from s3fs import S3FileSystem, S3Map

sys.path.append('/uufs/chpc.utah.edu/common/home/u6058223/git_dirs/env/')
import helpers as h
sys.path.append('/uufs/chpc.utah.edu/common/home/u6058223/git_dirs/ucrb-isnobal/scripts/')
import processing as proc

sns.set_palette('icefire')

# Define variables here
sk1 = '/uufs/chpc.utah.edu/common/home/skiles-group1'
sk3 = '/uufs/chpc.utah.edu/common/home/skiles-group3'
workdir = f'{sk3}/model_runs'
script_dir = f'{sk1}/jmhu/isnobal_scripts'
poly_dir = f'{sk1}/jmhu/ancillary/polys'
aso_dir = f'{sk3}/ASO'
nwm_dir = f'{sk3}/ancillary_sdswe_products/NWM'
swann_dir = f'{sk3}/ancillary_sdswe_products/SWANN_SWE_depth'

allsites_fn = f'{sk3}/ancillary_sdswe_products/SNOTEL/snotel_sites_32613.json'
nwm_projfn = f'{sk1}/jmhu/ancillary/NWM_datasets_proj4.txt' # nwm proj4 file


# Define functions here
def setup_pyprojdirs(CONDA_ENV='studio',
              miniconda_dir='/uufs/chpc.utah.edu/common/home/u6058223/software/pkg/miniconda3',
              verbose=True
              ):
    # Locate pyproj_datadir for studio env
    # From https://stackoverflow.com/questions/69630630/on-fresh-conda-installation-of-pyproj-pyproj-unable-to-set-database-path-pypr

    proj_version = h.fn_list(miniconda_dir, f'envs/{CONDA_ENV}/conda-meta/proj-[0-9]*.json')[0]
    VERSION = PurePath(proj_version).stem
    pyprojdatadir = f'{miniconda_dir}/pkgs/{VERSION}/share/proj'
    if verbose:
        print(pyprojdatadir)
    pyproj.datadir.set_data_dir(pyprojdatadir)

    # Set environmental variable for PROJ to directory where you can find proj.db
    os.environ['PROJ'] = pyprojdatadir
    os.environ['PROJLIB'] = pyprojdatadir
    os.environ['PROJ_LIB'] = pyprojdatadir
    return None

def retrieve_nwm(nwm_dir):
    '''Retrieve NWM data from csv for input basin and water year'''
    try:
        nwm_csv = h.fn_list(nwm_dir, f'{basin}/*{WY}.csv')[0]
    except IndexError:
        print(f'No NWM data found for {basin} basin for WY{WY}, run extract_nwm_timeseries.py')
        return None
    nwm_df = pd.read_csv(nwm_csv, index_col=0)
    nwm_df.index.name = 'Date'
    # Set as DatetimeIndex
    nwm_df.index = pd.to_datetime(nwm_df.index)
    nwm_daily_df = nwm_df.resample('1D').mean()
    return nwm_daily_df


def __main__():
    setup_pyprojdirs()
    # TODO add argparse and convert these to command line arguments
    # Basin-specific variables
    basin = 'blue'
    verbose = True

    # Select just for all variable output of time decay
    basindirs = h.fn_list(workdir, f'{basin}*/wy*/{basin}*/')

    # if no input WY, get the first one or
    # Get the WY from the directory name - assumes there is only one WY per basin right now
    WYs = [int(basindir.split('wy')[1][:4]) for basindir in basindirs]
    WYs = np.unique(WYs)
    if len(WYs) == 1:
        WY = WYs[0]
    else:
        print(f'Multiple water years in {basin} basin: {WYs}')
        print('Select WY')
        sys.exit(1)
    print(WY)

    # Extract SNOTEL site data within basin polygon
    # TODO check on ability to buffer in locate_snotel_in_poly, I think this exists
    # Locate SNOTEL sites within basin
    found_sites = proc.locate_snotel_in_poly(poly_fn=poly_fn, site_locs_fn=allsites_fn)

    # Get site names and site numbers
    sitenames = found_sites['site_name']
    sitenums = found_sites['site_num']
    if verbose:
        print(sitenames)

    ST_arr = ['CO'] * len(sitenums)
    gdf_metloom, snotel_dfs = proc.get_snotel(sitenums, sitenames, ST_arr, WY=WY)

    # Figure out filenames
    poly_fn = h.fn_list(script_dir, f'*{basin}*setup/polys/*shp')[0]
    if verbose:
        print(poly_fn)

    # TODO Could change this to save snow depth plot to png
    # snotel_dfs[sitenames.iloc[0]]["SNOWDEPTH_m"].plot()

    nwm_daily_df = retrieve_nwm(nwm_dir)

    # TODO Could change this to save snow depth plot to png
    # nwm_daily_df.plot()


    # Establish filename of time series based on this WY (netcdf much smaller than csv)
    swann_ts_fn = f'{swann_dir}/{basin}/{basin}_swann_snotelmetloom_DEPTH_wy{WY}.csv'
    print(swann_ts_fn)

    # If this file does not exist, process
    if not os.path.exists(swann_ts_fn):
        # Get this water year's file
        swann_wy_fn = h.fn_list(swann_dir, f'*{WY}*')

        # Read in the depth variable
        swann_ds = xr.open_mfdataset(swann_wy_fn, drop_variables=["SWE", "time_str"])

        # Reassign coordinate names
        swann_ds = swann_ds.rename({'lat': 'y', 'lon': 'x'})

        # Explicitly write out crs
        swann_ds.rio.write_crs(input_crs=swann_ds.crs.attrs['spatial_ref'],
                            inplace=True)

        # Transform  the x and y coords of snotel sites into this SWANN dst_crs
        gdf_metloom_reproj = gdf_metloom.to_crs(swann_ds.crs.attrs['spatial_ref'])

        # Extract values at snotel coordinates
        swann_data = swann_ds['DEPTH'].sel(x=list(gdf_metloom_reproj.geometry.x.values),
                                        y=list(gdf_metloom_reproj.geometry.y.values),
                                        method='nearest')

        # Convert to a dict and convert depth from millimeters to meters
        swann_dict = dict()
        for jdx, sitename in enumerate(sitenames):
            ds = swann_data[:, jdx, jdx]
            swann_dict[sitename] = ds.values / 1000

        # Turn it into a dataframe
        swann_datadf = pd.DataFrame(swann_dict, index=ds['time'].values)

        # Save the the dataframe as csv for easy access later
        if not os.path.exists(swann_ts_fn):
            swann_basin_dir = f'{swann_dir}/{basin}'
            if not os.path.exists(swann_basin_dir):
                os.makedirs(swann_basin_dir)

        # Write out for next time
        swann_datadf.to_csv(swann_ts_fn)

    else:
        swann_datadf = pd.read_csv(swann_ts_fn, index_col=0)
        swann_datadf.index.name = 'Date'
        # Set as DatetimeIndex
        swann_datadf.index = pd.to_datetime(swann_datadf.index)

    # Could change this to save snow depth plot to png
    # swann_datadf.plot()

if __name__ == '__main__':
    __main__()