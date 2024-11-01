#!/usr/bin/env python

'''Script to calculate per-pixel snow disappearance date based on calc_sdd func in processing.py

Usage: calc_basin_sdd.py basin_name 

# Defaults to calc_basin_sdd.py basin_name -t 10 <first water year detected, if multiple> <verbose>
# TODO add WY input, verbose flag as args
'''

import sys
import glob
import argparse

from typing import List

import numpy as np
import pandas as pd
import xarray as xr
from pathlib import PurePath
import copy
import json 

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

def get_dirs_filenames(basin, varfile='snow.nc', verbose=True, 
                           workdir='/uufs/chpc.utah.edu/common/home/skiles-group3/model_runs/'):
    '''Find basin directories, water year and list of daily snow.nc files for each model run'''
    basindirs = fn_list(workdir, f'{basin}*/*/*/')
    if verbose: 
        [print(b) for b in basindirs]

    # Get the WY from the directory name
    WY = int(PurePath(basindirs[0]).parents[0].stem.split('wy')[-1])
    if verbose:
        print(WY)
    # Update basindirs for the selected water year
    basindirs = fn_list(workdir, f'{basin}*/*{WY}/{basin}*/')
    wydir = PurePath(basindirs[0]).parents[0].as_posix()
    if verbose:
        [print(b) for b in basindirs]
    # list all the daily snow.nc files for a water year for each treatment
    nc_lists = [fn_list(basindir, f'*/{varfile}') for basindir in basindirs]
    if verbose: 
        print(len(nc_lists))

    return basindirs, wydir, WY, nc_lists

def load_snowdata(nc_lists, chunks='auto',
                      drop_var_list = ['snow_density', 'specific_mass', 
                                       'liquid_water', 'temp_surf', 'temp_lower', 
                                       'temp_snowcover', 'thickness_lower', 
                                       'water_saturation', 'projection'],
                      thisvar = 'thickness',
                      verbose=True
                     ):
        '''Read in snow depth data from variable filename lists and return as loaded, concatenated xarray dataset'''
        # read in snow depth data
        ds_lists = [[np.squeeze(xr.open_dataset(nc, 
                                                chunks=chunks, 
                                                drop_variables=drop_var_list)[thisvar]) for nc in nc_list] for nc_list in nc_lists]
        if verbose: 
            print(len(ds_lists[0]))

        # Concatenate the list of xarray datasets into a single xarray dataset
        ds_concat_list = [xr.concat(ds_list, dim='time') for ds_list in ds_lists]
        ds_concat_list = [ds.load() for ds in ds_concat_list]
        return ds_concat_list
    
def calculate_sdd(basindirs, wydir, wy, ds_concat_list, day_thresh, verbose=True):
        '''Calculate snow disappearance date and day of year using processing.py calc_sdd() func and input day_thresh.
        Generate dictionary of missing snow disappearance dates and pixel indices with basin model run type as dict key.'''
        missing_sdd_dict = dict()
        for basindir, ds in zip(basindirs, ds_concat_list):
            # Create an empty dataset of the same x and y dims to store the SDD values
            sdd_ds = copy.deepcopy(ds.isel(time=0))
            sdd_arr = sdd_ds.data
            print(sdd_arr.shape)

            # Create and empty list for keeping track of missing sdd pixels
            missing_list = []

            # fill the array with the sdd value if calculable
            for i in range(ds.x.size):
                for j in range(ds.y.size):
                    try:
                        sdd, _ = proc.calc_sdd(ds[:,j,i].to_series(), day_thresh=day_thresh)
                    except Exception as e:
                        e.add_note(f"Something wrong with SDD extract for {i, j} at day_thresh of {day_thresh}")
                        # store the pixel where sdd extraction is an issue
                        missing_list.append((i, j))

                        # add default bogus day to continue
                        sdd = pd.Timestamp(year=wy, month=12, day=25)

                    sdd_arr[j, i] = sdd.timestamp()

            # enter the missing_list into a dict using the basindir stems as keys
            missing_sdd_dict[PurePath(basindirs[0]).stem] = missing_list

            sdd_ds.data = sdd_arr

            # Update var name
            sdd_ds.name = 'sdd'

            # remove the time coordinates
            sdd_ds = sdd_ds.drop_vars('time')

            sdd_date_ds = sdd_ds.to_dataset()
            # Convert to datetime to access .dt.dayofyear for DOY calc
            # Needs to be in seconds, put up with nanosecond precision warning
            sdd_date_ds['sdd'] = sdd_date_ds['sdd'].astype('datetime64[s]')

            # Calculate Day of year
            sdd_date_ds['sdd_doy'] = sdd_date_ds['sdd'].dt.dayofyear

            # Clean up attributes
            sdd_date_ds['sdd'].attrs = dict()
            sdd_date_ds['sdd_doy'].attrs = dict()

            # Assign dt units in encoding, doing this in attributes creates issues
            sdd_date_ds['sdd'].attrs = dict(description='snow disappearance date for each pixel in the domain')
            sdd_date_ds['sdd'].encoding['units'] = "seconds since 1970-01-01 00:00:00"
            sdd_date_ds['sdd_doy'].attrs = dict(units='day of year', 
                                                description='snow disappearance day of year for each pixel in the domain')

            # write this out
            sdd_date_ds.to_netcdf(f'{wydir}/{PurePath(basindir).stem}_sdd_daythresh{day_thresh}_WY{wy}.nc')
        
        return missing_sdd_dict

def __main__():
    # Parse command line args
    parser = argparse.ArgumentParser(description='Basin-wide snow disappearance calculation')
    parser.add_argument('basin', type=str, help='name of basin')
    parser.add_argument('-t', '--day_thresh', type=str, help='Threshold of days \
                        to calculate snow disappearance. Default is 2 days', default=2)
    args = parser.parse_args()
    basin = args.basin
    day_thresh = args.day_thresh
    
    # Extract the basin directories, water year and list of daily snow.nc files for each model run
    basindirs, wydir, wy, nc_lists = get_dirs_filenames(basin, verbose=True)
    
    # Load the snow data
    ds_concat_list = load_snowdata(nc_lists, verbose=True)

    # Calculate the per-pixel snow disappearance date
    missing_sdd_dict = calculate_sdd(basindirs, wydir, wy, ds_concat_list, day_thresh)
    
    # Dump missing snow disappearance date dictionary to json file
    with open(f'{wydir}/missing_sdd_dict_daythresh{day_thresh}_WY{wy}.json', 'w') as fp:
        json.dump(missing_sdd_dict, fp)

if __name__ == "__main__":
    __main__()
