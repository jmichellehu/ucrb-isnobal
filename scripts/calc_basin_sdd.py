#!/usr/bin/env python

'''Script to calculate per-pixel snow disappearance date based on calc_sdd func in processing.py

Usage: calc_basin_sdd.py basin_name

Defaults to calc_basin_sdd.py basin_name -t 10 <first water year detected, if multiple> <verbose>
TODO add WY input, verbose flag as args
'''

import os
import sys
import glob
import argparse

from typing import List

import pandas as pd
import datetime

import xarray as xr
from pathlib import PurePath
import copy
import json

from tqdm import tqdm

sys.path.append('/uufs/chpc.utah.edu/common/home/u6058223/git_dirs/ucrb-isnobal/scripts/')

def fn_list(thisDir: str, fn_pattern: str, verbose: bool = False) -> List[str]:
    '''Match and sort filenames based on a regex pattern in specified directory

    Parameters
    ----------
        thisDir: directory path to search
        fn_pattern: regex pattern to match files
        verbose: print filenames

    Returns
    ----------
        fns: list of filenames matched and sorted
    '''
    fns = []
    for f in glob.glob(thisDir + "/" + fn_pattern):
        fns.append(f)
    fns.sort()
    if verbose:
         print(fns)
    return fns

def get_dirs_filenames(basin: str, WY:int, verbose: bool = False, res: int = 100,
                           workdir: str = '/uufs/chpc.utah.edu/common/home/skiles-group3/model_runs/'):
    '''Find basin directories and water year directory for each model run

    Parameters
    ----------
        basin: basin name
        WY: water year
        verbose: print filenames
        res: model resolution
        workdir: model run directory

    Returns
    ----------
        basindirs: list of basin directories
        wydir: water year directory
    '''
    basindirs = fn_list(workdir, f'{basin}*/*/{basin}*{res}*/')
    if verbose:
        [print(b) for b in basindirs]

    # Update basindirs for the selected water year
    basindirs = fn_list(workdir, f'{basin}*/*{WY}/{basin}*{res}*/')
    wydir = PurePath(basindirs[0]).parents[0].as_posix()
    if verbose:
        [print(b) for b in basindirs]
    # list all the daily snow.nc files for a water year for each treatment
    nc_lists = [fn_list(basindir, f'*/{varfile}') for basindir in basindirs]
    if verbose: 
        _ = [print(len(f)) for f in nc_lists]

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

def calc_sdd(snow_property: pd.Series, alg: str = "threshold", day_thresh: int = 10, verbose: bool = False):
    '''
    Calculate snow disappearance date from a pandas series of a snow property (snow depth or SWE).
    The snow disappearance date is represented by the last date at which the first derivative is nonzero.
    The "threshold" method ignores spurious late season events defined by occasions when the snow property
    is zero within a definable threshold (day_thresh) of preceding days.

    Parameters
    -------------
    snow_property: snow depth or SWE (measurement units in meters)
    alg: algorithm to use for snow all gone date calculation
        - "first": first date where snow property hits zero after the maximum value
        - "last": last date where snow property hits zero after the maximum value
        - "threshold": last date where the first derivative of snow property is negative for a definable threshold
    day_thresh: number of lookback days to consider for the threshold algorithm, defaults to 10
    verbose: boolean
        print additional information, defaults to False

    Returns
    -------------
        snow_all_gone_date: date at which the snow property disappears
        firstderiv: first derivative of the snow property
    '''
    # Calculate first derivative of snow property
    firstderiv = snow_property.diff() / snow_property.index.to_series().diff().dt.total_seconds()

    # Get list of dates with negative derivatives (declining snow property)
    deriv_dates = firstderiv[firstderiv < 0]

    # Determine last date at which derivative is negative
    current_SDD = deriv_dates.index[-1] + pd.Timedelta(days=1)

    # Based on algorithm, determine snow all gone date
    if verbose:
        print(f'Algorithm: {alg}')

    if alg == "first":
        # Pull index of maximum value
        max_depth_date = snow_property.idxmax()
        if verbose:
            print(f"Max depth date: {max_depth_date}")
            print(f"Max depth is: {snow_property.loc[max_depth_date]}")
        # Pull index of minimum value after max depth date
        # idxmin pulls the first value (date) if multiple meet the condition
        snow_all_gone_date = snow_property.loc[max_depth_date:].idxmin()
        if verbose:
            print(f'Found snow all gone date: {snow_all_gone_date}')
    elif alg == "last":
        # take the day after the last date where the first derivative is negative and
        snow_all_gone_date = current_SDD + datetime.timedelta(days=1)
        if verbose:
            print(f'Found snow all gone date: {snow_all_gone_date}')
    elif alg == "threshold":
        if verbose:
            print(f'Starting snow all gone date: {current_SDD}')
        # Loop through all dates where the first derivative is robustly negative
        # Starting with the second to last date in the series
        # Assign as "preceding_date"
        for f in range(-2, -len(deriv_dates), -1):
            preceding_date = deriv_dates.index[f]

            # If any of the `day_thresh` days preceding the current_SDD hit a snow depth of zero,
            # ==> this is a spurious late season event <==
            # change the current_SDD to the preceding date value and continue looping
            if (snow_property.loc[current_SDD - datetime.timedelta(days=day_thresh):current_SDD] == 0).any():
                current_SDD = preceding_date
                next
            else:
                # Now we should have the actual snow all gone date
                snow_all_gone_date = current_SDD + datetime.timedelta(days=1)
                if verbose:
                    print(f'Found snow all gone date: {snow_all_gone_date}')
                break

    return snow_all_gone_date, firstderiv

def calculate_sdd(basindirs: List[str], wydir: str, wy: int, verbose: bool = True, alg: str= 'first',
                        thisvar: str= 'thickness', varname: str= 'depth', varfile: str= 'snow.nc',
                        drop_var_list: List[str] = ['snow_density', 'specific_mass', 'liquid_water', 'temp_surf',
                                                    'temp_lower', 'temp_snowcover', 'thickness_lower',
                                                    'water_saturation', 'projection']) -> tuple[dict, xr.Dataset]:
        '''Calculate snow disappearance date and day of year using processing.py calc_sdd() func.
        Generate dictionary of missing snow disappearance dates and pixel indices with basin model run type as dict key.'''
        missing_sdd_dict = dict()
        for basindir, ds in zip(basindirs, ds_concat_list):
            # Create an empty dataset of the same x and y dims to store the SDD values
            sdd_ds = copy.deepcopy(ds.isel(time=0))
            sdd_arr = sdd_ds.data
            if verbose:
                print(sdd_arr.shape)

            # Create and empty list for keeping track of missing sdd pixels
            missing_list = []

            if verbose:
                print('Begin looping...')
            # fill the array with the sdd value if calculable
            for i in tqdm(range(sdd_ds.x.size)):
                for j in range(sdd_ds.y.size):
                    try:
                        sdd, _ = calc_sdd(ds[:,j,i].to_series(), alg=alg, verbose=False)
                    except Exception as e:
                        e.add_note(f"Something wrong with SDD extract for {i, j}")
                        # store the pixel where sdd extraction is an issue
                        missing_list.append((i, j))

                        # add default bogus day to continue
                        sdd = pd.Timestamp(year=wy, month=12, day=25)

                    sdd_arr[j, i] = sdd.timestamp()

            if verbose:
                print('Storing missing list in dict')
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
            if verbose:
                print('Converting SDD type to datetime64')
            sdd_date_ds['sdd'] = sdd_date_ds['sdd'].astype('datetime64[s]')

            if verbose:
                print('Calculating DOY')
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

            outname = f'{wydir}/{PurePath(basindir).stem}_sdd_{varname}_wy{wy}_{alg}.nc'
            if verbose:
                print(f'Writing out netcdf...\n{outname}')
            # write this out
            sdd_date_ds.to_netcdf(f'{outname}')
        
        return missing_sdd_dict

        return missing_sdd_dict, sdd_date_ds

def parse_arguments():
    """Parse command line arguments.

    Returns:
        argparse.Namespace: Parsed command line arguments.
    """
    parser = argparse.ArgumentParser(description='Basin-wide snow disappearance calculation')
    parser.add_argument('basin', type=str, help='name of basin')
    args = parser.parse_args()
    basin = args.basin
    
    print('Getting dirs_filenames')
    # Extract the basin directories, water year and list of daily snow.nc files for each model run
    basindirs, wydir, wy, nc_lists = get_dirs_filenames(basin, verbose=True)
    
    print('Load snow data')
    # Load the snow data
    ds_concat_list = load_snowdata(nc_lists, verbose=True)

    print('Calculate SDD')
    ending = f'WY{wy}'
    # Calculate the per-pixel snow disappearance date
    missing_sdd_dict = calculate_sdd(basindirs, ending, wydir, wy, ds_concat_list)
    
    print('Write out to json')
    # Dump missing snow disappearance date dictionary to json file
    with open(f'{wydir}/missing_sdd_dict_{ending}.json', 'w') as fp:
        json.dump(missing_sdd_dict, fp)

if __name__ == "__main__":
    __main__()
