#!/usr/bin/env python

'''Script to calculate per-pixel date of peak snow property based on calc_peak func in processing.py and calc_basin_sdd.py script
Defaults to calc_basin_peak.py basin_name wy <varfile> <varname> <verbose>
Example usage: calc_basin_peak.py animas 2023 -varfile snow.nc -varname depth -verbose True
'''

import sys
import glob
import argparse

from typing import List

import pandas as pd
# import datetime

import xarray as xr
from pathlib import PurePath
import copy
import json

from tqdm import tqdm

sys.path.append('/uufs/chpc.utah.edu/common/home/u6058223/git_dirs/ucrb-isnobal/scripts/')

def fn_list(thisDir: str, fn_pattern: str, verbose: bool = False) -> List[str]:
    """Match and sort filenames based on a regex pattern in specified directory

    Parameters:
        thisDir: directory path to search
        fn_pattern: regex pattern to match files
        verbose: print filenames

    Returns:
        fns: list of filenames matched and sorted
    """
    fns = []
    for f in glob.glob(thisDir + "/" + fn_pattern):
        fns.append(f)
    fns.sort()
    if verbose:
         print(fns)
    return fns

def get_dirs_filenames(basin: str, WY:int, verbose: bool = True, res: int = 100,
                           workdir: str = '/uufs/chpc.utah.edu/common/home/skiles-group3/model_runs/'):
    """Find basin directories and water year directory for each model run

    Parameters:
        basin: basin name
        WY: water year
        verbose: print filenames
        res: model resolution
        workdir: model run directory

    Returns:
        basindirs: list of basin directories
        wydir: water year directory
    """
    basindirs = fn_list(workdir, f'{basin}*/*/{basin}*{res}*/')
    if verbose:
        [print(b) for b in basindirs]

    # Update basindirs for the selected water year
    basindirs = fn_list(workdir, f'{basin}*/*{WY}/{basin}*{res}*/')
    wydir = PurePath(basindirs[0]).parents[0].as_posix()
    if verbose:
        [print(b) for b in basindirs]

    return basindirs, wydir

def calc_peak(snow_property: pd.Series, verbose: bool = False, snow_name: str = None, units: str ='m') -> tuple:
    """Finds the date of the maximum snow value from a pandas series.

    Parameters:
        snow_property: A pandas series containing snow depth or SWE values.
        verbose: If True, prints additional information. Defaults to False.
        snow_name: The name of the snow property. Defaults to None.
        units: The units of the snow property. Defaults to 'm'.

    Returns:
        tuple: A tuple containing the peak date and the maximum value of the snow property.
    """
    # Determine date of maximum value in snow property
    peak_date = snow_property.idxmax()

    # Pull the value of the snow property at peak date
    max_val = snow_property.loc[peak_date]

    if verbose:
        if snow_name is None:
            snow_name = 'Snow property value'
        print(f'Peak {snow_name} date: {peak_date.strftime("%Y-%m-%d")}')
        print(f'Peak {snow_name} value: {max_val} {units}')

    return peak_date, max_val

def calculate_peak_date(basindirs, wydir, wy, verbose=True,
                        thisvar='thickness', varname='depth', varfile='snow.nc',
                        drop_var_list = ['snow_density', 'specific_mass', 'liquid_water', 'temp_surf',
                                         'temp_lower', 'temp_snowcover', 'thickness_lower',
                                         'water_saturation', 'projection']):
    """Calculate peak date and day of year of input snow property  using processing.py calc_peak() func.
    Generate dictionary of missing peak snow dates and pixel indices with basin model run type as dict key.

    Parameters:
        basindirs: list of basin directories
        wydir: water year directory
        wy: water year
        verbose: print filenames
        thisvar: snow property to calculate peak date
        varname: snow property name
        varfile: iSnobal output filename
        drop_var_list: list of variables to drop from the dataset

    Returns:
        missing_dt_dict: dictionary of missing peak snow dates and pixel indices
        peak_date_ds: xarray dataset containing peak date and day of year of input snow property
    """
    # Load the snow data
    ds_concat_list = [xr.open_mfdataset(f'{basindir}*/{varfile}', decode_coords="all", drop_variables=drop_var_list).load() for basindir in basindirs]

    missing_dt_dict = dict()
    for basindir, ds in zip(basindirs, ds_concat_list):
        # Create an empty dataset of the same x and y dims to store the peak values
        peak_ds = copy.deepcopy(ds.isel(time=0))
        peak_ds = peak_ds[thisvar]
        ds = ds[thisvar]
        peak_arr = peak_ds.data
        if verbose:
            print(peak_arr.shape)

        # Create an empty list for keeping track of missing peak pixels
        missing_list = []

        if verbose:
            print('Begin looping...')
        # fill the array with the peak value if calculable
        for i in tqdm(range(peak_ds.x.size)):
            for j in range(peak_ds.y.size):
                try:
                    # if verbosity true, output is unmanageable with tqdm
                    peak, _ = calc_peak(ds[:,j,i].to_series(), verbose=False)
                except Exception as e:
                    e.add_note(f"Something wrong with peak extract for {i, j}")
                    # store the pixel where peak extraction is an issue
                    missing_list.append((i, j))

                    # add default bogus day to continue
                    peak = pd.Timestamp(year=wy, month=12, day=25)

                peak_arr[j, i] = peak.timestamp()

        if verbose:
            print('Storing missing list in dict')
        # enter the missing_list into a dict using the basindir stems as keys
        missing_dt_dict[PurePath(basindirs[0]).stem] = missing_list
        peak_ds.data = peak_arr

        # Update var name
        peak_ds.name = 'peak'

        # remove the time coordinates
        peak_ds = peak_ds.drop_vars('time')

        peak_date_ds = peak_ds.to_dataset()
        # Convert to datetime to access .dt.dayofyear for DOY calc
        # Needs to be in seconds, put up with nanosecond precision warning
        if verbose:
            print('Converting peak type to datetime64')
        peak_date_ds['peak'] = peak_date_ds['peak'].astype('datetime64[s]')

        if verbose:
            print('Calculating DOY')
        # Calculate Day of year
        peak_date_ds['peak_doy'] = peak_date_ds['peak'].dt.dayofyear

        # Clean up attributes
        peak_date_ds['peak'].attrs = dict()
        peak_date_ds['peak_doy'].attrs = dict()

        # Assign dt units in encoding, doing this in attributes creates issues
        peak_date_ds['peak'].attrs = dict(description='peak {varname} date for each pixel in the domain')
        peak_date_ds['peak'].encoding['units'] = "seconds since 1970-01-01 00:00:00"
        peak_date_ds['peak_doy'].attrs = dict(units='day of year',
                                            description=f'peak {varname} day of year for each pixel in the domain')

        outname = f'{wydir}/{PurePath(basindir).stem}_peak_{varname}_wy{wy}.nc'
        if verbose:
            print(f'Writing out netcdf...\n{outname}')
        # write this out
        peak_date_ds.to_netcdf(f'{outname}')

    return missing_dt_dict, peak_date_ds

def parse_arguments():
    """Parse command line arguments.

    Returns:
        argparse.Namespace: Parsed command line arguments.
    """
    parser = argparse.ArgumentParser(description='Basin-wide peak snow property calculation')
    parser.add_argument('basin', type=str, help='Basin name')
    parser.add_argument('wy', type=int, help='Water year of interest')
    parser.add_argument('-varfile', type=str, help='iSnobal output filename', default='snow.nc')
    parser.add_argument('-varname', type=str, help='variable common name', choices=['depth', 'swe', 'density'], default='depth')
    parser.add_argument('-v', '--verbose', default=True, help='Print filenames')
    return parser.parse_args()

def __main__():
    args = parse_arguments()
    basin = args.basin
    WY = args.wy
    varfile = args.varfile
    varname = args.varname
    verbose = args.verbose
    if varname == 'depth':
        thisvar = 'thickness'
    elif varname == 'swe':
        thisvar = 'specific_mass'
    elif varname == 'density':
        thisvar = 'snow_density'
    else:
        print('Invalid variable name, exiting...')
        sys.exit(1)

    print('Getting basin directories and water year directory')
    # Extract the basin directories, water year and list of daily snow.nc files for each model run
    basindirs, wydir = get_dirs_filenames(basin, WY, verbose=verbose)

    print('Load snow data and calculate peak snow date')
    ending = f'WY{WY}'
    # Calculate the per-pixel snow date
    missing_peak_dict, _ = calculate_peak_date(basindirs, wydir, wy=WY, verbose=verbose,
                                               thisvar=thisvar, varname=varname, varfile=varfile)

    print('Write out to json')
    # Dump missing snow date dictionary to json file
    with open(f'{wydir}/missing_peak_dict{ending}.json', 'w') as fp:
        json.dump(missing_peak_dict, fp)

if __name__ == "__main__":
    __main__()