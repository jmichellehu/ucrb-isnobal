#!/usr/bin/env python

'''Script to calculate per-pixel date of peak snow property based on calc_peak func in processing.py and calc_basin_sdd.py script
Defaults to calc_basin_peak.py basin_name wy <varfile> <varname> <verbose>
Example usage: calc_basin_peak.py animas 2023 -varfile snow.nc -varname depth -verbose True
'''

import sys
import os
import glob
import argparse

from typing import List

import pandas as pd

import xarray as xr
from pathlib import PurePath
import copy

from tqdm import tqdm

sys.path.append('/uufs/chpc.utah.edu/common/home/u6058223/git_dirs/ucrb-isnobal/scripts/')

def fn_list(thisDir: str, fn_pattern: str, verbose: bool = False) -> List[str]:
    """Match and sort filenames based on a regex pattern in specified directory

    Parameters:
    ----------
        thisDir: directory path to search
        fn_pattern: regex pattern to match files
        verbose: print filenames

    Returns:
    ----------
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
    ----------
        basin: basin name
        WY: water year
        verbose: print filenames
        res: model resolution
        workdir: model run directory

    Returns:
    ----------
        basindirs: list of basin directories
        wydir: water year directory
    """
    # Update basindirs for the selected water year
    basindirs = fn_list(workdir, f'{basin}*/wy{WY}/{basin}*{res}*/')
    wydir = PurePath(basindirs[0]).parents[0].as_posix()
    if verbose:
        [print(b) for b in basindirs]

    return basindirs, wydir

def calc_peak(snow_property: pd.Series, verbose: bool = False, snow_name: str = None, units: str ='m') -> tuple:
    """Finds the date of the maximum snow value from a pandas series.

    Parameters:
    ----------
        snow_property: A pandas series containing snow depth or SWE values.
        verbose: If True, prints additional information. Defaults to False.
        snow_name: The name of the snow property. Defaults to None.
        units: The units of the snow property. Defaults to 'm'.

    Returns:
    ----------
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

def calculate_peak_date(basindir, wydir, wy, outname, verbose=True,
                        thisvar='thickness', varname='depth', varfile='snow.nc',
                        drop_var_list = ['thickness', 'snow_density', 'specific_mass', 'liquid_water', 'temp_surf',
                                         'temp_lower', 'temp_snowcover', 'thickness_lower',
                                         'water_saturation', 'projection']):
    """Calculate peak date and day of year of input snow property  using calc_peak() func.
    Generates csvs of missing peak snow dates and pixel indices by basin model run type and water year.

    Parameters:
    ----------
        basindir: basin directory
        wydir: water year directory
        wy: water year
        verbose: print filenames
        outname: output filename
        thisvar: snow property to calculate peak date
        varname: snow property name
        varfile: iSnobal output filename
        drop_var_list: list of variables to drop from the dataset

    Returns:
    ----------
        peak_date_ds: xarray dataset containing peak date and day of year of input snow property
    """
    # Amend drop_var_list depending on the snow property
    # (Keep all but the variable in the list of variables to drop)
    drop_var_list = [var for var in drop_var_list if var != thisvar]

    # Load the snow data
    ds = xr.open_mfdataset(f'{basindir}*/{varfile}', decode_coords="all", drop_variables=drop_var_list).load()

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
        print('Saving missing list to csv')

    # Dump missing peak snow date list to file
    ending = f'wy{wy}'
    missing_name = f'{wydir}/missing_peak_{varname}_{PurePath(basindir).stem}_{ending}.csv'
    with open(missing_name, 'w') as fp:
        fp.write(",".join(missing_list))

    peak_ds.data = peak_arr

    # Update var name
    peak_ds.name = 'peak'

    # remove the time coordinates if they exist
    try:
        peak_ds = peak_ds.drop_vars('time')
    except:
        pass

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

    if verbose:
        print(f'Writing out netcdf...\n{outname}')
    # write this out
    peak_date_ds.to_netcdf(f'{outname}')

    return peak_date_ds

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
    parser.add_argument('-ow', '--overwrite', help='Overwrite existing files', default=False)
    return parser.parse_args()

def __main__():
    args = parse_arguments()
    basin = args.basin
    WY = args.wy
    varfile = args.varfile
    varname = args.varname
    verbose = args.verbose
    overwrite = args.overwrite

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

    # Calculate the per-pixel peak snow date for each basin directory
    for basindir in basindirs:
        outname = f'{wydir}/{PurePath(basindir).stem}_peak_{varname}_wy{WY}.nc'
        if os.path.exists(outname) and not overwrite:
            print(f'Output file already exists: {outname}')
            pass
        else:
            print('Load snow data and calculate peak snow date')
            _ = calculate_peak_date(basindir, wydir, wy=WY, outname=outname, verbose=verbose,
                                    thisvar=thisvar, varname=varname, varfile=varfile)

if __name__ == "__main__":
    __main__()