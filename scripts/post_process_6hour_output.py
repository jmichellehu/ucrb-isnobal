#!/usr/bin/env python
'''
This script post-processes 6 hour iSnobal outputs into 4 timesteps per day, each consisting of 6 hours

Current Timestep breakdown for example Day 91
5 total timesteps per day, unequally distributed:
hour 00 = SWI flux value at hour 00 to save, computed since end of previous day's hour 23 (covers 1 hour)
hour 06 = flux values summed over hours 01 - 06 (6 hours)
hour 12 = flux summed over hours 07 - 12 (6 hours)
hour 18 = flux summed over hours 13 - 18 (6 hours)
hour 23 = flux summed over hours 19 - 23 (5 hours)

Idealized timestep breakdown for example Day 91
h06 (01-06, 6 hours)
h12 (07-12, 6 hours)
h18 (13-18, 6 hours)
"new h24"(19-23 + h 00 from Day 92, 6 hours)

'''
import os
import glob
import argparse

import numpy as np
import xarray as xr
import pandas as pd

from typing import List, Tuple, Union
import datetime

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

def pretty_time(ds: xr.Dataset, hours_only: bool = False) -> List[str]:
    """Convert time values of input dataset to 'YYYY-MM-DD HH' format strings

    Parameters
    -------------
    ds: input dataset with time coordinate
    hours_only: if True, return only hour values as 'hXX' format,
                if False, return full 'YYYY-MM-DD HH' format

    Returns
    -------------
    list of formatted time strings
    """
    if hours_only:
        return [f'h{np.datetime_as_string(t.values, unit="h")[-2:]}' for t in ds.time]
    else:
        return ds.time.values.astype('M8[h]').astype(str)

def check_sums(ds: xr.Dataset, data_var: str = 'SWI', daily_only: bool = False,
               return_sums: bool = False) -> Union[None, Tuple[List[float], float]]:
    """Compute array sum for each timestep, daily sum, and optionally return daily sums

    Parameters
    -------------
    ds: input dataset containing data variable(s)
    data_var: name of data variable to sum
    daily_only: if True, only print daily total; if False, print each timestep
    return_sums: if True, return tuple of daily sum of timestep sums

    Returns
    -------------
    daily_total: timestep sums list if return_sums is True,
          otherwise None
    """
    pretty_hours = pretty_time(ds[data_var], hours_only=True)
    if not daily_only:
        for jdx, t in enumerate(ds[data_var].time):
            # Print just the hours, removing YYYY-MM-DD part and
            # Sum at this timestep
            print(f'{pretty_hours[jdx]}: {ds[data_var].sel(time=t).sum().values:.0f}')
        # Print the daily total
    else:
        daily_total = ds[data_var].sum(dim='time').sum().values
        print(f"Daily total: {daily_total:.0f}")
    print('---')
    if return_sums:
        return daily_total

def combine_hour_23_and_00(day_data: xr.Dataset, data_var: str = 'SWI',
                          debug: bool = False) -> xr.Dataset:
    """Combine hour 23 and hour 00 data into a single timestep as new hour 00. Removes hour 23.

    Parameters
    -------------
    day_data: dataset containing data for a single day with hour 23 and 00
    data_var: name of data variable to combine
    debug: if True, print debug information during combination

    Returns
    -------------
    combined_data: dataset with hour 23 and 00 combined into single timestep
    """
    # Get hours
    hours = day_data.time.dt.hour
    if debug:
        print(hours.values)
    # Sum variable data for hour 23 and hour 00
    value_23 = day_data[data_var].sel(time=hours == 23).squeeze('time')
    value_00 = day_data[data_var].sel(time=hours == 0).squeeze('time')
    combined = value_23 + value_00
    if debug:
        print(f"Hour 23 value: {value_23.values}")
        print(f"Hour 00 value: {value_00.values}")
        print(f"Combined value: {combined.values}")
    # Create modified dataset
    if debug:
        print("Creating modified dataset, removing hour 23...")
    ds_modified = day_data.sel(time=hours != 23)  # Remove hour 23
    if debug:
        print("...and assigning combined value to hour 00")
        print(ds_modified.time[ds_modified.time.dt.hour == 0])
    ds_modified[data_var].loc[dict(time=ds_modified.time[ds_modified.time.dt.hour == 0])] = combined

    return ds_modified

def remove_hour_23(day_data: xr.Dataset) -> xr.Dataset:
    """Remove hour 23 from the dataset. Should use for state variables

    Parameters
    -------------
    day_data: dataset containing data for a single day

    Returns
    -------------
    ds_modified: dataset with hour 23 removed
    """
    hours = day_data.time.dt.hour
    ds_modified = day_data.sel(time=hours != 23)  # Remove hour 23
    return ds_modified

def process_year_state(ds: xr.Dataset, data_var: str = 'specific_mass',
                       output_dir: str = "daily_output") -> Tuple[List[datetime.date], List[datetime.date]]:
    """Remove hour 23 for each day and save as separate daily netCDF files
    Midnight (hour 00) is grouped with the previous day.

    Parameters
    -------------
    ds: input dataset with time coordinate and data variable
    data_var: name of data variable to process
    output_dir: directory path to save processed netCDF files

    Returns
    -------------
    processed_days: list of successfully processed dates
    skipped_days: list of dates skipped in processing due to errors
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Shift time by 1 hour so midnight belongs to previous day
    shifted_time = ds.time - pd.Timedelta(hours=1)

    # Group by date using shifted time
    daily_groups = ds.groupby(shifted_time.dt.date)
    processed_days = []
    skipped_days = []

    for date, day_data in daily_groups:
        try:
            # Remove hour 23
            processed_day = remove_hour_23(day_data)

            # Save to netCDF
            filename = f"{output_dir}/{data_var}_{date.strftime('%Y%m%d')}.nc"
            processed_day.to_netcdf(filename)
            print(f"Saved: {filename}")

            processed_days.append(date)

        except Exception as e:
            print(f"Error processing {date}: {e}")
            skipped_days.append(date)
    return processed_days, skipped_days

def process_year_flux(ds: xr.Dataset, data_var: str = 'SWI', output_dir: str = "daily_output",
                     debug: bool = False) -> Tuple[List[datetime.date], List[datetime.date], List[datetime.date]]:
    """Apply hour 23/00 combination for each day and save as separate netCDF files.
    Midnight (hour 00) is grouped with the previous day.

    Parameters
    -------------
    ds: input dataset with time coordinate and data variable
    data_var: name of data variable to process
    output_dir: directory path to save processed netCDF files
    debug: if True, print debug information during processing

    Returns
    -------------
    mismatch_days: list of mismatched dates
    processed_days: list of successfully processed dates
    skipped_days: list of dates skipped in processing due to errors
    """
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Shift time by 1 hour so midnight belongs to previous day
    shifted_time = ds.time - pd.Timedelta(hours=1)

    # Group by date using shifted time
    daily_groups = ds.groupby(shifted_time.dt.date)
    mismatch_days = []
    processed_days = []
    skipped_days = []

    for date, day_data in daily_groups:
        try:
            # Only process if day has both hour 23 and hour 00
            hours = day_data.time.dt.hour
            if debug:
                print(hours)
            if (23 in hours.values) and (0 in hours.values):
                before = check_sums(day_data, data_var=data_var, daily_only=True, return_sums=True)
                if debug:
                    print("Attempting processing")
                processed_day = combine_hour_23_and_00(day_data, data_var=data_var, debug=False)
                if debug:
                    print("Processing complete")
                after = check_sums(processed_day, data_var=data_var, daily_only=True, return_sums=True)
                if before != after:
                    print(f"Warning: Daily sum mismatch on {date}: before={before}, after={after}")
                    mismatch_days.append(date)
            else:
                # Keep day unchanged if it doesn't have both hours
                processed_day = day_data
            # Save to netCDF
            filename = f"{output_dir}/{data_var}_{date.strftime('%Y%m%d')}.nc"
            processed_day.to_netcdf(filename)
            print(f"Saved: {filename}")

            processed_days.append(date)

        except Exception as e:
            print(f"Error processing {date}: {e}")
            skipped_days.append(date)
    return mismatch_days, processed_days, skipped_days

def determine_processing_flow(basin: str, wy: int, varname: str, model_output_dir: str, output_dir: str,
                              verbose: bool = False, return_vars: bool = False) -> Union[None, Tuple[List[datetime.date],
                                                                                                      List[datetime.date],
                                                                                                      List[datetime.date]]]:
    """Use variable type to determine filenames, variables to drop, and to execute appropriate processing flow

    Parameters
    -------------
    basin: name of basin for output directory naming
    wy: water year to process
    varname: name of variable to process ('SWI' or 'specific_mass')
    model_output_dir: directory path containing model output files
    output_dir: directory path to save processed files
    verbose: if True, print processing information
    return_vars: if True, return processing results lists instead of None

    Returns
    -------------
    processing_results: None if return_vars is False, otherwise tuple of processing results.
                       For 'SWI': (mismatch_days, processed_days, skipped_days)
                       For other variables: (processed_days, skipped_days)
    """
    # Build a dict using thisvar as keys
    var_dict = {
        'SWI': {
            'filename': 'em',
            'dropvarlist': ['net_rad', 'sensible_heat', 'latent_heat', 'snow_soil', 'precip_advected',
                            'sum_EB', 'evaporation', 'snowmelt', 'cold_content', 'projection']
        },
        'specific_mass': {
            'filename': 'snow',
            'dropvarlist': ['thickness', 'snow_density', 'liquid_water', 'temp_surf', 'temp_lower',
                            'temp_snowcover', 'thickness_lower', 'water_saturation', 'projection']
        }
    }
    # Extract filename and dropvarlist based on varname
    filename = var_dict[varname]['filename']
    dropvarlist = var_dict[varname]['dropvarlist']
    if verbose:
        print(basin, wy, varname)
        print(filename, dropvarlist)

    # Open multiple netCDF files as a single xarray dataset
    ds = xr.open_mfdataset(fn_list(model_output_dir, f'*/{filename}.nc'), drop_variables=dropvarlist, parallel=False)

    # Process based on variable type
    if varname == 'SWI':
        # Process flux variable with hour 23/00 combination
        mismatch_days, processed_days, skipped_days = process_year_flux(ds, data_var=varname, output_dir=f"{output_dir}/{basin}_wy{wy}_{varname}")
        if return_vars:
            return mismatch_days, processed_days, skipped_days
    else:
        # Process state variable by removing hour 23 and shifting hour 00
        processed_days, skipped_days = process_year_state(ds, data_var=varname, output_dir=f"{output_dir}/{basin}_wy{wy}")
        if return_vars:
            return processed_days, skipped_days

def parse_arguments():
        """Parse command line arguments.

        Returns:
        argparse.Namespace: Parsed command line arguments.
        """
        parser = argparse.ArgumentParser(description='Reprocess 6 hour iSnobal outputs into 4 timesteps per day, each consisting of 6 hours')
        parser.add_argument('basin', type=str, help='Basin')
        parser.add_argument('wy', type=int, help='Water year')
        parser.add_argument('-modeldir', type=str, help='Model runs directory',
                            default='/uufs/chpc.utah.edu/common/home/skiles-group2/model_runs_jmh')
        parser.add_argument('-var', '--variable', type=str, help='iSnobal output variable to process',
                            choices=['SWI', 'specific_mass', 'all'], default='all')
        parser.add_argument('-v', '--verbose', help='Print filenames', default=True)
        return parser.parse_args()

def __main__():
    args = parse_arguments()
    basin = args.basin
    wy = args.wy
    model_dir = args.modeldir
    varname = args.variable
    verbose = args.verbose

    # Find RTI directory for input basin and water year
    # rti_dir = fn_list(model_dir, f'{basin}*rti/wy{wy}/*100m/')[0]

    # Specify the output directory
    # output_dir = fn_list(model_dir, f'{basin}*rti/')[0]

    # For black river, specifically
    # output_dir = f'/uufs/chpc.utah.edu/common/home/skiles-group3/model_runs/{basin}*rti/'
    # for roaring fork located here
    # /uufs/chpc.utah.edu/common/home/skiles-group1/RoaringFork/wy2021/erw_isnobal/wy2021/erw
    rti_dir = fn_list(model_dir, f'wy{wy}/{basin}')[0]
    output_dir = f'/uufs/chpc.utah.edu/common/home/skiles-group1/roaringfork/wy${wy}'

    # Use variable of interest to determine how to process and output daily files
    # If 'all', process both SWI and specific_mass
    if varname == 'all':
        determine_processing_flow(basin=basin, wy=wy, model_output_dir=rti_dir, varname='SWI',
                                  output_dir=output_dir, verbose=verbose, return_vars=False)
        determine_processing_flow(basin=basin, wy=wy, model_output_dir=rti_dir, varname='specific_mass',
                                  output_dir=output_dir, verbose=verbose, return_vars=False)
    else:
        # Process specified variable
        # SWI is summed fluxes for the time period and requires combining hour 23 and the following hour 00
        # specific_mass is a state variable and only requires removing hour 23 and shifting hour 00 to the previous day
        determine_processing_flow(basin=basin, wy=wy, model_output_dir=rti_dir, varname=varname,
                                  output_dir=output_dir, verbose=verbose, return_vars=False)

if __name__ == "__main__":
    __main__()