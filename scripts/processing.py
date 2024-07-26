# Helper functions for working with isnobal outputs, ASO data products, and SNOTEL files
# TODO - need to organize

import os
import sys
from pathlib import PurePath
import pathlib as p

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import xarray as xr
import geopandas as gpd
from shapely.geometry import Point
from shapely.ops import unary_union

import pyproj

sys.path.append('/uufs/chpc.utah.edu/common/home/u6058223/git_dirs/env/')
import helpers as h

def get_varnc(basin=None, indir=None, indate=None, templatefn=None, returnvar=None):
    '''
    Obtains specified output files based on template file or basin and date.
    
    Parameters:
        basin (str): The name of the basin containing iSnobal runs (e.g., erw_newbasin_isnobal/). Default is None.
        indir (str): The input directory containing daily runs (e.g., erw_newbasin_isnobal/wy2018/erw_newbasin/). Default is None.
        indate (str): The input date in YYYYMMDD. Default is None.
        templatefn (str): The template file name. Default is None.
        returnvar (str): The variable to return. Default is None and returns all files.
        
    Returns:
        list: A list of output files based on the specified parameters.

    TODO:maybe remove basin and add returnvar handling for an input list
    '''
        
    # first check if you are looking for specific date based on template fn    
    if templatefn is not None:
        datedir = p.PurePath(templatefn).parent.as_posix()

    # or an input directory and specific date    
    elif (indir is not None) & (indate is not None):
        datedir = h.fn_list(indir, f'*{indate}/')[0]

    if returnvar is None:
        return h.fn_list(datedir, '*nc')
    else:
        return h.fn_list(datedir, f'{returnvar}.nc')
    
def extract_ravel_varvals(ds, varname='thickness'):
    '''Extract variable values from input DataArray.

    Parameters:
    ds (xarray.DataArray): Input DataArray containing the variable values.
    varname (str, optional): Name of the variable to extract. Defaults to 'thickness'.

    Returns:
    numpy.ndarray: Flattened array of variable values.
    '''
    return np.ravel(ds[varname].values)

def extract_boxplot_vals(rav_arr):
    '''
    Extracts outlier thresholds, 25th percentile, 50th percentile, and 75th percentile from the given array.

    Parameters:
    rav_arr (numpy.ndarray): The input array from which the statistics will be extracted.

    Returns:
    tuple: A tuple containing the following values in order: 
        - lowwhisk (float): The lower whisker threshold for outliers.
        - p25 (float): The 25th percentile value.
        - p50 (float): The 50th percentile value (median).
        - p75 (float): The 75th percentile value.
        - highwhisk (float): The upper whisker threshold for outliers.

    TODO: consider using robust thresholds for this (16–84p)
    '''
    
    p25 = np.nanpercentile(rav_arr, 25) # first quartile
    p50 = np.nanmedian(rav_arr)
    p75 = np.nanpercentile(rav_arr, 75) # third quartile
    iqr = p75 - p25
    
    # Compute outlier thresholds
    lowwhisk = p25 - 1.5 * iqr
    highwhisk = p75 + 1.5 * iqr
    
    return lowwhisk, p25, p50, p75, highwhisk

def fn2boxval(fn, varname='thickness'):
    '''
    Convert input filename (e.g., snow.nc) to boxplot values of specified variable value.

    Parameters:
        fn (str): The filename of the snow.nc file.
        varname (str, optional): The name of the variable to extract values from. Defaults to 'thickness'.

    Returns:
        list: The boxplot values of the specified variable value.
    '''
    ds = xr.open_dataset(fn)
    raveled_vals = extract_ravel_varvals(ds, varname=varname)
    boxvals = extract_boxplot_vals(raveled_vals)
    
    return boxvals


def extract_dt(fn, inputvar="_snowdepth"):
    '''
    Extracts date from filename (ASO snow depth or swe) and stores it in the corresponding DataSet.
    
    Args:
        fn (str): The filename from which to extract the date.
        inputvar (str, optional): The input variable to use for extraction. 
                                    Defaults to "_snowdepth". Change inputvar to '_swe' if using SWE files.
    
    Returns:
        pd.DatetimeIndex: A pandas DatetimeIndex object representing the extracted date.
    '''
    
    # get date from filename and store in dataset as a single dimension?
    dt_file = PurePath(fn).name.split(inputvar)[0].split("_")[-1]
    year = int(dt_file[:4])
    month = int(dt_file[4:-2])
    day = int(dt_file[-2:])
    # Create a dataframe from the parsed dates
    df = pd.DataFrame(columns=['Year', 'Month', 'Day'], 
                      data=np.expand_dims(np.array([year, month, day]), 1).T)
    
    if df.Month.dtype != 'int64':
        # Create dictionary of month abbreviations with associated numeric 
        # from https://stackoverflow.com/questions/42684530/convert-a-column-in-a-python-pandas-from-string-month-into-int
        d = dict(zip(pd.date_range('2000-01-01', freq='ME', periods=12).strftime('%b'), range(1,13)))
            
        # Map numeric month using month abbreviation dictionary
        df.Month = df.Month.map(d)
    
    dt = pd.to_datetime(df)
    return dt

def assign_dt(ds, dt):
    '''
    Assigns a datetime value to a new time dimension for the input dataset.

    Parameters:
        ds (xarray.Dataset): The input dataset.
        dt (datetime): The date value to assign.

    Returns:
        xarray.Dataset: The dataset with the new time dimension assigned.
    '''
    ds = ds.expand_dims(time=dt)
    return ds

def calc_sdd(snow_property, verbose=False, snow_name=None):
    '''
    Calculate snow disappearance date from a pandas series of snow depth or SWE.
    
    Args:
        snow_property (pandas.Series): A pandas series of snow depth or SWE (measurement units in meters).
        verbose (bool, optional): If True, print additional information. Defaults to False.
        snow_name (str, optional): Name of the snow property. Defaults to None.
    
    Returns:
        tuple: A tuple containing the snow disappearance date and the first derivative of the snow property.
    
    Notes:
        - This function calculates the first derivative of the snow property.
        - It determines the last date at which the derivative is nonzero, which represents the snow disappearance date.
        - This method is not robust to late season blips. See Hoosier Pass (531) and Grizzly Peak (505) for WY 2022 example.
    
    Modified from https://stackoverflow.com/questions/22000882/find-last-non-zero-elements-index-in-pandas-series
    '''

    # Calculate first derivative of snow property
    firstderiv = snow_property.diff() / snow_property.index.to_series().diff().dt.total_seconds()
    
    # Determine last date at which derivative is nonzero
    snow_all_gone_date = firstderiv[firstderiv != 0].index[-1]
    
    if verbose:
        print(f'Snow all gone date {snow_all_gone_date.strftime('%Y-%m-%d')}')
        print(f'Derivative: {firstderiv.loc[snow_all_gone_date]} m/d')
        if snow_name is None:
            snow_name = 'Snow property value'
        print(f'{snow_name}: {snow_property.loc[snow_all_gone_date]} m')
    
    return snow_all_gone_date, firstderiv

def calc_peak(snow_property, verbose=False, snow_name=None, units='m'):
    '''
    Finds the date of the maximum snow value from a pandas series of snow depth or SWE.

    Parameters:
        snow_property (pandas.Series): A pandas series containing snow depth or SWE values.
        verbose (bool, optional): If True, prints additional information. Defaults to False.
        snow_name (str, optional): The name of the snow property. Defaults to None.
        units (str, optional): The units of the snow property. Defaults to 'm'.

    Returns:
        tuple: A tuple containing the peak date and the maximum value of the snow property.
    '''
    # Determine date of maximum value in snow property
    peak_date = snow_property.idxmax()
    
    # Pull the value of the snow property at peak date
    max_val = snow_property.loc[peak_date]
    
    if verbose:
        if snow_name is None:
            snow_name = 'Snow property value'
        print(f'Peak {snow_name} date: {peak_date.strftime('%Y-%m-%d')}')
        print(f'Peak {snow_name} value: {max_val} {units}')
    
    return peak_date, max_val

def locate_snotel_in_poly(poly_fn: str, site_locs_fn: str):
    '''
    Extract snotel sites located within a given polygon.

    Args:
        poly_fn (str): The filepath of the polygon.
        site_locs_fn (str): The filepath of the snotel sites.

    Returns:
        geopandas.GeoDataFrame: A geodataframe of snotel sites located within the polygon.
    
    Notes:
        site_locs_fn:   EPSG 4326 /uufs/chpc.utah.edu/common/home/skiles-group3/SNOTEL/snotel_sites.json
                        EPSG 32613 /uufs/chpc.utah.edu/common/home/skiles-group3/SNOTEL/snotel_sites_32613.json
        poly_fn: '/uufs/chpc.utah.edu/common/home/skiles-group1/jmhu/ancillary/polys/yampa.shp'
    '''
    sites_gdf = gpd.read_file(site_locs_fn)
    poly_gdf = gpd.read_file(poly_fn)

    # Merge geometries if multipart polygon
    if len(poly_gdf.geometry) > 1:
        poly_geom = unary_union(poly_gdf.geometry)
    else:        
        poly_geom = poly_gdf.iloc[0].geometry
    
    # Check if sites are located within polygon
    idx = sites_gdf.intersects(poly_geom)
    
    # Extract sites located within polygon
    poly_sites = sites_gdf.loc[idx]
    
    return poly_sites