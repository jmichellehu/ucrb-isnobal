# Helper functions for working with isnobal outputs, ASO data products, and SNOTEL files
# TODO - need to organize

import os
import sys
import glob
from pathlib import PurePath
import pathlib as p

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import xarray as xr
import geopandas as gpd
from shapely.geometry import Point
from shapely.ops import unary_union
import datetime

import pyproj

from s3fs import S3FileSystem, S3Map


month_dict = {
    'Jan': 1,
    'Feb': 2,
    'Mar': 3,
    'Apr': 4,
    'May': 5,
    'Jun': 6,
    'Jul': 7,
    'Aug': 8,
    'Sep': 9,
    'Oct': 10,
    'Nov': 11,
    'Dec': 12,
}

def fn_list(thisDir, fn_pattern, verbose=False):
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
    fns=[]
    for f in glob.glob(thisDir + "/" + fn_pattern):
        fns.append(f)
    fns.sort()
    if verbose: print(fns)
    return fns


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
        datedir = fn_list(indir, f'*{indate}/')[0]

    if returnvar is None:
        return fn_list(datedir, '*nc')
    else:
        return fn_list(datedir, f'{returnvar}.nc')
    
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
    try:
        month = int(dt_file[4:-2])
    except:
        # newer collection, potentially collected over multiple days, 
        # access via 3 alpha code using month_dict
        month = month_dict[dt_file[4:-2][:3]]
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

def calc_sdd(snow_property, verbose=False, snow_name=None, day_thresh=10):
    '''
    Calculate snow disappearance date from a pandas series of snow depth or SWE.
    
    Args:
        snow_property (pandas.Series): A pandas series of snow depth or SWE (measurement units in meters).
        verbose (bool, optional): If True, print additional information. Defaults to False.
        snow_name (str, optional): Name of the snow property. Defaults to None.
    
    Returns:
        tuple: A tuple containing the snow disappearance date and the first derivative of the snow property.
    
    Notes:
        - This function calculates the first derivative of the snow property and expects input units of meters.
        - It determines the last date at which the derivative is nonzero , which represents the snow disappearance date.
        - This method is robust to a definable threshold (day_thresh), where the preceding sequential days must also have negative derivatives
    
    Modified from https://stackoverflow.com/questions/22000882/find-last-non-zero-elements-index-in-pandas-series
    '''
    # Calculate first derivative of snow property
    firstderiv = snow_property.diff() / snow_property.index.to_series().diff().dt.total_seconds()
    # Get list of dates with negative derivatives (declining snow property)
    # deriv_dates = firstderiv[firstderiv < -1e-7]
    deriv_dates = firstderiv[firstderiv < 0]

    # Determine last date at which derivative is robustly negative, starting with the last date in the series
    snow_all_test_date = deriv_dates.index[-1]
    counter = 0
    if verbose: 
        print(f'Starting snow all gone date: {snow_all_test_date}')

    # Loop through all dates where the first derivative is negative
    for f in range(-2, -len(deriv_dates), -1):
        preceding_date = deriv_dates.index[f-1]
        if verbose:
            print(snow_all_test_date, preceding_date)
        
        # If the preceding date is not the date before, reset the counter,
        # otherwise increment the counter
        if snow_all_test_date - preceding_date != datetime.timedelta(days=1):    
            if verbose: 
                print('Did not pass test, resetting counter')
            counter = 0
        else:
            counter+=1
        
        # Reassign snow_all_test_date to preceding date value
        snow_all_test_date = preceding_date

        # If the counter exceeds the sequential days threshold, 
        # break the loop and readjust the index to the now robustly found date
        if counter >= day_thresh:
            snow_all_gone_date = deriv_dates.index[f+counter-1]
            # print(f, counter)
            if verbose: 
                print(f'Found snow all gone date: {snow_all_gone_date}')
            break
        # print(f'{counter}\n')
    
    if verbose:
        print(f'Snow all gone date {snow_all_gone_date.strftime("%Y-%m-%d")}')
        print(f'Derivative: {firstderiv.loc[snow_all_gone_date]*86400} m/d')
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
        print(f'Peak {snow_name} date: {peak_date.strftime("%Y-%m-%d")}')
        print(f'Peak {snow_name} value: {max_val} {units}')
    
    return peak_date, max_val

def locate_snotel_in_poly(poly_fn: str, site_locs_fn: str, buffer: int = 0):
    '''
    Extract snotel sites located within a given polygon.

    Args:
        poly_fn (str): The filepath of the polygon.
        site_locs_fn (str): The filepath of the snotel sites.
        buffer (int): The buffer distance to pad the polygon geometry

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
    
    # Buffer this polygon geometry
    poly_geom = poly_geom.buffer(distance=buffer)
    
    # Check if sites are located within polygon
    idx = sites_gdf.intersects(poly_geom)
    
    # Extract sites located within polygon
    poly_sites = sites_gdf.loc[idx]
    
    return poly_sites

def get_nwm_retrospective_LDAS(site_gdf, start=None, end=None, var='SNOWH'):
    '''
    Retrieves NWM retrospective LDAS (NoahMP land surface model output) data for a given site or sites.

    Parameters:
        site_gdf (GeoDataFrame): A GeoDataFrame containing the site locations.
        start (str, optional): The start date of the data to retrieve. Defaults to None. Format: YYYY-MM-DD
        end (str, optional): The end date of the data to retrieve. Defaults to None. Format: YYYY-MM-DD
        var (str, optional): The variable to retrieve. Defaults to snow depth 'SNOWH'.

    Returns:
        list: A list of xarray Datasets containing the retrieved data for each input site.

    '''
    bucket = 's3://noaa-nwm-retrospective-3-0-pds/CONUS/zarr'
    fs = S3FileSystem(anon=True)
    ds = xr.open_dataset(S3Map(f"{bucket}/ldasout.zarr", s3=fs), engine='zarr')
    
    if var is not None:
        ds = ds[var]
      
    # Extract data by each site location
    ds_list = [np.squeeze(ds.sel(x=x, y=y, method='nearest')) for x, y in zip(site_gdf.geometry.x.values, site_gdf.geometry.y.values)]

    if start is not None and end is not None:
        ds_list = [ds.sel(time=slice(f'{start}T00:00:00', f'{end}T23:00:00')) for ds in ds_list]
    
    return ds_list


# def get_snotel_df_pt(jdx=0, sitenums=None, snotel_dir=None, WY=None, outepsg='epsg:32613', verbose=True):
#     '''Extract snotel data for specified water year and snotel site'''
#     sitenum = sitenums.iloc[jdx]
#     # If PoR, download and re-run function
#     if len(fn_list(snotel_dir, f'*site{sitenum}.csv')) == 0:
#         print(f'No record found for {sitenum}, need to download!')
#         print(f'Run: get_snotelPoR_csv.sh {sitenum} from {snotel_dir}')
#         print('Exiting...')
#         return
#     else:
#         snotelfn = fn_list(snotel_dir, f'*site{sitenum}.csv')[0]
#         df = pd.read_csv(snotelfn, skiprows=63, usecols=list(np.arange(0, 7)), parse_dates=["Date"])
#         # Copy date to new date indexing column
#         df['DateIndex'] = df['Date']
#         # reset index as Date
#         df = df.set_index('DateIndex')
#         # Clip to this water year
#         snotel_df = df[(df['Date']>=f'{int(WY) - 1}-10-01') & (df['Date']<f'{WY}-10-01')]
#         # Extract snotel point coords and plot
#         sitenums = [int(sitenum)] 
#         allsites_fn = fn_list(snotel_dir, '*active*csv')[0]
#         sites_df = pd.read_csv(allsites_fn, index_col=0)
#         # Extract the lats and lons based on these site numbers
#         snotellats = []
#         snotellons = []
#         for sitenum in sitenums:
#             # print(sitenum)
#             this_site = sites_df[sites_df['site_num']==sitenum]
#             lat, lon = this_site['lat'].values[0], this_site['lon'].values[0]
#             snotellats.append(lat)
#             snotellons.append(lon)

#         # Convert to UTM EPSG 32613
#         # Create a Geoseries based off of a list of a Shapely point using the lat and lon from the SNOTEL site
#         s = gpd.GeoSeries([Point(lon, lat) for lon, lat in zip(snotellons, snotellats)])
#         # Turn this into a geodataframe and specify the geom as the geoseries of the SNOTEL point
#         gdf = gpd.GeoDataFrame(geometry=s)
#         # Set the CRS inplace
#         gdf.set_crs('epsg:4326', inplace=True)
#         # Convert snotel coords' lat lon to UTM
#         gdf = gdf.to_crs(outepsg)

#         # Get sitename
#         sitename = snotel_df.columns[1].split(f' Snow Depth')[0]
#         if verbose:
#             print(f'Retrieved geodataframe of {sitename} SNOTEL site and dataframe for WY {WY}')
#         return snotel_df, gdf, sitenum, sitename

    
def get_snotel(sitenum, sitename, ST, WY, epsg=32613, snowvar='SNOWDEPTH'):
    '''Use metloom to pull snotel coordinates and return as geodataframe and daily data as dict of dataframes
    valid snow variables: SNOWDEPTH, SWE
    '''
    from metloom.pointdata import SnotelPointData
    import geopandas as gpd
    from shapely.geometry import Point
    from datetime import datetime
    
    # start and end date
    start_date = datetime(WY-1, 10, 1)
    end_date = datetime(WY, 9, 30)

    snotel_dfs = dict()
    snotellats = []
    snotellons = []
    for snotelNUM, snotelNAME, snotelST in zip(sitenum, sitename, ST):
        snotel_point = SnotelPointData(f"{snotelNUM}:{snotelST}:SNTL", f"{snotelNAME}")

        meta_df = snotel_point.metadata
        lon, lat = meta_df.x, meta_df.y
        snotellats.append(lat)
        snotellons.append(lon)
        
        # set up variable list
        if snowvar == "SNOWDEPTH":
            variables = [snotel_point.ALLOWED_VARIABLES.SNOWDEPTH]
        elif snowvar == "SWE":
            variables = [snotel_point.ALLOWED_VARIABLES.SWE]

        # request the data - use daily, the hourly data is too noisy and messes up SDD calcs
        df = snotel_point.get_daily_data(start_date, end_date, variables)
        # df = snotel_point.get_hourly_data(start_date, end_date, variables)

        # Convert to metric here
        if snowvar == "SNOWDEPTH":
            df['SNOWDEPTH_m'] = df['SNOWDEPTH'] * 0.0254
        if snowvar == "SWE":
            df['SWE_m'] = df['SWE'] * 0.0254
        
        # Reset the index 
        df = df.reset_index().set_index("datetime")

        # Store in dict
        snotel_dfs[snotelNAME] = df
    
    # Create a Geoseries based off of a list of a Shapely point using the lat and lon from the SNOTEL site
    s = gpd.GeoSeries([Point(lon, lat) for lon, lat in zip(snotellons, snotellats)])

    # Turn this into a geodataframe and specify the geom as the geoseries of the SNOTEL point
    gdf = gpd.GeoDataFrame(geometry=s)

    # Set the CRS inplace
    gdf.set_crs('epsg:4326', inplace=True)

    # Convert snotel coords' lat lon to UTM
    gdf = gdf.to_crs(f'epsg:{epsg}')

    return gdf, snotel_dfs