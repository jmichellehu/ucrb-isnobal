# Helper functions for working with isnobal outputs, ASO data products, and SNOTEL files
# TODO - need to organize

from pathlib import PurePath
import pandas as pd
import xarray as xr
import numpy as np

def get_varnc(basin=None, indir=None, indate=None, templatefn=None, returnvar=None):
    '''Returns specified output files based on template file or basin and date.
    If no specified variable, returns all files
    basin = erw_newbasin_isnobal/
    indir = erw_newbasin_isnobal/wy2018/erw_newbasin/
    indate = 20180109
    TODO
    maybe remove basin
    add returnvar handling for an input list

    '''
    import pathlib as p
        
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
    '''Extract variable values from input DataArray, default snow depth (thickness)'''
    return np.ravel(ds[varname].values)

def extract_boxplot_vals(rav_arr):
    '''Extract outlier thresholds, 25th p, 50th p, and 75th p
    Returns: lowwhisk, p25, p50, p75, highwhisk
    TODO: consider using robust thresholds for this
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
    '''from snow.nc filename to boxplot values of specified variable value'''
    ds = xr.open_dataset(fn)
    raveled_vals = extract_ravel_varvals(ds, varname=varname)
    boxvals = extract_boxplot_vals(raveled_vals)
    
    return boxvals


def extract_dt(fn, inputvar="_snowdepth"):
    """Extract date from ASO snow depth or swe filename and store in corresponding DataSet
    Change inputvar to '_swe' if using SWE"""

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
    """Assign date value to a new time dimension for input dataset"""
    ds = ds.expand_dims(time=dt)
    return ds

def calc_sdd(snow_property, verbose=False, snow_name=None):
    '''Calculate snow disappearance date from a pandas series of snow depth or SWE
    Expects measurement units of meters
    Modified from From https://stackoverflow.com/questions/22000882/find-last-non-zero-elements-index-in-pandas-series
    TODO: careful, this is not robust to late season blips, see Hoosier Pass (531) and Grizzly Peak (505) for WY 2022
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
    '''Finds date of maximum snow value from a pandas series of snow depth or SWE
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