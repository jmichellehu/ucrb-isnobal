#!/usr/bin/env python

'''Script to rescale and reset ndv for v2023.0e albedo products to match v03 standards 
Usage: modscag_albedo_reprocess.py in_file 

Defaults to modscag_albedo_reprocess.py in_file -o out_file -s 100 -n 255
with out_file using reprocess_albedo.tif as suffix
'''

import argparse
import copy
import xarray as xr
import numpy as np

def reset_and_rescale(ds: xr.Dataset, ndv: float = None, scale: int = 100, varname: str = 'band_data') -> xr.Dataset:
    '''Reset the no data values (ndv) in the dataset to 0 and rescale the values to a specified range.
    Intended to standardize issues found in v2023.0e MODSCAG albedo to v03 standards
    Range: 0-10000
    NDV: 0

    Parameters:
    ds: xarray.Dataset
        The dataset containing the variable to be reset and rescaled.
    ndv: float or None, optional
        The no data value to be reset to 0. If None, no ndv will be reset.
    scale: int, optional
        The scaling factor to rescale the values. Default is 100.
    varname: str, optional
        The name of the variable to be reset and rescaled. Default is 'band_data'.

    Returns:
    ds_reset: xarray.Dataset
        The dataset with the variable reset and rescaled.'''
    
    # Create a deep copy of the array 
    ds_reset = copy.deepcopy(ds)
    # Extract numpy array
    arr = ds_reset[varname].data
    # Reset ndv to 0
    arr[np.isnan(arr)] = 0
    if ndv is not None:
        arr[arr==ndv] = 0
    # Rescale to 0-10000 and assign to the dataset
    ds_reset[varname].data = arr * scale
    return ds_reset

def __main__():
    # Parse command line args
    parser = argparse.ArgumentParser(description='MODSCAG albedo reprocessing')
    parser.add_argument('input_file', type=str, help='Path to input file')
    parser.add_argument('-o', '--output_file', type=str, help='Path to output file', default=None)
    parser.add_argument('-s', '--scale_factor', type=int, help='Scale factor', default=100)
    parser.add_argument('-n', '--nodataval', help='No data value', default=255)
    args = parser.parse_args()

    in_file = args.input_file
    out_file = args.output_file
    scale = args.scale_factor
    ndv = args.nodataval

    if out_file is None:
        out_file = f'{in_file.split("_albedo")[0]}_reprocess_albedo.tif'

    # Open the input file
    ds = np.squeeze(xr.open_dataset(in_file))

    # Reset and rescale the dataset
    ds_reset = reset_and_rescale(ds, scale=scale, ndv=ndv)

    # Save the modified dataset to the output file
    ds_reset.rio.to_raster(out_file)

if __name__ == "__main__":
    __main__()