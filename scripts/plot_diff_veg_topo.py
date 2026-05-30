#!/usr/bin/env python

'''Script to plot snow depth differences against vegetation height, vegetation type, and slope.

Extracts LANDFIRE vegetation variables and DEM from topo.nc, calculates slope,
reads snow depth difference NetCDF files, and creates boxplots of differences
by veg height class, veg type, and slope range for each input difference file.

Usage: plot_diff_veg_topo.py basin_name water_year [--outdir DIR] [--verbose] [--overwrite]
'''
import os
import sys
import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import rioxarray as riox
from rasterio.enums import Resampling

sys.path.append('/uufs/chpc.utah.edu/common/home/u6058223/git_dirs/env/')
import helpers as h

sys.path.append('/uufs/chpc.utah.edu/common/home/u6058223/git_dirs/ucrb-isnobal/scripts/')
import processing as proc

def calculate_slope(dem):
    """Calculate slope in degrees from a DEM using gradient method.

    Parameters:
        dem: xarray DataArray containing elevation data

    Returns:
        slope: xarray DataArray with slope in degrees
    """
    # Get grid spacing (assume equal spacing in both directions)
    res = float(dem.rio.resolution()[0])  # resolution (absolute value)

    # Calculate gradients
    dz_dy, dz_dx = np.gradient(dem.values, res)

    # Calculate slope in degrees
    slope = np.degrees(np.arctan(np.sqrt(dz_dx**2 + dz_dy**2)))

    # Create DataArray with same coordinates
    slope_da = xr.DataArray(slope, coords=dem.coords, dims=dem.dims,
                           attrs={'units': 'degrees', 'long_name': 'Slope'})
    if dem.rio.crs is not None:
        slope_da = slope_da.rio.write_crs(dem.rio.crs)

    return slope_da

def reproj_match(ds, ref_ds, fallback_epsg, resampling='nearest'):
    try:
        return riox.rio.reproject_match(ds, ref_ds, resampling=resampling)
    except Exception as e:
        print(f'WARNING: Reprojection failed with error: {e}. Attempting fallback reprojection to {fallback_epsg}.')
        ds_reproj = ds.rio.reproject(fallback_epsg)
        return riox.rio.reproject_match(ds_reproj, ref_ds, resampling=resampling)

def plot_boxplot(topo_var, ds_fnlist):
#     if figsize is None: figsize=(cols*figmult/2, rows*figmult)

#     f,axa = plt.subplots(rows, cols, figsize=figsize, sharey=sharey)

#     for idx, ax in enumerate(axa.ravel()):
#         # Specify dataset
#         anomaly_map = iolib.ds_getma(ds_list[idx])

#         # Extract anomaly pixel values for each slope bin as a list
#         mid_val, var_list = slope_aggregator_box(slope, anomaly_map, slope_lims=slope_lims)
#         box_test = ax.boxplot(var_list, positions=mid_val, sym='', widths=boxwidth, patch_artist=True)

#         # Get pixel counts (and percentages of total pixels)
#         count = [var.size for var in var_list]
#         count_per = [(c/np.sum(count))*100 for c in count]
#         time_color = set_box_color(box_test, count_per,ax=ax,cmap=cmap, usenorm=usenorm)

#         # axis presentation
#         ax.set_xticks(np.arange(0,slope_lims[1]+1,10))
#         ax.set_xticklabels(np.arange(0,slope_lims[1]+1,10))
#         if max_anomaly2plot:
#             ax.set_ylim(max_anomaly2plot)
#         else:
#             ax.set_ylim(top=max_anomaly2plot)

#         for jdx, (pix_count, xloc, anom_vals) in enumerate(zip(count_per, mid_val, var_list)):
# #             med=np.nanmedian(anom_vals)
# #             maxiqr=np.percentile(anom_vals, 75)
# #             # median elevation anomaly placed above 75th percentile yloc
# #             ax.annotate(f'{med:0.1f}', xy=(xloc, maxiqr+ax.get_ylim()[1]*0.075),
# #                         xycoords='data', size=fontsize, ha='center', va="center", color='k', bbox=bbox_props)

#             # Bin pixel percent
#             ax.annotate(f'{pix_count:0.1f}%', xy=(xloc, pix_count/100*ax.get_ylim()[1]/2 + ax.get_ylim()[1]*2/3),
#                         xycoords='data', size=fontsize, ha='center', va="center", color=time_color[jdx], bbox=bbox_props)

    #     # Titles and labels
    #     if len(title)>1: ax.set_title(title[idx])
    #     else: ax.set_title(title[0])
    #     ax.set_xlabel(xlab)
    #     ax.set_ylabel(ylab)

    #     if addzeroline: ax.hlines(0, slope_lims[0], slope_lims[1], colors='gray', linestyles="-")

    #     ax.grid(color='lightgray', linestyle='-', linewidth=1)

    # plt.tight_layout();
    # plt.suptitle(f'Slope lims: {slope_lims}\nUsing refdem: {os.path.basename(ref_dem_fn)}', y=1.1, fontsize=18);


def parse_arguments():
    """Parse command line arguments.

    Returns:
        argparse.Namespace: Parsed command line arguments.
    """
    parser = argparse.ArgumentParser(
        description='Plot snow depth differences against vegetation height, vegetation type, and slope from topo.nc'
    )
    parser.add_argument('basin', type=str, help='Basin name')
    parser.add_argument('-dt', '--fulldate', type=int, help='Flight date of interest', default=None)
    parser.add_argument('-s', '--snowproperty', type=str, help='Snow property to analyze (e.g., depth, SWE)',
                        choices=['depth', 'SWE'], default='depth')
    parser.add_argument('-r', '--reproj', type=str, help='Reprojection method used for diff calculations (e.g., uniformreproj, original)',
                        choices=['uniformreproj', 'original'], default='original')
    parser.add_argument('-o', '--outdir', type=str, help='Output directory for figures',
                        default='/uufs/chpc.utah.edu/common/home/skiles-group3/jmhu/figures/spatial/diff_plots/depth_veg_topo')
    parser.add_argument('-script', '--scriptdir', type=str, help='Directory containing basin setup files',
                        default='/uufs/chpc.utah.edu/common/home/skiles-group3/jmhu/isnobal_scripts')
    parser.add_argument('-diff', '--diffdir', type=str, help='Directory containing difference files',
                        default='/uufs/chpc.utah.edu/common/home/skiles-group3/ASO/diffs')
    parser.add_argument('-v', '--verbose', action='store_true', help='Print verbose output', default=False)
    parser.add_argument('-ow', '--overwrite', action='store_true', help='Overwrite existing files', default=False)
    return parser.parse_args()

def __main__():
    args = parse_arguments()
    basin = args.basin
    dt = args.fulldate
    snowprop = args.snowproperty
    reproj_type = args.reproj
    outdir = args.outdir
    script_dir = args.scriptdir
    diff_dir = args.diffdir
    verbose = args.verbose
    overwrite = args.overwrite
    fallback_epsg = 'EPSG:32613'

    # Find and load topo.nc
    topo_fn = h.fn_list(script_dir, f'{basin}_setup/output_100m/topo.nc', verbose=verbose)
    topo = xr.open_dataset(topo_fn)

    # Locate and load difference files
    # defaults to all available dates in the basin if dt is None
    if dt is not None:
        diff_fn_list = h.fn_list(diff_dir, f'{basin}_wy*_diff_{dt}_{snowprop}_{reproj_type}.nc', verbose=verbose)
    else:
        diff_fn_list = h.fn_list(diff_dir, f'{basin}_wy*_diff_*_{snowprop}_{reproj_type}.nc', verbose=verbose)
    if len(diff_fn_list) == 0:
        print(f'ERROR: No difference files found in {diff_dir} for {basin}')
        sys.exit(1)

    # Reproject and match the topo ds to the first difference grid
    veg_height = reproj_match(topo['veg_height'], xr.open_dataset(diff_fn_list[0]), resampling=Resampling.average, dst_crs=fallback_epsg)
    veg_type = reproj_match(topo['veg_type'], xr.open_dataset(diff_fn_list[0]), resampling=Resampling.nearest, dst_crs=fallback_epsg)
    dem = reproj_match(topo['dem'], xr.open_dataset(diff_fn_list[0]), resampling=Resampling.average, dst_crs=fallback_epsg)

    # Calculate slope
    slope = calculate_slope(dem)

    # Define vegetation classes
    veg_height_classes = [0.00, 0.25, 0.75, 1.00, 2.00, 2.50, 3.00, 7.50, 17.50, 37.50]
    veg_type_classes = np.unique(np.ravel(veg_type.data))

    # Make slope bins in increments of 10 degrees
    slope_bins = proc.bin_slope(slope, plot_on=False)

    # Create output directory if it doesn't exist
    if outdir is not None and not os.path.exists(outdir):
        os.makedirs(outdir, exist_ok=True)
        if verbose:
            print(f'Created output directory: {outdir}')

    # Plot boxplots


if __name__ == '__main__':
    __main__()