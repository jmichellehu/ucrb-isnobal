#!/usr/bin/env python

'''Script to plot exploratory analyses with topo.nc LANDFIRE veg variables

Usage: plot_veg_analysis.py basin_name water_year <snowmetric> <snowproperty> <outdir> <script> <palette> <verbose> <savefig>
'''
import os
import sys
import argparse

from pathlib import PurePath
from typing import List

import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import seaborn as sns

sys.path.append('/uufs/chpc.utah.edu/common/home/u6058223/git_dirs/env/')
import helpers as h

sys.path.append('/uufs/chpc.utah.edu/common/home/u6058223/git_dirs/ucrb-isnobal/scripts/')

SMALL_SIZE = 14
SMEDIUM_SIZE = 16
MEDIUM_SIZE = 22
BIGGER_SIZE = 24
BIGGEST_SIZE = 28

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=BIGGER_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGEST_SIZE)  # fontsize of the figure title

def get_dirs_filenames(basin: str, WY:int, verbose: bool = False, res: int = 100,
                           workdir: str = '/uufs/chpc.utah.edu/common/home/skiles-group3/model_runs/',
                           script_dir: str = '/uufs/chpc.utah.edu/common/home/skiles-group3/jmhu/isnobal_scripts/',
                           ):
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
        basin_setupdir: basin setup directory
    """
    basindirs = h.fn_list(workdir, f'{basin}*/*/{basin}*{res}*/')
    # Update basindirs for the selected water year
    basindirs = h.fn_list(workdir, f'{basin}*/*{WY}/{basin}*{res}*/')
    wydir = PurePath(basindirs[0]).parents[0].as_posix()

    # Locate basin setup dir
    basin_setupdir = h.fn_list(script_dir, f'*{basin}*setup')[0]
    if verbose:
        print(basin_setupdir)
    if verbose:
        [print(b) for b in basindirs]
    return basindirs, wydir, basin_setupdir


def get_basin_terrain(basin_setupdir, verbose=True):
    # Locate topo.nc file within setupdir
    topo_file = h.fn_list(basin_setupdir, '*/*topo.nc')[0]
    if verbose:
        print(topo_file)
    topo = xr.open_dataset(topo_file)
    # topo

    # Extract the DEM for this basin
    dem = topo['dem']
    # dem.plot.imshow()
    return dem

def reproj_match_tcc_to_basin(basin, dem, product='nlcd',
                               veg_dir='/uufs/chpc.utah.edu/common/home/u6058223/veg_data',
                               verbose=True, EPSG='epsg:32613', ploton=True):
    outname = f'{veg_dir}/{product}_{basin}_reprojmatch.tif'
    if os.path.exists(outname):
        print(f'Output file {outname} already exists. Skipping reprojection, loading directly.')
        return np.squeeze(xr.open_dataset(outname))
    else:
        # NLCD_tcc_conus_2021_v2021-4
        # science_tcc_conus_wgs84_v2023-5_20230101_20231231
        tcc_fn = h.fn_list(veg_dir, f'{product}*{basin}clip.tif')[0]
        if verbose:
            print(tcc_fn)
        tcc = np.squeeze(xr.open_dataset(tcc_fn))
        # reproject to default EPSG (epsg 32613)
        tcc = tcc.rio.reproject(EPSG)

        if ploton:
            fig, axa = plt.subplots(1, 2, figsize=(12, 6), sharex=True, sharey=True)
            ax = axa[0]
            dem.plot.imshow(ax=ax)
            ax = axa[1]
            tcc['band_data'].plot.imshow(ax=ax)
            for ax in axa.flatten():
                ax.set_aspect('equal')
        # Set CRS of DEM
        dem.rio.write_crs(32613, inplace=True)

        # Reproject and match TCC product to the DEM basin
        tcc_reprojmatch = tcc.rio.reproject_match(dem)

        if ploton:
            fig, axa = plt.subplots(1, 2, figsize=(12, 6), sharex=True, sharey=True)
            ax = axa[0]
            dem.plot.imshow(ax=ax)
            ax = axa[1]
            tcc_reprojmatch['band_data'].plot.imshow(ax=ax)
            for ax in axa.flatten():
                ax.set_aspect('equal')

        # Write this out
        tcc_reprojmatch.rio.to_raster(outname)
        return tcc_reprojmatch

def extract_basin_sdd(wydir, vtype='sdd', snowprop='depth'):
    # Get the calculated files
    # Calculate the timing of peak snow depth for each pixel and plot (histogram, map) - read from file! use calc_basin_peak.py to generate
    calc_fns = h.fn_list(wydir, f'*{vtype}*{snowprop}*.nc')

    # Read them in
    ds_list = [xr.open_dataset(calc_fn) for calc_fn in calc_fns]
    return ds_list

def plot_sdd_tcc_hexbin(basin, WY, xvar, ds_list, product, var='sdd_doy', outdir=None,
                              jointcolor='seagreen', jointheight=4,
                              kind='hex', runtypes=['Baseline', 'HRRR-SPIReS'],
                              marginal_kws = dict(bins=12, fill=False)):
    x = np.ravel(xvar.data)
    for mdx, ds in enumerate(ds_list):
        yvar = ds[var]
        y = np.ravel(yvar.data)

        s = sns.jointplot(x=x, y=y, color=jointcolor,
                          kind=kind, height=jointheight,
                          gridsize=25,
                          )

        # JointGrid has a convenience function
        s.set_axis_labels(f'{basin.capitalize()} % Canopy Cover', f'{runtypes[mdx]} SDD (DOY)', fontsize=12, fontweight='bold')

        ax = s.ax_joint
        # Calculate correlation coefficient
        corr = np.corrcoef(x, y)[0, 1]
        print(f'Correlation coefficient: {corr:.2f}')

        # Add to plot
        ax.text(0.05, 0.95, f'Correlation: {corr:.2f}', transform=ax.transAxes,
                fontsize=12, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.5))
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 200)
        sns.regplot(ax=ax, x=x, y=y, color=jointcolor, scatter=False)
        if outdir is not None:
            outname = f'{outdir}/{basin}_{product}_tcc_{runtypes[mdx]}_sdd_wy{WY}_hexbin.png'
            print(f'Saving figure to {outname}')
            plt.savefig(outname, dpi=300, bbox_inches='tight')

def parse_arguments():
    """Parse command line arguments.

    Returns:
        argparse.Namespace: Parsed command line arguments.
    """
    parser = argparse.ArgumentParser(description='Basin-wide snow disappearance calculation')
    parser.add_argument('basin', type=str, help='Basin name')
    parser.add_argument('wy', type=int, help='Water year of interest')
    parser.add_argument('-vtype', '--snowmetric', type=str, help='Snow metric of interest, defaults to sdd', choices=['sdd', 'peak'], default='sdd')
    parser.add_argument('-snowprop', '--snowproperty', type=str, help='Snow property of interest, defaults to depth', choices=['depth', 'density', 'swe'], default='depth')
    parser.add_argument('-out', '--outdir', default='/uufs/chpc.utah.edu/common/home/skiles-group3/jmhu/figures', help='Directory to save output files')
    parser.add_argument('-script', '--scriptdir', default='/uufs/chpc.utah.edu/common/home/skiles-group3/jmhu/isnobal_scripts', help='Directory to search for basin topos')
    parser.add_argument('-p', '--palette', default='tab20', help='Seaborn palette, defaults to tab20')
    parser.add_argument('-v', '--verbose', default=True, help='Print filenames, defaults to True')
    parser.add_argument('-s', '--savefig', default=True, help='Save figures, defaults to True')
    return parser.parse_args()

def __main__():
    args = parse_arguments()
    basin = args.basin
    WY = args.wy
    vtype = args.snowmetric
    snowprop = args.snowproperty
    outdir = args.outdir
    script_dir = args.scriptdir
    palette = args.palette
    verbose = args.verbose

    var = f'{vtype}_doy'#'sdd_doy' #'peak_doy'
    # if vtype == 'sdd':
    #     longname = "Disappearance"
    # elif vtype == 'peak':
    #     longname = "Peak"

    sns.set_palette(palette)

    if verbose:
        print('Getting dirs_filenames')

    # Extract the basin directories and water year for each model run
    _, wydir, basin_setupdir = get_dirs_filenames(basin, WY, verbose=verbose, script_dir=script_dir)

    # Get the calculated SDD files and read them in
    ds_list = extract_basin_sdd(wydir=wydir, snowprop=snowprop)

    # Extract the basin DEM
    dem = get_basin_terrain(basin_setupdir=basin_setupdir)

    # Extract tree canopy cover
    # product = 'nlcd'
    product = 'science'
    nlcd_reprojmatch = reproj_match_tcc_to_basin(basin=basin, dem=dem, product=product, ploton=False)

    # for NCLD
    plot_sdd_tcc_hexbin(basin=basin, WY=WY,
                        xvar=nlcd_reprojmatch['band_data'],
                        var=var, product=product,
                        ds_list=ds_list, outdir=outdir)


if __name__ == "__main__":
    __main__()