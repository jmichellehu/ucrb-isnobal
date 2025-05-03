#!/usr/bin/env python
'''Script to plot isnobal outputs of snow depth against ASO, NWM, UA snow depth.
TODO: add type hints, docstrings, and better comments
TODO: add more consistent plots, titles, etc.
TODO: add better console printing
TODO: review for clarity and conciseness
'''
import os
import sys

import argparse
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import geopandas as gpd
import xarray as xr
import rioxarray

from s3fs import S3FileSystem, S3Map
from typing import List, Tuple

sys.path.append('/uufs/chpc.utah.edu/common/home/u6058223/git_dirs/env/')
import helpers as h

sys.path.append('/uufs/chpc.utah.edu/common/home/u6058223/git_dirs/ucrb-isnobal/scripts/')
import processing as proc

from rasterio.enums import Resampling
import seaborn as sns

def specify_vars(var, compute_density, compute_SWE):
    # pull depth only
    if var == 'depth':
        NWM_var = 'SNOWH'
        UA_var = 'DEPTH'
        thisvar = 'thickness'
        aso_var = 'snowdepth'
        band_name = 'snow_depth'
    elif var == 'SWE':
        compute_SWE = True
        # pull SWE only
        NWM_var = 'SNEQV'
        UA_var = 'SWE'
        thisvar = 'specific_mass'
        aso_var = 'swe'
        band_name = 'swe'
    else:
        NWM_var = var.upper()
        UA_var = var.upper()
        compute_density = True
        thisvar = 'snow_density'
        aso_var = var
        band_name = var
    return NWM_var, UA_var, thisvar, compute_density, compute_SWE, aso_var, band_name

def clean_axes(ax, ticksoff=True, labelsoff=True, gridon=True, fc='k', aspect='equal'):
    if ticksoff:
        ax.set_xticks([])
        ax.set_yticks([])
    if labelsoff:
        ax.set_xlabel('')
        ax.set_ylabel('')
    if fc is not None:
        ax.set_facecolor(fc)
    if gridon:
        ax.grid(True)
    ax.set_aspect(aspect)

def locate_aso(aso_dir, state, basinname, WY, aso_var, band_name, inputvar, verbose=False):
    # Water year collections should all be post January so this should work
    aso_depth_fns = h.fn_list(aso_dir, f'{state}/*{basinname}*{WY}*{aso_var}*tif')

    date_list = retrieve_dates(aso_depth_fns, inputvar)

    # Load depth arrays and squeeze out single dimensions
    aso_depth_list = [np.squeeze(xr.open_dataset(fn)) for fn in aso_depth_fns]

    if len(aso_depth_fns) > 0:
        # Rename band_data to descriptive snow_depth
        aso_depth_list = [ds.rename_vars({'band_data': band_name}) for ds in aso_depth_list]

        # Deal with adding time input for ASO data
        aso_depth_list = [np.squeeze(proc.assign_dt(ds, proc.extract_dt(fn, inputvar=inputvar))) for ds, fn in zip(aso_depth_list, aso_depth_fns)]

        if verbose:
            _ = [print(f) for f in aso_depth_fns]
    else:
        print(f'No ASO data found for {basinname} in {WY} for {aso_var}, using default date list')

    return aso_depth_list, date_list

def retrieve_dates(aso_depth_fns, inputvar, date_list=None):
    if len(aso_depth_fns) > 0:
        # Get dates, could easily just pull from filenames, but this is fine
        date_list = [proc.extract_dt(fn, inputvar=inputvar)[0] for fn in aso_depth_fns]
        date_list = [f.strftime('%Y-%m-%d') for f in date_list]
    else:
        # Use set default dates since no ASO for this WY (e.g., yampa 2021–2023 and animas 2022-2024)
        date_list = ['2020-04-01', '2020-05-25']
    return date_list

def locate_isnobal_outputs(basindirs, dt, thisvar, epsg='32613', verbose=False):
    # find corresponding isnobal daily output for this dt and compare directly
    isnobal_rundirs = [h.fn_list(basindir, f"run{''.join(str(dt).split('-'))}")[0] for basindir in basindirs]
    if verbose:
        print(isnobal_rundirs)

    # Read in snow.nc file
    depth_fns = [proc.fn_list(sdir, "snow.nc")[0] for sdir in isnobal_rundirs]
    if verbose:
        print(depth_fns)

    # Extract the variable
    depths = [np.squeeze(xr.open_mfdataset(depth_fn, decode_coords="all")[thisvar]) for depth_fn in depth_fns]

    # Standardize the units
    # Convert specific mass kg/m2 to meters of SWE assuming density of 1000 kg/m3
    #  X kg/m2 * 1 m3/1000 kg = X/1000 m
    if thisvar == 'specific_mass':
        depths = [depth / 1000 for depth in depths]
        # Rename the variable to meters SWE, accordingly
        depths = [depth.rename('SWE [m]') for depth in depths]

    # Set CRS for iSnobal output
    depths = [depth.rio.write_crs(f'epsg:{epsg}', inplace=True) for depth in depths]

    return depths

def locate_nwm(dt, WY, NWM_var, proj_fn, poly_fn, epsg='32613', verbose=False):
    # As of March 2025, NWM version 3 only runs till Jan 2023, so no full WY 2023 or 2024!
    # https://registry.opendata.aws/nwm-archive/
    if WY < 2023:
        bucket = 's3://noaa-nwm-retrospective-3-0-pds/CONUS/zarr'
        fs = S3FileSystem(anon=True)
        ds = xr.open_dataset(S3Map(f"{bucket}/ldasout.zarr", s3=fs), engine='zarr')

        # Read in NWM proj4 string
        with open(proj_fn, "r") as f:
            proj4 = f.read()
        if verbose:
            print(proj4)

        # Read in poly_fn for spatial slicing
        poly_gdf = gpd.read_file(poly_fn)
        poly_gdf.set_crs(f'epsg:{epsg}', inplace=True, allow_override=True)

        # Convert to NWM proj4 coords
        poly_gdf = poly_gdf.to_crs(crs=proj4)

        # Crop the dataset to the input polygon extent
        cropped_ds = ds.sel(x=slice(poly_gdf.bounds.minx.values[0], poly_gdf.bounds.maxx.values[0]),
                            y=slice(poly_gdf.bounds.miny.values[0], poly_gdf.bounds.maxy.values[0]))

        # Slice time from NWM spatial crop
        nwm_depth = cropped_ds[NWM_var].sel(time=dt)

        # Standardize the units
        # Convert specific mass kg/m2 to meters of SWE assuming density of 1000 kg/m3
        #  X kg/m2 * 1 m3/1000 kg = X/1000 m
        if NWM_var == 'SNEQV':
            nwm_depth = nwm_depth / 1000
            # Rename the variable to meters SWE, accordingly
            nwm_depth = nwm_depth.rename('Snow Water Equivalent [m]')

        # Establish CRS inplace
        nwm_depth = nwm_depth.rio.write_crs(input_crs=proj4, inplace=True)
        return nwm_depth

def locate_ua(dt, WY, UA_var, poly_fn, use4k=False, epsg='32613', verbose=False):
    # Get matched UA files based on WY and ASO dates and subsequent date_lit
    ua_dir = '/uufs/chpc.utah.edu/common/home/skiles-group3/ancillary_sdswe_products/UA_SWE_depth'
    # flag to use the 4 km data
    use4k = False
    if UA_var == 'DEPTH':
        dropvar = 'SWE'
    elif UA_var == 'SWE':
        dropvar = 'DEPTH'
    else:
        dropvar = False

    # Establish filename of time series based on this WY (netcdf much smaller than csv)
    if use4k:
        # This is for the 4k downloaded netcdfs
        ua_fn = h.fn_list(ua_dir, f'*{WY}*')[0]
        # Read in matched UA dataset, dropping the SWE and time_str variables
        ds = xr.open_dataset(ua_fn, drop_variables=[dropvar, 'time_str'])
        thistime = dt
    else:
        # You can pull this in on the fly
        thisdt = ''.join(str(dt).split('-'))
        if verbose:
            print(thisdt)
        ua_fn = f'https://climate.arizona.edu/data/UA_SWE/DailyData_800m/WY{WY}/UA_SWE_Depth_800m_v1_{thisdt}_stable.nc'
        _, ds = rioxarray.open_rasterio(ua_fn, masked=True, chunks='auto')
        thistime = thisdt
    if verbose:
        print(ua_fn)

    # Slice time
    ds = np.squeeze(ds.sel(time=thistime)).load()

    # Read in poly_fn for spatial slicing
    poly_gdf = gpd.read_file(poly_fn)
    poly_gdf.set_crs(f'epsg:{epsg}', inplace=True, allow_override=True)

    # Convert to UA crs
    poly_gdf = poly_gdf.to_crs(crs=ds.crs.spatial_ref)

    if use4k:
        # Crop the dataset to the input polygon extent
        cropped_ds = ds[UA_var].sel(lon=slice(poly_gdf.bounds.minx.values[0], poly_gdf.bounds.maxx.values[0]),
                            lat=slice(poly_gdf.bounds.miny.values[0], poly_gdf.bounds.maxy.values[0]))
    else:
        # Crop the dataset to the input polygon extent
        # Watch the y ordering
        cropped_ds = ds[UA_var].sel(x=slice(poly_gdf.bounds.minx.values[0], poly_gdf.bounds.maxx.values[0]),
                            y=slice(poly_gdf.bounds.maxy.values[0], poly_gdf.bounds.miny.values[0]))

    # Establish CRS inplace
    ua_depth = cropped_ds.rio.write_crs(input_crs=ds.crs.spatial_ref, inplace=True)

    # Adjust units from mm SWE to m SWE
    ua_depth.data = ua_depth.data / 1000

    # Adjust unit in attributes too
    ua_depth.attrs['units'] = 'm'

    if use4k:
        # change lat and lon to x, y
        ua_depth = ua_depth.rename({'lon':'x', 'lat':'y'})

    return ua_depth

def get_arrays(basin, WY, process_nwm, depths, nwm_depth, ua_depth, nwm_reproj, ua_reproj, aso_var, aso_depth_list, ddx, labels):
    if basin == 'yampa' and WY != 2024:
        if process_nwm:
            arrs_original = depths + [nwm_depth] + [ua_depth]
            arrs_reproj = depths + [nwm_reproj] + [ua_reproj]
            titles = labels + ['NWM', 'ua']
        else:
            arrs_original = depths + [ua_depth]
            arrs_reproj = depths + [ua_reproj]
            titles = labels + ['ua']
    else:
        if process_nwm:
            arrs_original = depths + [nwm_depth] + [ua_depth] + [aso_depth_list[ddx][aso_var]]
            arrs_reproj = depths + [nwm_reproj] + [ua_reproj] + [aso_depth_list[ddx][aso_var]]
            titles = labels + ['NWM', 'UA', 'ASO']
        else:
            arrs_original = depths + [ua_depth] + [aso_depth_list[ddx][aso_var]]
            arrs_reproj = depths + [ua_reproj] + [aso_depth_list[ddx][aso_var]]
            titles = labels + ['UA', 'ASO']
    return arrs_original, arrs_reproj, titles

def extract_diffs(basin, WY, dt, arrs, titles, diff_dir, var, original, verbose=False, overwrite=True):
    # Convert dt back to YYYYMMDD
    thisdt = pd.to_datetime(dt).strftime('%Y%m%d')
    aso_reproj_list = []
    arrs_aso_clipped = []
    diff_dict = dict()
    for ldx, (arr, title) in enumerate(zip(arrs, titles)):
        if ldx < len(arrs) - 1:
            # Reproject and match ASO to comparison array
            aso_reproj = arrs[-1].rio.reproject_match(arr, resampling=Resampling.average)
            if verbose:
                print(title, aso_reproj.shape)

            # store this in a list for later
            aso_reproj_list.append(aso_reproj)

            # Compute diff
            diff = arr - aso_reproj

            # Compute the original array, but clipped to the ASO extent by adding ASO back in
            arrs_aso_clip = diff + aso_reproj
            arrs_aso_clipped.append(arrs_aso_clip)

            # Specify data variable name
            diff = diff.rename(f'{var}_diff')

            # Add attributes to the diff
            diff.attrs['units'] = 'm'
            diff.attrs['description'] = f'{title} - ASO {var}'
            diff.attrs['basin'] = basin.capitalize()
            diff.attrs['WY'] = WY
            diff.attrs['date'] = str(dt)
            diff.attrs['model'] = title

            # Store in dict
            diff_dict[title] = diff

            # Re-format title for output filename
            retitle = title.replace('-', '').lower()

            # Save each diff to file (.nc)
            if original:
                outname = f'{diff_dir}/{basin}_wy{WY}_{retitle}_diff_{thisdt}_{var}_original.nc'
            else:
                outname = f'{diff_dir}/{basin}_wy{WY}_{retitle}_diff_{thisdt}_{var}_uniformreproj.nc'
            if not os.path.exists(outname) or overwrite:
                if os.path.exists(outname):
                    print(f'File exists, but overwrite flag is {overwrite}, removing...')
                    os.remove(outname)
                if verbose:
                    print(f'Saving to {outname}')
                diff.to_netcdf(outname, format='NETCDF4', engine='netcdf4',
                                        encoding={f'{var}_diff': {'dtype': 'float32', 'zlib': True, 'complevel': 5}},
                                        compute=True)
            else:
                print(f'{outname} already exists, skipping')

    return diff_dict, aso_reproj_list, arrs_aso_clipped

def calc_volumes(arrs_aso_clipped, titles, aso_depth_list, ddx, aso_var, pix_res_list=[100, 100, 1000, 800, 50]):
    arrs2calc = arrs_aso_clipped + [aso_depth_list[ddx][aso_var]]
    # modify for missing NWM
    if len(arrs2calc) == 4:
        pix_res_list = [100, 100, 800, 50]
    for pix_res, arr, title in zip(pix_res_list, arrs2calc, titles):
        flat_arr = arr.values.flatten()
        flat_arr = flat_arr[~np.isnan(flat_arr)]
        volume = pix_res ** 2 * flat_arr # per-pixel volume = per-pixel area * per-pixel depth
        total_volume = volume.sum()
        print(f'{title}: {arr.size} pixels, {total_volume / 1e6:.0f} km^3')

def set_fontsizes(MEDIUM_SIZE, BIGGER_SIZE):
    plt.rc('font', size=BIGGER_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=BIGGER_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=MEDIUM_SIZE)  # fontsize of the figure title

# Add plot_combo_fig to this?

def plot_depthdiff_hist(basin, WY, dt, diff_dict, var, outdir=None, figsize=(16, 4), binrange=(-5, 5), nbins=50, alpha=0.5, logon=False,
                    verbose=True, overwrite=False):
    # Plot nonzero snow depth distribution
    _, axa = plt.subplots(1, len(diff_dict.keys()), figsize=figsize, sharex=True, sharey=True)
    for ldx, f in enumerate(diff_dict.keys()):
        ax = axa.flatten()[ldx]
        med = np.nanmedian(diff_dict[f])
        diff_dict[f].plot.hist(ax=ax, range=binrange, bins=nbins, alpha=alpha, ec=sns.color_palette()[ldx-1],
                                facecolor=sns.color_palette()[ldx-1],
                                label=f)
        ax.annotate(f, xycoords='axes fraction', xy=(0.05, 0.9), fontsize=16)
        ax.annotate(f'm={med:.2f} m', xycoords='axes fraction', xy=(0.05, 0.8), fontsize=14)
        ax.set_title('')
        ax.axvline(0, ymin=0, ymax=ax.get_ylim()[1], color='k', linestyle='--', linewidth=2)
        ax.set_xlabel(f'{var} difference [m]')
    if logon:
        plt.yscale('log')
    plt.suptitle(f'{basin.capitalize()} {str(dt)} \nDifference from ASO {var}', y=0.95, fontsize=18)
    plt.tight_layout()
    thisdt = pd.to_datetime(dt).strftime('%Y%m%d')
    if outdir is not None:
        outname = f'{outdir}/{basin}_wy{WY}_{thisdt}_{var}diff_hist.png'
        if verbose:
            print(outname)
        if not os.path.exists(outname) or overwrite:
            plt.savefig(outname, dpi=300, bbox_inches='tight')

def plot_depth_hist(basin, WY, dt, hist_arrs, aso_reproj_list, titles, var, outdir=None, figsize=(16, 4), alpha=0.5, ec='k',
                    binrange=(0, 3.5), nbins=35, thresh=0, logon=False,
                    verbose=True, overwrite=False):
    # Plot nonzero snow depth distribution
    _, axa = plt.subplots(1, len(hist_arrs), figsize=figsize, sharex=True, sharey=True)

    for ldx, arr in enumerate(hist_arrs):
        ax = axa.flatten()[ldx]
        nonzeroarr = np.ravel(arr)
        nonzeroarr = nonzeroarr[nonzeroarr>thresh]

        ax.hist(nonzeroarr, range=binrange, bins=nbins, alpha=alpha, ec=ec, label=titles[ldx], facecolor=sns.color_palette()[ldx-1])
        # Use each reprojected ASO set for each array
        nonzeroref = np.ravel(aso_reproj_list[ldx])
        nonzeroref = nonzeroref[nonzeroref>thresh]

        ax.hist(nonzeroref, range=binrange, bins=nbins, alpha=alpha/2, ec=ec, label='ASO', facecolor=sns.color_palette()[2])
        ax.legend(fontsize=14)
    if logon:
        plt.yscale('log')
        plt.ylim(1e0, 1e5)
    plt.xlabel(f'{var} [m]')
    plt.suptitle(f'{basin.capitalize()} {str(dt)} {var}', fontsize=18)
    plt.tight_layout()

    if outdir is not None:
        thisdt = pd.to_datetime(dt).strftime('%Y%m%d')
        outname = f'{outdir}/{basin}_wy{WY}_{thisdt}_{var}_hist.png'
        if verbose:
            print(outname)
        if not os.path.exists(outname) or overwrite:
            plt.savefig(outname, dpi=300, bbox_inches='tight')

def plot_diffs(diff_dict, titles, var, original=True, outdir=None, figsize=(16, 4), cmap='RdYlBu', plot_agu=False, vmax=3,
               verbose=True, overwrite=False):
    # Pull the requisite data based on diff_dict entries
    basin = diff_dict['Baseline'].attrs['basin']
    WY = diff_dict['Baseline'].attrs['WY']
    dt = diff_dict['Baseline'].attrs['date']
    thisdt = pd.to_datetime(dt).strftime('%Y%m%d')

    # Plot the diffs in a nice row
    if original:
        fig, axa = plt.subplots(1, len(diff_dict.keys()), figsize=figsize)
    else:
        fig, axa = plt.subplots(1, len(diff_dict.keys()), figsize=figsize, sharex=True, sharey=True)
    for mdx, f in enumerate(diff_dict.keys()):
        diff = diff_dict[f]
        title = titles[mdx]
        ax = axa.flatten()[mdx]

        # Plot the diff
        diff_im = diff.plot.imshow(ax=ax, cmap=cmap, add_colorbar=False,
                                vmin=-vmax/2, vmax=vmax/2)

        # add a new colorbar using separate fig axis
        if mdx == len(diff_dict.keys()) - 1:
            # set colorbar size to be same height as the plot
            cb_ax = fig.add_axes([0.91, 0.22, 0.005, 0.56])
            fig.colorbar(diff_im, orientation='vertical', cax=cb_ax, label=f'{var} \ndifference [m]',)

        if plot_agu:
            ax.set_title('')
            clean_axes(ax)
        else:
            ax.set_title(title)
            clean_axes(ax)
        if not plot_agu:
            plt.suptitle(f'{basin.capitalize()} {str(dt)}')

    if outdir is not None:
        if original:
            outname = f'{outdir}/{basin.lower()}_wy{WY}_{thisdt}_{var}_diffs_original.png'
        else:
            outname = f'{outdir}/{basin.lower()}_wy{WY}_{thisdt}_{var}_diffs_uniformreproj.png'
        if verbose:
            print(outname)
        if not os.path.exists(outname) or overwrite:
            plt.savefig(outname, dpi=300, bbox_inches='tight')

def locate_terrain(script_dir, basin, depth, verbose=False):
    terrain_fns = h.fn_list(script_dir, f'{basin}*_setup/data/*100m*.tif')
    if verbose:
        _= [print(tf) for tf in terrain_fns]
    # Load files
    terrain_list = [h.load(f) for f in terrain_fns]

    # Crop to the same extent as the depth data
    cropped_list = [ds.rio.reproject_match(depth) for ds in terrain_list]

    # Assign individual varnames for ease of use below
    dem, aspect, hs, slope = cropped_list

    return dem, aspect, hs, slope, terrain_list

def plotit(dem_elev_ranges: dict, clip_arr: xr.DataArray, dem: xr.DataArray, ax: plt.Axes,
           pixel_res: float = None, label='label', markerstyle: str = 'P', depths: bool = True) -> Tuple[List, List]:
    '''Plot the snow depth data binned by elevation range, with optional pixel resolution for volume calculations.
    Parameters
    ---------
    dem_elev_ranges: Dictionary of elevation ranges to bin the data
    clip_arr: DataArray to plot
    dem: DEM DataArray for the basin
    ax: matplotlib.axes.Axes to plot on
    pixel_res: Pixel resolution for calculating volume
    markerstyle: marker style for plotting
    depths: Plot depths (True) or volumes (False)
    Returns
    ---------
    mean_elevs: mean elevations for each bin
    mean_depths: mean depths for each bin if depths is True
    mean_volumes: mean volumes for each bin if depths is False
    '''
    mean_elevs = []
    mean_depths = []
    total_volumes = []

    for kdx, elev_range in enumerate(dem_elev_ranges):
        # Extract min and max elevations in that bin
        low, high = dem_elev_ranges[elev_range]
        elev_slice = clip_arr.data[(dem.data>=low) & (dem.data<high)]
        elev_slice = elev_slice[~np.isnan(elev_slice)]

        mean_elev = (low + high) / 2
        mean_depth = elev_slice.mean()
        mean_elevs.append(mean_elev)
        mean_depths.append(mean_depth)

        if pixel_res is not None:
            volume = pixel_res ** 2 * elev_slice # per-pixel volume = per-pixel area * per-pixel depth
            total_volume = volume.sum() # total volume in this elevation range
            total_volumes.append(total_volume)

    if depths:
        ax.scatter(mean_elevs, mean_depths, marker=markerstyle, s=80, linewidths=0.5, label=f'{label}: {np.nanmean(clip_arr):.2f} m')
        ax.plot(mean_elevs, mean_depths)
        return mean_elevs, mean_depths
    else:
        if pixel_res is not None:
            ax.scatter(mean_elevs, total_volumes, marker=markerstyle, s=80, linewidths=0.5, label=f'{label}: {np.nansum(total_volumes)/1e6:.0f} km$^{3}$')
            ax.plot(mean_elevs, total_volumes)
            return mean_elevs, total_volumes

def plot_depths_against_elev(diff_dict, aso_reproj_list, dem_elev_ranges, dem, terrain_list, arrs_original, titles, basin, WY, dt, var, area_slices,
                             markerstyles=['d', 'o', 'x',  'v', 's', 'd', 'P'], outdir=None, verbose=False, overwrite=False, run_aso=True):
    # Also plot this up
    _, ax = plt.subplots(1, figsize=(8, 6))
    print(f'Mean {var} by elevation')

    if run_aso:
        # Pull the array representing difference from ASO
        for kdx, k in enumerate(diff_dict.keys()):
            # print('\n', k)
            diff_arr = diff_dict[k].load()
            # Add back in ASO to get the original array, but still clipped to ASO bounds :D
            clip_arr = diff_arr + aso_reproj_list[kdx]
            mean_elevs, mean_depths = plotit(dem_elev_ranges=dem_elev_ranges, ax=ax,
                                            clip_arr=clip_arr, dem=dem, label=f'{k}',
                                            markerstyle=markerstyles[kdx], depths=True)

        # First, plot ASO!
        kdx = kdx + 1
        mean_elevs, mean_depths = plotit(dem_elev_ranges=dem_elev_ranges, ax=ax,
                                        clip_arr=aso_reproj_list[0], dem=dem,
                                        markerstyle='s', label='ASO', depths=True)
    else:
        # Crop to the same extent as the depth data
        cropped_original_list = [ds.rio.reproject_match(depth) for ds in terrain_list for depth in arrs_original]

        for kdx, k in enumerate(titles[:-1]):
            clip_arr = arrs_original[kdx].load()
            dem_orig = cropped_original_list[kdx]
            mean_elevs, mean_depths = plotit(dem_elev_ranges=dem_elev_ranges, clip_arr=clip_arr, ax=ax,
                                             markerstyle=markerstyles[kdx], dem=dem_orig, label=f'{k}', depths=True)

    # Put legend outside of the plot
    ax.legend(bbox_to_anchor=(1.8, 1), title=f'Basin-wide mean {var}', frameon=False)
    if var == 'depth':
        ylims = 0, 2.5
    elif var == 'SWE':
        ylims = 0, 0.5
    ax.set_ylim(ylims)
    ax.set_ylabel(f'Mean {var}')
    ax.set_xlabel('Binned mean elevation [m]')
    ax.set_title(f'{basin.capitalize()}: mean {var} by elevation bin {dt}')

    # Basin area by elevation
    basinarea_alpha = 0.5
    ax2 = ax.twinx()
    area_proportion = area_slices / area_slices.sum() * 100
    ax2.bar(x=mean_elevs, height=area_proportion, width=100, alpha=0.06, color='k')
    ax2.set_ylim(0, 25)

    # Add annotation for each area bar
    annotate = False
    if annotate:
        for idx, area in enumerate(area_slices):
            if idx == 6:
                adjustment = -0.02
            elif idx >= 7:
                adjustment = -area_proportion[idx] + 0.0075
            else:
                adjustment = 0.01
            ax2.annotate(f'{area_proportion[idx]:.2f}', xy=(mean_elevs[idx], area_proportion[idx]+adjustment),
                        ha='center', va='center', color='k', alpha=basinarea_alpha, fontsize=12)

    ax2.set_ylabel('Basin proportion (%)', labelpad=10, color='k', alpha=basinarea_alpha)
    # change ax2 ticks, tick labels and axis label color
    ax2.tick_params(axis='y', colors='gray')

    # rearrange legend items, moving last item to top
    handles, labels = ax.get_legend_handles_labels()
    handles = handles[-1:] + handles[:-1]

    # Save to file
    if outdir is not None:
        thisdt = pd.to_datetime(dt).strftime('%Y%m%d')
        outname = f'{outdir}/{basin}_wy{WY}_{thisdt}_{var}_against_elev.png'
        if verbose:
            print(outname)
        if not os.path.exists(outname) or overwrite:
            plt.savefig(outname, dpi=300, bbox_inches='tight')
    return mean_depths

def plot_volumes_against_elev(diff_dict, arrs, aso_reproj_list, dem_elev_ranges, dem, titles, basin, WY, dt, var, area_slices, pixel_res=100,
                      markerstyles=['d', 'o', 'x',  'v', 's', 'd', 'P'], outdir=None, verbose=False, overwrite=False, run_aso=True):
    _, ax = plt.subplots(1, figsize=(8, 6))
    print('Snow volume by elevation')
    if run_aso:
        # Pull the array representing difference from ASO
        for kdx, k in enumerate(diff_dict.keys()):
            # print('\n', k)
            diff_arr = diff_dict[k].load()
            # Add back in ASO to get the original array, but still clipped to ASO bounds :D
            clip_arr = diff_arr + aso_reproj_list[kdx]
            mean_elevs, total_volumes = plotit(dem_elev_ranges=dem_elev_ranges, clip_arr=clip_arr, ax=ax, label=f'{k}',
                                            dem=dem, pixel_res=pixel_res, markerstyle=markerstyles[kdx], depths=False)
        k = 'ASO'
        kdx = kdx + 1 # fix this in the function in the future, this is how markerstyles get updated and withou
        mean_elevs, total_volumes = plotit(dem_elev_ranges=dem_elev_ranges, clip_arr=aso_reproj_list[0], ax=ax, label=f'{k}',
                                            dem=dem, pixel_res=pixel_res, markerstyle=markerstyles[kdx], depths=False)


    else:
        for kdx, k in enumerate(titles[:-1]):
            # print('\n', k)
            clip_arr = arrs[kdx].load()
            mean_elevs, total_volumes = plotit(dem_elev_ranges=dem_elev_ranges, clip_arr=clip_arr, ax=ax, label=f'{k}',
                                            dem=dem, pixel_res=pixel_res, markerstyle=markerstyles[kdx], depths=False)

    ax.legend(bbox_to_anchor=(1.8, 1), title=f'Basin-wide total {var} volume', frameon=False)
    if var == 'depth':
        ylims = 0, 3.5e8
        if basin == 'yampa':
            ylims = 0, 8e8
    elif var == 'SWE':
        ylims = 0, 1e8

    ax.set_ylim(ylims)
    ax.set_ylabel(f'{var} volume')
    ax.set_xlabel('Binned mean elevation [m]')
    ax.set_title(f'{basin.capitalize()}: {var} volume by elevation bin {dt}')

    # Basin area by elevation
    basinarea_alpha = 0.5
    ax2 = ax.twinx()
    area_proportion = area_slices / area_slices.sum() * 100
    ax2.bar(x=mean_elevs, height=area_proportion, width=100, alpha=0.06, color='k')
    ax2.set_ylim(0, 25)

    ax2.set_ylabel('Basin proportion (%)', labelpad=10, color='k', alpha=basinarea_alpha)
    # change ax2 ticks, tick labels and axis label color
    ax2.tick_params(axis='y', colors='gray')
    # Save to file
    if outdir is not None:
        thisdt = pd.to_datetime(dt).strftime('%Y%m%d')
        outname = f'{outdir}/{basin}_wy{WY}_{thisdt}_{var}_volumes_against_elev.png'
        if verbose:
            print(outname)
        if not os.path.exists(outname) or overwrite:
            plt.savefig(outname, dpi=300, bbox_inches='tight')
    return total_volumes

def basin_terrain_snow_stats_df(basin, WY, dt, var, low_elevs, total_areas, area_slices, mean_elevs, mean_depths, total_volumes, dem_elev_ranges, n_bins, outdir=None):
            """Create a DataFrame with snow depth and volume statistics by elevation range.
            Outputs to csv in output directory with default verbose and no overwrite
            """
            df = pd.DataFrame({'low_thresh_elevation_m': low_elevs, 'cumulative_rel_area': total_areas,
                    'total_area_m2': area_slices,
                    'mean_elevation_m': mean_elevs,
                    'mean_depth': mean_depths, 'total_volume_m3': total_volumes,
                    'elev_range_m': list(dem_elev_ranges.values())
                    })
            thisdt = pd.to_datetime(dt).strftime('%Y%m%d')
            outname = f'{outdir}/{basin}_terrain_wy{WY}_{thisdt}_{var}_stats_{n_bins}elevbins.csv'
            if not os.path.exists(outname):
                print(f'Saving {outname}')
                df.to_csv(outname, index=False)
            else:
                print(f'{outname} already exists, skipping save')

def parse_arguments():
        """Parse command line arguments.

        Returns:
        argparse.Namespace: Parsed command line arguments.
        """
        parser = argparse.ArgumentParser(description='Plot spatial comparisons for a given basin and water year.')
        parser.add_argument('basin', type=str, help='Basin name')
        parser.add_argument('wy', type=int, help='Water year of interest')
        parser.add_argument('-shp', '--shapefile', type=str, help='Shapefile of basin polygon', default=None)
        parser.add_argument('-st', '--state', type=str, help='State abbreviation', default='CO')
        parser.add_argument('-e', '--epsg', type=str, help='EPSG of AOI', default='32613')
        parser.add_argument('-p', '--palette', type=str, help='Seaborn color palette', default='icefire')
        parser.add_argument('-o', '--outdir', type=str, help='Output directory for figures and files',
                            default='/uufs/chpc.utah.edu/common/home/skiles-group3/jmhu/figures/spatial')
        parser.add_argument('-v', '--verbose', help='Print filenames', default=True)
        parser.add_argument('-ow', '--overwrite', help='Overwrite existing files', default=False)
        return parser.parse_args()

def __main__():
    args = parse_arguments()
    basin = args.basin
    poly_fn = args.shapefile
    WY = args.wy
    state = args.state
    epsg = args.epsg
    palette = args.palette
    outdir = args.outdir
    verbose = args.verbose
    overwrite = args.overwrite

    sns.set_palette(palette)

    workdir = '/uufs/chpc.utah.edu/common/home/skiles-group3/model_runs/'
    script_dir = '/uufs/chpc.utah.edu/common/home/skiles-group3/jmhu/isnobal_scripts'
    aso_dir = '/uufs/chpc.utah.edu/common/home/skiles-group3/ASO'
    diff_dir = '/uufs/chpc.utah.edu/common/home/skiles-group3/ASO/diffs'

    # nwm proj4 file
    proj_fn = "/uufs/chpc.utah.edu/common/home/skiles-group3/jmhu/ancillary/NWM_datasets_proj4.txt"

    # Determine if using original or reprojected data
    original = True
    process_nwm = True
    if WY >= 2023:
        process_nwm = False
    plot_agu = False

    basindirs = h.fn_list(workdir, f'{basin}*/wy{WY}/{basin}*/')
    if verbose:
        _ = [print(b) for b in basindirs]
    if poly_fn is None:
        poly_fn = h.fn_list(script_dir, f'*{basin}*setup/polys/*shp')[0]
    if verbose:
        print(poly_fn)

    # can use default and no aso! fix that later, right now it just doesn't process
    if basin == 'yampa' and WY !=2024:
        sys.exit('No ASO data for Yampa in 2021-2023, exiting')
    elif basin == 'animas' and WY != 2021:
        sys.exit('No ASO data for Animas in 2022-2024, exiting')

    var = 'SWE'
    labels = ['Baseline', 'HRRR-SPIReS']

    # Prepare variables for extraction and analysis
    NWM_var, UA_var, thisvar, compute_density, compute_SWE, aso_var, band_name = specify_vars(var, compute_density=False, compute_SWE=False)

    basinname = basin.capitalize()
    inputvar = f'_{aso_var}'

    # Fetch ASO data for this basin and water year
    print('\nFetching ASO data...')
    aso_depth_list, date_list = locate_aso(aso_dir, state, basinname, WY, aso_var=aso_var, band_name=band_name, inputvar=inputvar, verbose=verbose)

    # Based on validation timesteps, select a time slice
    for ddx, dt in enumerate(date_list):
        print(f'============ {dt} ============')
        # convert to datetime object
        dt = np.datetime64(dt)

        # Get iSnobal snow depth
        print('\nFetching iSnobal data...')
        depths = locate_isnobal_outputs(basindirs=basindirs, dt=dt, thisvar=thisvar, epsg=epsg)

        print('\nFetching NWM and UA data...')
        if process_nwm:
            # Get NWM snow depth
            nwm_depth = locate_nwm(dt=dt, WY=WY, NWM_var=NWM_var, proj_fn=proj_fn, poly_fn=poly_fn, epsg=epsg)
            # Reproject to match iSnobal outputs for shared coord plotting
            nwm_reproj = nwm_depth.rio.reproject_match(depths[0])
        else:
            nwm_depth, nwm_reproj = None, None

        # Get UA snow depth
        ua_depth = locate_ua(dt=dt, WY=WY, UA_var=UA_var, poly_fn=poly_fn, use4k=False, epsg=epsg, verbose=verbose)
        # Reproject to match iSnobal outputs for shared coord plotting
        ua_reproj = ua_depth.rio.reproject_match(depths[0])

        print('\nReprojecting arrays...')
        arrs_original, arrs_reproj, titles = get_arrays(basin=basin, WY=WY, process_nwm=process_nwm,
                                                        depths=depths, nwm_depth=nwm_depth, ua_depth=ua_depth,
                                                        nwm_reproj=nwm_reproj, ua_reproj=ua_reproj, aso_var=aso_var,
                                                        aso_depth_list=aso_depth_list, ddx=ddx, labels=labels)
        del nwm_depth, ua_depth, nwm_reproj, ua_reproj

        # Extract difference arrays using the original arrays and reprojecting ASO to match
        arrs = arrs_original
        print('\nExtracting difference arrays...')
        diff_dict, aso_reproj_list, arrs_aso_clipped = extract_diffs(basin=basin, WY=WY, dt=dt, arrs=arrs, titles=titles,
                                                                     diff_dir=diff_dir, var=var, original=original,
                                                                     verbose=verbose, overwrite=overwrite)
        print('\nCalculating volumes...')
        calc_volumes(arrs_aso_clipped=arrs_aso_clipped, titles=titles, aso_depth_list=aso_depth_list, ddx=ddx, aso_var=aso_var)

        # Plot all of the depth and depth difference plots
        print('\nPlotting depth and depth difference figures...')

        # Set up the figure size and font sizes
        set_fontsizes(MEDIUM_SIZE=15, BIGGER_SIZE=18)

        plot_depthdiff_hist(basin=basin, WY=WY, dt=dt, diff_dict=diff_dict, logon=True, var=var,
                            outdir=outdir, verbose=verbose, overwrite=overwrite)
        plot_depth_hist(basin=basin, WY=WY, dt=dt, hist_arrs=arrs_aso_clipped, titles=titles, var=var,
                        aso_reproj_list=aso_reproj_list, logon=True, outdir=outdir, verbose=verbose, overwrite=overwrite)
        plot_diffs(diff_dict=diff_dict, titles=titles, var=var, original=original, plot_agu=plot_agu,
                   outdir=outdir, verbose=verbose, overwrite=overwrite)
        del arrs_aso_clipped

        # TERRAIN SECTION
        # Change arrays to reprojected for this section to work
        # Would need to completely rework all the functions for each dataset if using original arrays
        arrs = arrs_reproj
        # need to recalculate the diff_dict as well
        diff_dict, aso_reproj_list, _ = extract_diffs(basin=basin, WY=WY, dt=dt, arrs=arrs, titles=titles,
                                                      diff_dir=diff_dir, var=var, original=False,
                                                      verbose=verbose, overwrite=overwrite)
        # Extract terrain data
        print('\nExtracting terrain data...')
        dem, _, _, _, terrain_list = locate_terrain(script_dir=script_dir, basin=basin, depth=depths[0], verbose=verbose)
        del depths

        # Bin by elevation range into n_bins
        n_bins = 10
        _, dem_elev_ranges = proc.bin_elev(dem=dem, basinname=basin, p=n_bins, plot_on=False)
        total_areas, low_elevs, mean_elevs, area_slices = proc.bin_elev_range(dem=dem, dem_elev_ranges=dem_elev_ranges)

        # Plot the hypsometry and area-elevation curve
        print('\nPlotting elevation and area figures...')
        proc.plot_hypsometry(basin=basin, total_areas=total_areas, low_elevs=low_elevs,
                             outdir=outdir, verbose=verbose, overwrite=overwrite)
        proc.plot_aec(basin, area_slices, mean_elevs, kmflag=False, outdir=outdir,
                      verbose=verbose, overwrite=overwrite)

        # Plot the snow depth and volume by elevation range
        print('\nPlotting snow depth and volume by elevation range...')
        mean_depths = plot_depths_against_elev(diff_dict=diff_dict, aso_reproj_list=aso_reproj_list, dem_elev_ranges=dem_elev_ranges,
                                               dem=dem, terrain_list=terrain_list, arrs_original=arrs_original,
                                               titles=titles, basin=basin, WY=WY, dt=dt, var=var, area_slices=area_slices,
                                               outdir=outdir, verbose=verbose, overwrite=overwrite)
        total_volumes = plot_volumes_against_elev(diff_dict=diff_dict, arrs=arrs, aso_reproj_list=aso_reproj_list,
                                                  dem_elev_ranges=dem_elev_ranges, dem=dem, titles=titles, basin=basin,
                                                  WY=WY, dt=dt, var=var, area_slices=area_slices,
                                                  outdir=outdir, verbose=verbose, overwrite=overwrite)

        # Save the snow depth and volume statistics to a csv
        basin_terrain_snow_stats_df(basin=basin, WY=WY, dt=dt, var=var, low_elevs=low_elevs, total_areas=total_areas,
                                    area_slices=area_slices, mean_elevs=mean_elevs, mean_depths=mean_depths,
                                    total_volumes=total_volumes, dem_elev_ranges=dem_elev_ranges,
                                    n_bins=n_bins, outdir=outdir)

        # Delete remaining variables in this loop
        del dem, dem_elev_ranges, total_areas, low_elevs, mean_elevs, area_slices, mean_depths, total_volumes
        del arrs, arrs_original, arrs_reproj, titles, diff_dict, aso_reproj_list, terrain_list

if __name__ == '__main__':
    __main__()