#!/usr/bin/env python
'''NEEDS WORK: Script to plot snow disappearance date over shaded relief

Usage: plot_sdd.py basin wy
TODO: fix verbosity calls in functions
TODO: correct spacing when verbose=True
'''

import os
import sys
import argparse

import copy
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import xarray as xr
from pathlib import PurePath
import datetime
import seaborn as sns

sys.path.append('/uufs/chpc.utah.edu/common/home/u6058223/git_dirs/env/')
import helpers as h

sys.path.append('/uufs/chpc.utah.edu/common/home/u6058223/git_dirs/ucrb-isnobal/scripts/')
import processing as proc

# turn off the warnings
import warnings
warnings.filterwarnings("ignore")

def locate_sdd_files(basindirs, wydir, alg='first', snowvar='depth', verbose=True):
    # Pull variables from directory name
    basin = PurePath(wydir).parents[0].stem.split('_')[0]
    WY = int(wydir.split('wy')[1])
    # Instantiate empty lists
    sdd_date_ds_list = [] # sdd datasets
    sdd_fn_list = []
    for basindir in basindirs:
        try:
            sdd_fn = f'{wydir}/{PurePath(basindir).stem}_sdd_{snowvar}_wy{WY}_{alg}.nc'
            if verbose:
                print(sdd_fn)
            sdd_fn_list.append(sdd_fn)
            sdd_date_ds = xr.open_dataset(sdd_fn)
            # Set the name of the array as the isnobal implementation
            if sdd_fn.split('basin_100m_')[1].split('_')[0] == 'solar':
                sdd_name = 'HRRR-SPIReS'
            else:
                sdd_name = 'Baseline'
            # Specify the iSnobal run in the attributes
            sdd_date_ds.attrs['iSnobal run'] = sdd_name
            sdd_date_ds.attrs['Basin name'] = basin.capitalize()
            sdd_date_ds.attrs['Water Year'] = WY
            sdd_date_ds.attrs['SDD algorithm'] = alg

            sdd_date_ds.rio.write_crs("epsg:32613", inplace=True)
            sdd_date_ds_list.append(sdd_date_ds)
        except FileNotFoundError:
            print(f"{sdd_fn} does not exist, process snow.nc files")
            sys.exit(1)
    if verbose:
        print('\n')
    return sdd_date_ds_list

def locate_terrain(basin, sdd_date_ds, script_dir='/uufs/chpc.utah.edu/common/home/skiles-group3/jmhu/isnobal_scripts', verbose=True):
    terrain_fns = h.fn_list(script_dir, f'{basin}*_setup/data/*100m*.tif')
    if verbose:
        _ = [print(t) for t in terrain_fns]
        print('\n')
    # Load files
    terrain_list = [h.load(f) for f in terrain_fns]

    # Crop to the same extent as the depth data
    cropped_list = [ds.rio.reproject_match(sdd_date_ds['sdd_doy']) for ds in terrain_list]
    terraintitles = [PurePath(f).stem for f in terrain_fns]
    return cropped_list, terraintitles

# def plot_terrain(cropped_list, terraintitles, basin, cmaps=['magma', 'rainbow', 'Greys_r', 'viridis'], figsize=(20, 4)):
#     # Plot them up for a quick view
#     fig, axa = plt.subplots(1, len(cropped_list), figsize=figsize)
#     for jdx, ax in enumerate(axa.flatten()):
#         h.plot_one(cropped_list[jdx], cmap=cmaps[jdx], specify_ax=(fig, ax), title=terraintitles[jdx])
#     plt.suptitle(f'{basin}')
#     plt.tight_layout()

def check_missing_sdd(sdd_date_ds_list, WY, oct1_doy=274, verbose=True):
    """Check on the missing sdd pixels or the largest values and set to nan in both for comparison
    Current approach: omit the pixels with SDD before January 1st of that WY.
    TODO: Revisit this approach
    These are all pixels between Oct 1 and Dec 31
    DOY for Oct 1 is 274
    DOY for Dec 31 is 365/66
    Use > oct1_doy as the threshold
    """
    sdd_date_ds_ma_list = []
    for jdx, sdd_date_ds in enumerate(sdd_date_ds_list):
        # Count up the number of values greater than the threshold
        pixels2omit = sdd_date_ds['sdd_doy'].data[sdd_date_ds['sdd_doy'].data >= oct1_doy].size
        if verbose:
            print('\n')
            print(f'{sdd_date_ds.attrs["iSnobal run"]} has {pixels2omit} pixels with SDD between Oct 1 and Dec 31 of the current WY')
            print(f'Latest valid SDD is: {pd.Timestamp(sdd_date_ds['sdd'].max().values).strftime("%Y-%m-%d")}')
        # Set the pixels with SDD greater than the threshold to nan
        sdd_date_ds_ma = sdd_date_ds.where(sdd_date_ds['sdd'].dt.date>=datetime.date(WY, 1, 1))
        sdd_date_ds_ma_list.append(sdd_date_ds_ma)
        if verbose:
            print(f"{sdd_date_ds['sdd'].min().values} | {sdd_date_ds_ma['sdd'].min().values}")
        quickcheck = np.ravel(sdd_date_ds_ma['sdd'])
        if verbose:
            print(f"{sdd_date_ds['sdd'].size} | {quickcheck[~np.isnat(quickcheck)].size}")
            print(f"Invalid pixel counts match? {(sdd_date_ds['sdd'].size - quickcheck[~np.isnat(quickcheck)].size) == pixels2omit}")
    return sdd_date_ds_ma_list

def clean_axes(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_aspect('equal')
    ax.set_facecolor('k')

def calcnplot_sdd_shift(wydir, sdd_date_ds_list, hs, diffval=45, diff_cmap='RdYlBu', verbose=True, outname=None):
    # Pull variables from directory name
    basin = PurePath(wydir).parents[0].stem.split('_')[0]
    WY = int(wydir.split('wy')[1])
    title = f'{basin.capitalize()} WY {WY} SDD shift: HRRR-SPIReS minus Baseline'
    # Calculate diff in DOY
    sdd_diff = proc.calc_doydiff(sdd_date_ds_list, xmas=359, ndv=-9999)

    # Rename the variable
    sdd_diff.name = 'sdd_shift'

    fig, ax = plt.subplots(1, 1, figsize=(6,6))
    h.plot_one(hs, cmap='gray', cbaron=False,
            turnofflabels=True, turnoffaxes=True,
            alpha=0.8, specify_ax=(fig, ax))

    h.plot_one(sdd_diff, vmin=-diffval, vmax=diffval,
            cmap=diff_cmap, title=None, turnofflabels=True, turnoffaxes=True,
            alpha=0.6, specify_ax=(fig, ax))
    plt.title(title)

    if outname is not None:
        plt.savefig(outname, bbox_inches='tight', dpi=300)
        # Write out the sdd_diff as sdd_shift.nc
        out_fn = f'{wydir}/{basin}_sdd_shift_wy{WY}.nc'
        if not os.path.exists(out_fn):
            if verbose:
                print(f'Saving {out_fn}')
            sdd_diff.to_netcdf(out_fn)
    return sdd_diff

def plot_sdd_shift_hist(wydir, sdd_date_ds_list, sdd_diff, labels, elapsed_sec, bins=50, alpha=0.6, outname=None):
    # Pull variables from directory name
    basin = PurePath(wydir).parents[0].stem.split('_')[0]
    WY = int(wydir.split('wy')[1])
    title = f'{basin.capitalize()} WY {WY} SDD shift: HRRR-SPIReS compared to Baseline'

    fig, axa = plt.subplots(1, 2, figsize=(12, 3))
    # Plot up a histogram of differences
    ax = axa[0]
    h.plot_hist(sdd_date_ds_list[0]['sdd_doy'], range=(0, 365), specify_ax=(fig, ax), title=None, xlabel='SDD DOY', label=f'{labels[0]}', color='firebrick', alpha=alpha, bins=52)
    h.plot_hist(sdd_date_ds_list[1]['sdd_doy'], range=(0, 365), specify_ax=(fig, ax), title=None, xlabel='SDD DOY', label=f'{labels[1]}', color='gold', alpha=alpha, bins=52)
    data_slice = sdd_date_ds_list[1]['sdd_doy'] - sdd_date_ds_list[0]['sdd_doy']
    ax.grid(color='grey', linestyle='--', linewidth=0.5)

    # Add seconds since 1970 and convert to seconds for xlabel
    ax.set_xticklabels([pd.Timestamp((c * 24 * 3600 + elapsed_sec).astype('datetime64[s]')).strftime("%d %b") for c in ax.get_xticks()])
    ax.legend()
    ax.set_xlabel('Snow Disappearance Date')

    ax = axa[1]
    h.plot_hist(sdd_diff, specify_ax=(fig, ax), range=(-100, 100), title=None, label=f'{labels[1]} minus \n{labels[0]}', xlabel='SDD day difference', bins=bins)
    ymin, ymax = ax.get_ylim()
    ax.axvline(x=0, ymin=ymin, ymax=ymax, linestyle='--', color='k')
    ax.grid(color='grey', linestyle='--', linewidth=0.5)

    # Add some annotation
    # base this on the max ylim
    ax.annotate(text=f'{labels[1]} \nmelts later', xy=(75, ymax * 0.75), ha='center')
    ax.annotate(text=f'{labels[1]} \nmelts earlier', xy=(-75, ymax * 0.75), ha='center')
    ax.annotate(f'Difference: \nMedian: {np.nanmedian(data_slice):.1f} days \nMean: {np.nanmean(data_slice):.1f} days \nStDev: {np.nanstd(data_slice):.1f} days',
            xy=(65, ymax * 0.325), color='k', ha='center')
    plt.suptitle(title)
    if outname is not None:
        plt.savefig(outname, bbox_inches='tight', dpi=300)

def plot_sdd_maps(sdd_date_ds_list, hs, vmax, elapsed_sec, labels, figsize=(12, 6), cmap='magma_r', outname=None):
    fig, axa = plt.subplots(1, 2, figsize=figsize)
    for jdx, sdd_date_ds in enumerate(sdd_date_ds_list):
        ax = axa[jdx]
        hs.plot.imshow(cmap='gray', ax=ax, add_colorbar=False)
        s = sdd_date_ds['sdd_doy'].plot.imshow(cmap=cmap, alpha=0.7, ax=ax, vmin=0, vmax=vmax, add_colorbar=False)
        clean_axes(ax)
        # add scale bar
        h.add_scalebar(ax, scale=1, location='lower left')

        # add a new colorbar using separate fig axis
        if jdx == len(sdd_date_ds_list) - 1:
            cb_ax = fig.add_axes([0.91, 0.205, 0.015, 0.58])
            cbar = fig.colorbar(s, orientation='vertical', cax=cb_ax)

            # Add seconds since 1970 and convert to seconds
            cbar.set_ticklabels([pd.Timestamp((c * 24 * 3600 + elapsed_sec).astype('datetime64[s]')).strftime("%d %b %Y") for c in cbar.get_ticks()])

        ax.set_title(f'{labels[jdx]}')
        # else:
        #     ax.set_title(f'{basin.capitalize()}\n Snow Disappearance Date')
    if outname is not None:
        plt.savefig(outname, bbox_inches='tight', dpi=300)

def parse_by_aspect(aspect, basin, WY, sdd_date_ds_list, plot_on=False, compass_rose=['North', 'East', 'South', 'West'], aspect_labels=['N', 'E', 'S', 'W'], verbose=True):
    # bin by aspect
    aspect_bin = proc.bin_aspect(aspect, basinname=basin, plot_on=plot_on)
    # Count up percentages
    aspect_proportions = dict()
    # aspect_dataslices = []
    aspect_bin_arr = copy.deepcopy(aspect_bin.data)
    aspect_bin_arr = aspect_bin_arr[~np.isnan(aspect_bin_arr)]

    # Calculate some stats
    for r, direction in zip(range(len(aspect_labels)), compass_rose):
        # Get aspect proportions for the basin
        pixelcount = aspect_bin_arr[aspect_bin_arr==r+1].size
        percent = pixelcount / aspect_bin_arr.size
        if verbose:
            print(f"{direction}: {pixelcount} pixels, {percent*100:.1f}% of all pixels")
        aspect_name = aspect_labels[r]

        # Pull the sdd data from baseline and hrrr-spires for each aspect slice
        baseline = sdd_date_ds_list[0]['sdd_doy'].data[aspect_bin.data == r+1].flatten()
        hrrrspires = sdd_date_ds_list[1]['sdd_doy'].data[aspect_bin.data == r+1].flatten()

        # Calculate the sdd_diff ("shift") data for each slice
        sdd_diff = hrrrspires - baseline

        # The datasets have set values to nans and nats, so invalid values need to be removed for calculations
        # print(f" {baseline[np.isnan(baseline)].size} invalid values")
        # print(f" {hrrrspires[np.isnan(hrrrspires)].size} invalid values")
        # print(f" {sdd_diff[np.isnan(sdd_diff)].size} invalid values")

        # Mask out all of the nan values
        valid_mask = ~np.isnan(baseline) & ~np.isnan(hrrrspires) & ~np.isnan(sdd_diff)
        baseline = baseline[valid_mask]
        hrrrspires = hrrrspires[valid_mask]
        sdd_diff = sdd_diff[valid_mask]

        # print(f" {baseline[np.isnan(baseline)].size} invalid values")
        # print(f" {hrrrspires[np.isnan(hrrrspires)].size} invalid values")
        # print(f" {sdd_diff[np.isnan(sdd_diff)].size} invalid values")

        # # save all the dataslices? any ohter use for aspect_dataslices?
        # df = pd.DataFrame({f'baseline{aspect_name}': baseline, f'hrrrspires{aspect_name}': hrrrspires, f'sdd_diff_{aspect_name}': sdd_diff})
        # aspect_dataslices.append(df)

        # Calculate the median and stdev for each aspect slice for baseline, hrrr-spires and the shift
        med_list = [np.nanmedian(f) for f in [baseline, hrrrspires, sdd_diff]]
        mean_list = [round(np.nanmean(f), ndigits=1) for f in [baseline, hrrrspires, sdd_diff]]
        stdev_list = [round(np.nanstd(f), ndigits=1) for f in [baseline, hrrrspires, sdd_diff]]
        aspect_proportions[aspect_name] = (pixelcount, percent, med_list, mean_list, stdev_list)

        # abbrev = direction[0]
        # Figure the date for this
        if verbose:
            td_sdd = pd.to_datetime(med_list[0]-1, unit='D', origin=pd.Timestamp(f'01-01-{WY}'))
            hm_sdd = pd.to_datetime(med_list[1]-1, unit='D', origin=pd.Timestamp(f'01-01-{WY}'))
            td_doy = med_list[0]
            hm_doy = med_list[1]
            # print(f"{direction}:")
            print(f" {med_list[-1]} day median shift ..... {td_sdd.strftime('%Y-%m-%d')} to {hm_sdd.strftime('%Y-%m-%d')} ({td_doy} to {hm_doy})")

            td_sdd = pd.to_datetime(mean_list[0]-1, unit='D', origin=pd.Timestamp(f'01-01-{WY}'))
            hm_sdd = pd.to_datetime(mean_list[1]-1, unit='D', origin=pd.Timestamp(f'01-01-{WY}'))
            td_doy = mean_list[0]
            hm_doy = mean_list[1]
            print(f" {mean_list[-1]} day mean shift ....... {td_sdd.strftime('%Y-%m-%d')} to {hm_sdd.strftime('%Y-%m-%d')} ({td_doy} to {hm_doy})\n")
    return aspect_bin, aspect_proportions

def plot_sdd_by_aspect(aspect_bin, sdd_date_ds_list, labels, wydir, aspect_proportions,
                       compass_rose=['North', 'East', 'South', 'West'],
                       colors=['mediumpurple', 'gold', 'tomato', 'b', 'teal', 'gray'],
                       darkcolors=['indigo', 'goldenrod', 'darkred', 'midnightblue', 'darkslategray', 'dimgray'],
                       bins=50, binrange=(0, 250), fontsize=11, figsize=(12, 7), alpha=0.6, outname=None):
    # Pull variables from directory name
    basin = PurePath(wydir).parents[0].stem.split('_')[0]
    WY = int(wydir.split('wy')[1])
    title = f'{basin.capitalize()} WY {WY} \nSDD DOY by aspect Median [Stdev]'
    aspect_bin_arr = copy.deepcopy(aspect_bin.data)
    aspect_bin_arr = aspect_bin_arr[~np.isnan(aspect_bin_arr)]
    # Loop through array of binned aspect and plot the differences in sdd
    fig, axa = plt.subplots(2, 2, figsize=figsize, sharex=True, sharey=True)
    aspect_slices = []
    for jdx, f in enumerate(np.unique(aspect_bin_arr)):
        ax = axa.flatten()[jdx]
        # print(compass_rose[jdx])
        data_slice = sdd_date_ds_list[0]['sdd_doy'].data[aspect_bin.data == f]
        # Remove nans
        data_slice = data_slice[~np.isnan(data_slice)]
        aspect_slices.append(data_slice)
        h.plot_hist(data_slice, specify_ax=(fig, ax),
                    color=darkcolors[jdx],
                    label=f'{labels[0]}',
                    xlabel='SDD DOY',
                    title=compass_rose[jdx],
                    range=binrange, bins=bins)
        data_slice = sdd_date_ds_list[1]['sdd_doy'].data[aspect_bin.data == f]
        aspect_slices.append(data_slice)

        h.plot_hist(data_slice, specify_ax=(fig, ax),
                    color=colors[jdx],
                    label=f'{labels[1]}',
                    xlabel='SDD DOY',
                    alpha=alpha,
                    title=compass_rose[jdx],
                    range=binrange, bins=bins)
        ax.grid(color='lightgrey', linestyle='--', linewidth=0.5)
        ax.set_axisbelow(True)

    # Determine xy based on max values
    _, ymax = ax.get_ylim()
    xy = (-10, ymax*0.9)
    hrrrspires_xy = (-10, ymax*0.825)
    xshift = 65
    yshift = xy[1] - hrrrspires_xy[1]
    for jdx, f in enumerate(np.unique(aspect_bin_arr)):
        direction = compass_rose[jdx][0]
        ax = axa.flatten()[jdx]
        # Determine xy based on max values
        ax.annotate(f'{labels[0]}:', xy=xy, color=darkcolors[jdx], ha='left', fontsize=fontsize, weight='semibold', fontstyle='italic')
        ax.annotate(f'{np.nanmedian(aspect_slices[jdx*2]):.1f} [{np.nanstd(aspect_slices[jdx*2]):.0f}]',
                    xy=(xy[0]+xshift, xy[1]), color=darkcolors[jdx], ha='left', fontsize=fontsize)
        ax.annotate(f'{labels[1]}:', xy=hrrrspires_xy, color=colors[jdx], ha='left', fontsize=fontsize, weight='semibold', fontstyle='italic')
        ax.annotate(f'{np.nanmedian(aspect_slices[jdx*2+1]):.1f} [{np.nanstd(aspect_slices[jdx*2+1]):.0f}]',
                    xy=(hrrrspires_xy[0]+xshift, hrrrspires_xy[1]), color=colors[jdx], ha='left', fontsize=fontsize)

        # Add pixel count and proportion of basin
        ax.annotate(f'{aspect_proportions[direction][0]} pixels\n{aspect_proportions[direction][1]*100:.1f}%', xy=(0, hrrrspires_xy[1] - 2*yshift),
                    ha='left', fontsize=10, color='k')
    plt.suptitle(title)
    plt.tight_layout()
    if outname is not None:
        plt.savefig(outname, bbox_inches='tight', dpi=300)

def plot_sdd_by_aspect_line(basindirs, aspect_proportions, elapsed_sec, labels,
                            figsize=(6,4), markerstyles=['x', 'o'], compass_rose=['North', 'East', 'South', 'West'],
                            ylims=(30, 220), statname='mean', outname=None):
    # Plot up the days and differences per aspect
    stat_doy_list = []
    basin = PurePath(basindirs[0]).stem.split('_')[0]
    _, ax = plt.subplots(1, figsize=figsize)
    # Loop through each implementation
    for jdx, _ in enumerate(basindirs):
        mean_aspects = range(len(compass_rose))
        if statname == 'mean':
            # Mean
            stat_doys = [aspect_proportions[direction[0]][3][jdx] for direction in compass_rose]
        elif statname == 'median':
            # Median
            stat_doys = [aspect_proportions[direction[0]][2][jdx] for direction in compass_rose]
        stat_doy_list.append(np.array(stat_doys))
        ax.scatter(mean_aspects, stat_doys, marker=markerstyles[jdx], linewidths=1, label=labels[jdx])
        ax.plot(mean_aspects, stat_doys)
    ax.set_ylim(ylims)

    # Add seconds since 1970 and convert to seconds for xlabel
    ax.set_yticklabels([pd.Timestamp((c * 24 * 3600 + elapsed_sec).astype('datetime64[s]')).strftime("%d %b") for c in ax.get_yticks()])

    diffs_by_bin = stat_doy_list[1] - stat_doy_list[0]
    # add text of diff by bin for each mean elev bin
    for i, txt in enumerate(diffs_by_bin):
        # if the magnitude of the difference is less than 15, set the position to fixed shift
        if abs(txt) < 15:
            if txt < 0:
                shift = -15
            else:
                shift = 15
            ax.annotate(f'{txt:.0f}', (mean_aspects[i], stat_doys[i] + shift), color='k')
        else:
            ax.annotate(f'{txt:.0f}', (mean_aspects[i], stat_doys[i] - txt/2), color='k')
    ax.grid(color='lightgrey', linestyle='--', linewidth=0.5)
    ax.legend(loc='lower right')
    ax.set_ylabel('Snow disappearance date')
    ax.set_xlabel('Aspect')
    ax.set_title(f'{basin.capitalize()}: {statname} snow disappearance date by aspect')
    if outname is not None:
        plt.savefig(outname, bbox_inches='tight', dpi=300)

def parse_by_elev(dem, basin, basindirs, WY, sdd_date_ds_list, plot_on=False, verbose=True):
    # bin by elevation range
    # store the binned doy values, and mean doy, mean sdd for that bin in a dict
    _, dem_elev_ranges = proc.bin_elev(dem, basinname=basin, plot_on=plot_on)
    if verbose:
        print('Mean DOY and snow disappearance dates by elevation and treatment')

    sdd_elev_date_dict = dict()

    for basindir, sdd_date_ds in zip(basindirs, sdd_date_ds_list):
        if verbose:
            print('\n', PurePath(basindir).stem)
        # sdd_date_arr = sdd_date_ds['sdd'].data
        sdd_doy_arr = sdd_date_ds['sdd_doy'].data

        for elev_range in dem_elev_ranges:
            # Extract min and max elevations in that bin
            low, high = dem_elev_ranges[elev_range]
            # sdd_slice = sdd_date_arr[(dem_crop.data>=low) & (dem_crop.data<high)]
            doy_slice = sdd_doy_arr[(dem.data>=low) & (dem.data<high)]
            # Remove the nan values
            doy_slice = doy_slice[~np.isnan(doy_slice)]

            mean_doy = doy_slice.mean()
            mean_sdd = pd.to_datetime(mean_doy-1, unit='D', origin=pd.Timestamp(f'01-01-{WY}'))

            if verbose:
                print(f'{low, high}:\
                {mean_doy:.1f}, {mean_sdd.strftime("%Y-%m-%d")}')

            sdd_elev_date_dict[f'{PurePath(basindir).stem}_{low}_{high}'] = (doy_slice, mean_doy, mean_sdd)
    return sdd_elev_date_dict, dem_elev_ranges

def plot_sdd_by_elev_line(basindirs, dem_elev_ranges, sdd_elev_date_dict, elapsed_sec, labels,
                          markerstyles=['x', 'o'], figsize=(6,4), ylims=(30, 220), statname='mean', outname=None):
    # Pull variables from directory name
    wydir = PurePath(basindirs[0]).parents[0].as_posix()
    basin = PurePath(wydir).parents[0].stem.split('_')[0]
    WY = int(wydir.split('wy')[1])
    title = f'{basin.capitalize()} WY {WY} \n{statname} snow disappearance date by elevation'
    # Plot this up
    mean_doy_list = []
    _, ax = plt.subplots(1, figsize=figsize)
    for jdx, basindir in enumerate(basindirs):
        mean_elevs = []
        mean_doys = []
        for kdx, elev_range in enumerate(dem_elev_ranges):
            low, high = dem_elev_ranges[elev_range]
            mean_elev = (low + high) / 2
            doy_slice, mean_doy, mean_sdd = sdd_elev_date_dict[f'{PurePath(basindir).stem}_{low}_{high}']
            del doy_slice, mean_sdd # keep this so you know how the dict is organized
            mean_elevs.append(mean_elev)
            mean_doys.append(mean_doy)
        mean_doy_list.append(np.array(mean_doys))
        ax.scatter(mean_elevs, mean_doys, marker=markerstyles[jdx], linewidths=1, label=labels[jdx])
        ax.plot(mean_elevs, mean_doys)
    ax.set_ylim(ylims)
    # Add seconds since 1970 and convert to seconds for xlabel
    ax.set_yticklabels([pd.Timestamp((c * 24 * 3600 + elapsed_sec).astype('datetime64[s]')).strftime("%d %b") for c in ax.get_yticks()])

    diffs_by_bin = mean_doy_list[1] - mean_doy_list[0]
    # add text of diff by bin for each mean elev bin
    for i, txt in enumerate(diffs_by_bin):
        # if the magnitude of the difference is less than 10, set the position to fixed shift
        if abs(txt) < 15:
            if txt < 0:
                shift = -15
            else:
                shift = 15
            ax.annotate(f'{txt:.0f}', (mean_elevs[i] - 30, mean_doys[i] + shift), color='k')
        else:
            ax.annotate(f'{txt:.0f}', (mean_elevs[i] - 30, mean_doys[i] - txt/2), color='k')
    ax.grid(color='lightgrey', linestyle='--', linewidth=0.5)
    ax.legend(loc='lower right')
    ax.set_ylabel('Snow disappearance date')
    ax.set_xlabel('Binned mean elevation [m]')
    ax.set_title(title)
    if outname is not None:
        plt.savefig(outname, bbox_inches='tight', dpi=300)
    return mean_elevs

def plot_sdd_by_elev_boxplots(basindirs, dem_elev_ranges, sdd_elev_date_dict, mean_elevs, outname=None):
    # Plot the paired boxplots by elevation bin
    # get basin and wy from directory name
    wydir = PurePath(basindirs[0]).parents[0].as_posix()
    basin = PurePath(wydir).parents[0].stem.split('_')[0]
    WY = int(wydir.split('wy')[1])
    # Plot this up
    bval_list = []
    colnames = []
    for kdx, elev_range in enumerate(dem_elev_ranges):
        for jdx, basindir in enumerate(basindirs):
            low, high = dem_elev_ranges[elev_range]
            doy_slice, _, _ = sdd_elev_date_dict[f'{PurePath(basindir).stem}_{low}_{high}']
            bvals = proc.extract_boxplot_vals(doy_slice)
            bval_list.append(bvals)
            colnames.append(f'{PurePath(basindir).stem}_{low}_{high}')

    _, ax = plt.subplots(1, figsize=(12, 6))
    df = pd.DataFrame(data=np.array(bval_list).T, columns=colnames)
    df.index = ['low_whisk', 'p25', 'p50', 'p75', 'high_whisk']
    sns.boxplot(df, palette='icefire_r', ax=ax, width=0.5)
    ax.set_xticks(ticks=np.linspace(0.5, 18.5, 10), labels=mean_elevs, rotation=45)
    ax.grid(color='lightgrey', linestyle='--', linewidth=0.5)
    ax.set_ylim(0, df.loc['high_whisk'][0::2].max() + 25)
    ax.set_xlabel('Binned mean elevation [m]')
    ax.set_ylabel('Snow disappearance DOY')

    # Annotate with difference between treatments: HRRR-SPIReS minus Baseline
    for d, w, e in zip(df.loc['p50'].diff()[1::2], df.loc['high_whisk'][0::2], np.linspace(0.5, 18.5, 10)):
        ax.annotate(text=f'{d} days', xy=(e-0.5, w+15),
                    ha='left',
                    bbox=dict(facecolor='white', alpha=1,
                            edgecolor='gray', boxstyle='round, pad=0.2'))
    ax.set_title(f'{basin.capitalize()} WY {WY}: median SDD shift by elevation')
    if outname is not None:
        plt.savefig(outname, bbox_inches='tight', dpi=300)

def parse_by_slope(basin, slope, slope_bin_categories=['0-10', '10-20', '20-30', '30-40', '40-50', '50-60'], plot_on=False, verbose=True):
    slope_bin = proc.bin_slope(slope, basinname=basin, plot_on=plot_on)
    # Count up percentages
    slope_proportions = dict()
    for r in range(len(np.unique(slope_bin))):
        pixelcount = slope_bin.data[slope_bin.data==r+1].size
        percent = pixelcount / slope_bin.size
        if verbose:
            print(f"{slope_bin_categories[r]}: {pixelcount} pixels, {percent*100:.1f}% of all pixels")
        slope_proportions[r+1] = (pixelcount, percent)
    return slope_bin, slope_proportions

def plot_sdd_by_slope(basin, wydir, slope_bin, sdd_date_ds_list, labels, slope_proportions,
                      slope_bin_categories=['0-10', '10-20', '20-30', '30-40', '40-50', '50-60'],
                      colors=['mediumpurple', 'gold', 'tomato', 'b', 'teal', 'gray'],
                      darkcolors=['indigo', 'goldenrod', 'darkred', 'midnightblue', 'darkslategray', 'dimgray'],
                      binrange=(0, 250), bins=50, fontsize=11, alpha=0.6, verbose=True, outname=None
                      ):
    # Pull variables from directory name
    basin = PurePath(wydir).parents[0].stem.split('_')[0]
    WY = int(wydir.split('wy')[1])
    title = f'{basin.capitalize()} WY {WY} median SDD by slope'
    # Loop through array of binned aspect and plot the differences in sdd
    fig, axa = plt.subplots(3, 2, figsize=(12,11), sharex=True, sharey=True)
    slope_slices = []
    for jdx, f in enumerate(np.unique(slope_bin.data[~np.isnan(slope_bin.data)])):
        ax = axa.flatten()[jdx]
        if verbose:
            print(slope_bin_categories[jdx])
        data_slice = sdd_date_ds_list[0]['sdd_doy'].data[slope_bin.data == f]
        h.plot_hist(data_slice, specify_ax=(fig, ax),
                    color=darkcolors[jdx],
                    label=f'{labels[0]}',
                    xlabel='SDD DOY',
                    title=f'{slope_bin_categories[jdx]} degrees',
                    range=binrange, bins=bins)
        # Remove the nans, even though calcualtions omit nans
        data_slice = data_slice[~np.isnan(data_slice)]
        slope_slices.append(data_slice)

        data_slice = sdd_date_ds_list[1]['sdd_doy'].data[slope_bin.data == f]
        h.plot_hist(data_slice, specify_ax=(fig, ax),
                    color=colors[jdx],
                    label=f'{labels[1]}',
                    xlabel='SDD DOY',
                    alpha=alpha,
                    title=f'{slope_bin_categories[jdx]}˚',
                    range=binrange, bins=bins)
        # Remove the nans, even though calcualtions omit nans
        data_slice = data_slice[~np.isnan(data_slice)]
        slope_slices.append(data_slice)

        ax.grid(color='lightgrey', linestyle='--', linewidth=0.5)
        ax.set_axisbelow(True)

    # Determine xy based on max values
    _, ymax = ax.get_ylim()
    xy = (-10, ymax*0.9)
    hrrrspires_xy = (-10, ymax*0.83)
    xshift = 65
    yshift = xy[1] - hrrrspires_xy[1]
    for jdx, f in enumerate(np.unique(slope_bin.data[~np.isnan(slope_bin.data)])):
        ax = axa.flatten()[jdx]
        # Determine xy based on max values
        ax.annotate(f'{labels[0]}:', xy=xy, color=darkcolors[jdx], ha='left', fontsize=fontsize, weight='semibold', fontstyle='italic')
        ax.annotate(f'{np.nanmedian(slope_slices[jdx*2]):.1f} [{np.nanstd(slope_slices[jdx*2]):.0f}]', xy=(xy[0]+xshift, xy[1]), color=darkcolors[jdx], ha='left', fontsize=fontsize)
        ax.annotate(f'{labels[1]}:', xy=hrrrspires_xy, color=colors[jdx], ha='left', fontsize=fontsize, weight='semibold', fontstyle='italic')
        ax.annotate(f'{np.nanmedian(slope_slices[jdx*2+1]):.1f} [{np.nanstd(slope_slices[jdx*2+1]):.0f}]', xy=(hrrrspires_xy[0]+xshift, hrrrspires_xy[1]), color=colors[jdx], ha='left', fontsize=fontsize)

        # Add pixel count and proportion of basin
        ax.annotate(f'{slope_proportions[jdx+1][0]} pixels\n{slope_proportions[jdx+1][1]*100:.1f}%', xy=(0, hrrrspires_xy[1] - 2*yshift),
                    ha='left', fontsize=10, color='k')

    if len(np.unique(slope_bin.data[~np.isnan(slope_bin.data)])) == 5:
        # Turn off the last axis
        axa.flatten()[-1].axis('off')
    plt.suptitle(title)
    plt.tight_layout()
    if outname is not None:
        plt.savefig(outname, bbox_inches='tight', dpi=300)

def plot_sdd_by_slope_line(basindirs, sdd_date_ds_list, slope_bin, elapsed_sec, labels,
                           slope_bin_categories=['0-10', '10-20', '20-30', '30-40', '40-50', '50-60'],
                           figsize=(6,4), markerstyles=['x', 'o'], ylims=(30, 220), outname=None):
    # Pull variables from directory name
    wydir = PurePath(basindirs[0]).parents[0].as_posix()
    basin = PurePath(wydir).parents[0].stem.split('_')[0]
    WY = int(wydir.split('wy')[1])
    title = f'{basin.capitalize()} WY {WY}: mean snow disappearance DOY by slope'
    # Plot this up
    mean_doy_list = []
    sdd_slope_date_dict = dict()
    _, ax = plt.subplots(1, figsize=figsize)
    for jdx, (basindir, sdd_date_ds) in enumerate(zip(basindirs, sdd_date_ds_list)):
        sdd_doy_arr = sdd_date_ds['sdd_doy'].data
        mean_slopes = []
        mean_doys = []
        for kdx, slope_range in enumerate(slope_bin_categories):
            low, high = slope_range.split('-')
            low = int(low)
            high = int(high)
            mean_slope = ( low + high ) / 2
            slope_slice = sdd_doy_arr[slope_bin.data==kdx+1]
            # Remove  nan values
            slope_slice = slope_slice[~np.isnan(slope_slice)]
            mean_doy = slope_slice.mean()
            mean_sdd = pd.to_datetime(mean_doy-1, unit='D', origin=pd.Timestamp(f'01-01-{WY}'))
            # Build the dict
            sdd_slope_date_dict[f'{PurePath(basindir).stem}_{low}_{high}'] = slope_slice, mean_doy, mean_sdd

            mean_slopes.append(mean_slope)
            mean_doys.append(mean_doy)
        mean_doy_list.append(np.array(mean_doys))
        ax.scatter(mean_slopes, mean_doys, marker=markerstyles[jdx], linewidths=1, label=labels[jdx])
        ax.plot(mean_slopes, mean_doys)
    ax.set_ylim(ylims)
    # Add seconds since 1970 and convert to seconds for ylabel
    ax.set_yticklabels([pd.Timestamp((c * 24 * 3600 + elapsed_sec).astype('datetime64[s]')).strftime("%d %b") for c in ax.get_yticks()])

    diffs_by_bin = mean_doy_list[1] - mean_doy_list[0]

    # add text of diff by bin for each mean elev bin
    for i, txt in enumerate(diffs_by_bin):
        # if the magnitude of the difference is less than 10, set the position to fixed shift
        if abs(txt) < 15:
            if txt < 0:
                shift = -15
            else:
                shift = 15
            ax.annotate(f'{txt:.0f}', (mean_slopes[i], mean_doys[i] + shift), color='k')
        else:
            ax.annotate(f'{txt:.0f}', (mean_slopes[i], mean_doys[i] - txt/2), color='k')

    ax.grid(color='lightgrey', linestyle='--', linewidth=0.5)
    ax.legend(loc='lower right')
    ax.set_ylabel('Snow disappearance DOY')
    ax.set_xlabel('Binned mean slope')
    ax.set_title(title)
    if outname is not None:
        plt.savefig(outname, bbox_inches='tight', dpi=300)
    return sdd_slope_date_dict, mean_slopes

# Plot the paired boxplots by slope bin
def plot_sdd_by_slope_boxplots(basindirs, sdd_slope_date_dict, mean_slopes,
                               slope_bin_categories=['0-10', '10-20', '20-30', '30-40', '40-50', '50-60'],
                               figsize=(8, 5), outname=None):
    wydir = PurePath(basindirs[0]).parents[0].as_posix()
    basin = PurePath(wydir).parents[0].stem.split('_')[0]
    WY = int(wydir.split('wy')[1])
    title = f'{basin.capitalize()} WY {WY}: median SDD shift by slope'
    # Plot this up
    bval_list = []
    colnames = []
    for kdx, slope_range in enumerate(slope_bin_categories):
        for jdx, basindir in enumerate(basindirs):
            low, high = slope_range.split('-')
            low = int(low)
            high = int(high)
            doy_slice, _, _ = sdd_slope_date_dict[f'{PurePath(basindir).stem}_{low}_{high}']
            bvals = proc.extract_boxplot_vals(doy_slice)
            bval_list.append(bvals)
            colnames.append(f'{PurePath(basindir).stem}_{low}_{high}')

    _, ax = plt.subplots(1, figsize=figsize)
    df = pd.DataFrame(data=np.array(bval_list).T, columns=colnames)
    df.index = ['low_whisk', 'p25', 'p50', 'p75', 'high_whisk']
    sns.boxplot(df, palette='icefire_r', ax=ax, width=0.5)

    ax.set_xticks(ticks=np.linspace(0.5, 12.5, 7), labels=mean_slopes + ['>60'], rotation=45)
    ax.grid(color='lightgrey', linestyle='--', linewidth=0.5)
    ax.set_ylim(0, df.loc['high_whisk'][0::2].max() + 25)
    ax.set_xlabel('Binned mean slope [m]')
    ax.set_ylabel('Snow disappearance DOY')

    # Annotate with difference between treatments: HRRR-SPIReS minus Baseline
    for d, w, e in zip(df.loc['p50'].diff()[1::2], df.loc['high_whisk'][0::2], np.linspace(0.5, 12.5, 7)):
        ax.annotate(text=f'{d} days', xy=(e-0.5, w+10),
                    ha='left',
                    bbox=dict(facecolor='white', alpha=1,
                            edgecolor='gray', boxstyle='round, pad=0.2'))
    ax.set_title(title)
    if outname is not None:
        plt.savefig(outname, bbox_inches='tight', dpi=300)

def parse_arguments():
    """Parse command line arguments.

    Returns:
    argparse.Namespace: Parsed command line arguments.
    """
    parser = argparse.ArgumentParser(description='Calculate and plot snow disappearance dates for a given basin and water year.')
    parser.add_argument('basin', type=str, help='Basin name')
    parser.add_argument('wy', type=int, help='Water year of interest')
    parser.add_argument('-p', '--palette', type=str, help='Seaborn color palette', default='icefire')
    parser.add_argument('-o', '--outdir', type=str, help='Output directory',
                        default='/uufs/chpc.utah.edu/common/home/skiles-group3/jmhu/figures/sdd')
    parser.add_argument('-v', '--verbose', help='Print filenames', default=True)
    return parser.parse_args()

def __main__():
    # Parse command line args
    args = parse_arguments()
    basin = args.basin
    WY = args.wy
    palette = args.palette
    outdir = args.outdir
    verbose = args.verbose
    sns.set_palette(palette)

    workdir = '/uufs/chpc.utah.edu/common/home/skiles-group3/model_runs/'
    script_dir = '/uufs/chpc.utah.edu/common/home/skiles-group3/jmhu/isnobal_scripts'
    # Use this to thin out the basindirs
    basindirs = h.fn_list(workdir, f'{basin}*/*{WY}/{basin}*/')
    if verbose:
        _ = [print(b) for b in basindirs]

    wydir = PurePath(basindirs[0]).parents[0].as_posix()
    labels = ['Baseline', 'HRRR-SPIReS']

    # Identify SDD files
    print('Finding SDD files and terrain files')
    sdd_date_ds_list = locate_sdd_files(basindirs=basindirs, wydir=wydir, verbose=verbose)
    cropped_list, _ = locate_terrain(basin=basin, script_dir=script_dir, sdd_date_ds=sdd_date_ds_list[0], verbose=verbose)

    # Assign individual varnames for ease of use below
    dem, aspect, hs, slope = cropped_list

    # Reassign missing values to NaN and NaT
    sdd_date_ds_list = check_missing_sdd(sdd_date_ds_list=sdd_date_ds_list, WY=WY, verbose=verbose)

    # Plot SDD shift map
    print('\nPlotting SDD shift maps')
    outname = f'{outdir}/{basin}_sdd_shift_map_wy{WY}.png'
    sdd_diff = calcnplot_sdd_shift(wydir=wydir, sdd_date_ds_list=sdd_date_ds_list, hs=hs, outname=outname)

    # Calculate seconds since up to january first of this water year
    elapsed_sec = pd.to_datetime([f'{WY}-01-01 00:00:00']).astype(int) / 10**9
    elapsed_sec = int(elapsed_sec[0])

    # Plot SDD shift histogram
    outname = f'{outdir}/{basin}_sdd_shift_hist_wy{WY}.png'
    plot_sdd_shift_hist(wydir=wydir, sdd_date_ds_list=sdd_date_ds_list, sdd_diff=sdd_diff, labels=labels, elapsed_sec=elapsed_sec, outname=outname)

    outname = f'{outdir}/{basin}_sdd_maps_wy{WY}.png'
    aug1_doy = 213
    plot_sdd_maps(sdd_date_ds_list=sdd_date_ds_list, hs=hs, vmax=aug1_doy, elapsed_sec=elapsed_sec, labels=labels, outname=outname)

    # Move to parsing by terrain variables
    print('\nMoving to terrain variables')

    # Aspect
    print('Aspect')
    aspect_bin, aspect_proportions = parse_by_aspect(aspect=aspect, basin=basin, sdd_date_ds_list=sdd_date_ds_list, WY=WY, verbose=verbose)
    outname = f'{outdir}/{basin}_sdd_by_aspect_wy{WY}.png'
    plot_sdd_by_aspect(aspect_bin=aspect_bin, sdd_date_ds_list=sdd_date_ds_list, labels=labels, wydir=wydir, aspect_proportions=aspect_proportions, outname=outname)
    outname = f'{outdir}/{basin}_sdd_by_aspect_line_wy{WY}.png'
    plot_sdd_by_aspect_line(basindirs=basindirs, aspect_proportions=aspect_proportions, elapsed_sec=elapsed_sec, labels=labels, outname=outname)

    # Elevation
    print('Elevation')
    sdd_elev_date_dict, dem_elev_ranges = parse_by_elev(dem=dem, basin=basin, basindirs=basindirs, sdd_date_ds_list=sdd_date_ds_list, WY=WY)
    outname = f'{outdir}/{basin}_sdd_by_elev_line_wy{WY}.png'
    mean_elevs = plot_sdd_by_elev_line(basindirs=basindirs, dem_elev_ranges=dem_elev_ranges, sdd_elev_date_dict=sdd_elev_date_dict,
                                       elapsed_sec=elapsed_sec, labels=labels, outname=outname)
    outname = f'{outdir}/{basin}_sdd_by_elev_boxplots_wy{WY}.png'
    plot_sdd_by_elev_boxplots(basindirs=basindirs, dem_elev_ranges=dem_elev_ranges, sdd_elev_date_dict=sdd_elev_date_dict, mean_elevs=mean_elevs, outname=outname)

    # Slope
    print('\nSlope')
    slope_bin, slope_proportions = parse_by_slope(basin=basin, slope=slope)
    outname = f'{outdir}/{basin}_sdd_by_slope_wy{WY}.png'
    plot_sdd_by_slope(basin=basin, wydir=wydir, slope_bin=slope_bin, sdd_date_ds_list=sdd_date_ds_list, labels=labels, slope_proportions=slope_proportions, outname=outname)
    outname = f'{outdir}/{basin}_sdd_by_slope_line_wy{WY}.png'
    sdd_slope_date_dict, mean_slopes = plot_sdd_by_slope_line(basindirs=basindirs, sdd_date_ds_list=sdd_date_ds_list, slope_bin=slope_bin, elapsed_sec=elapsed_sec, labels=labels, outname=outname)
    outname = f'{outdir}/{basin}_sdd_by_slope_boxplots_wy{WY}.png'
    plot_sdd_by_slope_boxplots(basindirs=basindirs, sdd_slope_date_dict=sdd_slope_date_dict, mean_slopes=mean_slopes, outname=outname)

if __name__ == "__main__":
    __main__()