#!/usr/bin/env python

'''Script to plot exploratory analyses with topo.nc LANDFIRE veg variables

Usage: plot_veg_analysis.py basin_name water_year <snowmetric> <snowproperty> <outdir> <script> <palette> <verbose> <savefig>
'''
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

def get_dirs_filenames(basin: str, WY:int, verbose: bool = False, res: int = 100,
                           workdir: str = '/uufs/chpc.utah.edu/common/home/skiles-group3/model_runs/'):
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
    """
    basindirs = h.fn_list(workdir, f'{basin}*/*/{basin}*{res}*/')
    # Update basindirs for the selected water year
    basindirs = h.fn_list(workdir, f'{basin}*/*{WY}/{basin}*{res}*/')
    wydir = PurePath(basindirs[0]).parents[0].as_posix()
    if verbose:
        [print(b) for b in basindirs]
    return basindirs, wydir

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

def plot_var_scatter(ds_list, veglist, var, calc_fns, xlab, ylab):
    """Don't call this one, not a super helpful plot"""
    for mdx, ds in enumerate(ds_list):
        # Plot veg height against snow var
        fig, axes = plt.subplots(1, 3, figsize=(6*3, 4))
        xvar = veglist
        yvar = ds[var]
        titles = [f'{ylab} vs. Veg Height', f'{ylab} vs. Veg Type', f'{ylab} vs. Veg Transmissivity']

        for jdx, ax in enumerate(axes.flatten()):
            ax.scatter(xvar[jdx].values.flatten(), yvar.values.flatten(), color='k', s=20, alpha=0.2)
            ax.set_xlabel(xlab[jdx])
            ax.set_ylabel(ylab)
            ax.set_title(titles[jdx])
        plt.suptitle(PurePath(calc_fns[mdx]).stem, fontsize=14, y=1.02)

def plot_var_hist(ds_list, veglist, var, abbrev_lab, calc_fns, beginning, xlab, ylab, figsize=(12, 4), palette = 'tab20',
                  histtype = 'step', histbins = 52, histlw = 1.5, annotcolor = 'k', annotsize = 12, annotcenterx = 250,
                  suptitlesize = 14, suptitley = 1.02, savefig: bool = False
                  ):
    """Plot histogram of snow var by veg type"""
    for mdx, ds in enumerate(ds_list):
        xvar = veglist
        yvar = ds[var]
        for zdx, xv in enumerate(xvar):
            # Skip veg type, there are too many types
            if zdx == 1:
                pass
            else:
                unique_vals = np.unique(xvar[zdx])
                fig, ax = plt.subplots(figsize=figsize)
                for kdx, c in enumerate(unique_vals):
                    this_class = yvar.where(xvar[zdx] == c).values
                    this_class = this_class[~np.isnan(this_class)]
                    ax.hist(this_class, histtype=histtype, bins=histbins, linewidth=histlw, label=f'{c:.2f}')
                ymean = ax.get_ylim()[1] / 3 * 2

                # Annotate the median doy for each class c in the color of the histogram
                classcolors = sns.color_palette(palette, len(unique_vals))
                _ = [ax.annotate(f'{c:.2f}: {np.nanmedian(yvar.where(xvar[zdx] == c).values)}', xy=(annotcenterx, ymean-kdx*ymean*0.1),
                                c=classcolors[kdx], fontsize=annotsize, ha='center') for kdx, c in enumerate(unique_vals)]
                # add the "legend" for annotations
                ax.annotate(f'{xlab[zdx]}: {abbrev_lab}', xy=(annotcenterx, ymean+ymean*0.1), c=annotcolor, fontsize=annotsize, ha='center')
                plt.legend()
                plt.title(f'{ylab} Histogram by {xlab[zdx]}')
                plt.suptitle(PurePath(calc_fns[mdx]).stem, fontsize=suptitlesize, y=suptitley)

                # Save fig
                vtype = PurePath(beginning).stem.split('_')[2]
                fig_fn = f'{beginning}_{xlab[zdx].split("Veg ")[1].split(" ")[0].lower()}{PurePath(calc_fns[mdx]).stem.split("_100m")[1].split(f"{vtype}_")[0]}hist.png'
                print(fig_fn)
                if savefig:
                    fig.savefig(fig_fn, dpi=300, bbox_inches='tight')

def plot_var_heatmap(ds_list, veglist, var, xlab, ylab, calc_fns, beginning,
                     suptitlesize=14, suptitley=1.02, figsize=(18, 4), savefig: bool = False):
        """Plot heatmap of snow var by veg type"""
        for mdx, ds in enumerate(ds_list):
            # Now plot it as a heat map
            fig, axes = plt.subplots(1, 3, figsize=figsize)
            xvar = veglist
            yvar = ds[var]
            for jdx, ax in enumerate(axes.flatten()):
                sns.histplot(x=xvar[jdx].values.flatten(), y=yvar.values.flatten(), ax=ax)
                ax.set_xlabel(xlab[jdx])
                ax.set_ylabel(ylab)
                ax.set_title(f'{ylab} vs. {xlab[jdx]}')
            plt.suptitle(PurePath(calc_fns[mdx]).stem, fontsize=suptitlesize, y=suptitley)

            # Save fig
            vtype = PurePath(beginning).stem.split('_')[2]
            fig_fn = f'{beginning}{PurePath(calc_fns[mdx]).stem.split("_100m")[1].split(f"{vtype}_")[0]}heat.png'
            print(fig_fn)
            if savefig:
                fig.savefig(fig_fn, dpi=300, bbox_inches='tight')

def plot_median_shift(veglist: List, diff, xlab, ylab, abbrev_lab, beginning, basin, WY, 
                      histtype = 'step', histbins = 52, histlw = 1.5, annotcolor = 'k', annotsize = 12, palette = 'tab20',
                      xlims = (-150, 150), suptitlesize = 14, suptitley = 1.02, diffannotcenterx = 75,
                      savefig: bool = False, figsize: tuple = (12, 4)):
        """Calculate median shift per class"""
        xvar = veglist
        yvar = diff
        for zdx, xv in enumerate(xvar):
            if zdx == 1:
                pass
            else:
                unique_vals = np.unique(xvar[zdx])
                fig, ax = plt.subplots(figsize=(8,4))
                # Desired bin width
                bin_width = 2
                # Set a standard bin width despite variability and range in class values
                for kdx, c in enumerate(unique_vals):
                    this_class = yvar.where(xvar[zdx] == c).values
                    this_class = this_class[~np.isnan(this_class)]
                    # Calculate bin edges
                    min_val = np.min(this_class)
                    max_val = np.max(this_class)
                    bins = np.arange(min_val, max_val + bin_width, bin_width)
                    # ax.hist(this_class, histtype=histtype, bins=histbins * 2, linewidth=histlw, label=f'{c:.2f}')
                    ax.hist(this_class, histtype=histtype, bins=bins, linewidth=histlw, label=f'{c:.2f}')
                # Annotate the median doy for each class c in the color of the histogram
                classcolors = sns.color_palette(palette, len(unique_vals))
                ymean = ax.get_ylim()[1] / 3 * 2.5
                print(ymean)
                _ = [ax.annotate(f'{c:.2f}: {np.nanmedian(yvar.where(xvar[zdx] == c).values)}', xy=(diffannotcenterx, ymean-kdx*ymean*0.07),
                                    c=classcolors[kdx], fontsize=annotsize, ha='center') for kdx, c in enumerate(unique_vals)]
                # add the "legend" for annotations
                ax.annotate(f'{xlab[zdx]}: {abbrev_lab}', xy=(diffannotcenterx, ymean+ymean*0.1), c=annotcolor, fontsize=annotsize, ha='center')
                # Add dashed zero line
                ax.axvline(0, linestyle='--', color='k', alpha=0.3)
                ax.set_xlim(xlims)
                ax.annotate(f'{basin.capitalize()} WY {WY}', xy=(-145, ymean*1.1), c=annotcolor, fontweight='bold', fontsize=14, ha='left')

                # Plot annotation showing earlier melt vs. later melt than Baseline
                ax.annotate('HRRR-SPIReS \nmelts earlier', xy=(-100, ymean/5), c='goldenrod', fontsize=annotsize, ha='center')
                ax.annotate('HRRR-SPIReS \nmelts later', xy=(100, ymean/5), c='b', fontsize=annotsize, ha='center')

                # Save fig
                fig_fn = f'{beginning}_{xlab[zdx].split("Veg ")[1].split(" ")[0].lower()}_hist_shift.png'
                print(fig_fn)
                if savefig:
                    fig.savefig(fig_fn, dpi=300, bbox_inches='tight')

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
    savefig = args.savefig

    var = f'{vtype}_doy'#'sdd_doy' #'peak_doy'
    abbrev_lab = var.upper()
    xlab = ['Veg Height (m)', 'Veg Type', 'Veg Transmissivity']
    if vtype == 'sdd':
        longname = "Disappearance"
    elif vtype == 'peak':
        longname = "Peak"
    ylab = f'Snow {longname} DOY'

    sns.set_palette(palette)

    if verbose:
        print('Getting dirs_filenames')

    # Extract the basin directories and water year for each model run
    _, wydir = get_dirs_filenames(basin, WY, verbose=verbose)

    # Locate basin setup dir
    basin_setupdir = h.fn_list(script_dir, f'*{basin}*setup')[0]
    if verbose:
        print(basin_setupdir)

    # Locate topo.nc file within setupdir
    topo_file = h.fn_list(basin_setupdir, '*/*topo.nc')[0]
    if verbose:
        print(topo_file)

    # Extract vegetation height, vegetation type, transmissivity from the topo.nc
    topo = xr.open_dataset(topo_file)
    veglist = [topo['veg_height'], topo['veg_type'], topo['veg_tau']]

    # Get the calculated files
    calc_fns = h.fn_list(wydir, f'*{vtype}*{snowprop}*.nc')

    # Read them in
    ds_list = [xr.open_dataset(calc_fn) for calc_fn in calc_fns]

    # Calculate the difference (HRRR-MODIS - Baseline)
    diff = ds_list[1][var] - ds_list[0][var]

    # Set the beginning of output filenames
    beginning = f'{outdir}/{basin}_wy{WY}_{vtype}_{snowprop}_veg'

    # Plot the exploratory analyses
    # plot_var_hist(ds_list=ds_list, veglist=veglist, var=var, xlab=xlab, ylab=ylab, abbrev_lab=abbrev_lab, calc_fns=calc_fns, beginning=beginning, savefig=savefig)
    # plot_var_heatmap(ds_list=ds_list, veglist=veglist, var=var, xlab=xlab, ylab=ylab, calc_fns=calc_fns, beginning=beginning, savefig=savefig)
    plot_median_shift(veglist=veglist, diff=diff, xlab=xlab, ylab=ylab, abbrev_lab=abbrev_lab,
                      beginning=beginning, basin=basin, WY=WY, savefig=savefig)

if __name__ == "__main__":
    __main__()