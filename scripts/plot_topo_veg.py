#!/usr/bin/env python

'''Script to plot topo.nc LANDFIRE veg information

Usage: plot_topo_veg.py basin_name <verbose>
'''
import os
import sys
import argparse

import matplotlib.pyplot as plt
import xarray as xr

sys.path.append('/uufs/chpc.utah.edu/common/home/u6058223/git_dirs/env/')
import helpers as h

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

def plot(basin, topo, outdir: str = '.', figsize: tuple = (12, 4),
         titles = ['Vegetation Height (m)', 'Vegetation Type Code', 'Veg Type-Derived Transmissivity'],
         savefig: bool = False):
     # Quick plot
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    h.plot_one(topo['veg_height'], specify_ax=(fig, axes[0]), title='Vegetation Height (m)', cmap='Greens', vmin=0, vmax=20)
    h.plot_one(topo['veg_type'], specify_ax=(fig, axes[1]), title='Vegetation Type Code')
    h.plot_one(topo['veg_tau'], specify_ax=(fig, axes[2]), title='Veg Type-Derived Transmissivity', cmap='magma', vmin=0, vmax=1)
    for jdx, ax in enumerate(axes.flatten()):
            clean_axes(ax)
    plt.suptitle(f'{basin.capitalize()} Basin LANDFIRE Veg Height, Type, Tau', y=1.02)
    plt.tight_layout()

    # Save fig
    if savefig:
        fig_fn = f'{outdir}/{basin}_veg_height_type_tau_maps.png'
        fig.savefig(fig_fn, dpi=300, bbox_inches='tight')

    veg_list = [topo['veg_height'], topo['veg_type'], topo['veg_tau']]
    fig, axes = plt.subplots(1, 3, figsize=figsize, sharey=True)
    for jdx, ax in enumerate(axes.flatten()):
        veg_list[jdx].plot.hist(ax=ax, ec='k')
        ax.set_title(titles[jdx])
    plt.suptitle(f'{basin.capitalize()} Basin LANDFIRE Veg Height, Type, Tau', y=1.02)
    plt.tight_layout()

    # Save fig
    if savefig:
        fig_fn = f'{outdir}/{basin}_veg_height_type_tau_hist.png'
        fig.savefig(fig_fn, dpi=300, bbox_inches='tight')

def parse_arguments():
    """Parse command line arguments.

    Returns:
        argparse.Namespace: Parsed command line arguments.
    """
    parser = argparse.ArgumentParser(description='Basin-wide snow disappearance calculation')
    parser.add_argument('basin', type=str, help='Basin name')
    parser.add_argument('-out', '--outdir', default='.', help='Directory to save output files')
    parser.add_argument('-v', '--verbose', default=True, help='Print filenames')
    parser.add_argument('-o', '--overwrite', default=False, help='Overwrite existing files')
    return parser.parse_args()

def __main__():
    args = parse_arguments()
    basin = args.basin
    outdir = args.outdir
    verbose = args.verbose
    overwrite = args.overwrite

    # Locate basin setup dir
    script_dir = '/uufs/chpc.utah.edu/common/home/skiles-group3/jmhu/isnobal_scripts'
    basin_setupdir = h.fn_list(script_dir, f'*{basin}*setup')[0]

    # Locate topo.nc file within setupdir
    topo_file = h.fn_list(basin_setupdir, '*/*topo.nc')[0]
    if verbose:
        print(topo_file)

    # Extract vegetation height, vegetation type, transmissivity from the topo.nc
    topo = xr.open_dataset(topo_file)

    outname = f'{outdir}/{basin}_veg_height_type_tau_hist.png'
    if os.path.exists(outname) and not overwrite:
        print(f'Output file already exists: {outname}')
        sys.exit(1)
    else:
        print(f'Plotting now...{outname}')
        plot(basin, topo, outdir, savefig=True)

if __name__ == "__main__":
    __main__()