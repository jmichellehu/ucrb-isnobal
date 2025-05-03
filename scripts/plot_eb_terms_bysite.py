#!/usr/bin/env python

import sys
import os
import pandas as pd
import argparse

import xarray as xr
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append('/uufs/chpc.utah.edu/common/home/u6058223/git_dirs/env/')
import helpers as h

sys.path.append('/uufs/chpc.utah.edu/common/home/u6058223/git_dirs/ucrb-isnobal/scripts/')
import processing as proc

def prep_snotel_sites(basin, script_dir, snotel_dir, WY, poly_fn=None, epsg=None, buffer=200, ST_abbrev='CO', verbose=True):
    if poly_fn is None:
        # Basin polygon file
        poly_fn = h.fn_list(script_dir, f'*{basin}*setup/polys/*shp')[0]

    # SNOTEL all sites geojson fn
    allsites_fn = h.fn_list(snotel_dir, 'snotel_sites_32613.json')[0]

    # Locate SNOTEL sites within basin
    if epsg is not None:
        found_sites = proc.locate_snotel_in_poly(poly_fn=poly_fn, site_locs_fn=allsites_fn, buffer=buffer, epsg=epsg)
    else:
        found_sites = proc.locate_snotel_in_poly(poly_fn=poly_fn, site_locs_fn=allsites_fn, buffer=buffer)

    # Get site names and site numbers
    sitenames = found_sites['site_name']
    sitenums = found_sites['site_num']
    if verbose:
        print(sitenames)

    ST_arr = [ST_abbrev] * len(sitenums)
    gdf_metloom, _ = proc.get_snotel(sitenums, sitenames, ST_arr, WY=int(WY))
    return gdf_metloom, sitenames

def plot_em_by_site(sitenames, baseline_netsolar_list, hs_netsolar_list, em_list, cmap, WY, basin, outdir, verbose, overwrite):
    from matplotlib import patheffects
    data_vars = ['sum_EB', 'net_rad', 'net_solar', 'sensible_heat', 'latent_heat', 'snow_soil', 'precip_advected']
    # Develop fixed names for the sites when writing to file
    fixed_names = [sitename.replace(' ', '_').replace('(', '').replace(')', '').replace('#', '') for sitename in sitenames]
    num_plots = len(data_vars)
    for sdx, sitename in enumerate(sitenames):
        _, axa = plt.subplots(num_plots, 1, figsize=(8, 1.2*num_plots), sharex=True, sharey=True)
        for jdx, f in enumerate(data_vars):
            ax = axa[jdx]
            if jdx == 2:
                baseline_netsolar_list[0][f][:, sdx, sdx].plot(ax=ax, label='Baseline', color=cmap[jdx], linewidth=3, alpha=0.4)
                hs_netsolar_list[0][f][:, sdx, sdx].plot(ax=ax, label='HRRR-SPIReS', color=cmap[jdx], linewidth=1)
            else:
                em_list[0][f][:, sdx, sdx].plot(ax=ax, label='Baseline', color=cmap[jdx], linewidth=3, alpha=0.4)
                em_list[1][f][:, sdx, sdx].plot(ax=ax, label='HRRR-SPIReS', color=cmap[jdx], linewidth=1)
            # Annotate f in upper lefthand corner inside plot and add white buffer
            ax.annotate(f, xy=(0.985, 0.8), xycoords='axes fraction', ha='right', c=cmap[jdx], fontsize=10,
                        path_effects=[patheffects.withStroke(linewidth=5, foreground="w")])
            ax.set_xlabel('')
            ax.set_ylabel('W m-2')
            # Trim everything to August
            xmin, xmax = pd.Timestamp(f'{WY-1}-10-01'), pd.Timestamp(f'{WY}-7-15')
            # Format limits and title
            ax.set_xlim(xmin, xmax)
            ax.set_title('')
            ax.set_ylim(-100, 200)
            # Add zero line
            ax.axhline(0, color='black', linewidth=0.5)
            # Add grid
            ax.grid(color='grey', linestyle='--', linewidth=1, which='both', alpha=0.3)
        plt.suptitle(f'{basin.capitalize()} WY {WY} - {sitename}', fontsize=11, fontstyle='italic')
        plt.tight_layout()
        outname = f'{outdir}/{basin}_wy{WY}_{fixed_names[sdx]}_energy_balance_terms_daily.png'
        if verbose:
            print(outname)
        if os.path.exists(outname) and not overwrite:
            print(f'File exists: {outname}, skipping...')
        else:
            plt.savefig(outname, dpi=300)

def parse_arguments():
        """Parse command line arguments.

        Returns:
        argparse.Namespace: Parsed command line arguments.
        """
        parser = argparse.ArgumentParser(description='Plot energy balance terms for a given basin and water year.')
        parser.add_argument('basin', type=str, help='Basin name')
        parser.add_argument('wy', type=int, help='Water year of interest')
        parser.add_argument('-shp', '--shapefile', type=str, help='Shapefile of basin polygon', default=None)
        parser.add_argument('-st', '--state', type=str, help='State abbreviation', default='CO')
        parser.add_argument('-e', '--epsg', type=str, help='EPSG of AOI', default=None)
        parser.add_argument('-p', '--palette', type=str, help='Seaborn color palette', default='plasma')
        parser.add_argument('-o', '--outdir', type=str, help='Output directory',
                            default='/uufs/chpc.utah.edu/common/home/skiles-group3/jmhu/figures/')
        parser.add_argument('-v', '--verbose', help='Print filenames', default=True)
        parser.add_argument('-ow', '--overwrite', help='Overwrite existing files', default=False)
        return parser.parse_args()

def __main__():
    args = parse_arguments()
    basin = args.basin
    WY = args.wy
    poly_fn = args.shapefile
    state_abbrev = args.state
    epsg = args.epsg
    palette = args.palette
    outdir = args.outdir
    verbose = args.verbose
    overwrite = args.overwrite

    workdir = '/uufs/chpc.utah.edu/common/home/skiles-group3/jmhu/data_extracts'
    snotel_dir = '/uufs/chpc.utah.edu/common/home/skiles-group3/ancillary_sdswe_products/SNOTEL'
    script_dir = '/uufs/chpc.utah.edu/common/home/skiles-group3/jmhu/isnobal_scripts'

    _, sitenames = prep_snotel_sites(basin, script_dir, snotel_dir, WY, ST_abbrev=state_abbrev, epsg=epsg,
                                               poly_fn=poly_fn, verbose=verbose)
    # get cmap from palette
    sns.set_palette(palette)
    cmap = sns.color_palette(n_colors=7, palette=palette)

    # Energy balance
    if verbose:
        print('Locating snotel site energy balance extracts')
    em_list = [xr.open_dataset(f) for f in h.fn_list(workdir, f'*{basin}*em_{WY}.nc')]
    hs_netsolar_list= [xr.open_dataset(f).resample(time='1D').mean() for f in h.fn_list(workdir, f'net_HRRR_SPIReS*{basin}_{WY}_snotel.nc')]
    baseline_netsolar_list = [xr.open_dataset(f).resample(time='1D').mean() for f in h.fn_list(workdir, f'*{basin}*smrf_energy_balance_{WY}.nc')] # make sure this is extracted separately for plotting
    if verbose:
        print(len(em_list), len(hs_netsolar_list), len(baseline_netsolar_list))

    # Plot them up
    if verbose:
        print('Plotting energy balance terms by snotel site')
    plot_em_by_site(sitenames, baseline_netsolar_list, hs_netsolar_list,
                    em_list, cmap, WY, basin, outdir, verbose, overwrite)

if __name__ == '__main__':
    __main__()