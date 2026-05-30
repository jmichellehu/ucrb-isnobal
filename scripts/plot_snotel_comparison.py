#!/usr/bin/env python
'''Script to plot isnobal outputs of snow depth against snotel sites.'''

import sys
import argparse
from pathlib import PurePath

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sys.path.append('/uufs/chpc.utah.edu/common/home/u6058223/git_dirs/env/')
import helpers as h

sys.path.append('/uufs/chpc.utah.edu/common/home/u6058223/git_dirs/ucrb-isnobal/scripts/')
import processing as proc

def pull_snotel_sites(basin, script_dir, snotel_dir, WY, poly_fn=None, epsg=32613, buffer=200, ST_abbrev='CO', verbose=True):
    if poly_fn is None:
        try:
            # Basin polygon file
            poly_fn = h.fn_list(script_dir, f'*{basin}*setup/polys/*shp')[0]
        except IndexError:
            try:
                ancillary_poly_dir = '/uufs/chpc.utah.edu/common/home/skiles-group3/jmhu/ancillary/polys'
                poly_fn = h.fn_list(ancillary_poly_dir, f'*{basin}*{epsg}*shp')[0]
            except IndexError as e:
                raise FileNotFoundError('No polygon file found for basin %s in %s or %s, exiting.' % (basin, script_dir, ancillary_poly_dir)) from e

    # SNOTEL all sites geojson fn
    allsites_fn = h.fn_list(snotel_dir, f'snotel_sites_{epsg}.json')[0]

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
    _, snotel_dfs = proc.get_snotel(sitenums, sitenames, ST_arr, WY=int(WY))
    return snotel_dfs, sitenames

def get_isnobal_depth(wydir, basin, WY, thisvar='thickness'):
     # Read in iSnobal output
    ds_dict = dict()
    # labels = ['iSnobal-HRRR', 'HRRR-MODIS'] # to follow csv output extract nomenclature
    labels = ['unified']
    for kdx, label in enumerate(labels):
        model_ts_fn = h.fn_list(wydir, f'{basin}_{label}*{thisvar}_snotelmetloom_wy{WY}.csv')[0]
        print(model_ts_fn)
        df = pd.read_csv(model_ts_fn, index_col=0)
        df.index.name = 'Date'
        # Set as DatetimeIndex
        df.index = pd.to_datetime(df.index)
        ds_dict[f'{label}_{thisvar}'] = df
    return ds_dict

def plot_snotel_comparison(sitenames, snotel_dfs, ds_dict, basin, WY, outdir, outname=None,
                           ylims=(0, 2), linestyle='-', linewidth=0.5, gridon=True,
                           marker='.', snowvar_col='SNOWDEPTH_m'):
    fig, axes = plt.subplots(len(sitenames), 1, figsize=(12, 2 * len(sitenames)), sharex=True, sharey=True)
    # Plot by site
    for jdx, sitename in enumerate(sitenames):
        ax = axes[jdx]
        # Extract dfs
        snotel_df = snotel_dfs[sitename]
        unified_df = ds_dict['unified_thickness'][sitename]
        # baseline_df = ds_dict['iSnobal-HRRR_thickness'][sitename]
        # hrrrmodis_df = ds_dict['HRRR-MODIS_thickness'][sitename]
        # hrrrmodis_df = ds_dict['HRRR-SPIReS-prtest_thickness'][sitename]
        (snotel_df[snowvar_col]).plot(ax=ax,
                                    label=f'{sitename} Snow Depth [m]',
                                    linestyle=linestyle,
                                    linewidth=linewidth,
                                    color='gray',
                                    marker=marker,
                                    markersize=4,
                                    )
        unified_df.plot(ax=ax,
                        label='iSnobal Snow Depth [m]',
                        linestyle=linestyle,
                        linewidth=1.2,
                        )
        # baseline_df.plot(ax=ax,
        #                 label='Baseline Snow Depth [m]',
        #                 linestyle=linestyle,
        #                 linewidth=1.2,
        #                 )
        # hrrrmodis_df.plot(ax=ax,
        #                 label='HRRR-SPIReS Snow Depth [m]',
        #                 linestyle=linestyle,
        #                 linewidth=1.2,
        #                 )
        ax.set_ylim(ylims)
        # Ensure all months are plotted
        ax.set_xlim(snotel_df.index[0], snotel_df.index[-1])
        # Put legend outside of plot
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        if gridon:
            ax.grid(color='lightgrey', linewidth=0.5)
    plt.suptitle(f'{basin} WY {WY} SNOTEL Snow Depth')
    plt.tight_layout()
    if outname is None:
        outname = f'{outdir}/{basin}_WY{WY}_snowdepth_comparison.png'
    plt.savefig(outname, dpi=300, bbox_inches='tight')

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
        parser.add_argument('-e', '--epsg', type=str, help='EPSG of AOI', default=32613)
        parser.add_argument('-p', '--palette', type=str, help='Seaborn color palette', default='icefire')
        parser.add_argument('-o', '--outdir', type=str, help='Output directory',
                            default='/uufs/chpc.utah.edu/common/home/skiles-group3/jmhu/figures/temporal/')
        parser.add_argument('-v', '--verbose', help='Print filenames', default=True)
        return parser.parse_args()

def __main__():
    args = parse_arguments()
    basin = args.basin
    poly_fn = args.shapefile
    WY = args.wy
    state_abbrev = args.state
    epsg = args.epsg
    palette = args.palette
    outdir = args.outdir
    verbose = args.verbose
    sns.set_palette(palette)

    # workdir = '/uufs/chpc.utah.edu/common/home/skiles-group3/model_runs/'
    # thp runs
    workdir = '/uufs/chpc.utah.edu/common/home/skiles-group2/model_runs_jmh/thp/'
    snotel_dir = '/uufs/chpc.utah.edu/common/home/skiles-group3/ancillary_sdswe_products/SNOTEL'
    script_dir = '/uufs/chpc.utah.edu/common/home/skiles-group3/jmhu/isnobal_scripts'
    # basindirs = h.fn_list(workdir, f'{basin}*isnobal/wy{WY}/*/')
    # thp runs
    basindirs = h.fn_list(workdir, f'{basin}*/wy{WY}/*/')
    # # Remove the prtest directories
    # basindirs = [d for d in basindirs if 'prtest' not in d]
    snotel_dfs, sitenames = pull_snotel_sites(basin, script_dir, snotel_dir, WY, ST_abbrev=state_abbrev, epsg=epsg,
                                               poly_fn=poly_fn, verbose=verbose)
    wydir = PurePath(basindirs[0]).parents[0].as_posix()
    ds_dict = get_isnobal_depth(wydir, basin, WY)
    plot_snotel_comparison(sitenames, snotel_dfs, ds_dict, basin, WY, outdir)

if __name__ == '__main__':
    __main__()
