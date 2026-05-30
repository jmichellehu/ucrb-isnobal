#!/usr/bin/env python

import sys
import os
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

def trim_datasets(ds1, ds2, verbose=True):
        # Trim the datasets to the same time period
        if verbose:
            print(f'Dataset start lengths: {len(ds1)}, {len(ds2)}')
        # if HRRR-MODIS ends earlier than Baseline, trim Baseline time period
        if ds1.time.values[-1] > ds2.time.values[-1]:
            ds1 = ds1.sel(time=slice(ds1.time.values[0], ds2.time.values[-1]))
        # if HRRR-MODIS ends later than Baseline, trim HRRR-MODIS time period
        elif ds2.time.values[-1] > ds1.time.values[-1]:
            ds2 = ds2.sel(time=slice(ds2.time.values[0], ds1.time.values[-1]))
        # if HRRR-MODIS starts later than Baseline, trim Baseline time period
        if ds1.time.values[0] < ds2.time.values[0]:
            ds1 = ds1.sel(time=slice(ds2.time.values[0], ds1.time.values[-1]))
        # if HRRR-MODIS starts earlier than Baseline, trim HRRR-MODIS time period
        elif ds2.time.values[0] < ds1.time.values[0]:
            ds2 = ds2.sel(time=slice(ds2.time.values[0], ds1.time.values[-1]))
        return ds1, ds2

def extract_em_data(gdf_metloom, em_ds_list, basin, WY,
                    resampling='nearest', labels=['Baseline', 'Unified'],
                    outname=None):
    '''
    '''
    big_em_dict = dict()
    for kdx, em_ds in enumerate(em_ds_list):
        label = labels[kdx]
        outname = f'{basin}_{label}_em_{WY}.nc'
        print(outname)
        if not os.path.exists(outname):
            # Extract all data values at snotel coordinates
            em_data = em_ds.sel(x=list(gdf_metloom.geometry.x.values),
                                y=list(gdf_metloom.geometry.y.values),
                                method=resampling)
            # Save extract to file
            em_data.to_netcdf(outname)
        else:
            # Load it
            em_data = xr.open_dataset(outname)
        big_em_dict[label] = em_data.load()
    return big_em_dict

def plot_em_by_site(sitenames, big_em_dict, outdir, basin, WY, overwrite=True):
     # fig, axes = plt.subplots(len(sitenames), 1, figsize=(12,1.5 * len(sitenames)), sharex=True, sharey=True)
    data_vars = big_em_dict['Baseline'].data_vars
    # Develop fixed names for the sites when writing to file
    fixednames = [sitename.replace(' ', '_').replace('(', '').replace(')', '').replace('#', '') for sitename in sitenames]
    print(len(sitenames))
    for jdx, sitename in enumerate(sitenames):
        print(jdx)
        fixedname = fixednames[jdx]
        outname = f'{outdir}/{basin}_snotel_{fixedname}_em_{WY}.png'
        print(outname)
        if os.path.exists(outname) and not overwrite:
            pass
        else:
            # Plot the energy balance terms for each site
            _, axes = plt.subplots(len(data_vars), 1, figsize=(10,1.2 * len(data_vars)), sharex=True, sharey=True)
            for kdx, data_var in enumerate(data_vars):
                print(data_var, kdx)
                ax = axes[kdx]
                # Plot the var for the site - TODO fix so it takes in a labels argument and has a default color and marker scheme based on the label
                big_em_dict['Baseline'][data_var][:, jdx, jdx].plot(ax=ax, label='Baseline', color='r', marker='.', markersize=2, linewidth=0)
                big_em_dict['Unified'][data_var][:, jdx, jdx].plot(ax=ax, label='Unified', color='royalblue', linewidth=1, linestyle='-')
                # big_em_dict['HRRR-MODIS-preciptest'][data_var][:, jdx, jdx].plot(ax=ax, label='HRRR-MODIS', color='mediumpurple', marker='|', markersize=3, linewidth=0)
                ax.annotate(data_var, xy=(0.01, 1.05), xycoords='axes fraction')
                ax.set_ylim(-50, 100)
                ax.set_title('')
                ax.set_xlabel('')
                ax.set_ylabel('')
                ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
                # Add more ticks
                ax.minorticks_on()
                ax.grid(color='lightgrey', linewidth=0.3, which='both')
            plt.suptitle(f'{basin} WY {WY} SNOTEL site {sitename}')
            plt.tight_layout()
            plt.savefig(outname, dpi=300, bbox_inches='tight')
            plt.close()

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
        parser.add_argument('-p', '--palette', type=str, help='Seaborn color palette', default='icefire')
        parser.add_argument('-o', '--outdir', type=str, help='Output directory',
                            default='/uufs/chpc.utah.edu/common/home/skiles-group3/jmhu/figures/unified')
        parser.add_argument('-v', '--verbose', help='Print filenames', default=True)
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
    sns.set_palette(palette)

    # workdir = '/uufs/chpc.utah.edu/common/home/skiles-group3/model_runs/'
    workdir = '/uufs/chpc.utah.edu/common/home/skiles-group2/model_runs_jmh/'
    snotel_dir = '/uufs/chpc.utah.edu/common/home/skiles-group3/ancillary_sdswe_products/SNOTEL'
    script_dir = '/uufs/chpc.utah.edu/common/home/skiles-group3/jmhu/isnobal_scripts'
    basindirs = h.fn_list(workdir, f'{basin}*isnobal/wy{WY}/*/')
    gdf_metloom, sitenames = prep_snotel_sites(basin, script_dir, snotel_dir, WY, ST_abbrev=state_abbrev, epsg=epsg,
                                               poly_fn=poly_fn, verbose=verbose)
    # Energy balance
    if verbose:
        print('Extracting energy balance at snotel sites')
    em_ds_list = [xr.open_mfdataset(h.fn_list(basindir, '*/em.nc'), drop_variables=['projection', 'snowmelt', 'SWI', 'cold_content']) for basindir in basindirs]

    # ensure datasets are the same length
    if verbose:
        print('Trimming datasets')
    em_ds_list[0], em_ds_list[1] = trim_datasets(em_ds_list[0], em_ds_list[1])

    # extract at snotel sites
    if verbose:
        print('Extracting energy balance data and saving to file')
    big_em_dict = extract_em_data(gdf_metloom, em_ds_list, basin, WY)

    # Plot them up
    if verbose:
        print('Plotting energy balance terms by snotel site')
    plot_em_by_site(sitenames, big_em_dict, outdir, basin, WY)

if __name__ == '__main__':
    __main__()
