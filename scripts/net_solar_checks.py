#!/usr/bin/env python

import sys
import argparse

import xarray as xr
import matplotlib.pyplot as plt
import pandas as pd
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

def extract_em_data(gdf_metloom, sitenames, em_ds_list, em_var='net_rad',
                    resampling='nearest', labels=['Baseline', 'HRRR-MODIS']):
    '''em_var: net_rad, net_solar, albedo
    fix this func --> do not turn into df, extract snotel by sampling all ds vars at once
    might be best to save these extracts to file
    then put into a dict of model runs - baseline, hrrr-modis
    '''
    em_dict = dict()
    for kdx, em_ds in enumerate(em_ds_list):
        label = labels[kdx]
        if em_var =='albedo':
            data_varnames = [f for f in em_ds_list[kdx].data_vars if em_var in f]
            for data_var in data_varnames:
                # Extract values at snotel coordinates
                em_data = em_ds[data_var].sel(x=list(gdf_metloom.geometry.x.values),
                                                y=list(gdf_metloom.geometry.y.values),
                                                method='nearest')
                # Convert to a dict
                for jdx, sitename in enumerate(sitenames):
                    ds = em_data[:, jdx, jdx]
                    em_dict[f'{sitename}_{label}_{data_var}'] = ds.values
        else:
            # Extract values at snotel coordinates
            em_data = em_ds[em_var].sel(x=list(gdf_metloom.geometry.x.values),
                                            y=list(gdf_metloom.geometry.y.values),
                                            method=resampling)
            # Convert to a dict
            for jdx, sitename in enumerate(sitenames):
                ds = em_data[:, jdx, jdx]
                em_dict[f'{sitename}_{label}'] = ds.values

            # Pull the DSWRF variable from the HRRR-MODIS dataset if em_var is net_solar
            if em_var == 'net_solar':
                em_ds = em_ds_list[1]
                em_data = em_ds['DSWRF'].sel(x=list(gdf_metloom.geometry.x.values),
                                            y=list(gdf_metloom.geometry.y.values),
                                            method=resampling)
                for jdx, sitename in enumerate(sitenames):
                    ds = em_data[:, jdx, jdx]
                    em_dict[f'{sitename}_DSWRF'] = ds.values

    # Turn it into a dataframe
    em_datadf = pd.DataFrame(em_dict, index=ds['time'].values)
    return em_datadf

def plot_by_site(sitenames, em_datadf, em_var, outdir, basin, WY, outname=None):
    # Plot specified variable site by site
    fig, axes = plt.subplots(len(sitenames), 1, figsize=(12,1.5 * len(sitenames)), sharex=True, sharey=True)
    for jdx, sitename in enumerate(sitenames):
        ax = axes[jdx]
        site_df = em_datadf[[col for col in em_datadf.columns if sitename in col]]
        # Plot baseline
        site_df[[col for col in site_df.columns if 'Baseline' in col]].plot(ax=ax, linewidth=1.5)
        # Plot HRRR-MODIS
        site_df[[col for col in site_df.columns if 'HRRR' in col]].plot(ax=ax, linewidth=0.75, linestyle='--')
        # Stop at July
        # ax.set_xlim(pd.Timestamp('2020-10-01'), pd.Timestamp('2021-06-30'))
        ax.set_ylim(-50, 150)
        # Put legend outside of plot
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.suptitle(f'{basin} WY {WY} SNOTEL site [{em_var}]')
    plt.tight_layout()
    if outname is None:
        outname = f'{outdir}/{basin}_snotel_{em_var}_{WY}.png'
    plt.savefig(outname, dpi=300, bbox_inches='tight')

def plot_net_solar(datadf, sitenames, outdir, basin, WY, ylims=(-50, 200), em_var='net_solar', outname=None):
    # Plot site by site
    fig, axes = plt.subplots(len(sitenames), 1, figsize=(12,1.5 * len(sitenames)), sharex=True, sharey=True)
    for jdx, sitename in enumerate(sitenames):
        ax = axes[jdx]
        site_df = datadf[[col for col in datadf.columns if sitename in col]]
        # Plot baseline
        site_df[[col for col in site_df.columns if 'Baseline' in col]].plot(ax=ax, linewidth=1.5, color='gold')
        # Plot HRRR-MODIS
        site_df[[col for col in site_df.columns if 'HRRR' in col]].plot(ax=ax, linewidth=0.75, linestyle='--', color='firebrick')
        # Stop at July
        # ax.set_xlim(pd.Timestamp('2020-10-01'), pd.Timestamp('2021-06-30'))
        ax.set_ylim(ylims)
        # Put legend outside of plot
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.suptitle(f'{basin} WY {WY} SNOTEL site [{em_var}]')
    plt.tight_layout()
    if outname is None:
        outname = f'{outdir}/{basin}_snotel_{em_var}_{WY}.png'
    plt.savefig(outname, dpi=300, bbox_inches='tight')

def plot_net_solar_diff(datadf, sitenames, outdir, basin, WY, ylims=(-100, 100), em_var='net_solar', outname=None):
    # Plot site by site
    fig, axes = plt.subplots(len(sitenames), 1, figsize=(12,1.5 * len(sitenames)), sharex=True, sharey=True)
    for jdx, sitename in enumerate(sitenames):
        ax = axes[jdx]
        site_df = datadf[[col for col in datadf.columns if sitename in col]]
        # Calculate the difference between HRRR-MODIS and Baseline
        diff = site_df[[col for col in site_df.columns if 'HRRR' in col]] - site_df[[col for col in site_df.columns if 'Baseline' in col]]
        diff.plot(ax=ax, linewidth=0.75, linestyle='--')
        ax.set_ylim(ylims)
        # Put legend outside of plot
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.suptitle(f'{basin} WY {WY} SNOTEL site [{em_var} difference]')
    plt.tight_layout()
    if outname is None:
        outname = f'{outdir}/{basin}_snotel_{em_var}_diff_{WY}.png'
    plt.savefig(outname, dpi=300, bbox_inches='tight')

def plot_net_solar_dswrf(datadf, sitenames, outdir, basin, WY, ylims=(-50, 550), em_var='net_solar', outname=None):
    # Plot site by site
    fig, axes = plt.subplots(len(sitenames), 1, figsize=(12,1.5 * len(sitenames)), sharex=True, sharey=True)
    for jdx, sitename in enumerate(sitenames):
        ax = axes[jdx]
        site_df = datadf[[col for col in datadf.columns if sitename in col]]
        # Plot baseline
        site_df[[col for col in site_df.columns if 'Baseline' in col]].plot(ax=ax, linewidth=1.5, color='gold')
        # Plot HRRR-MODIS
        site_df[[col for col in site_df.columns if 'HRRR' in col]].plot(ax=ax, linewidth=0.75, linestyle='--', color='firebrick')
        # Plot DSWRF
        site_df[[col for col in site_df.columns if 'DSWRF' in col]].plot(ax=ax, linewidth=0.5, color='gray')
        # Stop at July
        # ax.set_xlim(pd.Timestamp('2020-10-01'), pd.Timestamp('2021-06-30'))
        ax.set_ylim(ylims)
        # Put legend outside of plot
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.suptitle(f'{basin} WY {WY} SNOTEL site [{em_var}]')
    plt.tight_layout()
    if outname is None:
        outname = f'{outdir}/{basin}_snotel_{em_var}_dswrf_{WY}.png'
    plt.savefig(outname, dpi=300, bbox_inches='tight')

def plot_albedo(datadf, sitenames, outdir, basin, WY, em_var='albedo', outname=None):
    # Plot site by site
    fig, axes = plt.subplots(len(sitenames), 1, figsize=(12,1.5 * len(sitenames)), sharex=True, sharey=True)
    for jdx, sitename in enumerate(sitenames):
        ax = axes[jdx]
        site_df = datadf[[col for col in datadf.columns if sitename in col]]
        # Plot baseline albedo vis
        site_df[[col for col in site_df.columns if '_vis' in col]].plot(ax=ax, linewidth=0.75)
        # Plot baseline albedo IR
        site_df[[col for col in site_df.columns if '_ir' in col]].plot(ax=ax, linewidth=0.75)
        # Plot HRRR-MODIS and adjust for albedo units
        (site_df[[col for col in site_df.columns if 'HRRR' in col]] / 10000).plot(ax=ax, linewidth=0.75, linestyle='--')
        # Put legend outside of plot
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.suptitle(f'{basin} WY {WY} SNOTEL site [{em_var}]')
    plt.tight_layout()
    if outname is None:
        outname = f'{outdir}/{basin}_{em_var}_{WY}.png'
    plt.savefig(outname, dpi=300, bbox_inches='tight')

def plot_albedo_solar(datadf, net_solar_datadf, sitenames, outdir, basin, WY, ylog=False, em_var='albedo', outname=None):
    ### Plot albedo on top of DSWRF and net solar for HRRR-MODIS
    # Plot site by site
    fig, axes = plt.subplots(len(sitenames), 1, figsize=(12,1.5 * len(sitenames)), sharex=True, sharey=True)
    for jdx, sitename in enumerate(sitenames):
        ax = axes[jdx]
        # Get the solar bits
        site_df = net_solar_datadf[[col for col in net_solar_datadf.columns if sitename in col]]
        # Plot baseline
        site_df[[col for col in site_df.columns if 'Baseline' in col]].plot(ax=ax, linewidth=1.5, color='gold')
        # Plot HRRR-MODIS
        site_df[[col for col in site_df.columns if 'HRRR' in col]].plot(ax=ax, linewidth=0.75, linestyle='--', color='firebrick')
        ax.legend(loc='center left', bbox_to_anchor=(1.1, 0.6))
        if ylog:
            # Turn the yaxis into a log scale
            ax.set_yscale('log')

        # Now plot the albedo
        ax2 = ax.twinx()
        site_df = datadf[[col for col in datadf.columns if sitename in col]]
        # Plot HRRR-MODIS and adjust for albedo units
        (site_df[[col for col in site_df.columns if 'HRRR' in col]] / 10000).plot(ax=ax2, linewidth=0.75, linestyle='--', color = 'k')
        # Put legend outside of plot
        ax2.legend(loc='center left', bbox_to_anchor=(1.1, 0.275))
    plt.suptitle(f'{basin} WY {WY} SNOTEL site [{em_var}]')
    plt.tight_layout()
    if outname is None:
        outname = f'{outdir}/{basin}_{em_var}_solar_{WY}.png'
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
        parser.add_argument('-e', '--epsg', type=str, help='EPSG of AOI', default=None)
        parser.add_argument('-p', '--palette', type=str, help='Seaborn color palette', default='icefire')
        parser.add_argument('-o', '--outdir', type=str, help='Output directory',
                            default='/uufs/chpc.utah.edu/common/home/skiles-group3/jmhu/figures/hrrrmodis_checks/animas_focus')
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

    workdir = '/uufs/chpc.utah.edu/common/home/skiles-group3/model_runs/'
    snotel_dir = '/uufs/chpc.utah.edu/common/home/skiles-group3/ancillary_sdswe_products/SNOTEL'
    script_dir = '/uufs/chpc.utah.edu/common/home/skiles-group3/jmhu/isnobal_scripts'
    basindirs = h.fn_list(workdir, f'{basin}*isnobal/wy{WY}/*/')
    gdf_metloom, sitenames = prep_snotel_sites(basin, script_dir, snotel_dir, WY, ST_abbrev=state_abbrev, epsg=epsg,
                                               poly_fn=poly_fn, verbose=verbose)

    # Net radiation
    if verbose:
        print('Extracting net radiation')
    em_var = 'net_rad'
    em_ds_list = [xr.open_mfdataset(h.fn_list(basindir, '*/em.nc'), drop_variables=['projection', 'snowmelt', 'SWI', 'cold_content']) for basindir in basindirs]
    # ensure datasets are the same length
    em_ds_list[0], em_ds_list[1] = trim_datasets(em_ds_list[0], em_ds_list[1])
    em_datadf = extract_em_data(gdf_metloom, sitenames, em_ds_list, em_var)
    plot_by_site(sitenames, em_datadf, em_var, outdir, basin, WY)
    del em_var, em_ds_list, em_datadf

    # Net solar
    if verbose:
        print('Extracting net solar')
    em_var = 'net_solar'
    netsolar_ds_list = [xr.open_mfdataset(h.fn_list(basindirs[0], '*/*energy_balance*.nc'))] + \
        [xr.open_mfdataset(h.fn_list(basindirs[1], '*/net_solar.nc'), drop_variables=['illumination_angle'])]
    # Resample hourly to daily timesteps
    for kdx, ds in enumerate(netsolar_ds_list):
        netsolar_ds_list[kdx] = ds.resample(time='1D').mean().load()
    # ensure datasets are the same length
    netsolar_ds_list[0], netsolar_ds_list[1] = trim_datasets(netsolar_ds_list[0], netsolar_ds_list[1])
    net_solar_datadf = extract_em_data(gdf_metloom, sitenames, em_ds_list=netsolar_ds_list, em_var=em_var)
    plot_net_solar(net_solar_datadf, sitenames, outdir, basin, WY)
    plot_net_solar_diff(net_solar_datadf, sitenames, outdir, basin, WY)
    plot_net_solar_dswrf(net_solar_datadf, sitenames, outdir, basin, WY)
    del em_var

    # Albedo
    if verbose:
        print('Extracting albedo')
    em_var = 'albedo'
    albedo_datadf = extract_em_data(gdf_metloom, sitenames, em_ds_list=netsolar_ds_list, em_var=em_var)
    plot_albedo(albedo_datadf, sitenames, outdir, basin, WY)
    plot_albedo_solar(albedo_datadf, net_solar_datadf, sitenames, outdir, basin, WY)
    del em_var, netsolar_ds_list, albedo_datadf, net_solar_datadf

    # DSWRF
    # TODO: Check that DSWRF in net_solar.nc is identical to DSWRF  in hrrr.yyyymmdd.dswrf.nc

if __name__ == '__main__':
    __main__()