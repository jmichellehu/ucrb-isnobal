#!/usr/bin/env python

import sys
import os
import re
import pandas as pd
import argparse

import xarray as xr
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append('/uufs/chpc.utah.edu/common/home/u6058223/git_dirs/env/')
import helpers as h

sys.path.append('/uufs/chpc.utah.edu/common/home/u6058223/git_dirs/ucrb-isnobal/scripts/')
import processing as proc


def _concat_by_simtype(file_list, basin, label, resample_daily=False):
    """Group files by simulation type and concatenate all WYs along time.

    Expected filename format:
    <basin>_<simulation_type>_<label>_<water_year>.nc
    """
    pattern = re.compile(rf"{re.escape(basin)}_(?P<sim>.+)_{re.escape(label)}_(?P<wy>\d{{4}})\.nc$")
    grouped = {}
    for fn in sorted(file_list):
        match = pattern.search(os.path.basename(fn))
        if not match:
            continue
        sim_type = match.group('sim')
        wy = int(match.group('wy'))
        grouped.setdefault(sim_type, []).append((wy, fn))

    concatenated = []
    sim_types = []
    for sim_type in sorted(grouped):
        wy_and_files = sorted(grouped[sim_type], key=lambda x: x[0])
        datasets = []
        for _, fn in wy_and_files:
            ds = xr.open_dataset(fn)
            if resample_daily:
                ds = ds.resample(time='1D').mean()
            datasets.append(ds)
        merged = xr.concat(datasets, dim='time').sortby('time')
        merged = merged.isel(time=~merged.get_index('time').duplicated())
        concatenated.append(merged)
        sim_types.append(sim_type)

    return concatenated, sim_types


def _concat_files_by_wy(file_list, resample_daily=False):
    """Concatenate datasets across multiple WYs, sorted by time.

    Used for file types that don't embed simulation type in their names.
    """
    if not file_list:
        return []
    datasets = []
    for fn in sorted(file_list):
        ds = xr.open_dataset(fn)
        if resample_daily:
            ds = ds.resample(time='1D').mean()
        datasets.append(ds)
    if not datasets:
        return []
    merged = xr.concat(datasets, dim='time').sortby('time')
    merged = merged.isel(time=~merged.get_index('time').duplicated())
    return [merged]


def _ensure_ordered_sims(datasets, sim_types, preferred_order=None):
    """Reorder datasets to match preferred simulation type order.

    Args:
        datasets: List of xarray datasets
        sim_types: List of simulation type names
        preferred_order: List of simulation type names in desired order (default: Baseline, HRRR-SPIReS)

    Returns:
        Tuple of (reordered_datasets, reordered_sim_types)
    """
    if preferred_order is None:
        preferred_order = ['Baseline', 'HRRR-SPIReS']

    if len(datasets) <= 1:
        return datasets, sim_types

    ordered_ds = []
    ordered_types = []
    for key in preferred_order:
        if key in sim_types:
            idx = sim_types.index(key)
            ordered_ds.append(datasets[idx])
            ordered_types.append(sim_types[idx])
    for idx, sim_type in enumerate(sim_types):
        if sim_type not in ordered_types:
            ordered_ds.append(datasets[idx])
            ordered_types.append(sim_type)
    return ordered_ds, ordered_types


def _wy_label_from_data(ds_list, fallback_wy):
    """Create a WY label (single year or range) from time coordinates."""
    times = []
    for ds in ds_list:
        if 'time' in ds.coords and ds.sizes.get('time', 0) > 0:
            times.extend(pd.to_datetime(ds['time'].values).tolist())
    if not times:
        return str(fallback_wy)

    dt_index = pd.DatetimeIndex(times)
    wy_values = sorted(set([t.year + 1 if t.month >= 10 else t.year for t in dt_index]))
    if not wy_values:
        return str(fallback_wy)
    if len(wy_values) == 1:
        return str(wy_values[0])
    return f"{wy_values[0]}-{wy_values[-1]}"

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

def plot_em_by_site(sitenames, baseline_netsolar_list, hs_netsolar_list, em_list, cmap, wy_fallback, basin, outdir, verbose, overwrite):
    """Plot energy balance terms for each site across all water years.

    Args:
        sitenames: List of SNOTEL site names
        baseline_netsolar_list: List with one dataset containing concatenated baseline net solar across all WYs
        hs_netsolar_list: List with one dataset containing concatenated HRRR-SPIReS net solar across all WYs
        em_list: List of datasets [baseline_em, hrrr_em] concatenated across all WYs
        cmap: Color palette (list of colors)
        wy_fallback: Water year for fallback label (actual years plotted are auto-detected from data)
        basin: Basin name
        outdir: Output directory for plots
        verbose: Print output filenames
        overwrite: Overwrite existing plots
    """
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
            # Use the full concatenated time span when plotting multiple WYs.
            xmin = pd.Timestamp(em_list[0]['time'].values.min())
            xmax = pd.Timestamp(em_list[0]['time'].values.max())
            # Format limits and title
            ax.set_xlim(xmin, xmax)
            ax.set_title('')
            ax.set_ylim(-100, 200)
            # Add zero line
            ax.axhline(0, color='black', linewidth=0.5)
            # Add grid
            ax.grid(color='grey', linestyle='--', linewidth=1, which='both', alpha=0.3)
        wy_label = _wy_label_from_data(em_list, wy_fallback)
        plt.suptitle(f'{basin.capitalize()} WY {wy_label} - {sitename}', fontsize=11, fontstyle='italic')
        plt.tight_layout()
        outname = f'{outdir}/{basin}_wy{wy_label}_{fixed_names[sdx]}_energy_balance_terms_daily.png'
        if verbose:
            print(outname)
        if os.path.exists(outname) and not overwrite:
            print(f'File exists: {outname}, skipping...')
        else:
            plt.savefig(outname, dpi=300)

def _extract_wy_from_files(file_list, pattern_str):
    """Extract water year(s) from filenames matching pattern.

    Returns the first (earliest) WY found, or None if no matches.
    """
    pattern = re.compile(pattern_str)
    wys = []
    for fn in sorted(file_list):
        match = pattern.search(os.path.basename(fn))
        if match:
            wys.append(int(match.group('wy')))
    return min(wys) if wys else None


def parse_arguments():
    """Parse command line arguments.

    Returns:
        argparse.Namespace: Parsed command line arguments.
    """
    parser = argparse.ArgumentParser(
        description='Plot energy balance terms for basin(s) across multiple water years. '
                    'Water year is auto-detected from available data files.'
    )
    parser.add_argument('basin', type=str, help='Basin name')
    parser.add_argument('-shp', '--shapefile', type=str, help='Shapefile of basin polygon', default=None)
    parser.add_argument('-st', '--state', type=str, help='State abbreviation', default='CO')
    parser.add_argument('-e', '--epsg', type=str, help='EPSG of AOI', default=None)
    parser.add_argument('-p', '--palette', type=str, help='Seaborn color palette', default='plasma')
    parser.add_argument('-o', '--outdir', type=str, help='Output directory',
                        default='/uufs/chpc.utah.edu/common/home/skiles-group3/jmhu/figures/energy_balance_terms_daily_por')
    parser.add_argument('-v', '--verbose', action='store_true', help='Print verbose output', default=False)
    parser.add_argument('-ow', '--overwrite', action='store_true', help='Overwrite existing files', default=False)
    return parser.parse_args()

def __main__():
    args = parse_arguments()
    basin = args.basin
    poly_fn = args.shapefile
    state_abbrev = args.state
    epsg = args.epsg
    palette = args.palette
    outdir = args.outdir
    verbose = args.verbose
    overwrite = args.overwrite

    # Data directories (consider moving to config later)
    workdir = '/uufs/chpc.utah.edu/common/home/skiles-group3/jmhu/data_extracts'
    snotel_dir = '/uufs/chpc.utah.edu/common/home/skiles-group3/ancillary_sdswe_products/SNOTEL'
    script_dir = '/uufs/chpc.utah.edu/common/home/skiles-group3/jmhu/isnobal_scripts'

    # Load and concatenate energy balance data across all water years
    if verbose:
        print('Locating snotel site energy balance extracts')

    # EM files: grouped by simulation type, concatenated by WY
    em_files = h.fn_list(workdir, f'*{basin}*em_*.nc')
    pattern_str = rf"{re.escape(basin)}_(?P<sim>.+)_em_(?P<wy>\d{{4}})\.nc$"
    wy_for_snotel = _extract_wy_from_files(em_files, pattern_str)
    if wy_for_snotel is None:
        raise ValueError(f'No EM files found for basin={basin}. Expected pattern: *{basin}*_em_*.nc')

    em_list, em_simtypes = _concat_by_simtype(em_files, basin=basin, label='em')
    em_list, em_simtypes = _ensure_ordered_sims(em_list, em_simtypes)

    # Fetch SNOTEL sites for the detected water year
    _, sitenames = prep_snotel_sites(basin, script_dir, snotel_dir, wy_for_snotel, ST_abbrev=state_abbrev, epsg=epsg,
                                     poly_fn=poly_fn, verbose=verbose)
    # get cmap from palette
    sns.set_palette(palette)
    cmap = sns.color_palette(n_colors=7, palette=palette)

    # Net solar files: concatenated by WY
    hs_netsolar_files = h.fn_list(workdir, f'net_HRRR_SPIReS*{basin}*snotel.nc')
    hs_netsolar_list = _concat_files_by_wy(hs_netsolar_files, resample_daily=True)

    baseline_netsolar_files = h.fn_list(workdir, f'*{basin}*smrf_energy_balance*.nc')
    baseline_netsolar_list = _concat_files_by_wy(baseline_netsolar_files, resample_daily=True)

    # Validation
    if len(em_list) < 2:
        raise ValueError(f'Expected at least 2 EM simulation types for {basin}, found {len(em_list)}: {em_simtypes}')
    if len(hs_netsolar_list) < 1 or len(baseline_netsolar_list) < 1:
        raise ValueError(f'Missing net solar datasets for basin={basin}')

    if verbose:
        print(f'  WY for SNOTEL sites: {wy_for_snotel}')
        print(f'  EM list: {len(em_list)} types: {em_simtypes}')
        print(f'  HRRR-SPIReS net solar: {len(hs_netsolar_list)} datasets')
        print(f'  Baseline net solar: {len(baseline_netsolar_list)} datasets')

    # Plot them up
    if verbose:
        print('Plotting energy balance terms by snotel site')
    plot_em_by_site(sitenames, baseline_netsolar_list, hs_netsolar_list,
                    em_list, cmap, wy_for_snotel, basin, outdir, verbose, overwrite)

if __name__ == '__main__':
    __main__()