#!/usr/bin/env python
'''
Script to plot the zonal means comparisons of isnobal data with CBRFC SNOW17 for a given basin and water year.
'''
import sys

import numpy as np
import matplotlib.pyplot as plt
import argparse

import pandas as pd
import geopandas as gpd
import seaborn as sns
import copy

sys.path.append('/uufs/chpc.utah.edu/common/home/u6058223/git_dirs/env/')
import helpers as h

# Set seaborn palette
sns.set_palette('icefire')

def prepare_isnobal_data(basin, WY, var, data_dir):
    # Read in basin data
    zonal_basin_data_fns = h.fn_list(data_dir, f'isnobal_zonal/*{basin}*WY{WY}*{var}*gpkg')
    zonal_basin_data_list = [gpd.read_file(zonal_fn) for zonal_fn in zonal_basin_data_fns]
    # Convert to meters rather than equivalent to mm SWE for all columns that are not zone, zonal_classification or geometry
    for zonal_basin_data in zonal_basin_data_list:
        for col in zonal_basin_data.columns:
            if col not in ['zone', 'zonal_classification', 'geometry']:
                zonal_basin_data[col] = zonal_basin_data[col] / 1000.0  # Convert from mm to m
    return zonal_basin_data_list

def prepare_cbrfc_data(zones_abbrev, data_dir, WY, verbose=False):
    # Read in SNOW17 data csvs only if csv name contains one of the zone names
    basin_csv_fns = []
    for zone in zones_abbrev:
        zone_csv = h.fn_list(data_dir, f'{zone}*csv')[0]
        basin_csv_fns.append(zone_csv)
    df_list = [pd.read_csv(csv, index_col=0) for csv in basin_csv_fns]
    if verbose:
        _ = [print(df.shape) for df in df_list]

    # Process each df in the list
    wy_list = [process_cbrfc_df(df, WY) for df in df_list]

    # Now recompile the snow17 wy list dataframes into a single one by moving the opid and appending to the swe column
    snow17_wy = pd.DataFrame()

    # For each unique "subzone" in opid column, split into new df
    # These should remain in zone order as above since deriving
    # zone names from np.unique
    for df in wy_list:
        subzones = np.unique(df['opid'])
        for subzone in subzones:
            if verbose:
                print(subzone)
            sub_df = df[df['opid'] == subzone]
            # Append subzone name to the swe column
            sub_df = sub_df.rename(columns={'swe': f'{subzone}_swe'})
            # Drop opid column
            sub_df.drop(columns=['opid'], inplace=True)
            if verbose:
                print(sub_df.shape)
            # Convert date column to index
            sub_df.set_index('date', inplace=True)
            # Concatenate to the larger snow17_wy DataFrame
            snow17_wy = pd.concat([snow17_wy, sub_df], axis=1)
    # Convert from inches to meters
    snow17_wy = snow17_wy / 39.37
    return snow17_wy

def process_cbrfc_df(in_df, WY):
    '''
    Generate date column and filter to specified water year (WY)
    in_df: pandas DataFrame with columns 'cal_yr', 'mon', and 'zday'
    WY: Water year to filter the DataFrame to, e.g., 2022 for WY 2022
    '''
    df = copy.deepcopy(in_df)
    # Generate a date column from the cal_yr, mon and zday columns by using a lambda function to do row-wise conversion
    df['date'] = pd.to_datetime(df.apply(lambda row: f"{row['cal_yr']}-{row['mon']:02d}-{row['zday']:02d}", axis=1))

    # Drop the cal_yr, mon, and zday columns
    df.drop(columns=['cal_yr', 'mon', 'zday'], inplace=True)
    # Trim to the specified water year Oct 1 through September 30
    df = df[(df['date'] >= f'{WY - 1}-10-01') & (df['date'] <= f'{WY}-09-30')]
    return df

def process_isnobal_zone_df(zone_row, verbose=False):
    # convert to series, dropping geometry, zonal classification
    zone_row = zone_row.drop(columns=['geometry', 'zonal_classification'])
    # Move the prepending model run to the zone column
    model_run = np.unique([c.split('_')[0] for c in zone_row.columns if c not in ['zone']])[0]
    if verbose:
        print(model_run)
    # Rename the columns to remove the prepending model run
    zone_row.rename(columns={c: c.split('_')[1] for c in zone_row.columns if '_' in c}, inplace=True)
    # Rename zone
    zone_row.rename(columns={'zone': f'zone_{model_run}_run'}, inplace=True)
    # Transpose zone_row
    zone_row = zone_row.T
    # Pull the zone name from the first row
    colname = zone_row.iloc[0]
    # Drop the first row
    zone_row = zone_row.drop(index=colname.name)
    # Rename column to zone name
    zone_row.columns = [colname]
    # Set zone_baseline_run as the index and convert to datetime
    zone_row.index = pd.to_datetime(zone_row.index)
    return zone_row

def plot_time_series_comparison(zones, zonal_basin_data_list, snow17_wy, basin, WY, data_dir, var, nrows=4, ncols=2, savefig=False):
    # Now compare the data for each zone
    # fig, axa = plt.subplots(nrows, ncols, figsize=(ncols*5, nrows*3), sharex=True, sharey=True)
    # for ax, zone in zip(axa.flatten(), zones):
    for zone in zones:
        fig, ax = plt.subplots(1, 1, figsize=(5, 3), sharex=True, sharey=True)
        # Plot time series of: Baseline, HRRR-SPIReS
        for df, label in zip(zonal_basin_data_list, ['Baseline', 'HRRR-SPIReS']):
            # Get the row for the zone
            zone_row = df[df['zone'] == zone]
            # Process the zone_row to get it in the right format
            zone_row = process_isnobal_zone_df(zone_row)
            # Plot the zone_row
            ax.plot(zone_row, label=label)
        # Plot CBRFC snow-17 data
        snow17_wy[f'{zone}_swe'].plot(ax=ax, label='CBRFC SNOW-17', color='black', linestyle='--')
        # TODO: On the second subplot, plot the difference from CBRFC
        # ax.set_title(f'Zone: {zone}')
        ax.set_ylabel('SWE (m)')
        ax.annotate(zone, xy=(0.96, 0.84), xycoords='axes fraction', ha='right', fontsize=11, fontstyle='italic')
        # # if this is the first plot, add a legend
        # if ax == axa[0, 0]:
        #     ax.legend(loc='upper left', fontsize=10)
        ax.legend(loc='upper left', fontsize=10)
        if basin == 'animas':
            ax.set_ylim(0, 0.95)
        elif basin == 'yampa':
            ax.set_ylim(0, 1.35)
        elif basin == 'blue':
            ax.set_ylim(0, 0.65)
        if savefig:
            fig_fn = f'{data_dir}/{basin}_wy{WY}_{var}_{zone}_snow17comp.png'
            print(f'Saving figure to {fig_fn}')
            fig.savefig(fig_fn, dpi=300, bbox_inches='tight')
    # plt.tight_layout()
    # plt.suptitle(f'{basin} WY{WY} iSnobal comparison with CBRFC SNOW-17', fontsize=12, y=1)
    if savefig:
        fig_fn = f'{data_dir}/{basin}_wy{WY}_{var}_snow17comp.png'
        print(f'Saving figure to {fig_fn}')
        fig.savefig(fig_fn, dpi=300, bbox_inches='tight')

def parse_arguments():
    """Parse command line arguments.

    Returns:
        argparse.Namespace: Parsed command line arguments.
    """
    parser = argparse.ArgumentParser(description='CBRFC zonal calculations for iSnobal output')
    parser.add_argument('basin', type=str, help='Basin name')
    parser.add_argument('wy', type=int, help='Water year of interest')
    parser.add_argument('-var', type=str, default='thickness',
                        choices=['thickness', 'snow_density', 'specific_mass',
                                 'liquid_water', 'temp_surf', 'temp_lower',
                                 'temp_snowcover', 'thickness_lower', 'water_saturation'],
                        help='Variable to calculate zonal means for (default: thickness)')
    parser.add_argument('-data_dir', type=str, default='/uufs/chpc.utah.edu/common/home/skiles-group3/ancillary_sdswe_products/CBRFC_SNOW17',
                        help='Directory to save data')
    return parser.parse_args()

def __main__():
    args = parse_arguments()
    data_dir = args.data_dir
    basin = args.basin
    WY = args.wy
    var = args.var
    verbose = True

    zonal_basin_data_list = prepare_isnobal_data(basin=basin, WY=WY, var=var, data_dir=data_dir)
    # If this list is empty, exit script on error
    if len(zonal_basin_data_list) == 0:
        print(f'No zonal data found for basin {basin} and water year {WY}. Exiting.')
        sys.exit(1)

    # Grab the zone names from these basins
    zones = np.unique(zonal_basin_data_list[0]['zone'].values)
    # Trim the last three characters to match the zone names in the csv files
    zones_abbrev = np.unique([zone[:-3] for zone in zones])
    if verbose:
        print(zones, len(zones))
        print(zones_abbrev)

    snow17_wy = prepare_cbrfc_data(zones_abbrev=zones_abbrev, data_dir=data_dir, WY=WY, verbose=verbose)
    # Derive number of rows and columns based on the number of zones
    nrows = int(np.ceil(len(zones) / 2.0))
    print(f'Number of rows: {nrows}, Number of columns: 2')
    plot_time_series_comparison(zones=zones, zonal_basin_data_list=zonal_basin_data_list, snow17_wy=snow17_wy,
                                basin=basin, WY=WY, var=var, data_dir=data_dir,
                                nrows=nrows, ncols=2, savefig=True)

if __name__ == '__main__':
    __main__()