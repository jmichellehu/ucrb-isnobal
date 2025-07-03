#!/usr/bin/env python
'''
Script to calculate the zonal means of isnobal data for CBRFC outputs for a given basin and water year.
'''
import os
import sys
import argparse

import numpy as np
import pandas as pd
import geopandas as gpd
import xarray as xr

sys.path.append('/uufs/chpc.utah.edu/common/home/u6058223/git_dirs/env/')
import helpers as h

def clip_2_basin(basin_outline_fn, basin, data_dir, polys):
    # Load the basin outline
    basin_outline = gpd.read_file(basin_outline_fn)

    # Reproject the polygon outline for intended clipping
    basin_outline = basin_outline.to_crs(polys.crs)

    # Clip the dataset by the upper yampa outline
    polys_clipped = polys.clip(basin_outline.geometry.iloc[0])

    # Write this to file, dropping the index again
    out_fn =f'{data_dir}/basin_zones/CBRFC_SNOW17_{basin}.gpkg'
    polys_clipped.to_file(out_fn, driver='GPKG', index=False)
    print(f'Wrote {out_fn}')

def prep_basin_data(basin, WY,
                    data_dir='/uufs/chpc.utah.edu/common/home/skiles-group3/ancillary_sdswe_products/CBRFC_SNOW17',
                    workdir='/uufs/chpc.utah.edu/common/home/skiles-group3/model_runs/',
                    script_dir='/uufs/chpc.utah.edu/common/home/skiles-group3/jmhu/isnobal_scripts/'
                    ):
    # Get the basin outline file
    basin_outline_fn = h.fn_list(script_dir, f'*{basin}*/polys/*{basin}_*.shp')[0]
    print(f'Basin outline file: {basin_outline_fn}')
    # Clip the polygons to the basin if the file does not already exist
    basin_gpkg_fn = f'{data_dir}/basin_zones/CBRFC_SNOW17_{basin}.gpkg'
    if not os.path.exists(basin_gpkg_fn):
        print(f'Clipping polygons to basin {basin}...')
        polys = gpd.read_file(f'{data_dir}/CBRFC_SNOW17_select_segments.gpkg')
        clip_2_basin(basin_outline_fn, basin, data_dir, polys)
    else:
        # Load the geopackage based on basin
        print(f'Found basin geopackage: {basin_gpkg_fn}')

    # Read in the basin polygons
    basin_polys = gpd.read_file(basin_gpkg_fn)

    # Get the basin directories for the specified water year
    basindirs = h.fn_list(workdir, f'{basin}*/wy{WY}/{basin}*/')

    return basindirs, basin_polys

def load_isnobal_data(basindirs, nc_file='snow.nc'):
    day_fns = [h.fn_list(basindir, f'run*/{nc_file}') for basindir in basindirs]
    print(len(day_fns[0]), len(day_fns[1]))
    # Load the datasets for each day
    day_list = [xr.open_mfdataset(day_fn_list, chunks='auto') for day_fn_list in day_fns]
    return day_list

def locate_topo_zones(basin, basin_polys,
                      topo_dir='/uufs/chpc.utah.edu/common/home/skiles-group3/jmhu/ancillary/snow17_topos',
                      zone_name='cbrfc_zone'):
    # Load the new topos from joe with cbrfc zones embedded
    topo = xr.open_dataset(h.fn_list(topo_dir, f'{basin}*.nc')[0])

    # Set zero values as nans to leverage range plotting - can remove this later
    topo = topo.where(topo != 0)

    # Now use these zones as a mask to filter with xarray
    zones = np.unique(topo[zone_name].values)

    # Remove nan value
    zones = zones[~np.isnan(zones)]
    # Remove the following blue river zones
    # zones 974 GMRC2LLF and 985 SLAC2HUF
    zones = zones[zones != 974]
    zones = zones[zones != 985]
    # Get the corresponding basin polygons
    zone_names_sorted = np.sort(basin_polys['zone'])
    print(basin)
    # Drop the zones if the following substrings are in them
    # animas - CDRC2LOF
    # Yampa MBLC2 and SLF, and LSRC2 zones
    drop_substrings = ['MBLC2', 'SLFC2', 'LSRC2', 'CDRC2LOF']
    for substring in drop_substrings:
        zone_names_sorted = [zone for zone in zone_names_sorted if substring not in zone]

    return topo, zones, zone_names_sorted

def build_cbrfc_var_gdf(basin_polys, basin, WY, ds, topo, zones,
                        # zone_names_sorted,
                        data_dir,
                        var='thickness', runtype='Baseline',
                        zone_name='cbrfc_zone', save_file=None):
    # This is quicker with a geodataframe and single variable
    # Ultimately, construct a dataset with columns:
    # - zonal_classification: the classified zonal value from Joe
    # - mean_thickness_YYYYMMDD: mean daily thickness values for that zone for that day
    # - zone: the name of the zone (e.g., DGOC2LLF)
    # - geometry: the geometry of the zone polygon
    print(np.unique(topo[zone_name]))
    print(zones)
    # Ok, now generate a geodataframe where each row is a zone and the columns are the daily spatial means for each variable
    zonal_gdf = gpd.GeoDataFrame(columns=['zone', 'zonal_classification', 'geometry'])
    zonal_gdf['zone'] = basin_polys['zone']
    zonal_gdf['geometry'] = basin_polys['geometry']
    drop_substrings = ['MBLC2', 'SLFC2', 'LSRC2', 'CDRC2LOF']
    for substring in drop_substrings:
        # Remove rows where the zone name contains the substring
        zonal_gdf = zonal_gdf[~zonal_gdf['zone'].str.contains(substring, na=False)]
    # Sort the gdf on zone name and reset indices
    zonal_gdf = zonal_gdf.sort_values(by='zone').reset_index(drop=True)
    # Assign zonal_classification by unique zone values
    zonal_gdf['zonal_classification'] = zones

    # Reduce the dataset to only the variables we care about
    ds_var = ds[var].compute()
    # Loop through each day
    for jdx, day in enumerate(ds_var.time):
        # Grab date in YYYYMMDD format
        date_str = day.dt.strftime('%Y%m%d').values
        # print(date_str)
        date_list = []
        # Loop through each zone
        for zone in zones:
            # Calculate the daily spatial mean for this zone, ignoring nans
            daily_mean = np.nanmean(ds_var.isel(time=jdx).where(topo[zone_name] == zone, drop=True))
            # Append the mean thickness to this date column list
            date_list.append(daily_mean)
        print(date_str)

        # Concatenate this day's list as a new column in the zonal_gdf by first converting to dataframe
        zonal_gdf = pd.concat([zonal_gdf, pd.DataFrame({f'{runtype}_{date_str}': date_list})], axis=1)

    # Save the zonal_gdf to a geopackage
    if save_file is not None:
        out_fn = f'{data_dir}/isnobal_zonal/CBRFC_SNOW17_{basin}_WY{WY}_{runtype}_zonal_{var}_gdf.gpkg'
        # Default do not overwrite
        if not os.path.exists(out_fn):
            zonal_gdf.to_file(out_fn, driver='GPKG', index=False)
            print(f'Wrote {out_fn}')
        else:
            print(f'File {out_fn} already exists. Skipping.')
    return zonal_gdf

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
    parser.add_argument('-workdir', type=str, default='/uufs/chpc.utah.edu/common/home/skiles-group3/model_runs/',
                        help='Directory containing basin directories')
    return parser.parse_args()

def __main__():
    args = parse_arguments()
    data_dir = args.data_dir
    workdir = args.workdir
    basin = args.basin
    WY = args.wy
    var = args.var

    basindirs, basin_polys = prep_basin_data(basin=basin, WY=WY, data_dir=data_dir, workdir=workdir)
    day_list = load_isnobal_data(basindirs)
    # Reproject polygon to EPSG 32613
    basin_polys = basin_polys.to_crs('EPSG:32613')
    topo, zones, zone_names = locate_topo_zones(basin, basin_polys)
    # if these lengths are not equal, flag and exit
    if len(zones) != len(zone_names):
        print(f'Error: Number of zones ({len(zones)}) does not match number of zone names ({len(zone_names)}).')
        print('Zones:', zones)
        print('Zone names:', zone_names)
        sys.exit(1)

    zonal_gdf_list = []
    for ds, runtype in zip(day_list, ['baseline', 'hrrrspires']):
        zonal_gdf = build_cbrfc_var_gdf(basin_polys=basin_polys, ds=ds, topo=topo,
                                        basin=basin, WY=WY,
                                        runtype=runtype, zones=zones,
                                        data_dir=data_dir, var=var, zone_name='cbrfc_zone',
                                        save_file=True)
        zonal_gdf_list.append(zonal_gdf)

if __name__ == '__main__':
    __main__()
