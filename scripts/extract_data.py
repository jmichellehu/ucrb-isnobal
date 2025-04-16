#!/usr/bin/env python
"""This script extracts data from isnobal model output files (em.nc or snow.nc ) and saves the extracted data to netCDF files."""
import sys
import os
import argparse
from pathlib import PurePath
import xarray as xr
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

def extract_data(gdf_metloom, ds_list, basin, WY, outdir, varname='snow',
                    resampling='nearest', labels=['Baseline', 'HRRR-MODIS']):
    '''
    '''
    for kdx, ds in enumerate(ds_list):
        label = labels[kdx]
        outname = f'{outdir}/{basin}_{label}_{varname}_{WY}.nc'
        print(outname)
        if not os.path.exists(outname):
            # Extract all data values at snotel coordinates
            data_extract = ds.sel(x=list(gdf_metloom.geometry.x.values),
                                y=list(gdf_metloom.geometry.y.values),
                                method=resampling)
            # Save extract to file
            data_extract.to_netcdf(outname)
        else:
            print(f'File {outname} already exists. Skipping extraction.')

def parse_arguments():
        """Parse command line arguments.

        Returns:
        argparse.Namespace: Parsed command line arguments.
        """
        parser = argparse.ArgumentParser(description='Plot energy balance terms for a given basin and water year.')
        parser.add_argument('basin', type=str, help='Basin name')
        parser.add_argument('wy', type=int, help='Water year of interest')
        parser.add_argument('-var', '--varname', type=str, help='input netcdf file', default='em', choices=['em', 'snow', 'smrf_2', 'smrf_energy_balance'])
        parser.add_argument('-shp', '--shapefile', type=str, help='Shapefile of basin polygon', default=None)
        parser.add_argument('-st', '--state', type=str, help='State abbreviation', default='CO')
        parser.add_argument('-e', '--epsg', type=str, help='EPSG of AOI', default=None)
        parser.add_argument('-p', '--palette', type=str, help='Seaborn color palette', default='icefire')
        parser.add_argument('-o', '--outdir', type=str, help='Output directory',
                            default='/uufs/chpc.utah.edu/common/home/skiles-group3/jmhu/data_extracts')
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
    varname = args.varname
    verbose = args.verbose
    sns.set_palette(palette)

    workdir = '/uufs/chpc.utah.edu/common/home/skiles-group3/model_runs/'
    snotel_dir = '/uufs/chpc.utah.edu/common/home/skiles-group3/ancillary_sdswe_products/SNOTEL'
    script_dir = '/uufs/chpc.utah.edu/common/home/skiles-group3/jmhu/isnobal_scripts'
    basindirs = h.fn_list(workdir, f'{basin}*isnobal/wy{WY}/{basin}*/')
    gdf_metloom, _ = prep_snotel_sites(basin, script_dir, snotel_dir, WY, ST_abbrev=state_abbrev, epsg=epsg,
                                               poly_fn=poly_fn, verbose=verbose)
    # Energy balance
    if verbose:
        print('Stacking data files')
    if varname == 'smrf_energy_balance':
        # this only exists in baseline time decay isnobal-hrrr directory
        basindir = [b for b in basindirs if len(PurePath(b).stem.split('100m')[1]) == 0][0]
        ds_list = [xr.open_mfdataset(h.fn_list(basindir, f'*/{varname}*.nc'), drop_variables=['projection'], combine='by_coords')]
        # extract at snotel sites
        if verbose:
            print('Extracting data at snotel sites and saving to file')
        extract_data(gdf_metloom, ds_list, basin, WY, outdir, varname=varname, labels=['Baseline'])
    else:
        if varname == 'snow':
            drop_var_list = ['projection', 'snow_density', 'specific_mass', 'liquid_water', 'temp_surf', 'temp_lower', 'temp_snowcover', 'thickness_lower', 'water_saturation']
        elif varname == 'smrf_2':
            drop_var_list = ['projection', 'air_temp', 'percent_snow', 'precip_temp', 'precip', 'snow_density', 'storm_days', 'vapor_pressure', 'wind_speed']
        else:
            drop_var_list = ['projection']
        # ds_list = [xr.open_mfdataset(h.fn_list(basindir, f'*/{varname}*.nc'), drop_variables=drop_var_list, combine='by_coords', parallel=True) for basindir in basindirs]
        ds_list = [xr.open_mfdataset(h.fn_list(basindir, f'*/{varname}*.nc'), drop_variables=drop_var_list, combine='by_coords') for basindir in basindirs]

        # extract at snotel sites
        if verbose:
            print('Extracting data at snotel sites and saving to file')
        extract_data(gdf_metloom, ds_list, basin, WY, outdir, varname=varname)

if __name__ == '__main__':
    __main__()