#!/usr/bin/env python
"""This script extracts data from smeshr files used in the HRRR-SPIReS net solar implementation
and saves the extracted data to netCDF files."""

import sys
import os
import argparse
import xarray as xr

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
        parser.add_argument('-s', '--smesh', type=str, help='Path to SMESHR directory',
                            default='/uufs/chpc.utah.edu/common/home/skiles-group3/SMESHR')
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
    smeshdir = args.smesh
    outdir = args.outdir
    verbose = args.verbose

    snotel_dir = '/uufs/chpc.utah.edu/common/home/skiles-group3/ancillary_sdswe_products/SNOTEL'
    script_dir = '/uufs/chpc.utah.edu/common/home/skiles-group3/jmhu/isnobal_scripts'
    smeshdir = f'{smeshdir}/{basin}'
    dswrfdir = f'{smeshdir}/DSWRF'
    model_dir = '/uufs/chpc.utah.edu/common/home/skiles-group3/model_runs/'

    gdf_metloom, _ = prep_snotel_sites(basin, script_dir, snotel_dir, WY, ST_abbrev=state_abbrev, epsg=epsg,
                                               poly_fn=poly_fn, verbose=verbose)

    # Extract start and end dates from WY
    startdate = f'{WY - 1}1001'
    enddate = f'{WY}0930'
    if verbose:
        print(startdate, enddate)

    # extract smeshr files at snotel sites without resampling to daily values
    if verbose:
        print('Extracting data at snotel sites and saving to file')
    outname = f'{outdir}/DSWRF_{basin}_{WY}_snotel.nc'
    if not os.path.exists(outname):
        print(f'Extracting DSWRF at snotel sites and saving to {outname}')
        # List comprehension is faster than the mf_dataset with parallel=True for some reason
        print('Loading data files...')
        raw_dswrf_list = [xr.open_dataset(f, chunks='auto', drop_variables=['projection', 'illumination_angle', 'azimuth', 'zenith']) for f in h.fn_list(dswrfdir, f'hrrr.{WY-1}1*') + h.fn_list(dswrfdir, f'hrrr.{WY}0*')]
        # Sample at SNOTEL sites without resampling to daily values
        print('Sampling at SNOTEL sites...')
        raw_dswrf_sampled = [dswrf.sel(x=list(gdf_metloom.geometry.x.values), y=list(gdf_metloom.geometry.y.values), method='nearest') for dswrf in raw_dswrf_list]
        # Concatenate the sampled datasets
        print('Concatenating datasets...')
        raw_dswrf = xr.concat(raw_dswrf_sampled, dim='time')
        # Write to file
        print('Writing to file...')
        raw_dswrf.to_netcdf(outname)
        del raw_dswrf
    else:
        print(f'File {outname} already exists. Skipping extraction.')
    outname = f'{outdir}/net_HRRR_SPIReS_{basin}_{WY}_snotel.nc'
    if not os.path.exists(outname):
        print(f'Extracting net solar radiation at snotel sites and saving to {outname}')
        # Just open for the water year by going into the model dir!
        # List comprehension is faster than the mf_dataset with parallel=True for some reason
        print('Loading data files...')
        net_HRRR_SPIReS_list = [xr.open_dataset(f, chunks='auto', drop_variables=['illumination_angle', 'transverse_mercator']) for f in h.fn_list(model_dir, f'{basin}*/wy{WY}/*albedo/run*/net_solar.nc')]
        # Sample at SNOTEL sites without resampling to daily values
        print('Sampling at SNOTEL sites...')
        net_HRRR_SPIReS_sampled = [net_HRRR_SPIReS.sel(x=list(gdf_metloom.geometry.x.values), y=list(gdf_metloom.geometry.y.values), method='nearest') for net_HRRR_SPIReS in net_HRRR_SPIReS_list]
        # Concatenate the sampled datasets
        print('Concatenating datasets...')
        net_HRRR_SPIReS = xr.concat(net_HRRR_SPIReS_sampled, dim='time')
        # Write to file
        print('Writing to file...')
        net_HRRR_SPIReS.to_netcdf(outname)
        del net_HRRR_SPIReS
    else:
        print(f'File {outname} already exists. Skipping extraction.')

if __name__ == '__main__':
    __main__()