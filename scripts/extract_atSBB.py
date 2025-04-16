#!/usr/bin/env python
"""This script is a one-off to extract albedo data from HRRR-MODIS smeshr files at Senator Beck Basin sites and saves the extracted data to netCDF files."""
import sys
import os
import argparse
from pathlib import PurePath
import xarray as xr
import seaborn as sns
import geopandas as gpd
from shapely.geometry import Point

sys.path.append('/uufs/chpc.utah.edu/common/home/u6058223/git_dirs/env/')
import helpers as h

sys.path.append('/uufs/chpc.utah.edu/common/home/u6058223/git_dirs/ucrb-isnobal/scripts/')
import processing as proc

def prep_SBB_gdf(outcrs: int = 32613):
    """
    Prepare the SBB geodataframe with the coordinates of the Senator Beck Study Plot (Alpine)
    and Swamp Angel Study Plot (Subalpine).
    """
    # Senator Beck Study Plot (Alpine): 37.906900624490504, -107.72625795815301
    # Swamp Angel Study Plot (Subalpine): 37.906965992923794, -107.71133466651288
    SBSP_lat = 37.906900624490504
    SBSP_lon = -107.72625795815301
    SASP_lat = 37.906965992923794
    SASP_lon = -107.71133466651288

    # Turn into a geodataframe of points
    SBB_gdf = gpd.GeoDataFrame(geometry=[Point(SASP_lon, SASP_lat),
                                         Point(SBSP_lon, SBSP_lat)], crs='EPSG:4326')

    # Reproject to specified crs
    SBB_gdf = SBB_gdf.to_crs(crs=outcrs)
    return SBB_gdf

def extract_sbb_data(SBB_gdf, basin, wy, outdir, model_dir, verbose=True,
                    resampling='nearest'):
    '''
    '''
    outname = f'{outdir}/net_HRRR_MODIS_albedo_{basin}_sampled_SBB_{wy}.nc'
    if not os.path.exists(outname):
        if verbose:
            print('Loading data files')
        net_HRRR_MODIS_albedo_list = xr.open_mfdataset(h.fn_list(model_dir, f'{basin}*/wy{wy}/*albedo/run*/net_solar.nc'))

        if verbose:
            print('Resampling to daily values and extracting at the SBB sites')
        # Resample to daily values and extract at the SBB sites
        net_HRRR_MODIS_albedo_sampled = net_HRRR_MODIS_albedo_list.resample(time='1D').mean().sel(x=list(SBB_gdf.geometry.x.values),
                                                                                                  y=list(SBB_gdf.geometry.y.values), method=resampling)

        # Save to file
        if verbose:
            print(f'Saving to {outname}')
        net_HRRR_MODIS_albedo_sampled.to_netcdf(outname, mode='w',
                                                format='NETCDF4', engine='netcdf4', encoding={'albedo': {'zlib': True, 'complevel': 5}})
        del net_HRRR_MODIS_albedo_sampled
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
        parser.add_argument('-e', '--epsg', type=int, help='EPSG of AOI', default=32613)
        parser.add_argument('-p', '--palette', type=str, help='Seaborn color palette', default='icefire')
        parser.add_argument('-o', '--outdir', type=str, help='Output directory',
                            default='/uufs/chpc.utah.edu/common/home/skiles-group3/jmhu/data_extracts')
        parser.add_argument('-v', '--verbose', help='Print filenames', default=True)
        return parser.parse_args()

def __main__():
    args = parse_arguments()
    basin = args.basin
    WY = args.wy
    outcrs = args.epsg
    palette = args.palette
    outdir = args.outdir
    verbose = args.verbose
    sns.set_palette(palette)

    modeldir = '/uufs/chpc.utah.edu/common/home/skiles-group3/model_runs/'
    SBB_gdf = prep_SBB_gdf(outcrs=outcrs)
    extract_sbb_data(SBB_gdf=SBB_gdf, basin=basin, wy=WY, outdir=outdir, verbose=verbose, model_dir=modeldir)

if __name__ == '__main__':
    __main__()