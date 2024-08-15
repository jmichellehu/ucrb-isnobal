#!/usr/bin/env python

import pyproj
import os
import sys
sys.path.append('/uufs/chpc.utah.edu/common/home/u6058223/git_dirs/ucrb-isnobal/scripts/')
import processing as proc
import argparse
import pandas as pd
import glob
from typing import List

def fn_list(thisDir: str, fn_pattern: str, verbose: bool = False) -> List[str]:
    """Match and sort filenames based on a regex pattern in specified directory

    Parameters
    -------------
    thisDir: str
        directory path to search
    fn_pattern: str
        regex pattern to match files
    verbose: boolean
        print filenames
    
    Returns
    -------------
    fns: list
        list of filenames matched and sorted
    """
    fns=[]
    for f in glob.glob(thisDir + "/" + fn_pattern):
        fns.append(f)
    fns.sort()
    if verbose: print(fns)
    return fns

def parse_arguments():
        """Parse command line arguments.

        Returns:
        ----------
        argparse.Namespace
            Parsed command line arguments.
        """
        parser = argparse.ArgumentParser(description='Extract National Water Model snow depth ["SNOWH"] \
                                         at point sites [SNOTEL] within a basin for a given water year.')
        parser.add_argument('basin', type=str, help='Basin name')
        parser.add_argument('wy', type=int, help='Water year of interest')
        parser.add_argument('-shp', '--shapefile', type=str, help='Shapefile of basin polygon', default=None)
        parser.add_argument('-loc', '--sitelocs', type=str, help='json file of point locations', 
                            default='/uufs/chpc.utah.edu/common/home/skiles-group3/SNOTEL/snotel_sites_32613.json')
        parser.add_argument('-o', '--out_path', type=str, help='Output path', default=None)
        parser.add_argument('-v', '--verbose', action='store_true', help='Print filenames', default=False)
        return parser.parse_args()

def __main__():
    args = parse_arguments()
    verbose = args.verbose
    basin = args.basin
    wy = args.wy
    poly_fn = args.shapefile
    site_json = args.site_locs
    outname = args.out_path

    # NWM proj4 string
    proj4 = '+proj=lcc +lat_0=40 +lon_0=-97 +lat_1=30 +lat_2=60 +x_0=0 +y_0=0 +R=6370000 +units=m +no_defs'
    epsg = 'epsg:32613'
    
    # for NWM env
    pyprojdatadir = '/uufs/chpc.utah.edu/common/home/u6058223/software/pkg/miniconda3/pkgs/proj-9.4.1-hb784bbd_0/share/proj'
    pyproj.datadir.set_data_dir(pyprojdatadir)

    if poly_fn is None:
        poly_dir = '/uufs/chpc.utah.edu/common/home/skiles-group1/jmhu/ancillary/polys'
        poly_fn = fn_list(poly_dir, f'*{basin}*shp')[0]

    site_gdf = proc.locate_snotel_in_poly(poly_fn, site_json)
    if verbose:
        print(f'Located {len(site_gdf)} SNOTEL sites in {basin}')

    # Change the crs of the site_gdf to the NWM crs
    site_gdf.set_crs(epsg, inplace=True, allow_override=True)
    site_gdf = site_gdf.to_crs(crs=proj4)
    if verbose:
        print(site_gdf.crs)

    basin_nwm_ds = proc.get_nwm_retrospective_LDAS(site_gdf, start=f'{wy-1}-10-01', end=f'{wy}-09-30', var='SNOWH')

    # Turn it into a dict
    nwm_snowh_dict = dict()
    for jdx, ds in enumerate(basin_nwm_ds):
        nwm_snowh_dict[site_gdf['site_name'].values[jdx]] = ds.values

    # Turn it into a dataframe
    df = pd.DataFrame(nwm_snowh_dict, index=ds['time'].values)

    # Save the the dataframe as csv for easy access later
    if outname is None:
        nwm_basin_dir = f'/uufs/chpc.utah.edu/common/home/skiles-group3/NWM/{basin}'
        if not os.path.exists(nwm_basin_dir):
            os.makedirs(nwm_basin_dir)
        outname = f'{nwm_basin_dir}/{basin}_nwm_snotel_SNOWH_wy{wy}.csv'

    df.to_csv(outname)
    if verbose:
        print(outname)

if __name__ == "__main__":
    __main__()