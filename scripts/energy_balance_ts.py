#!/usr/bin/env python
 
import os
import argparse
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import glob
from typing import List
from pathlib import PurePath

import sys
sys.path.append('/uufs/chpc.utah.edu/common/home/u6058223/git_dirs/env/')
import helpers as h

sys.path.append('/uufs/chpc.utah.edu/common/home/u6058223/git_dirs/ucrb-isnobal/scripts/')
import processing as proc


# Set environmental variable for PROJ to directory where you can find proj.db
os.environ['PROJ']='/uufs/chpc.utah.edu/common/home/u6058223/software/pkg/miniconda3/pkgs/proj-9.3.1-h1d62c97_0/share/proj'
os.environ['PROJLIB']='/uufs/chpc.utah.edu/common/home/u6058223/software/pkg/miniconda3/pkgs/proj-9.3.1-h1d62c97_0/share/proj'

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
    if verbose: 
        print(fns)
    return fns

def parse_arguments():
    """Parse command line arguments.
    Returns: 
    ----------
    argparse.Namespace
        Parsed command line arguments.
    """
    parser = argparse.ArgumentParser(description='Plot energy balance terms for a given basin and water year.')
    parser.add_argument('basin', type=str, help='Basin name', default='animas')
    parser.add_argument('-b', '--basin_dir', type=str, help='Basin directory', default=None)
    parser.add_argument('-w', '--work_dir', type=str, help='Directory containing model runs', default='/uufs/chpc.utah.edu/common/home/skiles-group1/jmhu/model_runs/')
    parser.add_argument('-sn', '--snotel_dir', type=str, help='Directory containing SNOTEL period of record files', default='/uufs/chpc.utah.edu/common/home/skiles-group3/SNOTEL/')
    parser.add_argument('-sc', '--script_dir', type=str, help='Directory containing model runs', default='/uufs/chpc.utah.edu/common/home/skiles-group1/jmhu/isnobal_scripts/')
    parser.add_argument('-p', '--poly_fn', type=str, help='Polygon file for basin', default=None)
    parser.add_argument('-v', '--verbose', action='store_true', help='Print filenames', default=False)
    parser.add_argument('-o', '--outdir', type=str, help='Output directory', default='.')
    return parser.parse_args()

def __main__():
    args = parse_arguments()
    basin = args.basin
    workdir = args.work_dir
    basindir = args.basin_dir
    verbose = args.verbose
    outdir = args.outdir
    snotel_dir = args.snotel_dir
    script_dir = args.script_dir
    poly_fn = args.poly_fn

    colors = ['xkcd:coral', 'xkcd:aqua green', 'xkcd:warm grey', 'xkcd:pastel purple', 'xkcd:bright blue', 'k']
    linewidths = [0.75, 0.75, 1, 0.75, 0.75, 0.75]
    linestyles = ['-', '-', '-', '-', '-', '--']
    
    if basindir is None:
        # establish basin directories
        basindirs = h.fn_list(workdir, f'*{basin}*')
    else:
        basindirs = [basindir]

    # Get the WY from the directory name (only a single WY per basin right now, will need to add arg later)
    WY = int(fn_list(basindirs[0], '*')[0].split('/')[-1].split('wy')[-1])

    # Basin polygon file
    if poly_fn is None:
        try:
            poly_fn = h.fn_list(script_dir, f'*{basin}*setup/polys/*shp')[0]
        except IndexError:
            # exit script
            print(f'Polygon file not found for {basin}. Exiting...')
            return

    # SNOTEL all sites geojson fn
    allsites_fn = h.fn_list(snotel_dir, 'snotel_sites_32613.json')[0]

    # Locate SNOTEL sites within basin
    found_sites = proc.locate_snotel_in_poly(poly_fn=poly_fn, site_locs_fn=allsites_fn)

    # Get site names and site numbers
    sitenums = found_sites['site_num']

    for basindir in basindirs:
        if verbose:
            print(PurePath(basindir).name)
        em_fnlist = h.fn_list(basindir, f'*{WY}/*/run*/em.nc')
        em_list = [xr.open_dataset(em_fn) for em_fn in em_fnlist]
        if verbose:
            print(len(em_list))

        ds_list = em_list[0]
        var_ts_dict = dict()
        for kdx in range(len(sitenums)):
            _, gdf, _, sitename = proc.get_snotel_df_pt(snotel_dir=snotel_dir, sitenums=sitenums, WY=WY, jdx=kdx)
            
            # Plot the time series for all terms at this snotel site
            _, ax = plt.subplots(figsize=(18, 4))

            for jdx, thisvar in enumerate(ds_list.data_vars):
                # add spinner console print statements to track progress in for loop
                
                
                if jdx < len(ds_list.data_vars) - 5:                    
                    key_name = f'{sitename}_{thisvar}'
                    var_data = [ds[thisvar].sel(x=list(gdf.geometry.x.values), y=list(gdf.geometry.y.values), method='nearest') for ds in em_list]
                    # Concatenate all the days
                    var_data = np.squeeze(xr.concat(var_data, dim='time'))
                    var_ts_dict[key_name] = var_data
                    var_data.plot(x='time', ax=ax, label=thisvar, color=colors[jdx], linewidth=linewidths[jdx], linestyle=linestyles[jdx])
            
            ax.set_xlabel('')
            ax.set_ylabel('')
            ax.set_title('')
            plt.suptitle(f'{PurePath(basindir).name}\n{sitename}', y=1.0)
            plt.legend()
        
            # fix sitename for filename
            sitename = sitename.replace(' ', '_')
            sitename = sitename.replace('(', '_')
            sitename = sitename.replace('_', '_')
            sitename = sitename.replace('#', 'num')

            outpath = f'{outdir}/{PurePath(basindir).name}_{sitename}_eb_terms_wy{WY}.png'
            if verbose:
                print(outpath)
            plt.savefig(outpath, dpi=300, bbox_inches='tight')

if __name__ == '__main__':
    __main__()