#!/usr/bin/env python

import os
import argparse
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import glob
from typing import List

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
    parser = argparse.ArgumentParser(description='Plot domain maps of energy balance terms for a given basin and date.')
    parser.add_argument('basin', type=str, help='Basin name', default='animas')
    parser.add_argument('-w', '--workdir', type=str, help='Directory containing model runs', default='/uufs/chpc.utah.edu/common/home/skiles-group1/jmhu/model_runs/')
    parser.add_argument('-dt', '--date', type=str, help='Date of interest YYYYMMDD', default=None)
    parser.add_argument('-v', '--verbose', action='store_true', help='Print filenames', default=False)
    parser.add_argument('-o', '--outdir', type=str, help='Output directory', default='.')
    parser.add_argument('-c', '--cmap', type=str, help='Colormap', default='RdYlBu')
    parser.add_argument('-min', '--vmin', type=float, help='Minimum value for colormap', default=-50)
    parser.add_argument('-max', '--vmax', type=float, help='Maximum value for colormap', default=50)
    return parser.parse_args()

def __main__():
    args = parse_arguments()
    basin = args.basin
    workdir = args.workdir
    dt = args.date
    verbose = args.verbose
    outdir = args.outdir
    cmap = args.cmap
    vmin = args.vmin
    vmax = args.vmax

    # Get the basin directories
    basindirs = fn_list(workdir, f'{basin}*isnobal/')

    # Get the WY from the directory name (only a single WY per basin right now, will need to add arg later)
    WY = int(fn_list(basindirs[0], '*')[0].split('/')[-1].split('wy')[-1])
    if dt is None:
        dt = f'{WY-1}1015'
    em_fnlist = [fn_list(basindir, f'*{WY}/*/run{dt}/em.nc')[0] for basindir in basindirs]
    em_list = [np.squeeze(xr.open_dataset(em_fn)) for em_fn in em_fnlist]

    if verbose:
        print(em_fnlist)

    ds = em_list[0]
    fig, axa = plt.subplots(1, 6, figsize=(18,4), sharex=True, sharey=True)
    for jdx, thisvar in enumerate(em_list[0].data_vars):
        if jdx < len(em_list[0].data_vars) - 5:
            # print(thisvar)
            ax=axa.flatten()[jdx]
            # plot without colorbar
            ds[thisvar].plot.imshow(vmin=vmin, vmax=vmax, cmap=cmap, ax=ax, add_colorbar=False)
            ax.set_aspect('equal')
            ax.set_title(thisvar)
            ax.set_xlabel('')
            ax.set_ylabel('')
    plt.suptitle(f'{basin} {np.datetime_as_string(ds.time.values, unit="D")}')
    outpath = f'{outdir}/{basin}_eb_terms_{dt}.png'
    if verbose:
        print(outpath)
    plt.savefig(outpath, dpi=300, bbox_inches='tight')


if __name__ == '__main__':
    __main__()