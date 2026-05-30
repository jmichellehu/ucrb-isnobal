#!/usr/bin/env python
"""Build a full water-year Zarr data cube from daily iSnobal model output.

Combines all daily run directories (run20*/{dataset}.nc) into a single
time × y × x Zarr store, preserving the full spatial grid. Intended as a
one-time pre-processing step to eliminate per-file open overhead and enable
fast, chunked analysis with xarray + dask.

Usage
-----
    build_zarr_datacube.py animas 2022
    build_zarr_datacube.py animas 2022 -var snow -o /scratch/u6058223/zarr
    build_zarr_datacube.py animas 2022 --time-chunks 30 --xy-chunks 200

Reading back
------------
    import xarray as xr
    ds = xr.open_zarr('animas_unified_em_wy2022.zarr')   # lazy, no data loaded
    net_rad_mean = ds['net_rad'].mean(dim=['x', 'y']).compute()
"""
import sys
import os
import glob
import logging
import time
import argparse
from pathlib import PurePath
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import xarray as xr

LOGGER = logging.getLogger(__name__)

# Variables to drop on open (consistent with existing extract scripts)
DROP_VARS = {
    'em':                   ['projection'],
    'snow':                 ['projection'],
    'net_solar':                 ['projection']
}

# Cumulative mass-flux variables in em.nc that must be summed across h00 + h23
# to recover the full daily total. All other em variables are state / energy-flux
# quantities for which the h23 end-of-day value is used directly.
_EM_FLUX_VARS = frozenset(['snowmelt', 'evaporation', 'SWI'])


def _configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format='%(asctime)s %(levelname)s %(message)s')


def _fn_list(this_dir: str, fn_pattern: str) -> List[str]:
    fns = glob.glob(os.path.join(this_dir, fn_pattern))
    fns.sort()
    return fns


def _reduce_to_daily(ds: xr.Dataset) -> xr.Dataset:
    """Reduce the h00 / h23 two-timestep structure to one time step per day.

    iSnobal's daily files (after day 1) contain two timesteps:
      - h23: cumulative flux / end-of-day state for hours 01–23
      - h00: the 1-hour rollover from the previous day's h23 through midnight,
             attributed by iSnobal to the current day's accumulation

    Flux em variables (snowmelt, evaporation, SWI): daily total = h23 + h00.
    All other variables: end-of-day snapshot = h23 only.

    Returns a dataset with one timestep per calendar day, labelled at h23.
    """
    # Time coordinates are eagerly available even on dask-backed datasets
    hours = ds.time.dt.hour.values

    h23 = ds.isel(time=np.where(hours == 23)[0])
    h00 = ds.isel(time=np.where(hours == 0)[0])

    if h00.time.size == 0:
        LOGGER.debug('No h00 timesteps found — returning h23 only')
        return h23

    # Map each h00 timestamp to the h23 time on the same calendar date so they
    # can be aligned on a shared time axis before addition.
    h23_by_date = {pd.Timestamp(t).date(): t for t in h23.time.values}

    matched_mask = [pd.Timestamp(t).date() in h23_by_date for t in h00.time.values]
    n_dropped = sum(not m for m in matched_mask)
    if n_dropped:
        LOGGER.warning('%d h00 timestep(s) had no matching h23 date and will be dropped', n_dropped)
        h00 = h00.isel(time=np.where(matched_mask)[0])

    relabeled_times = [h23_by_date[pd.Timestamp(t).date()] for t in h00.time.values]
    h00 = h00.assign_coords(time=relabeled_times)

    # Reindex onto the full h23 time axis; day 1 (no h00) fills with 0 — correct for a flux sum
    h00 = h00.reindex(time=h23.time, fill_value=0)

    flux_vars_present = [v for v in _EM_FLUX_VARS if v in ds.data_vars]
    if not flux_vars_present:
        LOGGER.debug('No flux variables present in dataset — returning h23 only')
        return h23

    LOGGER.info('Summing h00 + h23 for flux variables: %s', flux_vars_present)
    updates = {var: h23[var] + h00[var] for var in flux_vars_present}
    return h23.assign(updates)


def _discover_basindirs(workdir: str, basin: str, wy: int) -> Tuple[List[str], List[str]]:
    """Return (basindirs, labels) for all simulations matching basin + water year."""
    basindirs = _fn_list(workdir, f'{basin}*hrrr_radiation/wy{wy}/{basin}*/') # currently for the unified thp runs
    if not basindirs:
        return [], []

    labels = []
    for bd in basindirs:
        stem = PurePath(bd).stem
        ending = stem.split('_')[-1]
        # Use the directory stem suffix as the label - this should default to the unified runs right now, which should just be the basin name
        # If so, change the label to "unified" for clarity
        if ending == basin:
            ending = 'unified'
        labels.append(ending)
    return basindirs, labels


def _build_zarr(
    basindir: str,
    label: str,
    basin: str,
    wy: int,
    dataset: str,
    outdir: str,
    chunks: dict,
    drop_vars: List[str],
    overwrite: bool,
) -> None:
    """Open all daily {dataset}.nc files and write a consolidated Zarr store."""
    store_name = f'{basin}_{label}_{dataset}_wy{wy}.zarr'
    outpath = os.path.join(outdir, store_name)

    if os.path.exists(outpath) and not overwrite:
        LOGGER.info('Store already exists, skipping: %s  (pass --overwrite to replace)', outpath)
        return

    file_list = _fn_list(basindir, f'run20*/{dataset}.nc')
    if not file_list:
        LOGGER.warning('No %s.nc files found in %s — skipping', dataset, basindir)
        return

    LOGGER.info('Found %d daily files  label=%s  dataset=%s', len(file_list), label, dataset)
    t0 = time.perf_counter()

    # Open lazily — data stays on disk until .to_zarr() triggers computation
    # parallel=False avoids fork-safety issues on CHPC login/compute nodes;
    # chunked dask arrays still parallelize the write via the threaded scheduler
    ds = xr.open_mfdataset(
        file_list,
        combine='by_coords',
        drop_variables=drop_vars,
        chunks=chunks,
        parallel=False,
        engine='netcdf4',
    )
    LOGGER.info('Dataset opened  dims=%s  variables=%s  elapsed=%.1fs',
                dict(ds.sizes), list(ds.data_vars), time.perf_counter() - t0)

    LOGGER.info('Processing em.nc: combining h00 + h23 for cumulative flux variables')
    ds = _reduce_to_daily(ds)
    LOGGER.info('Post-combination dims=%s', dict(ds.sizes))

    os.makedirs(outdir, exist_ok=True)
    write_mode = 'w' if overwrite else 'w-'

    LOGGER.info('Writing Zarr store: %s', outpath)
    t1 = time.perf_counter()
    ds.to_zarr(outpath, mode=write_mode, consolidated=True)
    ds.close()

    LOGGER.info('Write complete  elapsed=%.1fs  total=%.1fs',
                time.perf_counter() - t1, time.perf_counter() - t0)
    LOGGER.info('Store written: %s', outpath)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            'Combine daily iSnobal run directories into a single water-year '
            'Zarr data cube (time × y × x), preserving the full spatial grid.'
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('basin', type=str, help='Basin name')
    parser.add_argument('wy',    type=int, help='Water year')
    parser.add_argument(
        '-var', '--dataset',
        choices=['em', 'snow', 'net_solar'],
        default='em',
        help='Model output file to combine',
    )
    parser.add_argument(
        '-w', '--workdir',
        default='/uufs/chpc.utah.edu/common/home/skiles-group2/model_runs_jmh/thp/',
        help='Root directory containing basin model run subdirectories',
    )
    parser.add_argument(
        '-o', '--outdir',
        default='/uufs/chpc.utah.edu/common/home/skiles-group2/model_runs_jmh/thp/zarr_stores',
        help='Output directory for Zarr stores',
    )
    parser.add_argument(
        '--time-chunks', type=int, default=10,
        help=(
            'Chunk size along the time dimension. '
            'Increase (e.g., 30) for time-series-heavy workflows; '
            'decrease (e.g., 1) if you mostly access single time slices.'
        ),
    )
    parser.add_argument(
        '--xy-chunks', type=int, default=150,
        help=(
            'Chunk size along both x and y dimensions. '
            'Decrease for large grids to keep chunks under ~200 MB total.'
        ),
    )
    parser.add_argument(
        '--overwrite', action='store_true',
        help='Overwrite an existing Zarr store',
    )
    parser.add_argument(
        '-v', '--verbose', action='store_true',
        help='Enable DEBUG-level logging',
    )
    return parser.parse_args()


def __main__() -> None:
    args = parse_arguments()
    _configure_logging(args.verbose)

    chunks = {'time': args.time_chunks, 'x': args.xy_chunks, 'y': args.xy_chunks}
    drop_vars = DROP_VARS.get(args.dataset, ['projection'])

    LOGGER.info('Basin=%s  WY=%s  dataset=%s  chunks=%s', args.basin, args.wy, args.dataset, chunks)

    basindirs, labels = _discover_basindirs(args.workdir, args.basin, args.wy)
    if not basindirs:
        LOGGER.error('No basin directories found for %s wy%s in %s', args.basin, args.wy, args.workdir)
        sys.exit(1)

    LOGGER.info('Found %d simulation(s)', len(basindirs))
    for bd, lb in zip(basindirs, labels):
        LOGGER.debug('  label=%s  path=%s', lb, bd)

    for basindir, label in zip(basindirs, labels):
        LOGGER.info('--- Processing label=%s ---', label)
        _build_zarr(
            basindir=basindir,
            label=label,
            basin=args.basin,
            wy=args.wy,
            dataset=args.dataset,
            outdir=args.outdir,
            chunks=chunks,
            drop_vars=drop_vars,
            overwrite=args.overwrite,
        )

    LOGGER.info('Done.')


if __name__ == '__main__':
    __main__()
