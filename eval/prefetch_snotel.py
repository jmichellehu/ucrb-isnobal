"""Pre-fetch SNOTEL data locally and save to parquet cache.

Run from the eval/ directory with the snotel-fetch conda environment:
    conda activate snotel-fetch
    cd /path/to/ucrb-isnobal/eval
    python prefetch_snotel.py [--basin-poly PATH] [--snotel-json PATH]

The parquet is saved to the path set in config.yml under snotel.cache_parquet.
Copy it to CHPC afterwards so load_snotel() skips all network calls.
"""
import argparse
import pathlib
import sys

import numpy as np
import pandas as pd
import yaml
from tqdm.auto import tqdm

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from snotel_io import locate_snotel_in_poly, get_snotel


def main():
    parser = argparse.ArgumentParser(description='Pre-fetch SNOTEL data to parquet')
    parser.add_argument('--config',      default='config.yml')
    parser.add_argument('--basin-poly',  default=None, help='Override basin_poly path')
    parser.add_argument('--snotel-json', default=None, help='Override snotel_sites_geojson path')
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config))
    if args.basin_poly:
        cfg['paths']['basin_poly'] = args.basin_poly
    if args.snotel_json:
        cfg['paths']['snotel_sites_geojson'] = args.snotel_json

    cache_path = pathlib.Path(cfg['snotel']['cache_parquet'])

    print('Locating SNOTEL sites in basin polygon...')
    sites = locate_snotel_in_poly(
        cfg['paths']['basin_poly'],
        cfg['paths']['snotel_sites_geojson'],
        buffer=cfg['snotel']['buffer_m'],
        epsg=cfg['epsg'],
    )
    if sites.empty:
        print('No SNOTEL sites found in basin polygon.')
        return

    print(f'Found {len(sites)} site(s): {sites["site_name"].tolist()}')

    site_nums  = sites['site_num'].tolist()
    site_names = sites['site_name'].tolist()
    states     = sites['state'].tolist()
    snowvar    = cfg['snotel']['snowvar']
    wy         = cfg['water_year']

    name_to_xy, all_dfs = {}, {}
    for num, name, state in tqdm(
        zip(site_nums, site_names, states),
        total=len(site_nums),
        desc='Fetching SNOTEL',
        unit='site',
    ):
        coord_gdf, dfs = get_snotel(
            [num], [name], [state],
            WY=wy,
            epsg=cfg['epsg'],
            snowvar=snowvar,
        )
        if len(coord_gdf):
            name_to_xy[name] = (coord_gdf.iloc[0].geometry.x, coord_gdf.iloc[0].geometry.y)
        all_dfs.update(dfs)

    if not all_dfs:
        print('No data returned from any site.')
        return

    frames = []
    for site_name, df in all_dfs.items():
        if snowvar == 'SNOWDEPTH' and 'SNOWDEPTH_m' in df.columns:
            out = df[['SNOWDEPTH_m']].rename(columns={'SNOWDEPTH_m': 'thickness'})
        elif snowvar == 'SWE' and 'SWE_m' in df.columns:
            out = df[['SWE_m']].rename(columns={'SWE_m': 'specific_mass'})
        else:
            continue
        out['site_name'] = site_name
        x, y = name_to_xy.get(site_name, (np.nan, np.nan))
        out['x_utm'] = x
        out['y_utm'] = y
        frames.append(out)

    result = pd.concat(frames).sort_index()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(cache_path)
    print(f'Saved {len(result)} rows to {cache_path}')
    print('Sites:', result['site_name'].unique().tolist())


if __name__ == '__main__':
    main()
