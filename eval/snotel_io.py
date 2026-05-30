"""Standalone SNOTEL helpers extracted from processing.py.

Only depends on geopandas, shapely, metloom, numpy, datetime — no xarray/rioxarray/rasterio.
"""
import datetime
import logging

import numpy as np
import geopandas as gpd
from shapely.geometry import Point
from shapely.ops import unary_union
from metloom.pointdata import SnotelPointData

LOGGER = logging.getLogger(__name__)


def locate_snotel_in_poly(poly_fn, site_locs_fn, buffer=0, epsg=32613):
    """Return GeoDataFrame of SNOTEL sites within the basin polygon."""
    sites_gdf = gpd.read_file(site_locs_fn)
    if sites_gdf.crs.to_epsg() != epsg:
        sites_gdf = sites_gdf.to_crs(f'epsg:{epsg}')

    poly_gdf = gpd.read_file(poly_fn)
    poly_geom = unary_union(poly_gdf.geometry) if len(poly_gdf) > 1 else poly_gdf.iloc[0].geometry
    poly_geom = poly_geom.buffer(buffer)

    return sites_gdf.loc[sites_gdf.intersects(poly_geom)]


def get_snotel(sitenum, sitename, ST, WY, epsg=32613, snowvar='SNOWDEPTH'):
    """Fetch daily SNOTEL data via metloom. Returns (GeoDataFrame, dict[site_name -> DataFrame]).

    DataFrames have a DatetimeIndex and a SNOWDEPTH_m or SWE_m column (metric units).
    """
    if isinstance(WY, list):
        start = datetime.datetime(WY[0] - 1, 10, 1)
        end   = datetime.datetime(WY[-1], 9, 30)
    else:
        start = datetime.datetime(WY - 1, 10, 1)
        end   = datetime.datetime(WY, 9, 30)

    lons, lats, dfs = [], [], {}
    for num, name, state in zip(sitenum, sitename, ST):
        pt = SnotelPointData(f"{num}:{state}:SNTL", name)
        meta = pt.metadata
        lons.append(meta.x)
        lats.append(meta.y)

        if snowvar == 'SNOWDEPTH':
            variables = [pt.ALLOWED_VARIABLES.SNOWDEPTH]
        elif snowvar == 'SWE':
            variables = [pt.ALLOWED_VARIABLES.SWE]
        else:
            variables = [pt.ALLOWED_VARIABLES.SNOWDEPTH, pt.ALLOWED_VARIABLES.SWE]

        df = pt.get_daily_data(start, end, variables)
        if df is None or df.empty:
            LOGGER.warning('No data for %s between %s and %s', name, start, end)
            continue

        if snowvar in ('SNOWDEPTH', None):
            df['SNOWDEPTH_m'] = df['SNOWDEPTH'] * 0.0254
        if snowvar in ('SWE', None):
            df['SWE_m'] = df['SWE'] * 0.0254

        dfs[name] = df.reset_index().set_index('datetime')

    gdf = gpd.GeoDataFrame(
        geometry=gpd.GeoSeries([Point(lon, lat) for lon, lat in zip(lons, lats)])
    ).set_crs('epsg:4326').to_crs(f'epsg:{epsg}')

    return gdf, dfs
