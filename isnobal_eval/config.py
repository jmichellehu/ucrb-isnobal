"""Config schema and loader for isnobal_eval."""
from __future__ import annotations
from typing import List, Optional, Tuple
import yaml
from pydantic import BaseModel, field_validator


class PathsConfig(BaseModel):
    snow_zarr: str
    em_zarr: str
    net_solar_zarr: str
    topo_nc: str
    snotel_sites_geojson: str
    basin_poly: str
    output_dir: str


class TerrainFilterConfig(BaseModel):
    elev_range: Optional[Tuple[float, float]] = None
    aspect_range: Optional[Tuple[float, float]] = None
    slope_max: Optional[float] = None
    veg_classes: Optional[List[int]] = None


class SnotelConfig(BaseModel):
    variable_mapping: dict = {'SNOWDEPTH': 'thickness', 'SWE': 'specific_mass'}
    snowvar: str = 'SNOWDEPTH'
    buffer_m: int = 200


class EvalConfig(BaseModel):
    basin: str
    water_year: int
    paths: PathsConfig
    epsg: int
    state: str
    terrain_filter: TerrainFilterConfig = TerrainFilterConfig()
    snotel: SnotelConfig = SnotelConfig()


def load_config(path: str) -> dict:
    """Load YAML config, validate against EvalConfig schema, return plain dict."""
    with open(path) as f:
        raw = yaml.safe_load(f)
    EvalConfig(**raw)  # raises ValidationError if schema violated
    return raw
