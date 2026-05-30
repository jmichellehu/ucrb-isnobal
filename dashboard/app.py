"""iSnobal Evaluation Dashboard.

Launch:
    conda activate studio
    cd /uufs/.../ucrb-isnobal
    pip install -e .
    ISNOBAL_EVAL_CONFIG=eval/config.yml python dashboard/app.py
"""
import os
import panel as pn

from isnobal_eval import (
    load_config, load_snow, load_em, load_net_solar,
    load_topo, compute_aspect, load_snotel,
)
from dashboard.panels import (
    PointTemporalPanel,
    SpatialDistributionPanel,
    ConditionalTimeseriesPanel,
    ValidationPanel,
)

pn.extension('tabulator', sizing_mode='stretch_width')

# Load config and data once at startup
_config_path = os.environ.get('ISNOBAL_EVAL_CONFIG', 'eval/config.yml')
cfg      = load_config(_config_path)
snow_ds  = load_snow(cfg)
em_ds    = load_em(cfg)
ns_ds    = load_net_solar(cfg)
topo_ds  = compute_aspect(load_topo(cfg))
snotel_df = load_snotel(cfg)

header = pn.pane.Markdown(
    f"# iSnobal Evaluation — **{cfg['basin'].upper()}** WY{cfg['water_year']}",
    sizing_mode='stretch_width',
)

tabs = pn.Tabs(
    ('Point Temporal',          PointTemporalPanel(snow_ds, em_ds, ns_ds).panel()),
    ('Spatial Distribution',    SpatialDistributionPanel(snow_ds, em_ds, ns_ds, topo_ds).panel()),
    ('Conditional Time Series', ConditionalTimeseriesPanel(snow_ds, em_ds, ns_ds, topo_ds).panel()),
    ('Validation',              ValidationPanel(snow_ds, snotel_df).panel()),
    dynamic=True,
)

app = pn.Column(header, tabs, sizing_mode='stretch_width')

if __name__ == '__main__':
    pn.serve(app, port=5006, show=True, title='iSnobal Eval')
