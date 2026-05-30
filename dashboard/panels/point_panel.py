"""Interactive point temporal panel."""
from __future__ import annotations
import numpy as np
import panel as pn
import hvplot.xarray  # noqa: F401 — registers hvplot accessor
import hvplot.pandas  # noqa: F401
import xarray as xr

from isnobal_eval.analysis import extract_point_timeseries

ALL_VARS = [
    'thickness', 'specific_mass', 'snow_density', 'liquid_water',
    'net_rad', 'net_solar', 'net_LW', 'sensible_heat', 'latent_heat',
    'snowmelt', 'SWI', 'evaporation', 'cold_content',
]


class PointTemporalPanel:
    def __init__(self, snow_ds: xr.Dataset, em_ds: xr.Dataset, net_solar_ds: xr.Dataset):
        self.snow_ds    = snow_ds
        self.em_ds      = em_ds
        self.net_solar_ds = net_solar_ds

        cx = float(snow_ds.x.values[len(snow_ds.x) // 2])
        cy = float(snow_ds.y.values[len(snow_ds.y) // 2])

        self.x_input = pn.widgets.FloatInput(name='X (UTM)', value=cx, step=100.0, width=140)
        self.y_input = pn.widgets.FloatInput(name='Y (UTM)', value=cy, step=100.0, width=140)
        self.var_sel = pn.widgets.MultiSelect(
            name='Variables', width=200, size=8,
            options=ALL_VARS,
            value=['thickness', 'specific_mass', 'net_solar', 'snowmelt'],
        )

    def _plot(self, x, y, variables):
        try:
            df = extract_point_timeseries(
                self.snow_ds, self.em_ds, self.net_solar_ds, x, y,
                variables=variables or None,
            )
            available = [v for v in (variables or df.columns) if v in df.columns]
            if not available:
                return pn.pane.Str('No variables available.')
            return df[available].hvplot.line(
                x='datetime', responsive=True, height=350,
                title=f'Point ({x:.0f}, {y:.0f})',
            )
        except Exception as e:
            return pn.pane.Str(f'Error: {e}')

    def panel(self) -> pn.viewable.Viewable:
        plot_pane = pn.bind(self._plot, self.x_input, self.y_input, self.var_sel)
        controls = pn.Column(self.x_input, self.y_input, self.var_sel, width=220)
        return pn.Row(controls, pn.panel(plot_pane, sizing_mode='stretch_width'))
