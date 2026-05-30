"""Interactive conditional time series panel."""
from __future__ import annotations
import numpy as np
import panel as pn
import hvplot.pandas  # noqa: F401
import xarray as xr

from isnobal_eval.analysis import build_terrain_mask, compute_masked_timeseries

VARS = ['thickness', 'specific_mass', 'snow_density',
        'net_solar', 'net_rad', 'snowmelt', 'SWI']


class ConditionalTimeseriesPanel:
    def __init__(self, snow_ds: xr.Dataset, em_ds: xr.Dataset,
                 net_solar_ds: xr.Dataset, topo_ds: xr.Dataset):
        self.snow_ds    = snow_ds
        self.em_ds      = em_ds
        self.net_solar_ds = net_solar_ds
        self.topo_ds    = topo_ds

        dem = topo_ds['dem'].values
        elev_min, elev_max = float(np.nanmin(dem)), float(np.nanmax(dem))

        self.var_sel  = pn.widgets.Select(name='Variable', options=VARS, value='thickness', width=180)
        self.elev_sl  = pn.widgets.RangeSlider(
            name='Elevation range (m)', start=elev_min, end=elev_max,
            value=(elev_min + 0.2 * (elev_max - elev_min),
                   elev_min + 0.8 * (elev_max - elev_min)),
            step=50.0, width=350,
        )
        self.asp_sl   = pn.widgets.RangeSlider(name='Aspect range (°)', start=0, end=360, value=(315, 45), step=5, width=350)
        self.slope_sl = pn.widgets.FloatSlider(name='Max slope (°)', start=0, end=80, value=45, step=5, width=250)

    def _get_ds(self, variable):
        em_vars = {'net_rad', 'sensible_heat', 'latent_heat', 'snowmelt', 'SWI', 'evaporation'}
        ns_vars = {'net_solar'}
        if variable in em_vars:
            return self.em_ds
        if variable in ns_vars:
            return self.net_solar_ds
        return self.snow_ds

    def _plot(self, variable, elev_range, asp_range, slope_max):
        try:
            mask = build_terrain_mask(
                self.topo_ds,
                elev_range=elev_range,
                aspect_range=asp_range,
                slope_max=slope_max,
            )
            n_px = int(mask.values.sum())
            if n_px == 0:
                return pn.pane.Str('No pixels match the current filter.')
            ds = self._get_ds(variable)
            ts = compute_masked_timeseries(ds, variable, mask)
            plot = ts['mean'].hvplot.line(label='mean', responsive=True, height=350,
                                          title=f'{variable} — {n_px} pixels')
            band = ts.hvplot.area(x='datetime', y='q25', y2='q75', alpha=0.25, label='IQR')
            return (plot * band).opts(legend_position='top_left')
        except Exception as e:
            return pn.pane.Str(f'Error: {e}')

    def panel(self) -> pn.viewable.Viewable:
        plot_pane = pn.bind(self._plot, self.var_sel, self.elev_sl, self.asp_sl, self.slope_sl)
        controls  = pn.Column(self.var_sel, self.elev_sl, self.asp_sl, self.slope_sl, width=380)
        return pn.Row(controls, pn.panel(plot_pane, sizing_mode='stretch_width'))
