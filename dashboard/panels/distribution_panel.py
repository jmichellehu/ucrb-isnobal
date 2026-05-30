"""Interactive spatial distribution panel."""
from __future__ import annotations
import numpy as np
import panel as pn
import hvplot.pandas  # noqa: F401
import xarray as xr

from isnobal_eval.analysis import compute_terrain_distribution

SNOW_VARS = ['thickness', 'specific_mass', 'snow_density', 'liquid_water',
             'temp_surf', 'temp_snowcover']
EM_VARS   = ['net_rad', 'sensible_heat', 'latent_heat', 'snowmelt', 'SWI', 'evaporation']
NS_VARS   = ['net_solar']
STRAT_OPTS = ['elevation', 'aspect', 'slope', 'veg_class']


class SpatialDistributionPanel:
    def __init__(self, snow_ds: xr.Dataset, em_ds: xr.Dataset,
                 net_solar_ds: xr.Dataset, topo_ds: xr.Dataset):
        self.snow_ds    = snow_ds
        self.em_ds      = em_ds
        self.net_solar_ds = net_solar_ds
        self.topo_ds    = topo_ds

        n_times = len(snow_ds.time)
        dates = [str(snow_ds.time.values[i])[:10] for i in range(n_times)]

        self.var_sel   = pn.widgets.Select(name='Variable', options=SNOW_VARS + EM_VARS + NS_VARS, value='thickness', width=180)
        self.strat_sel = pn.widgets.Select(name='Stratify by', options=STRAT_OPTS, value='elevation', width=180)
        self.time_sl   = pn.widgets.DiscreteSlider(name='Date', options=dict(zip(dates, range(n_times))), value=120, width=400)

    def _get_ds(self, variable):
        if variable in EM_VARS:
            return self.em_ds
        if variable in NS_VARS:
            return self.net_solar_ds
        return self.snow_ds

    def _plot(self, variable, stratify_by, time_idx):
        try:
            ds = self._get_ds(variable)
            df = compute_terrain_distribution(ds, variable, self.topo_ds, stratify_by=stratify_by)
            row = df.iloc[time_idx]
            return row.hvplot.bar(
                title=f'{variable} by {stratify_by} — {df.index[time_idx].date()}',
                ylabel=f'Mean {variable}', rot=45,
                responsive=True, height=350,
            )
        except Exception as e:
            return pn.pane.Str(f'Error: {e}')

    def panel(self) -> pn.viewable.Viewable:
        plot_pane = pn.bind(self._plot, self.var_sel, self.strat_sel, self.time_sl)
        controls = pn.Row(self.var_sel, self.strat_sel, self.time_sl)
        return pn.Column(controls, pn.panel(plot_pane, sizing_mode='stretch_width'))
