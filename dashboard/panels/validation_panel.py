"""Interactive SNOTEL validation panel."""
from __future__ import annotations
import numpy as np
import pandas as pd
import panel as pn
import hvplot.pandas  # noqa: F401
import xarray as xr

from isnobal_eval.analysis import compare_snotel, compute_metrics


class ValidationPanel:
    def __init__(self, snow_ds: xr.Dataset, snotel_df: pd.DataFrame):
        self.snow_ds  = snow_ds
        self.snotel_df = snotel_df

        sites = [] if snotel_df.empty else sorted(snotel_df['site_name'].unique().tolist())
        self.var_sel  = pn.widgets.Select(name='Variable', options=['thickness', 'specific_mass'], value='thickness', width=180)
        self.site_sel = pn.widgets.MultiSelect(name='Sites', options=sites, value=sites[:4], size=8, width=220)

    def _update(self, variable, site_ids):
        if self.snotel_df.empty or not site_ids:
            empty = pn.pane.Str('No SNOTEL data available.')
            return pn.Row(empty)

        try:
            comp = compare_snotel(self.snow_ds, self.snotel_df, variable, site_ids=site_ids)
            if comp.empty:
                return pn.pane.Str('No paired data for selected sites/variable.')

            # Time series overlay
            plots = []
            for site, grp in comp.groupby('site_id'):
                grp = grp.set_index('date').sort_index()
                p = grp[['modeled', 'observed']].hvplot.line(
                    title=site, height=200, responsive=True,
                    color=['steelblue', 'darkorange'], line_dash=['solid', 'dashed'],
                )
                plots.append(p)
            ts_panel = pn.Column(*plots, sizing_mode='stretch_width')

            # Metrics table
            rows = []
            for site, grp in comp.groupby('site_id'):
                m = compute_metrics(grp['modeled'].values, grp['observed'].values)
                m['site'] = site
                rows.append(m)
            metrics_df = pd.DataFrame(rows).set_index('site')[['n', 'r', 'rmse', 'mae', 'kge']].round(3)
            table = pn.widgets.Tabulator(metrics_df, sizing_mode='stretch_width', height=250)

            return pn.Column(ts_panel, pn.pane.Markdown('**Metrics**'), table)
        except Exception as e:
            return pn.pane.Str(f'Error: {e}')

    def panel(self) -> pn.viewable.Viewable:
        content = pn.bind(self._update, self.var_sel, self.site_sel)
        controls = pn.Column(self.var_sel, self.site_sel, width=240)
        return pn.Row(controls, pn.panel(content, sizing_mode='stretch_width'))
