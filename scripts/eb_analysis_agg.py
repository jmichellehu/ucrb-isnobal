#!/usr/bin/env python
"""
Aggregate energy balance analysis script.

Unlike eb_analysis.py (which saves 6 figures per SNOTEL site), this script saves
6 total figures per basin/WY, with all SNOTEL sites arranged in a subplot grid
with 3 columns.

Saved figures:
  fluxes_by_condition_{basin}_{wy}_all_sites.png
  daily_mean_fluxes_{basin}_{wy}_all_sites.png
  daily_melt_fluxes_{basin}_{wy}_all_sites.png
  daily_positive_melt_flux_contributions_{basin}_{wy}_all_sites.png
  daily_positive_melt_proportions_{basin}_{wy}_all_sites.png
  daily_positive_melt_snowmelt_contributions_{basin}_{wy}_all_sites.png

Usage:
  python eb_analysis_agg.py --basin yampa --wy 2022 --output-dir /path/to/figs
"""
# pylint: disable=wrong-import-position
import argparse
import json
import math
import re
import sys
from pathlib import Path, PurePath

import matplotlib
matplotlib.use('Agg')
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyproj
import seaborn as sns

sys.path.append('/uufs/chpc.utah.edu/common/home/u6058223/git_dirs/env/')
import helpers as h

WORKDIR = '/uufs/chpc.utah.edu/common/home/skiles-group2/model_runs_jmh/thp'
MINICONDA_DIR = '/uufs/chpc.utah.edu/common/home/u6058223/software/pkg/miniconda3'
CONDA_ENV = 'studio'

FULL_EB_COLS = ['precip_advected', 'snow_soil', 'sensible_heat',
                'latent_heat', 'net_solar', 'net_LW']

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

def setup_pyproj():
    """Set the pyproj data directory for the studio conda environment."""
    proj_json = h.fn_list(MINICONDA_DIR, f'envs/{CONDA_ENV}/conda-meta/proj-[0-9]*.json')[0]
    version = PurePath(proj_json).stem
    pyproj.datadir.set_data_dir(f'{MINICONDA_DIR}/pkgs/{version}/share/proj')

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_em_data(basin, wy):
    """Load per-variable iSnobal EM CSVs and return a (site, variable) multi-index DataFrame."""
    wy_dirs = h.fn_list(WORKDIR, f'{basin}*/wy{wy}/')
    em_csvs = h.flatten([h.fn_list(d, '*_em_*.csv') for d in wy_dirs])

    def _varname(fn):
        return fn.split('_em_')[1].split('_snotel')[0]

    em_df = pd.concat(
        {_varname(f): pd.read_csv(f, index_col=0, parse_dates=True) for f in em_csvs},
        axis=1,
    )
    em_df = em_df.swaplevel(0, 1, axis=1).sort_index(axis=1)
    em_df = em_df[em_df.index.time != pd.to_datetime('00:00').time()]
    return em_df, wy_dirs


def add_net_solar(em_df, wy_dirs):
    """Compute daily-mean net solar from radiation CSVs and add it to em_df per site."""
    csvs = h.fn_list(wy_dirs[0], '*csv')
    csvs = [c for c in csvs if '_em_' not in c and 'thickness' not in c]
    var_df = pd.concat(
        {c.split('_snotel')[0].split('/')[-1]: pd.read_csv(c, index_col=0, parse_dates=True)
         for c in csvs},
        axis=1,
    )

    net_solar_key = next(
        (k for k in var_df.columns.get_level_values(0).unique() if k.endswith('net_solar')),
        None,
    )
    if net_solar_key is None:
        raise ValueError(f'No net_solar variable found in {wy_dirs[0]}')

    daily_net_solar = var_df[net_solar_key].resample('D').mean()
    em_dates = pd.to_datetime(em_df.index.date)
    daily_net_solar = daily_net_solar.reindex(em_dates)
    daily_net_solar.index = em_df.index

    for site in em_df.columns.get_level_values(0).unique():
        if site in daily_net_solar.columns:
            em_df.loc[:, (site, 'net_solar')] = daily_net_solar[site].values
        else:
            em_df.loc[:, (site, 'net_solar')] = np.nan

    return em_df

# ---------------------------------------------------------------------------
# Condition building
# ---------------------------------------------------------------------------

def build_conditions(sub_df, both_conditions_dates, tmax_freezing_periods, cold_snap_period):
    """Build boolean masks for each analysis condition."""
    conditions = {'melt': sub_df.index.isin(both_conditions_dates)}

    if tmax_freezing_periods:
        mask = np.zeros(len(sub_df), dtype=bool)
        for start, end in tmax_freezing_periods:
            mask |= (sub_df.index >= pd.to_datetime(start)) & (sub_df.index <= pd.to_datetime(end))
        conditions['tmax_freezing'] = mask

    if cold_snap_period:
        conditions['cold_snap'] = (
            (sub_df.index >= pd.to_datetime(cold_snap_period[0])) &
            (sub_df.index <= pd.to_datetime(cold_snap_period[1]))
        )

    exclude = np.zeros(len(sub_df), dtype=bool)
    for val in conditions.values():
        exclude |= val
    conditions['rest_of_season'] = ~exclude
    return conditions

# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _save(fig, path):
    """Save figure and close pyplot state."""
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved: {path}')


def _stacked_bar_dt(ax, df, cols, width=pd.Timedelta(hours=20)):
    """Stack positive and negative values separately on datetime x positions."""
    color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
    bottoms_pos = np.zeros(len(df))
    bottoms_neg = np.zeros(len(df))
    colors = []
    for i, col in enumerate(cols):
        color = color_cycle[i % len(color_cycle)]
        colors.append(color)
        vals = df[col].values.astype(float)
        bottom = np.where(vals >= 0, bottoms_pos, bottoms_neg)
        ax.bar(df.index, vals, width=width, bottom=bottom, color=color, label=col)
        bottoms_pos += np.where(vals >= 0, vals, 0)
        bottoms_neg += np.where(vals < 0, vals, 0)
    return colors


def _apply_dt_xaxis(ax):
    """Apply month ticks and month/day formatter on each axis."""
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')


def _make_grid(n_sites, n_cols=3, panel_width=5.2, panel_height=3.7):
    """Create a site subplot grid and return (fig, axes_flat)."""
    n_rows = max(1, math.ceil(n_sites / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * panel_width, n_rows * panel_height))
    if isinstance(axes, np.ndarray):
        axes = axes.ravel()
    else:
        axes = np.array([axes])
    return fig, axes


def _finalize_unused_axes(axes, used):
    """Hide any unused axes in the grid."""
    for idx in range(used, len(axes)):
        axes[idx].set_visible(False)


def _global_date_window(wy):
    """Return full WY x-axis bounds (Oct 1 to Sep 30) for aggregate plots."""
    return pd.Timestamp(f'{wy - 1}-10-01'), pd.Timestamp(f'{wy}-09-30')


def _resolve_md(md_str, wy):
    """Resolve MM-DD or MMDD to a WY-aware timestamp (Oct-Dec -> wy-1, Jan-Sep -> wy)."""
    md_str = md_str.strip()
    if re.fullmatch(r'\d{2}-\d{2}', md_str):
        month, day = map(int, md_str.split('-'))
    elif re.fullmatch(r'\d{4}', md_str):
        month = int(md_str[:2])
        day = int(md_str[2:])
    else:
        raise ValueError(f"Invalid month-day '{md_str}'. Expected MM-DD or MMDD.")

    year = wy - 1 if month >= 10 else wy
    try:
        return pd.Timestamp(year=year, month=month, day=day)
    except ValueError as exc:
        raise ValueError(f"Invalid month-day '{md_str}': {exc}") from exc


def _plot_suffix(plot_window, wy):
    """Build a filename suffix for non-default WY plot windows."""
    wy_start = pd.Timestamp(f'{wy - 1}-10-01')
    wy_end = pd.Timestamp(f'{wy}-09-30')

    if plot_window is None:
        return ''

    start_override, end_override = plot_window
    effective_start = start_override if start_override is not None else wy_start
    effective_end = end_override if end_override is not None else wy_end

    if effective_start == wy_start and effective_end == wy_end:
        return ''

    return f'_{effective_start:%Y%m%d}_{effective_end:%Y%m%d}'


def _capture_legend(ax, legend_handles, legend_labels):
    """Capture legend handles/labels once from an axis and return cache tuple."""
    if legend_handles is None:
        handles, labels_l = ax.get_legend_handles_labels()
        if handles:
            return handles, labels_l
    return legend_handles, legend_labels


def _add_side_legend(fig, handles, labels, title):
    """Add a shared figure legend to the right side if handles are present."""
    if handles:
        fig.legend(handles, labels, loc='center left', bbox_to_anchor=(1.01, 0.5), title=title)


def _collect_site_data(em_df, sites, tmax_freezing_periods, cold_snap_period):
    """Build all per-site analysis products once for aggregate plotting."""
    data = {}
    for site in sites:
        sub_df = em_df[site].copy()
        sub_df['net_LW'] = sub_df['net_rad'] - sub_df['net_solar']

        both_conditions_dates = (
            sub_df.index[sub_df['sum_EB'] > 0]
            .intersection(sub_df.index[sub_df['cold_content'] == 0])
        )
        conditions = build_conditions(sub_df, both_conditions_dates,
                                      tmax_freezing_periods, cold_snap_period)

        melt_df = sub_df.loc[both_conditions_dates].copy()
        mec = melt_df[FULL_EB_COLS].clip(lower=0).copy()
        mec['total_positive_flux'] = mec.sum(axis=1)
        for col in FULL_EB_COLS:
            mec[f'{col}_pct'] = np.where(mec['total_positive_flux'] > 0,
                                         mec[col] / mec['total_positive_flux'] * 100,
                                         np.nan)
            mec[f'{col}_melt_contribution'] = mec[f'{col}_pct'] / 100 * sub_df.loc[mec.index, 'snowmelt']

        data[site] = {
            'sub_df': sub_df,
            'conditions': conditions,
            'melt_df': melt_df,
            'mec': mec,
        }
    return data


# ---------------------------------------------------------------------------
# Figure functions
# ---------------------------------------------------------------------------

def fig_boxplot_all(site_data, output_dir, basin, wy, plot_window=None, file_suffix=''):
    """Aggregate figure: flux distributions by condition, one subplot per site."""
    sites = list(site_data.keys())
    fig, axes = _make_grid(len(sites), n_cols=3, panel_width=4.8, panel_height=3.8)

    has_tmax = any('tmax_freezing' in site_data[s]['conditions'] for s in sites)
    has_cold = any('cold_snap' in site_data[s]['conditions'] for s in sites)
    cond_order = []
    if has_tmax:
        cond_order.append('tmax_freezing')
    if has_cold:
        cond_order.append('cold_snap')
    cond_order.extend(['melt', 'rest_of_season'])
    legend_handles = None
    legend_labels = None

    for idx, site in enumerate(sites):
        ax = axes[idx]
        sub_df_full = site_data[site]['sub_df']
        sub_df = sub_df_full
        conditions = site_data[site]['conditions']

        if plot_window is not None:
            start_override, end_override = plot_window
            if start_override is not None:
                sub_df = sub_df.loc[sub_df.index >= start_override]
            if end_override is not None:
                sub_df = sub_df.loc[sub_df.index <= end_override]

        if len(sub_df) == 0:
            ax.set_title(f'{site} (no data in window)')
            ax.set_xlabel('')
            ax.set_ylabel('W m⁻²')
            continue

        labels = pd.Series('rest_of_season', index=sub_df_full.index)
        for cond in ['tmax_freezing', 'cold_snap', 'melt']:
            if cond in conditions:
                labels[conditions[cond]] = cond
        labels = labels.loc[sub_df.index]

        melted = sub_df[FULL_EB_COLS].copy()
        melted['condition'] = labels
        melted = melted.melt(id_vars='condition', value_vars=FULL_EB_COLS,
                             var_name='flux', value_name='value')

        sns.boxplot(
            data=melted,
            x='condition',
            y='value',
            hue='flux',
            order=cond_order,
            ax=ax,
            width=0.6,
            gap=0.15,
            showfliers=False,
        )
        ax.axhline(0, color='grey', linewidth=0.5, linestyle='--')
        ax.set_ylim(-250, 250)
        ax.set_ylabel('W m⁻²')
        ax.set_xlabel('')
        ax.set_title(site)
        ax.tick_params(axis='x', rotation=30)

        legend_handles, legend_labels = _capture_legend(ax, legend_handles, legend_labels)
        ax.legend_.remove()

    _finalize_unused_axes(axes, len(sites))
    _add_side_legend(fig, legend_handles, legend_labels, 'Flux')
    fig.suptitle(f'Energy Fluxes by Condition — {basin} WY{wy}', y=1.01)
    fig.tight_layout()
    _save(fig, output_dir / f'fluxes_by_condition_{basin}_{wy}_all_sites{file_suffix}.png')


def fig_daily_mean_all(site_data, output_dir, basin, wy, plot_window=None, file_suffix=''):
    """Aggregate figure: full-season daily mean stacked fluxes, one subplot per site."""
    sites = list(site_data.keys())
    fig, axes = _make_grid(len(sites), n_cols=3)

    global_start, global_end = (pd.Timestamp(f'{wy-1}-10-01'), pd.Timestamp(f'{wy}-09-30'))
    if plot_window is not None:
        start_override, end_override = plot_window
        global_start = start_override if start_override is not None else global_start
        global_end = end_override if end_override is not None else global_end

    legend_handles = None
    legend_labels = None
    for idx, site in enumerate(sites):
        ax = axes[idx]
        sub_df = site_data[site]['sub_df']
        _stacked_bar_dt(ax, sub_df, FULL_EB_COLS)
        ax.bar(sub_df.index, sub_df['sum_EB'], width=pd.Timedelta(hours=20),
               fill=False, hatch='///', edgecolor='black', linewidth=0.5, label='sum_EB')
        ax.axhline(0, color='grey', linewidth=0.5, linestyle='-')
        _apply_dt_xaxis(ax)
        ax.set_xlim(global_start, global_end)
        ax.set_ylim(-250, 250)
        ax.set_ylabel('W m⁻²')
        ax.set_title(site)

        legend_handles, legend_labels = _capture_legend(ax, legend_handles, legend_labels)

    _finalize_unused_axes(axes, len(sites))
    _add_side_legend(fig, legend_handles, legend_labels, 'Flux')
    fig.suptitle(f'Daily Mean Energy Fluxes — {basin} WY{wy}', y=1.01)
    fig.tight_layout()
    _save(fig, output_dir / f'daily_mean_fluxes_{basin}_{wy}_all_sites{file_suffix}.png')


def fig_daily_melt_all(site_data, output_dir, basin, wy, plot_window=None, file_suffix=''):
    """Aggregate figure: daily melt-period stacked fluxes, one subplot per site."""
    sites = list(site_data.keys())
    fig, axes = _make_grid(len(sites), n_cols=3)

    global_start, global_end = _global_date_window(wy)
    if plot_window is not None:
        start_override, end_override = plot_window
        global_start = start_override if start_override is not None else global_start
        global_end = end_override if end_override is not None else global_end

    legend_handles = None
    legend_labels = None
    for idx, site in enumerate(sites):
        ax = axes[idx]
        melt_df = site_data[site]['melt_df']
        if len(melt_df) > 0:
            _stacked_bar_dt(ax, melt_df, FULL_EB_COLS)
            ax.bar(melt_df.index, melt_df['sum_EB'], width=pd.Timedelta(hours=20),
                   fill=False, hatch='///', edgecolor='black', linewidth=0.5, label='sum_EB')
        ax.axhline(0, color='grey', linewidth=0.5, linestyle='-')
        _apply_dt_xaxis(ax)
        ax.set_xlim(global_start, global_end)
        ax.set_ylim(-250, 250)
        ax.set_ylabel('W m⁻²')
        ax.set_title(site)

        legend_handles, legend_labels = _capture_legend(ax, legend_handles, legend_labels)

    _finalize_unused_axes(axes, len(sites))
    _add_side_legend(fig, legend_handles, legend_labels, 'Flux')
    fig.suptitle(f'Daily Energy Fluxes During Melt (CC=0, sum_EB>0) — {basin} WY{wy}', y=1.01)
    fig.tight_layout()
    _save(fig, output_dir / f'daily_melt_fluxes_{basin}_{wy}_all_sites{file_suffix}.png')


def fig_positive_melt_contributions_all(site_data, output_dir, basin, wy, plot_window=None, file_suffix=''):
    """Aggregate figure: positive-only melt flux contributions (W m⁻²), one subplot per site."""
    sites = list(site_data.keys())
    fig, axes = _make_grid(len(sites), n_cols=3)

    global_start, global_end = _global_date_window(wy)
    if plot_window is not None:
        start_override, end_override = plot_window
        global_start = start_override if start_override is not None else global_start
        global_end = end_override if end_override is not None else global_end

    legend_handles = None
    legend_labels = None
    for idx, site in enumerate(sites):
        ax = axes[idx]
        mec = site_data[site]['mec']
        if len(mec) > 0:
            _stacked_bar_dt(ax, mec, FULL_EB_COLS)
        _apply_dt_xaxis(ax)
        ax.set_xlim(global_start, global_end)
        ax.set_ylim(0, 250)
        ax.set_ylabel('W m⁻²')
        ax.set_title(site)

        legend_handles, legend_labels = _capture_legend(ax, legend_handles, legend_labels)

    _finalize_unused_axes(axes, len(sites))
    _add_side_legend(fig, legend_handles, legend_labels, 'Positive Flux Contribution')
    fig.suptitle(f'Daily Positive Melt Flux Contributions — {basin} WY{wy}', y=1.01)
    fig.tight_layout()
    _save(fig, output_dir / f'daily_positive_melt_flux_contributions_{basin}_{wy}_all_sites{file_suffix}.png')


def fig_positive_melt_proportions_all(site_data, output_dir, basin, wy, file_suffix=''):
    """Aggregate figure: positive melt flux proportions (%), one subplot per site."""
    sites = list(site_data.keys())
    fig, axes = _make_grid(len(sites), n_cols=3)

    pct_cols = [f'{c}_pct' for c in FULL_EB_COLS]
    global_start, global_end = _global_date_window(wy)

    legend_handles = None
    legend_labels = None
    for idx, site in enumerate(sites):
        ax = axes[idx]
        mec = site_data[site]['mec']
        if len(mec) > 0:
            _stacked_bar_dt(ax, mec, pct_cols)
        _apply_dt_xaxis(ax)
        ax.set_xlim(global_start, global_end)
        ax.set_ylim(0, 100)
        ax.set_ylabel('%')
        ax.set_title(site)

        legend_handles, legend_labels = _capture_legend(ax, legend_handles, legend_labels)

    _finalize_unused_axes(axes, len(sites))
    _add_side_legend(fig, legend_handles, legend_labels, 'Positive Flux Proportion')
    fig.suptitle(f'Daily Positive Melt Flux Proportions — {basin} WY{wy}', y=1.01)
    fig.tight_layout()
    _save(fig, output_dir / f'daily_positive_melt_proportions_{basin}_{wy}_all_sites{file_suffix}.png')


def fig_snowmelt_contributions_all(site_data, output_dir, basin, wy, file_suffix=''):
    """Aggregate figure: snowmelt contributions (kg m⁻²), one subplot per site."""
    sites = list(site_data.keys())
    fig, axes = _make_grid(len(sites), n_cols=3)

    contrib_cols = [f'{c}_melt_contribution' for c in FULL_EB_COLS]
    global_start, global_end = _global_date_window(wy)

    global_ymax = 0
    for site in sites:
        mec = site_data[site]['mec']
        if len(mec) > 0:
            local_max = mec[contrib_cols].sum(axis=1).max()
            global_ymax = max(global_ymax, float(local_max))
    if global_ymax <= 0:
        global_ymax = 50
    else:
        global_ymax = math.ceil(global_ymax / 5.0) * 5.0

    legend_handles = None
    legend_labels = None
    for idx, site in enumerate(sites):
        ax = axes[idx]
        mec = site_data[site]['mec']
        if len(mec) > 0:
            _stacked_bar_dt(ax, mec, contrib_cols)
            season_total = mec[contrib_cols].sum().sum()
            ax.text(0.02, 0.93, f'Season total: {season_total:.1f} kg m⁻²',
                    transform=ax.transAxes, fontsize=8,
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='grey'))
        _apply_dt_xaxis(ax)
        ax.set_xlim(global_start, global_end)
        ax.set_ylim(0, global_ymax)
        ax.set_ylabel('kg m⁻²')
        ax.set_title(site)

        legend_handles, legend_labels = _capture_legend(ax, legend_handles, legend_labels)

    _finalize_unused_axes(axes, len(sites))
    _add_side_legend(fig, legend_handles, legend_labels, 'Snowmelt Contribution')
    fig.suptitle(f'Daily Positive Melt Snowmelt Contributions — {basin} WY{wy}', y=1.01)
    fig.tight_layout()
    _save(fig, output_dir / f'daily_positive_melt_snowmelt_contributions_{basin}_{wy}_all_sites{file_suffix}.png')

def main():
    """Main function to run the aggregate energy balance analysis and plotting."""
    parser = argparse.ArgumentParser(description='Aggregate energy balance analysis for iSnobal THP output')
    parser.add_argument('--basin', required=True, help='Basin name (e.g. yampa)')
    parser.add_argument('--wy', required=True, type=int, help='Water year (e.g. 2022)')
    parser.add_argument('--output-dir', default='.', help='Directory to save figures')
    parser.add_argument(
        '--tmax-freezing-periods', default=None,
        help='JSON list of [start, end] pairs, e.g. \'[["2021-12-08","2021-12-10"],...]\''
    )
    parser.add_argument(
        '--cold-snap-period', default=None,
        help='Comma-separated start,end dates, e.g. 2022-02-02,2022-02-03'
    )
    parser.add_argument(
        '--plot-start', default=None,
        help='Optional plot-window start in MM-DD or MMDD (WY-aware year assignment)'
    )
    parser.add_argument(
        '--plot-end', default=None,
        help='Optional plot-window end in MM-DD or MMDD (WY-aware year assignment)'
    )
    args = parser.parse_args()

    basin = args.basin
    wy = args.wy
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    basin_output_dir = output_dir / basin
    basin_output_dir.mkdir(parents=True, exist_ok=True)

    tmax_freezing_periods = json.loads(args.tmax_freezing_periods) if args.tmax_freezing_periods else None
    cold_snap_period = args.cold_snap_period.split(',') if args.cold_snap_period else None

    plot_start = _resolve_md(args.plot_start, wy) if args.plot_start else None
    plot_end = _resolve_md(args.plot_end, wy) if args.plot_end else None
    plot_window = (plot_start, plot_end) if (plot_start is not None or plot_end is not None) else None
    if plot_start is not None and plot_end is not None and plot_start > plot_end:
        raise ValueError(
            f'Invalid plot window: start {args.plot_start} resolves after end {args.plot_end} for WY {wy}'
        )
    file_suffix = _plot_suffix(plot_window, wy)
    if file_suffix:
        output_dir = basin_output_dir / file_suffix
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        output_dir = basin_output_dir

    setup_pyproj()

    em_df, wy_dirs = load_em_data(basin, wy)
    em_df = add_net_solar(em_df, wy_dirs)

    sites = list(em_df.columns.get_level_values(0).unique())
    print(f'Found {len(sites)} site(s): {sites}')

    site_data = _collect_site_data(em_df, sites, tmax_freezing_periods, cold_snap_period)

    fig_boxplot_all(site_data, output_dir, basin, wy, plot_window=plot_window, file_suffix=file_suffix)
    fig_daily_mean_all(site_data, output_dir, basin, wy, plot_window=plot_window, file_suffix=file_suffix)
    fig_daily_melt_all(site_data, output_dir, basin, wy, plot_window=plot_window, file_suffix=file_suffix)
    fig_positive_melt_contributions_all(
        site_data, output_dir, basin, wy, plot_window=plot_window, file_suffix=file_suffix
    )
    fig_positive_melt_proportions_all(site_data, output_dir, basin, wy, file_suffix=file_suffix)
    fig_snowmelt_contributions_all(site_data, output_dir, basin, wy, file_suffix=file_suffix)


if __name__ == '__main__':
    main()
