#!/usr/bin/env python
"""
Energy balance analysis script — converts eb_analysis.ipynb to executable form.

Saves six figures per basin/WY/site:
  fluxes_by_condition_{basin}_{wy}_{site}_boxplot.png
  daily_mean_fluxes_{basin}_{wy}_{site}_stacked.png
  daily_melt_fluxes_{basin}_{wy}_{site}_stacked.png
  daily_positive_melt_flux_contributions_{basin}_{wy}_{site}_stacked.png
  daily_positive_melt_proportions_{basin}_{wy}_{site}_stacked.png
  daily_positive_melt_snowmelt_contributions_{basin}_{wy}_{site}_sub.png

Usage:
  python eb_analysis.py --basin animas --wy 2022 --output-dir /path/to/figs
  python eb_analysis.py --basin animas --wy 2022 \\
      --tmax-freezing-periods '[["2021-12-08","2021-12-10"],["2021-12-24","2022-01-01"]]' \\
      --cold-snap-period '2022-02-02,2022-02-03'
"""

# pylint: disable=wrong-import-position
import argparse
import json
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

FLUX_COLS = ['precip_advected', 'snow_soil', 'sensible_heat', 'latent_heat', 'net_rad']
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


def slugify(name):
    """Replace spaces and non-alphanumeric characters with underscores for safe filenames."""
    return re.sub(r'[^\w]+', '_', name).strip('_')


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_em_data(basin, WY):
    """Load per-variable iSnobal EM CSVs and return a (site, variable) multi-index DataFrame."""
    wy_dirs = h.fn_list(WORKDIR, f'{basin}*/wy{WY}/')
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
    em_dates = pd.DatetimeIndex(em_df.index).normalize()
    daily_net_solar = daily_net_solar.reindex(em_dates)
    daily_net_solar.index = em_df.index

    for site in em_df.columns.get_level_values(0).unique():
        if site in daily_net_solar.columns:
            em_df.loc[:, (site, 'net_solar')] = daily_net_solar[site].values
        else:
            em_df.loc[:, (site, 'net_solar')] = np.nan
    # daily_net_solar.index = em_df.index

    # for site in em_df.columns.get_level_values(0).unique():
    #     em_df.loc[:, (site, 'net_solar')] = daily_net_solar.loc[:, site]

    return em_df


# ---------------------------------------------------------------------------
# Condition building
# ---------------------------------------------------------------------------

def build_conditions(sub_df, both_conditions_dates, tmax_freezing_periods, cold_snap_period):
    """
    Build boolean masks for each analysis condition.

    Always includes 'melt' (sum_EB>0 and CC=0) and 'rest_of_season'.
    Optionally adds 'tmax_freezing' and 'cold_snap' if date ranges are provided.
    """
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
    for v in conditions.values():
        exclude |= v
    conditions['rest_of_season'] = ~exclude
    return conditions


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _bar_xtick_labels(index):
    """Return month-day tick labels, non-empty only on the 1st of each month."""
    return [t.strftime('%b %d') if t.day == 1 else '' for t in index]


def _n_plot_bars(sub_df, WY):
    """Number of daily bars to show (Oct 1 – May 1 of WY)."""
    return int((sub_df.index <= pd.Timestamp(f'{WY}-05-01')).sum())


def _save(fig_or_ax, path):
    """Save the current figure and close all pyplot state."""
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close('all')
    print(f'Saved: {path}')


def _stacked_bar_dt(ax, df, cols, width=pd.Timedelta(hours=20)):
    """
    Plot a stacked bar chart on ax using datetime x-positions with correct temporal spacing.

    Positive and negative values are stacked separately so mixed-sign columns display correctly.
    Returns the color list used, in column order, for legend construction.
    """
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
    """Apply month-locator ticks and '%b %d' labels to a datetime x-axis."""
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')


# ---------------------------------------------------------------------------
# Figure functions
# ---------------------------------------------------------------------------

def fig_boxplot_by_condition(sub_df, conditions, output_dir, basin, WY, site):
    """Seaborn boxplot of full EB flux distributions grouped by season condition."""
    cond_order = [c for c in ['tmax_freezing', 'cold_snap', 'melt', 'rest_of_season']
                  if c in conditions]
    labels = pd.Series('rest_of_season', index=sub_df.index)
    for cond in ['tmax_freezing', 'cold_snap', 'melt']:
        if cond in conditions:
            labels[conditions[cond]] = cond

    melted = sub_df[FULL_EB_COLS].copy()
    melted['condition'] = labels
    melted = melted.melt(id_vars='condition', value_vars=FULL_EB_COLS,
                         var_name='flux', value_name='value')

    fig, ax = plt.subplots(figsize=(8, 4))
    sns.boxplot(data=melted, x='condition', y='value', hue='flux',
                order=cond_order, ax=ax, width=0.6, gap=0.2, showfliers=False)
    ax.axhline(0, color='grey', linewidth=0.5, linestyle='--')
    ax.set_ylim(-250, 250)
    ax.set_ylabel('W m⁻²')
    ax.set_title(f'{site} — {basin} WY{WY} — Energy Fluxes by Condition')
    ax.legend(title='Flux', bbox_to_anchor=(1.01, 1), loc='upper left')
    plt.tight_layout()
    _save(fig, output_dir / f'fluxes_by_condition_{basin}_{WY}_{slugify(site)}_boxplot.png')


def fig_daily_mean_stacked(sub_df, output_dir, basin, WY, site):
    """Stacked bar plot of daily mean energy fluxes across the full season."""
    ax = sub_df.plot(y=FULL_EB_COLS, kind='bar', stacked=True, figsize=(18, 6), legend=False)
    sub_df.plot(y='sum_EB', kind='bar', ax=ax, hatch='///', color='none',
                edgecolor='black', linewidth=1, label='sum_EB', legend=True)
    ax.axhline(0, color='grey', linewidth=0.5, linestyle='-')
    ax.set_xticks(range(len(sub_df)), _bar_xtick_labels(sub_df.index))
    ax.set_xlim(-1, _n_plot_bars(sub_df, WY))
    ax.set_ylim(-250, 250)
    ax.set_ylabel('W m⁻²')
    ax.set_title(f'{site} — {basin} WY{WY} — Daily Mean Energy Fluxes')
    plt.legend()
    plt.tight_layout()
    _save(ax, output_dir / f'daily_mean_fluxes_{basin}_{WY}_{slugify(site)}_stacked.png')


def fig_daily_melt_stacked(melt_df, output_dir, basin, WY, site, datetime_xaxis=False):
    """Stacked bar plot of daily energy fluxes restricted to active melt days."""
    fig, ax = plt.subplots(figsize=(18, 6))
    if datetime_xaxis:
        _stacked_bar_dt(ax, melt_df, FULL_EB_COLS)
        ax.bar(melt_df.index, melt_df['sum_EB'], width=pd.Timedelta(hours=20),
               fill=False, hatch='///', edgecolor='black', linewidth=0.5, label='sum_EB')
        _apply_dt_xaxis(ax)
    else:
        melt_df.plot(y=FULL_EB_COLS, kind='bar', stacked=True, ax=ax, legend=False)
        melt_df.plot(y='sum_EB', kind='bar', ax=ax, hatch='///', color='none',
                     edgecolor='black', linewidth=0.5, label='sum_EB', legend=True)
        ax.set_xticks(range(len(melt_df)), [t.strftime('%b %d') for t in melt_df.index])
    ax.axhline(0, color='grey', linewidth=0.5, linestyle='-')
    ax.set_ylim(-250, 250)
    ax.set_ylabel('W m⁻²')
    ax.set_title(f'{site} — {basin} WY{WY} — Daily Energy Fluxes During Melt (CC=0, sum_EB>0)')
    ax.legend()
    plt.tight_layout()
    _save(fig, output_dir / f'daily_melt_fluxes_{basin}_{WY}_{slugify(site)}_stacked.png')


def fig_positive_melt_contributions(mec, output_dir, basin, WY, site, datetime_xaxis=False):
    """Stacked bar of positive-only flux contributions during melt, annotated with percentages."""
    fig, ax = plt.subplots(figsize=(12, 6))
    if datetime_xaxis:
        _stacked_bar_dt(ax, mec, FULL_EB_COLS)
        _apply_dt_xaxis(ax)
        x_vals = mec.index
    else:
        mec.plot(y=FULL_EB_COLS, kind='bar', stacked=True, ax=ax, legend=False)
        ax.set_xticks(range(len(mec)), [t.strftime('%b %d') for t in mec.index])
        x_vals = range(len(mec))
    ax.set_ylabel('Positive Flux Melt Contributions (W m⁻²)')
    for x_pos, dt in zip(x_vals, mec.index):
        row = mec.loc[dt]
        cumsum = row[FULL_EB_COLS].cumsum()
        for col in FULL_EB_COLS:
            val = row[col]
            pct = row[f'{col}_pct']
            if val > 5:
                ax.text(x_pos, cumsum[col] - val / 2, f'{pct:.0f}%',
                        ha='center', va='center', fontsize=8, rotation=90,
                        fontweight='bold', color='white')
    ax.set_ylim(0, 250)
    ax.set_title(f'{site} — {basin} WY{WY}')
    ax.legend(bbox_to_anchor=(1.01, 1), title='Positive Flux Melt Contributions')
    plt.tight_layout()
    _save(fig, output_dir / f'daily_positive_melt_flux_contributions_{basin}_{WY}_{slugify(site)}_stacked.png')


def fig_positive_melt_proportions(mec, output_dir, basin, WY, site, datetime_xaxis=False):
    """Stacked bar of each flux's percentage share of total positive melt energy."""
    pct_cols = [c for c in mec.columns if c.endswith('_pct')]
    fig, ax = plt.subplots(figsize=(12, 6))
    if datetime_xaxis:
        _stacked_bar_dt(ax, mec, pct_cols)
        _apply_dt_xaxis(ax)
        x_vals = mec.index
    else:
        mec.plot(y=pct_cols, kind='bar', stacked=True, ax=ax, legend=False)
        ax.set_xticks(range(len(mec)), [t.strftime('%b %d') for t in mec.index])
        x_vals = range(len(mec))
    ax.set_ylabel('Percent Contribution to Positive Flux (%)')
    for x_pos, dt in zip(x_vals, mec.index):
        row = mec.loc[dt]
        cumsum = row[pct_cols].cumsum()
        for col in pct_cols:
            val = row[col]
            if val > 5:
                ax.text(x_pos, cumsum[col] - val / 2, f'{val:.0f}%',
                        ha='center', va='center', fontsize=8, rotation=90,
                        fontweight='bold', color='white')
    ax.set_ylim(0, 100)
    ax.set_title(f'{site} — {basin} WY{WY}')
    ax.legend(bbox_to_anchor=(1.01, 1), title='Positive Flux Melt Contributions')
    plt.tight_layout()
    _save(fig, output_dir / f'daily_positive_melt_proportions_{basin}_{WY}_{slugify(site)}_stacked.png')


def fig_snowmelt_contributions(mec, sub_df, output_dir, basin, WY, site):
    """Subplot per flux of its fractional snowmelt contribution (kg m⁻²), with season totals."""
    pct_cols = [c for c in mec.columns if c.endswith('_pct')]
    for col in pct_cols:
        mec[f'{col}_melt_contribution'] = (
            mec[col] / 100 * sub_df.loc[mec.index, 'snowmelt']
        )
    contrib_cols = [c for c in mec.columns if c.endswith('_melt_contribution')]

    fig, axes = plt.subplots(len(contrib_cols), 1,
                             figsize=(12, len(contrib_cols)), sharex=True, sharey=True)
    xtick_labels = [t.strftime('%b %d') for t in mec.index]
    for idx, col in enumerate(contrib_cols):
        ax = axes[idx]
        mec.plot(y=col, kind='bar', ax=ax, legend=False)
        ax.annotate(f'{col.split("_pct_")[0]} contribution',
                    xy=(0.01, 0.9), xycoords='axes fraction',
                    fontsize=10, fontweight='bold',
                    bbox=dict(boxstyle='round', facecolor='white', edgecolor='black'))
        cumulative = mec[col].sum()
        ax.annotate(f'Season Total:\n{cumulative:.1f} kg m⁻²',
                    xy=(0.5, 0.5), xycoords='axes fraction',
                    fontsize=10, color='dodgerblue')
        ax.set_xticks(range(len(mec)), xtick_labels)
        ax.set_ylim(0, 50)
    plt.suptitle(f'{site} — {basin} WY{WY}', y=1.01)
    plt.tight_layout()
    _save(fig, output_dir / f'daily_positive_melt_snowmelt_contributions_{basin}_{WY}_{slugify(site)}_sub.png')


# ---------------------------------------------------------------------------
# Per-site analysis
# ---------------------------------------------------------------------------

def run_site(em_df, site, basin, WY, output_dir, tmax_freezing_periods, cold_snap_period):
    """Run the full EB analysis pipeline and save all figures for a single site."""
    print(f'\n--- {site} ---')
    sub_df = em_df[site].copy()
    sub_df['net_LW'] = sub_df['net_rad'] - sub_df['net_solar']

    both_conditions_dates = (
        sub_df.index[sub_df['sum_EB'] > 0]
        .intersection(sub_df.index[sub_df['cold_content'] == 0])
    )

    conditions = build_conditions(sub_df, both_conditions_dates,
                                  tmax_freezing_periods, cold_snap_period)
    for cond, mask in conditions.items():
        sub_df.loc[mask, 'condition'] = cond

    melt_df = sub_df.loc[both_conditions_dates]

    mec = melt_df[FULL_EB_COLS].clip(lower=0).copy()
    mec['total_positive_flux'] = mec.sum(axis=1)
    for col in FULL_EB_COLS:
        mec[f'{col}_pct'] = mec[col] / mec['total_positive_flux'] * 100

    fig_boxplot_by_condition(sub_df, conditions, output_dir, basin, WY, site)
    fig_daily_mean_stacked(sub_df, output_dir, basin, WY, site)
    fig_daily_melt_stacked(melt_df, output_dir, basin, WY, site)
    fig_positive_melt_contributions(mec, output_dir, basin, WY, site)
    fig_positive_melt_proportions(mec, output_dir, basin, WY, site)
    fig_snowmelt_contributions(mec, sub_df, output_dir, basin, WY, site)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    """Parse command-line arguments, load data, and run analysis for each site in the specified basin/WY."""
    parser = argparse.ArgumentParser(description='Energy balance analysis for iSnobal THP output')
    parser.add_argument('--basin', required=True, help='Basin name (e.g. animas)')
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
    args = parser.parse_args()

    basin = args.basin
    WY = args.wy
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tmax_freezing_periods = json.loads(args.tmax_freezing_periods) if args.tmax_freezing_periods else None
    cold_snap_period = args.cold_snap_period.split(',') if args.cold_snap_period else None

    setup_pyproj()

    em_df, wy_dirs = load_em_data(basin, WY)
    em_df = add_net_solar(em_df, wy_dirs)

    sites = em_df.columns.get_level_values(0).unique()
    print(f'Found {len(sites)} site(s): {list(sites)}')

    for site in sites:
        run_site(em_df, site, basin, WY, output_dir, tmax_freezing_periods, cold_snap_period)


if __name__ == '__main__':
    main()
