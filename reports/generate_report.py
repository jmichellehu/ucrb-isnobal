"""Static report generator for iSnobal evaluation.

Usage:
    conda activate studio
    cd /uufs/.../ucrb-isnobal
    pip install -e .
    python reports/generate_report.py --config eval/config.yml
"""
import argparse
import pathlib
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from isnobal_eval import (
    load_config, load_snow, load_em, load_net_solar,
    load_topo, compute_aspect, load_snotel,
    extract_point_timeseries, compute_terrain_distribution,
    build_terrain_mask, compute_masked_timeseries,
    compare_snotel, compute_metrics,
)


def _save(fig, path, name):
    fig.savefig(path / name, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  saved {name}')


def run(cfg: dict, out_dir: pathlib.Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    basin = cfg['basin'].upper()
    wy    = cfg['water_year']

    print('Loading datasets...')
    snow_ds   = load_snow(cfg)
    em_ds     = load_em(cfg)
    ns_ds     = load_net_solar(cfg)
    topo_ds   = compute_aspect(load_topo(cfg))
    snotel_df = load_snotel(cfg)

    # ------------------------------------------------------------------
    # 1. Point temporal
    # ------------------------------------------------------------------
    print('Section 1: point temporal...')
    x_pt = float(snow_ds.x.values[len(snow_ds.x) // 2])
    y_pt = float(snow_ds.y.values[len(snow_ds.y) // 2])
    pt_df = extract_point_timeseries(snow_ds, em_ds, ns_ds, x_pt, y_pt)

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    for ax, rows, ylabel in zip(axes,
        [
            [('thickness','Depth (m)','steelblue'),('specific_mass','SWE/500','navy')],
            [('net_rad','Net Rad','orangered'),('net_solar','Net SW','gold'),
             ('net_LW','Net LW','coral'),('sensible_heat','Sensible','purple'),('latent_heat','Latent','mediumorchid')],
            [('snowmelt','Melt','tomato'),('SWI','SWI','darkorange'),('evaporation','Evap','teal')],
        ],
        ['Snow state','Energy flux (W/m²)','Mass flux (kg/m²/d)']
    ):
        for col, label, color in rows:
            if col in pt_df.columns:
                vals = pt_df[col] / 500 if col == 'specific_mass' else pt_df[col]
                ax.plot(pt_df.index, vals, label=label, color=color, lw=1.2)
        ax.set_ylabel(ylabel); ax.legend(fontsize=7, ncol=5); ax.grid(alpha=0.3)
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%b'))
    fig.suptitle(f'{basin} WY{wy} — Point ({x_pt:.0f}, {y_pt:.0f})', fontsize=12)
    fig.tight_layout()
    _save(fig, out_dir, 'point_timeseries.png')

    # ------------------------------------------------------------------
    # 2. Spatial distribution
    # ------------------------------------------------------------------
    print('Section 2: spatial distribution...')
    time_indices = [0, 120, 210]
    date_labels  = [str(snow_ds.time.values[i])[:10] for i in time_indices]
    colors       = ['steelblue', 'darkorange', 'green']
    strats       = ['elevation', 'aspect', 'slope', 'veg_class']

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    for ax, strat in zip(axes.ravel(), strats):
        try:
            df = compute_terrain_distribution(snow_ds, 'thickness', topo_ds, stratify_by=strat)
        except Exception:
            ax.set_visible(False); continue
        bin_cols = df.columns.tolist()
        x_pos = np.arange(len(bin_cols))
        w = 0.25
        for k, (ti, dl, c) in enumerate(zip(time_indices, date_labels, colors)):
            ax.bar(x_pos + k * w, df.iloc[ti][bin_cols].values, width=w, label=dl, color=c, alpha=0.8)
        ax.set_xticks(x_pos + w); ax.set_xticklabels(bin_cols, rotation=45, ha='right', fontsize=7)
        ax.set_title(f'By {strat}', fontsize=10); ax.set_ylabel('Mean depth (m)')
        ax.legend(fontsize=7); ax.grid(axis='y', alpha=0.3)
    fig.suptitle(f'{basin} WY{wy} — Snow Depth by Terrain', fontsize=12)
    fig.tight_layout()
    _save(fig, out_dir, 'spatial_distribution.png')

    # ------------------------------------------------------------------
    # 3. Conditional time series
    # ------------------------------------------------------------------
    print('Section 3: conditional time series...')
    tf = cfg['terrain_filter']
    mask = build_terrain_mask(
        topo_ds,
        elev_range=tf.get('elev_range'),
        aspect_range=tf.get('aspect_range'),
        slope_max=tf.get('slope_max'),
        veg_classes=tf.get('veg_classes'),
    )

    fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
    for ax, (ds, var, ylabel, color) in zip(axes, [
        (snow_ds, 'thickness',     'Depth (m)',    'steelblue'),
        (snow_ds, 'specific_mass', 'SWE (kg/m²)',  'navy'),
        (ns_ds,   'net_solar',     'Net SW (W/m²)','gold'),
    ]):
        ts = compute_masked_timeseries(ds, var, mask)
        ax.plot(ts.index, ts['mean'], color=color, lw=1.5)
        ax.fill_between(ts.index, ts['q25'], ts['q75'], color=color, alpha=0.25)
        ax.set_ylabel(ylabel); ax.grid(alpha=0.3)
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%b'))
    fig.suptitle(f"{basin} WY{wy} — Conditional TS (elev {tf.get('elev_range')}, aspect {tf.get('aspect_range')})", fontsize=11)
    fig.tight_layout()
    _save(fig, out_dir, 'conditional_timeseries.png')

    # ------------------------------------------------------------------
    # 4. SNOTEL validation
    # ------------------------------------------------------------------
    print('Section 4: SNOTEL validation...')
    if snotel_df.empty:
        print('  no SNOTEL sites, skipping.')
    else:
        comp = compare_snotel(snow_ds, snotel_df, 'thickness')
        if comp.empty:
            print('  no paired data, skipping.')
        else:
            metric_rows = []
            for site, grp in comp.groupby('site_id'):
                m = compute_metrics(grp['modeled'].values, grp['observed'].values)
                m['site_id'] = site
                metric_rows.append(m)
            metrics_df = pd.DataFrame(metric_rows).set_index('site_id')
            metrics_df[['n','r','r2','rmse','mae','kge']].round(3).to_csv(out_dir / 'snotel_metrics.csv')
            print('  saved snotel_metrics.csv')

            sites = comp['site_id'].unique()[:6]
            ncols = 2
            nrows = int(np.ceil(len(sites) / ncols))
            fig, axes = plt.subplots(nrows, ncols, figsize=(14, 3.5 * nrows))
            axes = np.array(axes).ravel()
            for ax, site in zip(axes, sites):
                grp = comp[comp['site_id'] == site].set_index('date').sort_index()
                ax.plot(grp.index, grp['modeled'],  color='steelblue',  lw=1.5, label='iSnobal')
                ax.plot(grp.index, grp['observed'], color='darkorange', lw=1.5, ls='--', label='SNOTEL')
                kge = metrics_df.loc[site, 'kge'] if site in metrics_df.index else float('nan')
                ax.set_title(f'{site}  KGE={kge:.2f}', fontsize=9)
                ax.set_ylabel('Depth (m)'); ax.legend(fontsize=7); ax.grid(alpha=0.3)
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
            for ax in axes[len(sites):]:
                ax.set_visible(False)
            fig.suptitle(f'{basin} WY{wy} — SNOTEL Validation', fontsize=12)
            fig.tight_layout()
            _save(fig, out_dir, 'snotel_comparison.png')

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print('\n--- Output summary ---')
    for f in sorted(out_dir.iterdir()):
        print(f'  {f.name}  ({f.stat().st_size // 1024} KB)')


def main():
    parser = argparse.ArgumentParser(description='Generate iSnobal evaluation report')
    parser.add_argument('--config',     required=True,  help='Path to config.yml')
    parser.add_argument('--output-dir', default=None,   help='Override output directory from config')
    args = parser.parse_args()

    cfg = load_config(args.config)
    out_dir = pathlib.Path(args.output_dir) if args.output_dir else pathlib.Path(cfg['paths']['output_dir'])
    run(cfg, out_dir)


if __name__ == '__main__':
    main()
