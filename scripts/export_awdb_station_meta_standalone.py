#!/usr/bin/env python3
"""
Export AWDB station metadata to a single CSV.
Calls the AWDB REST API directly — one request per network code, no tiling.
Co-created with Claude Code, with minor critiques from Copilot

Examples
--------
# SNOTEL only — full CONUS, should complete in seconds
python export_awdb_station_meta_standalone.py --networks SNTL --out sntl.csv

# All default networks, filtered to a Colorado bbox
python export_awdb_station_meta_standalone.py \
  --bounds -109.1 36.9 -102.0 41.0 \
  --out colorado_awdb.csv

# All networks, no geographic filter
python export_awdb_station_meta_standalone.py --out awdb_all.csv

# Re-fetch all networks and add any stations not already in the output file
python export_awdb_station_meta_standalone.py --out awdb_all.csv --resume

Notes
-----
- One request per network code; the response includes full metadata
  (lat, lon, elevation, name, HUC, etc.) — no separate metadata phase needed.
- --bounds filters results after fetching; it does not change the API query.
- REST API: https://wcc.sc.egov.usda.gov/awdbRestApi/services/v1
  Swagger: https://wcc.sc.egov.usda.gov/awdbRestApi/swagger-ui/index.html
- Requires: requests, tqdm, pandas
"""

import argparse
import sys
import threading
import time
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm


AWDB_BASE = "https://wcc.sc.egov.usda.gov/awdbRestApi/services/v1"
DEFAULT_NETWORK_CDS = ["SNTL", "SNTLT", "USGS", "BOR", "COOP", "SCAN"]

_thread_local = threading.local()


def _get_session() -> requests.Session:
    """Return a per-thread Session for connection reuse."""
    if not hasattr(_thread_local, "session"):
        _thread_local.session = requests.Session()
    return _thread_local.session


def _get(path: str, params: dict, *, timeout: int):
    """GET https://wcc.sc.egov.usda.gov/awdbRestApi/services/v1{path}, raise on non-2xx, return parsed JSON."""
    resp = _get_session().get(f"{AWDB_BASE}{path}", params=params, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def safe_call(fn, *, retries: int = 3, backoff_s: float = 1.0):
    """Retry helper for flaky network calls."""
    for attempt in range(retries):
        try:
            return fn()
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(backoff_s * (2**attempt))


def _log(msg: str, verbose: bool) -> None:
    # tqdm.write keeps messages from clobbering the progress bar
    if verbose:
        tqdm.write(msg, file=sys.stderr)


def _sort_df(df: pd.DataFrame) -> pd.DataFrame:
    """Sort by stationTriplet and move it to the first column."""
    if df.empty or "stationTriplet" not in df.columns:
        return df
    df = df.sort_values("stationTriplet").reset_index(drop=True)
    cols = ["stationTriplet"] + [c for c in df.columns if c != "stationTriplet"]
    return df[cols]


def fetch_all_stations(
    networks: Sequence[str],
    *,
    verbose: bool,
    workers: int = 10,
    timeout: int = 30,
    active_only: bool = True,
) -> list[dict]:
    """Fetch all stations for each network in parallel; deduplicate by stationTriplet."""
    n_workers = min(len(networks), workers)
    _log(f"[fetch] querying {len(networks)} networks (workers={n_workers}, timeout={timeout}s) ...", verbose)

    def fetch_network(network_cd: str) -> list[dict]:
        data = safe_call(lambda: _get("/stations", {
            "stationTriplets": f"::{network_cd}",
            "returnForecastPointMetadata": "false",
            "returnReservoirMetadata": "false",
            "returnStationElements": "false",
            "activeOnly": str(active_only).lower(),
        }, timeout=timeout))
        return data if isinstance(data, list) else []

    records: dict[str, dict] = {}
    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        future_map = {ex.submit(fetch_network, n): n for n in networks}
        with tqdm(total=len(networks), desc="networks", unit="network", disable=not verbose) as pbar:
            for fut in as_completed(future_map):
                network = future_map[fut]
                try:
                    for station in fut.result():
                        triplet = station.get("stationTriplet")
                        if triplet and triplet not in records:
                            records[triplet] = station
                    pbar.set_postfix(stations=len(records))
                except Exception as e:
                    tqdm.write(f"[fetch] WARNING: {network} failed after retries: {e}", file=sys.stderr)
                pbar.update(1)

    _log(f"[fetch] done: {len(records)} unique stations", verbose)
    return list(records.values())


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export AWDB station metadata to CSV (standalone).")
    p.add_argument("--out", default="awdb_station_metadata.csv", help="Output CSV path.")
    p.add_argument(
        "--networks",
        default=",".join(DEFAULT_NETWORK_CDS),
        help="Comma-separated network codes. Default: %(default)s",
    )
    p.add_argument(
        "--bounds",
        nargs=4,
        type=float,
        metavar=("MIN_LON", "MIN_LAT", "MAX_LON", "MAX_LAT"),
        default=None,
        help="Optional lat/lon bounding box applied after fetching (EPSG:4326). Default: no filter. CONUS: -125.0 24.0 -66.5 49.5",
    )
    p.add_argument("--include-inactive", action="store_true", help="Include inactive stations (default: active only).")
    p.add_argument("--limit", type=int, default=None, help="Cap on new stations fetched in this run (useful for testing).")
    p.add_argument("--workers", type=int, default=10, help="Parallel threads. Default: 10")
    p.add_argument("--timeout", type=int, default=30, help="Per-request timeout in seconds. Default: 30")
    p.add_argument("--no-progress", action="store_true", help="Disable progress bar and verbose log output (warnings always shown).")
    p.add_argument("--resume", action="store_true", help="Merge with existing --out file, skipping already-present stations. Note: --bounds is not applied to existing rows.")
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    networks = [n.strip() for n in args.networks.split(",") if n.strip()]
    verbose = not args.no_progress
    active_only = not args.include_inactive
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    _log(
        f"[config] networks={networks} bounds={args.bounds} active_only={active_only} "
        f"workers={args.workers} timeout={args.timeout}s limit={args.limit} resume={args.resume}",
        verbose,
    )
    t0 = time.time()

    # Load existing records when resuming
    existing_df = pd.DataFrame()
    already_fetched: set[str] = set()
    if args.resume and out_path.exists():
        existing_df = pd.read_csv(out_path)
        if "stationTriplet" in existing_df.columns:
            already_fetched = set(existing_df["stationTriplet"])
            _log(f"[resume] {len(already_fetched)} stations already in {out_path}", verbose)

    stations = fetch_all_stations(
        networks,
        verbose=verbose,
        workers=args.workers,
        timeout=args.timeout,
        active_only=active_only,
    )

    if already_fetched:
        before = len(stations)
        stations = [s for s in stations if s.get("stationTriplet") not in already_fetched]
        _log(f"[resume] skipping {before - len(stations)} already-present; {len(stations)} new", verbose)

    if args.bounds is not None:
        min_lon, min_lat, max_lon, max_lat = args.bounds
        stations = [
            s for s in stations
            if s.get("latitude") is not None
            and s.get("longitude") is not None
            and min_lat <= s["latitude"] <= max_lat
            and min_lon <= s["longitude"] <= max_lon
        ]
        _log(f"[bounds] {len(stations)} stations within {args.bounds}", verbose)

    if args.limit is not None:
        stations = stations[:args.limit]

    new_df = pd.DataFrame(stations) if stations else pd.DataFrame()
    frames = [df for df in (existing_df, new_df) if not df.empty]
    combined = _sort_df(pd.concat(frames, ignore_index=True)) if frames else pd.DataFrame()
    combined.to_csv(out_path, index=False)
    _log(f"[write] wrote {len(combined)} rows ({len(combined.columns)} fields) to {out_path}", verbose)
    _log(f"[done] total wall time: {time.time() - t0:.1f}s", verbose)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
