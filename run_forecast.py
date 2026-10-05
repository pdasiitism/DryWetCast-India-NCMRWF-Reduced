#!/usr/bin/env python3
"""DRY/WET probability forecast over India from NCMRWF NEPS + AIFS-ENS + GEFS
(model `ncmrwf_reduced`, XGBoost), for one 00Z cycle.

Inputs:
  NCMRWF  the NEPS ensemble file for the init date, <YYYYMMDD>.nc or
          <YYYYMMDD>_NPES.nc (23 members, precipitation_amount) — already on
          this machine: --ncmrwf-file FILE, or --ncmrwf-dir DIR to find it by date.
          Only this file is used (not the lagged-ensemble .lag.nc).
  AIFS    AIFS-ENS v2 rainfall: downloaded from ECMWF open data (default), or a
          local AIFS-ENS v2 zarr store via --aifs-zarr.
  GEFS    PWAT and CAPE ensemble mean/spread: downloaded from NOAA (82 small files).

Output: a PNG of the DRY and WET probability maps, and an .npz record next to it
with the exact probabilities (p_dry, p_wet; 129x135, NaN over sea).

Usage:
    python run_forecast.py --check
    python run_forecast.py --date 20261015 --ncmrwf-file /data/neps/20261015.nc
    python run_forecast.py --date 20261015 --ncmrwf-dir /data/neps --out-dir /scratch/dwf

    # compute nodes without internet: download first, then process as a batch job
    python run_forecast.py --date 20261015 --download-only --out-dir /scratch/dwf
    python run_forecast.py --date 20261015 --skip-download --out-dir /scratch/dwf --ncmrwf-dir /data/neps
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime

from pipeline import check

if sys.version_info < check.MIN_PYTHON:
    sys.exit(f'Python {check.MIN_PYTHON[0]}.{check.MIN_PYTHON[1]}+ required (this is '
             f'{sys.version_info.major}.{sys.version_info.minor}). The conda environment in '
             'environment.yml brings its own Python: conda env create -f environment.yml')

CYCLE = '00'   # the model was trained on 00Z runs of all three systems


def _init_date(value: str) -> str:
    try:
        datetime.strptime(value, '%Y%m%d')
    except ValueError:
        raise argparse.ArgumentTypeError(f'{value!r} is not a valid YYYYMMDD date (e.g. 20261015)')
    return value


def find_ncmrwf(ncmrwf_dir: str, date: str) -> str:
    """The NEPS file for `date` in ncmrwf_dir: <date>.nc or <date>_NPES.nc."""
    for name in (f'{date}.nc', f'{date}_NPES.nc'):
        path = os.path.join(ncmrwf_dir, name)
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f'No NCMRWF file for {date} in {ncmrwf_dir} '
                            f'(looked for {date}.nc and {date}_NPES.nc)')


def gefs_dir(out_dir, date):
    return os.path.join(out_dir, f'gefs_{date}_{CYCLE}z')


def aifs_dir(out_dir, date):
    return os.path.join(out_dir, f'aifs_{date}_{CYCLE}z')


def download(date: str, out_dir: str, aifs_zarr: str | None = None) -> None:
    from pipeline import download_aifs, download_gefs
    print(f'=== Downloading GEFS PWAT/CAPE ({date} {CYCLE}Z) ===', flush=True)
    t0 = time.time()
    download_gefs.download_cycle(date, gefs_dir(out_dir, date))
    print(f'  done in {time.time() - t0:.0f}s', flush=True)
    if aifs_zarr:
        print(f'=== AIFS: using local store {aifs_zarr} — nothing to download ===', flush=True)
        return
    print(f'=== Downloading AIFS-ENS v2 ({date} {CYCLE}Z) ===', flush=True)
    t0 = time.time()
    download_aifs.download_cycle(date, CYCLE, aifs_dir(out_dir, date))
    print(f'  done in {time.time() - t0:.0f}s', flush=True)


def run(date: str, ncmrwf_file: str, out_dir: str, fig_out: str, skip_download: bool = False,
        aifs_zarr: str | None = None, archive_dir: str | None = None) -> dict:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    from pipeline import aifs_features, archive, gefs_features, grid, inference, ncmrwf_features

    # NCMRWF first: a missing or wrong-date file stops the run before any download.
    print(f'=== NCMRWF rainfall features: {ncmrwf_file} ===', flush=True)
    t0 = time.time()
    if not os.path.exists(ncmrwf_file):
        raise FileNotFoundError(f'NCMRWF file {ncmrwf_file} does not exist')
    nc_mean, nc_spread = ncmrwf_features.compute_ncmrwf_mos(ncmrwf_file, init_date=date)
    nc_dry, nc_wet = ncmrwf_features.compute_ncmrwf_event_prob(ncmrwf_file, init_date=date)
    print(f'  done in {time.time() - t0:.0f}s', flush=True)

    if not skip_download:
        download(date, out_dir, aifs_zarr)

    print(f'=== AIFS rainfall features ({"local zarr" if aifs_zarr else "downloaded GRIB"}) ===', flush=True)
    t0 = time.time()
    aifs_input = aifs_zarr or aifs_dir(out_dir, date)
    aifs_mean, aifs_spread = aifs_features.compute_aifs_mos(aifs_input, CYCLE, through_day=4, init_date=date)
    aifs_dry, aifs_wet = aifs_features.compute_aifs_event_prob(aifs_input, CYCLE, init_date=date)
    print(f'  done in {time.time() - t0:.0f}s', flush=True)

    print('=== GEFS atmosphere features (PWAT, CAPE; 41 leads f003-f123) ===', flush=True)
    t0 = time.time()
    pw_mean, cape_mean, cape_spread = gefs_features.compute_gefs_pwat_cape(gefs_dir(out_dir, date))
    print(f'  done in {time.time() - t0:.0f}s', flush=True)

    print('=== Running the DRY and WET models ===', flush=True)
    land = grid.load_land_mask()
    feats = inference.build_feature_dict((nc_mean, nc_spread), (nc_dry, nc_wet),
                                         (pw_mean, cape_mean, cape_spread),
                                         (aifs_mean, aifs_spread), (aifs_dry, aifs_wet))
    p_dry = inference.predict('dry', feats, land)
    p_wet = inference.predict('wet', feats, land)

    print(f'=== Saving forecast product -> {fig_out} ===', flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(fig_out)), exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.5))
    for ax, (field, label) in zip(axes, ((p_dry, 'DRY'), (p_wet, 'WET'))):
        im = ax.pcolormesh(grid.TARGET_LONS, grid.TARGET_LATS, field, cmap='viridis', vmin=0, vmax=1, shading='auto')
        ax.set_title(f'{label} probability — NCMRWF + AIFS + GEFS (XGB)\nissued {date} {CYCLE}Z', fontsize=11)
        ax.set_aspect('equal')
        fig.colorbar(im, ax=ax, fraction=0.04, label='probability')
    plt.tight_layout()
    plt.savefig(fig_out, dpi=200, bbox_inches='tight')
    plt.close(fig)

    npz_out = os.path.splitext(fig_out)[0] + '.npz'
    archive.save(npz_out, inference.CONFIG, date, CYCLE, p_dry, p_wet)
    print(f'=== Saved forecast record -> {npz_out} ===', flush=True)
    if archive_dir:
        path = archive.archive_path(archive_dir, inference.CONFIG, date, CYCLE)
        archive.save(path, inference.CONFIG, date, CYCLE, p_dry, p_wet)
        print(f'=== Added to archive -> {path} ===', flush=True)
    return {'p_dry': p_dry, 'p_wet': p_wet}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--check', action='store_true', help='check this machine is set up correctly, then exit')
    p.add_argument('--date', type=_init_date, help='init date YYYYMMDD (00Z run)')
    src = p.add_mutually_exclusive_group()
    src.add_argument('--ncmrwf-file', help='the NEPS file for --date (<date>.nc or <date>_NPES.nc)')
    src.add_argument('--ncmrwf-dir', help='folder holding NEPS files named <date>.nc or <date>_NPES.nc')
    p.add_argument('--out-dir', default='./_scratch', help='where GEFS/AIFS downloads are kept')
    p.add_argument('--fig-out', default='./forecast_ncmrwf_reduced.png',
                   help='output PNG; the .npz record is written next to it')
    mode = p.add_mutually_exclusive_group()
    mode.add_argument('--download-only', action='store_true', help='download GEFS + AIFS for --date, then exit')
    mode.add_argument('--skip-download', action='store_true',
                      help='use data already downloaded under --out-dir (e.g. compute node without internet)')
    p.add_argument('--aifs-zarr', help='local AIFS-ENS v2 store init_<YYYYMMDD>T00.zarr instead of downloading AIFS')
    p.add_argument('--archive-dir', help='also save the record as <archive-dir>/ncmrwf_reduced/<date>_00.npz')
    p.add_argument('--workers', type=int, help='feature-extraction processes (default: CPUs available, max 16)')
    a = p.parse_args()

    if a.check:
        sys.exit(0 if check.run_check(a.out_dir) else 1)
    if a.date is None:
        p.error('--date is required')
    if a.aifs_zarr is not None:
        if not a.aifs_zarr.rstrip('/').endswith('.zarr'):
            p.error(f'--aifs-zarr must be a .zarr store (got {a.aifs_zarr})')
        if not os.path.isdir(a.aifs_zarr):
            p.error(f'--aifs-zarr: {a.aifs_zarr} does not exist')
    if a.download_only:
        download(a.date, a.out_dir, a.aifs_zarr)
        sys.exit(0)
    if not (a.ncmrwf_file or a.ncmrwf_dir):
        p.error('give the NCMRWF file: --ncmrwf-file FILE or --ncmrwf-dir DIR')
    try:
        nc_file = a.ncmrwf_file or find_ncmrwf(a.ncmrwf_dir, a.date)
    except FileNotFoundError as exc:
        p.error(str(exc))
    if a.workers is not None:
        from pipeline import aifs_features, gefs_features
        aifs_features.N_WORKERS = gefs_features.N_WORKERS = max(1, a.workers)
    run(a.date, nc_file, a.out_dir, a.fig_out, a.skip_download, a.aifs_zarr, a.archive_dir)
