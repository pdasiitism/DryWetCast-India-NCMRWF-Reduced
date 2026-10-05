"""Preflight check: can this machine run the ncmrwf_reduced forecast?

Uses only the standard library at import time, so it still works (and says
what's missing) when the scientific packages aren't installed yet.
"""

from __future__ import annotations

import importlib
import os
import shutil
import sys

from . import runtime

MIN_PYTHON = (3, 10)   # XGBoost 3.x (which saved the bundled models) and NumPy 2 both need it
PACKAGES = ['numpy', 'rasterio', 'xarray', 'netCDF4', 'scipy', 'xgboost', 'matplotlib', 'requests']
OPTIONAL_PACKAGES = {'zarr': 'only needed for --aifs-zarr (AIFS run locally)'}
FEEDS = {
    'GEFS  (NOAA S3)': 'https://noaa-gefs-pds.s3.amazonaws.com/',
    'AIFS  (ECMWF)  ': 'https://data.ecmwf.int/forecasts/',
}
MODELS = os.path.join(os.path.dirname(__file__), '..', 'models')
MODEL_FILES = ['ncmrwf_reduced/xgb_dry.json', 'ncmrwf_reduced/xgb_dry.mask.npz',
               'ncmrwf_reduced/xgb_wet.json', 'ncmrwf_reduced/xgb_wet.mask.npz', 'land_mask.npy']
MIN_FREE_GB = 2   # one cycle's AIFS download (~1 GB) + GEFS + cache


def _line(status, msg):
    print(f'  [{status}] {msg}', flush=True)


def run_check(out_dir: str) -> bool:
    """Print a report; return True if nothing blocks a run on this machine."""
    ok = True
    print('Python', flush=True)
    py = sys.version_info
    if py >= MIN_PYTHON:
        _line('OK  ', f'{py.major}.{py.minor}.{py.micro}')
    else:
        ok = False
        _line('FAIL', f'{py.major}.{py.minor} — need {MIN_PYTHON[0]}.{MIN_PYTHON[1]}+ '
                      '(the conda environment in environment.yml brings its own Python)')

    print('Packages', flush=True)
    for name in PACKAGES:
        try:
            mod = importlib.import_module(name)
            _line('OK  ', f'{name} {getattr(mod, "__version__", "")}')
        except Exception as exc:
            ok = False
            _line('FAIL', f'{name} — {exc.__class__.__name__}: {exc}')
    for name, why in OPTIONAL_PACKAGES.items():
        try:
            mod = importlib.import_module(name)
            _line('OK  ', f'{name} {getattr(mod, "__version__", "")} (optional)')
        except Exception:
            _line('INFO', f'{name} not installed — {why}; pip install {name}')

    print('Model files', flush=True)
    missing = [f for f in MODEL_FILES if not os.path.exists(os.path.join(MODELS, f))]
    if missing:
        ok = False
        for f in missing:
            _line('FAIL', f'missing models/{f}')
    else:
        _line('OK  ', 'ncmrwf_reduced DRY + WET models and land mask present')

    print('Internet access to data feeds (needed for downloads only)', flush=True)
    try:
        import requests
        for label, url in FEEDS.items():
            try:
                requests.head(url, timeout=10)
                _line('OK  ', f'{label} reachable')
            except Exception:
                _line('WARN', f'{label} not reachable — download on a machine with internet using '
                              '--download-only, then run here with --skip-download')
    except ImportError:
        _line('SKIP', 'requests not installed')

    print('Resources', flush=True)
    _line('INFO', f'{runtime.available_cpus()} CPUs available to this process -> '
                  f'{runtime.default_workers()} extraction workers (override with --workers)')
    probe = os.path.abspath(out_dir)
    while not os.path.exists(probe):
        probe = os.path.dirname(probe)
    free_gb = shutil.disk_usage(probe).free / 1e9
    _line('OK  ' if free_gb >= MIN_FREE_GB else 'WARN', f'{free_gb:.1f} GB free at {probe} (need ~{MIN_FREE_GB} GB per cycle)')

    print('\nReady to run.' if ok else '\nNot ready — fix the FAIL lines above.', flush=True)
    return ok
