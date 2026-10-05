"""GEFS atmosphere -> the three atmospheric features of the ncmrwf_reduced model.

  pw_mean_roll     PWAT, ensemble mean   (geavg), averaged over the 41 leads
  cape_mean_roll   CAPE, ensemble mean   (geavg), averaged over the 41 leads
  cape_spread_roll CAPE, ensemble spread (gespr), averaged over the 41 leads

The 41 leads are every 3 h from f003 to f123: the same D1-D5 window (03Z-03Z
days) as the rainfall, evenly weighted. (Names end in `_roll` for historical
reasons; each is a single 5-day mean.) Each field is regridded bilinearly from
GEFS's 0.5 deg grid to the 129x135 target grid, then averaged — as in training.
"""

import os
from multiprocessing import Pool

import numpy as np

from . import download_gefs, grid, runtime

N_WORKERS = runtime.default_workers()
BAND = {'PWAT': 1, 'CAPE': 2}   # message order written by download_gefs.py


def _paths(cycle_root, product):
    paths = [download_gefs.file_path(cycle_root, product, lead) for lead in download_gefs.LEADS]
    missing = [p for p in paths if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(f'{len(missing)} GEFS {product} files missing, e.g. {missing[0]} — '
                                'download them first (run without --skip-download)')
    return paths


def _regrid_bands(args):
    path, cache_dir, names = args
    return {n: grid.load_and_regrid_cached(path, cache_dir, band=BAND[n]) for n in names}


def compute_gefs_pwat_cape(cycle_root: str):
    """Return (pw_mean, cape_mean, cape_spread): each (129,135)."""
    cache_dir = os.path.join(cycle_root, '.regrid_cache')
    avg, spr = _paths(cycle_root, 'geavg'), _paths(cycle_root, 'gespr')
    with Pool(min(N_WORKERS, len(avg))) as pool:
        a = pool.map(_regrid_bands, [(p, cache_dir, ['PWAT', 'CAPE']) for p in avg])
        s = pool.map(_regrid_bands, [(p, cache_dir, ['CAPE']) for p in spr])
    return (np.mean([r['PWAT'] for r in a], axis=0),
            np.mean([r['CAPE'] for r in a], axis=0),
            np.mean([r['CAPE'] for r in s], axis=0))
