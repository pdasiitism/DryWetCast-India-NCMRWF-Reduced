"""Shared grid definitions and the GRIB2 read+regrid helper.

Target grid is the IMD-aligned 0.25 deg grid the model was trained on:
lat 6.5-38.5N (129 pts), lon 66.5-100.0E (135 pts) — the IMD 0.25 deg
gridded-rainfall grid itself.

GRIB2 reading uses `rasterio` (GDAL's built-in GRIB driver) rather than
cfgrib/eccodes: rasterio ships prebuilt wheels with GDAL bundled in, so
`pip install rasterio` just works on Windows/Mac/Linux with no separate
system library to compile or version-match — unlike eccodes, whose Python
bindings must link against a matching system `libeccodes`, which is the
single most common setup failure for this kind of pipeline. It's also
faster in practice here (~0.05-0.1s/file vs cfgrib's ~1-2s).

Regridding itself is plain bilinear interpolation (scipy
RegularGridInterpolator) on the decoded numpy array — not a subprocess call
to wgrib2's `-new_grid`, which was measured at ~5-17s per file (GRIB
decode/encode overhead).
"""

import os

import numpy as np
import rasterio
from scipy.interpolate import RegularGridInterpolator

N_LAT, N_LON = 129, 135
TARGET_LATS = np.arange(6.5, 38.51, 0.25)   # 129 pts
TARGET_LONS = np.arange(66.5, 100.01, 0.25)  # 135 pts

# Generous crop box around India applied before interpolation, purely to keep
# the interpolator's source array small (cheap) — not a physical boundary.
_CROP_LAT = (0.0, 45.0)
_CROP_LON = (55.0, 110.0)

LAND_MASK_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'land_mask.npy')



def load_land_mask() -> np.ndarray:
    """Return the (129,135) boolean land mask, bundled with the repo (no private-archive dependency)."""
    return np.load(LAND_MASK_PATH)


def regrid_field(arr: np.ndarray, src_lats: np.ndarray, src_lons: np.ndarray) -> np.ndarray:
    """Bilinear-regrid one (nlat,nlon) field onto the 129x135 target grid."""
    order = np.argsort(src_lats)
    src_lats = src_lats[order]
    arr = arr[order]

    lat_mask = (src_lats >= _CROP_LAT[0]) & (src_lats <= _CROP_LAT[1])
    lon_mask = (src_lons >= _CROP_LON[0]) & (src_lons <= _CROP_LON[1])
    sub = arr[np.ix_(lat_mask, lon_mask)]
    sub_lats = src_lats[lat_mask]
    sub_lons = src_lons[lon_mask]

    interp = RegularGridInterpolator((sub_lats, sub_lons), sub, method='linear',
                                      bounds_error=False, fill_value=np.nan)
    pts = np.array([[la, lo] for la in TARGET_LATS for lo in TARGET_LONS])
    return interp(pts).reshape(N_LAT, N_LON).astype(np.float32)


def _read_band(grib_path: str, band: int | None = None, element: str | None = None):
    """Return (arr, lats, lons) for one band of a GRIB2 file.

    `band` selects by 1-indexed position (default: 1, the common case for
    our single-message downloads). `element` instead selects by GRIB_ELEMENT
    tag (exact match), for multi-message files like the combined PWAT+CAPE
    downloads where band order isn't something to rely on.
    """
    with rasterio.open(grib_path) as ds:
        if element is not None:
            band = next(i for i in range(1, ds.count + 1)
                        if ds.tags(i).get('GRIB_ELEMENT', '').upper() == element.upper())
        arr = ds.read(band or 1).astype(np.float32)
        b = ds.bounds
        dx, dy = ds.transform.a, ds.transform.e   # dy is negative (north-to-south rows)
        nlat, nlon = arr.shape
        lats = (b.top + (np.arange(nlat) + 0.5) * dy).astype(np.float64)
        lons = (b.left + (np.arange(nlon) + 0.5) * dx).astype(np.float64)
    return arr, lats, lons


def load_and_regrid(grib_path: str, element: str | None = None, band: int | None = None) -> np.ndarray:
    """Read one GRIB2 message (by GRIB_ELEMENT name, band position, or band 1 if neither given) and regrid it.

    Use `band` when GRIB_ELEMENT is ambiguous (e.g. HGT appears at multiple
    pressure levels with no level info in the tag) — band position is
    reliable whenever you control the exact order messages were downloaded in.
    """
    arr, lats, lons = _read_band(grib_path, band=band, element=element)
    return regrid_field(arr, lats, lons)


def load_and_regrid_cached(grib_path: str, cache_dir: str, element: str | None = None,
                            band: int | None = None) -> np.ndarray:
    """Same as load_and_regrid, but memoized to disk under cache_dir.

    Makes repeat runs (and overlap between MOS and event-fraction extraction,
    which read the same member files) effectively free.
    """
    os.makedirs(cache_dir, exist_ok=True)
    key = grib_path.replace('/', '_') + (f'.{element}' if element else '') + (f'.b{band}' if band else '')
    cache_path = os.path.join(cache_dir, key + '.npy')
    if os.path.exists(cache_path):
        return np.load(cache_path)
    arr = load_and_regrid(grib_path, element=element, band=band)
    np.save(cache_path, arr)
    return arr
