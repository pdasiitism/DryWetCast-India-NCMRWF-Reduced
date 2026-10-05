"""Download the GEFS fields the ncmrwf_reduced model needs, for one 00Z cycle,
from NOAA's public S3 bucket (no credentials).

Only PWAT and CAPE are needed (pw_mean_roll, cape_mean_roll, cape_spread_roll),
from the 0.5 deg ensemble-mean (geavg) and ensemble-spread (gespr) products, at
41 leads: every 3 h from f003 to f123 — the D1-D5 window (03Z-03Z days) the
model was trained on. No GEFS rainfall is needed: rainfall comes from NCMRWF.

Each output file holds exactly two GRIB2 messages, fetched by HTTP byte range
using NOAA's `.idx` inventories, in a fixed order (band 1 = PWAT, band 2 = CAPE):
  <out_dir>/mean_spread/<geavg|gespr>/<product>.t00z.pgrb2a.0p50.f<LLL>.pwat_cape.grb2
82 files, a few MB in total.
"""

import argparse
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

BASE_URL = 'https://noaa-gefs-pds.s3.amazonaws.com'
CYCLE = '00'
LEADS = list(range(3, 124, 3))            # f003..f123 every 3 h (41 leads)
PRODUCTS = ('geavg', 'gespr')
# idx patterns, in band order — gefs_features.BAND relies on this order
MESSAGES = [':PWAT:entire atmosphere', ':CAPE:180-0 mb above ground:']
N_WORKERS = 16
TIMEOUT = 60
RETRIES = 6


def file_path(out_dir, product, lead):
    return os.path.join(out_dir, 'mean_spread', product,
                        f'{product}.t{CYCLE}z.pgrb2a.0p50.f{lead:03d}.pwat_cape.grb2')


def _get(session, url, headers=None):
    """GET with retries and capped exponential backoff (S3 throttles under load)."""
    last = None
    for attempt in range(RETRIES):
        if attempt:
            time.sleep(min(2 ** attempt, 30))
        try:
            r = session.get(url, headers=headers, timeout=TIMEOUT)
            if r.status_code in (200, 206):
                return r
            last = f'HTTP {r.status_code}'
            if r.status_code == 404:
                break
        except requests.RequestException as exc:
            last = str(exc)
    raise RuntimeError(f'{url}: {last}')


def _byte_range(idx_text, pattern):
    lines = idx_text.strip().splitlines()
    for i, line in enumerate(lines):
        if pattern in line:
            start = int(line.split(':')[1])
            end = int(lines[i + 1].split(':')[1]) - 1 if i + 1 < len(lines) else ''
            return start, end
    raise ValueError(f'{pattern!r} not in the .idx inventory')


def _download_one(args):
    date, product, lead, out_dir, session = args
    out = file_path(out_dir, product, lead)
    if os.path.exists(out) and os.path.getsize(out) > 0:
        return out
    os.makedirs(os.path.dirname(out), exist_ok=True)
    url = f'{BASE_URL}/gefs.{date}/{CYCLE}/atmos/pgrb2ap5/{product}.t{CYCLE}z.pgrb2a.0p50.f{lead:03d}'
    idx = _get(session, url + '.idx').text
    part = out + '.part'
    with open(part, 'wb') as f:
        for pattern in MESSAGES:
            start, end = _byte_range(idx, pattern)
            f.write(_get(session, url, headers={'Range': f'bytes={start}-{end}'}).content)
    os.replace(part, out)
    return out


def download_cycle(date: str, out_dir: str, n_workers: int = N_WORKERS) -> None:
    """Download the 82 PWAT/CAPE files for init `date` (YYYYMMDD, 00Z) into out_dir."""
    session = requests.Session()
    jobs = [(date, p, lead, out_dir, session) for p in PRODUCTS for lead in LEADS]
    failed = []
    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        futures = {pool.submit(_download_one, j): j for j in jobs}
        for fut in as_completed(futures):
            try:
                fut.result()
            except Exception as exc:
                failed.append((futures[fut][1:3], str(exc)))
    print(f'GEFS download: {len(jobs) - len(failed)}/{len(jobs)} OK, {len(failed)} failed', flush=True)
    for (product, lead), err in failed:
        print(f'  FAILED {product} f{lead:03d}: {err}', flush=True)
    if failed:
        raise RuntimeError(f'{len(failed)} GEFS files failed to download — rerun to retry just those')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--date', required=True, help='YYYYMMDD init date (00Z)')
    p.add_argument('--out', required=True, help='output cycle directory')
    a = p.parse_args()
    download_cycle(a.date, a.out)
