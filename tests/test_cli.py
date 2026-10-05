"""Command-line validation fails fast, before any download."""
import os
import subprocess
import sys

import pytest

from conftest import REPO


def cli(*args):
    return subprocess.run([sys.executable, os.path.join(REPO, 'run_forecast.py'), *args],
                          capture_output=True, text=True, timeout=120)


@pytest.mark.parametrize('args, message', [
    (['--date', '2026-10-15', '--ncmrwf-file', 'x.nc'], 'not a valid YYYYMMDD'),
    (['--ncmrwf-file', 'x.nc'], '--date is required'),
    (['--date', '20261015'], 'give the NCMRWF file'),
    (['--date', '20261015', '--ncmrwf-file', 'a.nc', '--ncmrwf-dir', '/tmp'], 'not allowed with'),
    (['--date', '20261015', '--ncmrwf-dir', '/nonexistent'], 'No NCMRWF file for 20261015'),
    (['--date', '20261015', '--download-only', '--skip-download'], 'not allowed with'),
    (['--date', '20261015', '--ncmrwf-file', 'x.nc', '--aifs-zarr', '/tmp/x.nc'], 'must be a .zarr'),
])
def test_bad_arguments(args, message):
    r = cli(*args)
    assert r.returncode == 2 and message in r.stderr, r.stderr
