"""Bundled grid, the .npz record round trip, and the AIFS unit guard."""
import numpy as np
import pytest

from pipeline import aifs_features, archive, grid


def test_bundled_grid():
    land = grid.load_land_mask()
    assert land.shape == (129, 135) and land.dtype == bool and land.any()


def test_archive_round_trip(tmp_path):
    rng = np.random.default_rng(1)
    p_dry, p_wet = rng.random((129, 135)), rng.random((129, 135))
    path = archive.archive_path(str(tmp_path), 'ncmrwf_reduced', '20261015', '00')
    archive.save(path, 'ncmrwf_reduced', '20261015', '00', p_dry, p_wet)
    rec = archive.read_archive(str(tmp_path))[0]
    assert (rec['date'], rec['config']) == ('20261015', 'ncmrwf_reduced')
    np.testing.assert_array_equal(rec['p_wet'], p_wet.astype(np.float32))


def test_aifs_unit_guard():
    daily_mm = np.random.default_rng(2).gamma(0.3, 10.0, size=(51, 5, 129, 135)).astype(np.float32)
    aifs_features._check_plausible(daily_mm, 'ok')
    with pytest.raises(ValueError, match='metres where mm'):
        aifs_features._check_plausible(daily_mm / 1000, 'metres as mm')
    with pytest.raises(ValueError, match='implausible'):
        aifs_features._check_plausible(daily_mm * 1000, 'mm as metres')
