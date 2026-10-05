"""The shipped DRY and WET models load and give sane probabilities."""
import numpy as np
import pytest

from pipeline import grid, inference


@pytest.mark.parametrize('event', ['dry', 'wet'])
def test_model_predicts(event):
    names = [str(n) for n in np.load(inference.model_paths(event)[1], allow_pickle=True)['feature_names']]
    land = grid.load_land_mask()
    rng = np.random.default_rng(0)
    feats = {n: rng.normal(size=land.shape).astype(np.float32) for n in names}
    p = inference.predict(event, feats, land)
    assert p.shape == (129, 135)
    assert np.all(np.isnan(p[~land])) and np.all((p[land] >= 0) & (p[land] <= 1))


def test_feature_dict_covers_both_models():
    z = np.zeros((129, 135), np.float32)
    feats = inference.build_feature_dict((np.zeros((5, 129, 135)), None), (z, z), (z, z, z),
                                         (np.zeros((5, 129, 135)), np.zeros((5, 129, 135))), (z, z))
    for event in ('dry', 'wet'):
        names = [str(n) for n in np.load(inference.model_paths(event)[1], allow_pickle=True)['feature_names']]
        assert set(names) <= set(feats), set(names) - set(feats)
