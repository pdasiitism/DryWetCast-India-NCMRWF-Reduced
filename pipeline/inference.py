"""Assemble the ncmrwf_reduced features and run the bundled XGBoost models.

Each model (DRY, WET) uses its own LASSO-selected subset of these features;
the subset and its order are stored next to the weights (xgb_<event>.mask.npz):

  ncmrwf_mean_D3, ncmrwf_mean_D4          NCMRWF ensemble-mean rainfall, day 3 / day 4
  ncmrwf_prob_dry_event/wet_event         NCMRWF ensemble event fraction
  aifs_ens_mean_D3                        AIFS ensemble-mean rainfall, day 3
  aifs_ens_spread_D1..D4                  AIFS ensemble spread, days 1-4
  aifs_ens_prob_dry_event/wet_event       AIFS ensemble event fraction
  pw_mean_roll, cape_mean_roll, cape_spread_roll   GEFS D1-D5 means (gefs_features.py)
"""

import os

import numpy as np
import xgboost as xgb

MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', 'models', 'ncmrwf_reduced')
CONFIG = 'ncmrwf_reduced'


def model_paths(event):
    return (os.path.join(MODELS_DIR, f'xgb_{event}.json'), os.path.join(MODELS_DIR, f'xgb_{event}.mask.npz'))


def build_feature_dict(ncmrwf_mos, ncmrwf_event_prob, gefs_pwat_cape, aifs_mos, aifs_event_prob):
    """Every named feature the two models can use, as (129,135) arrays."""
    rain_mean, _ = ncmrwf_mos
    prob_dry, prob_wet = ncmrwf_event_prob
    pw_mean, cape_mean, cape_spread = gefs_pwat_cape
    aifs_mean, aifs_spread = aifs_mos
    aifs_dry, aifs_wet = aifs_event_prob
    return {
        'ncmrwf_mean_D3': rain_mean[2], 'ncmrwf_mean_D4': rain_mean[3],
        'ncmrwf_prob_dry_event': prob_dry, 'ncmrwf_prob_wet_event': prob_wet,
        'aifs_ens_mean_D3': aifs_mean[2],
        'aifs_ens_spread_D1': aifs_spread[0], 'aifs_ens_spread_D2': aifs_spread[1],
        'aifs_ens_spread_D3': aifs_spread[2], 'aifs_ens_spread_D4': aifs_spread[3],
        'aifs_ens_prob_dry_event': aifs_dry, 'aifs_ens_prob_wet_event': aifs_wet,
        'pw_mean_roll': pw_mean, 'cape_mean_roll': cape_mean, 'cape_spread_roll': cape_spread,
    }


def predict(event, feat_dict, land_mask):
    """P(event) on the (129,135) grid; NaN over sea."""
    model_json, mask_npz = model_paths(event)
    names = [str(n) for n in np.load(mask_npz, allow_pickle=True)['feature_names']]
    rows, cols = np.where(land_mask)
    X = np.empty((len(rows), len(names)), dtype=np.float32)
    for j, name in enumerate(names):
        X[:, j] = feat_dict[name][rows, cols]
    booster = xgb.Booster()
    booster.load_model(model_json)
    out = np.full(land_mask.shape, np.nan, dtype=np.float32)
    out[rows, cols] = booster.inplace_predict(X)
    return out
