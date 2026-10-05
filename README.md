# DRY/WET probability forecast over India: NCMRWF + AIFS + GEFS (`ncmrwf_reduced`)

For one 00Z cycle, this forecasts the probability of two events at every
0.25° land grid point over India (129 × 135 grid, IMD-aligned), using an
XGBoost model trained on IMD observations:

- **DRY**: a spell of 3 consecutive dry days (< 1 mm/day) starting within days 1–3;
- **WET**: at least one wet day (≥ 1 mm) within days 1–5.

Day 1 is the 24 h ending 03Z (08:30 IST) on the day after the init date,
matching the IMD rainfall day.

## Inputs

| Source | What is used | Where it comes from |
|---|---|---|
| **NCMRWF NEPS** | rainfall, 23 members (1 control + 22 perturbed): ensemble mean rainfall for day 3 in the DRY model, plus DRY/WET event fractions | **your NEPS file** for the init date: `<YYYYMMDD>.nc` or `<YYYYMMDD>_NPES.nc` |
| **AIFS-ENS v2** (ECMWF) | rainfall, 51 members: mean, spread, event fractions | downloaded from ECMWF open data (default), or a local AIFS-ENS v2 zarr store |
| **GEFS** (NOAA) | PWAT and CAPE, ensemble mean/spread, averaged over days 1–5 | downloaded from NOAA's public S3 (82 small files) |

The operational NEPS file must contain `precipitation_amount(realization,
forecast_period, latitude, longitude)`: one 24 h total per day ending 03Z and
23 members (1 control + 22 perturbed), with `time` = init + 1 day 03Z,
then 03Z on each of the next four days. Every file is checked against
`--date`, so the wrong day's file is rejected. Only this file is used, not the
lagged-ensemble `.lag.nc`.

No credentials are needed; GEFS and AIFS are public.

## Setup

Python 3.10+. Either:

```bash
conda env create -f environment.yml && conda activate ncmrwf-dry-wet
# or
pip install -r requirements.txt
```

On macOS, XGBoost also needs `brew install libomp`. No GRIB system library
(eccodes) is needed.

Then check the machine:

```bash
python run_forecast.py --check
```

This reports the Python packages, the model files, whether NOAA and ECMWF can
be reached, the CPUs available, and the free disk space.

## Run a forecast

```bash
# NEPS file given directly
python run_forecast.py --date 20261015 --ncmrwf-file /data/neps/20261015.nc

# or: a folder of NEPS files, found by date (convenient for daily runs)
python run_forecast.py --date 20261015 --ncmrwf-dir /data/neps --out-dir /scratch/dwf \
    --fig-out /forecasts/ncmrwf_reduced_20261015_00.png --archive-dir /forecasts/archive
```

A run takes about 5 minutes: the AIFS download is the longest part (about 1 GB),
and feature extraction takes 1–2 minutes. NCMRWF is processed first, so a
missing or wrong-date NEPS file stops the run before anything is downloaded.

**When to run:** the 00Z GEFS data is usually on NOAA by about 05–06 UTC, and
AIFS-ENS on ECMWF open data by about 07–08 UTC. ECMWF keeps only the last few
days of open data, so run each day's forecast within a day or two.

**Compute nodes without internet:** download on a node that has internet, then
process anywhere:

```bash
python run_forecast.py --date 20261015 --download-only --out-dir /scratch/dwf
python run_forecast.py --date 20261015 --skip-download --out-dir /scratch/dwf --ncmrwf-dir /data/neps
```

`slurm/run_forecast.sh` submits these two steps as SLURM jobs, the second
waiting for the first. Edit the four paths at the top, and the partition,
then run `bash slurm/run_forecast.sh 20261015`, with the conda environment
activated.

**AIFS-ENS run locally:** pass `--aifs-zarr /path/to/init_20261015T00.zarr` and
AIFS isn't downloaded. The store must be in the format the model was trained on:
`tp(time, number, prediction_timedelta, lat, lon)` in **metres**, 6-hourly
amounts (not cumulative), on the 129 × 135 grid, with every 6 h lead from +6 h
to +126 h (the DRY/WET event fractions span all of days 1–5), at least 20 members, and the init date matching `--date`. Anything
else is refused with a message, including rainfall totals that are
implausible, which is the usual sign of mm and metres being mixed up.

## Output

- `<fig-out>.png`: the DRY and WET probability maps;
- `<fig-out>.npz`: the forecast record, loadable with `numpy.load`:
  `p_dry` and `p_wet` (129 × 135 float32, NaN over sea), `lats`, `lons`,
  `date`, `cycle`, `config`;
- with `--archive-dir`, a copy at `<archive-dir>/ncmrwf_reduced/<date>_00.npz`,
  so that a season of forecasts can be loaded with
  `pipeline.archive.read_archive(archive_dir)`.

## The model

Two XGBoost classifiers (DRY, WET) in `models/ncmrwf_reduced/`, trained on
2021–2024 (valid dates 15 October – 31 December) against IMD 0.25° gridded
rainfall. LASSO selected the features each model uses:

| DRY (8 features) | WET (8 features) |
|---|---|
| NCMRWF mean rainfall day 3; NCMRWF DRY event fraction | NCMRWF WET event fraction |
| AIFS mean rainfall day 3; AIFS spread days 2–3; AIFS DRY event fraction | AIFS spread days 1–4; AIFS WET event fraction |
| GEFS PWAT mean; GEFS CAPE mean | GEFS PWAT mean; GEFS CAPE spread |

The GEFS features are averages over 41 leads, every 3 h from +3 h to +123 h:
the same day 1–5 window as the rainfall. Their names end in `_roll` for
historical reasons; each one is a single 5-day mean. The event definitions
are the same for every source and for the IMD labels: dry day < 1 mm, wet day ≥ 1 mm.

**Skill** (Brier skill score against the 1991–2020 IMD climatology, held-out years):

| Event | 2020 | 2025 |
|---|---|---|
| DRY | 0.445 | 0.564 |
| WET | 0.439 | 0.615 |

## Limitations

- **Season:** trained on mid-October to late December only. Forecasts outside
  that window are extrapolation, with no skill estimate.
- **NCMRWF product:** the 2021–2024 training years used NCMRWF's earlier
  11-member GRIB ensemble. The current NetCDF product has 23 members
  (1 control + 22 perturbed) and exists from 2025; the 2025 scores above were
  obtained with it.
- **Missing upstream data:** if NOAA or ECMWF is missing part of a cycle, the
  download reports exactly which files failed. Rerunning retries only those.

## Tests

```bash
python -m pytest tests
```

These run offline in about 20 s and never touch the network:
- the model loads and predicts;
- every feature it needs is assembled;
- NCMRWF event definitions, date checks, and that a `.lag.nc` is ignored;
- the GEFS downloader against a stand-in for NOAA (throttling, missing files, message order);
- the AIFS unit guard;
- the `.npz` record;
- command-line validation.

## Files

```
run_forecast.py          command line: download -> features -> models -> map + record
pipeline/
  ncmrwf_features.py     NEPS file -> rainfall mean/spread and DRY/WET event fractions
  download_aifs.py       AIFS-ENS v2 rainfall from ECMWF open data
  aifs_features.py       AIFS (downloaded GRIB or local zarr) -> rainfall features
  download_gefs.py       GEFS PWAT/CAPE mean/spread from NOAA, 41 leads
  gefs_features.py       -> PWAT/CAPE 5-day means
  inference.py           feature assembly + the two XGBoost models
  grid.py                target grid, land mask, GRIB read + bilinear regrid
  archive.py             the .npz forecast record
  check.py, runtime.py   --check report; CPU detection (SLURM-aware)
models/                  ncmrwf_reduced/xgb_{dry,wet}.json + .mask.npz, land_mask.npy
slurm/run_forecast.sh    download + forecast as two SLURM jobs
tests/                   offline test suite
```
