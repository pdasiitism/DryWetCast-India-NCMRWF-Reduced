#!/bin/bash
# Daily forecast as two SLURM jobs: download (needs internet), then process.
# Edit the four settings below and the #SBATCH lines for your cluster, then:
#   bash slurm/run_forecast.sh 20261015          (date defaults to today, UTC)
set -euo pipefail
DATE=${1:-$(date -u +%Y%m%d)}
REPO=$(cd "$(dirname "$0")/.." && pwd)
NEPS_DIR=/path/to/neps          # folder with <date>.nc or <date>_NPES.nc
WORK=/path/to/scratch/dwf       # downloads (~1 GB/cycle) + regrid cache
OUT=/path/to/forecasts          # PNG + .npz per day
DL_PARTITION=general            # a partition whose nodes can reach the internet
mkdir -p "$REPO/slurm/logs" "$OUT"
common="--time=12:00:00 --output=$REPO/slurm/logs/%x_%j.out --chdir=$REPO"
dl=$(sbatch --parsable $common --job-name=dwf_dl_$DATE --partition=$DL_PARTITION --cpus-per-task=8 --mem=8G \
     --wrap="python run_forecast.py --date $DATE --download-only --out-dir $WORK")
sbatch $common -d afterok:$dl --job-name=dwf_run_$DATE --cpus-per-task=16 --mem=32G \
     --wrap="python run_forecast.py --date $DATE --skip-download --out-dir $WORK --ncmrwf-dir $NEPS_DIR --fig-out $OUT/ncmrwf_reduced_${DATE}_00.png --archive-dir $OUT/archive"
echo "submitted: download $dl, then the forecast (logs in $REPO/slurm/logs/)"
