#!/bin/bash
#SBATCH --job-name=sdm-geom-pilot
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/pilot_%j.out
#SBATCH --error=logs/pilot_%j.err

# Run the local-ID × calibration pilot on VEGA.
#
# Usage:
#   sbatch slurm/run_pilot_vega.sbatch <entity> <algo> <track> <level>
#
# Example:
#   sbatch slurm/run_pilot_vega.sbatch torrentium rf full 20
#
# Assumes:
#   - Project cloned at $HOME/sdm-geometry
#   - .venv created and dependencies installed
#   - Real data already staged in data/raw/ (or accessible via path)

set -euo pipefail

ENTITY="${1:?entity required}"
ALGO="${2:-rf}"
TRACK="${3:-full}"
LEVEL="${4:-20}"

cd "$HOME/sdm-geometry"
mkdir -p logs

# Activate venv
source .venv/bin/activate

echo "Host: $(hostname)"
echo "Date: $(date)"
echo "Cell: ${ENTITY}_${ALGO}_${TRACK}_L${LEVEL}"
echo "Python: $(python --version)"
echo

python scripts/02_pilot_single_species.py \
    --entity "$ENTITY" \
    --algorithm "$ALGO" \
    --track "$TRACK" \
    --level "$LEVEL"

echo
echo "Done at $(date)"
