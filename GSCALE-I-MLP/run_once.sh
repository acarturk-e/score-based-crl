#!/bin/bash

DATA_DIR=""
DEVICE="cpu"
#DEVICE="cuda:0"

# After parsing options, shift to process additional arguments
shift $((OPTIND - 1))

# Any remaining arguments after options are positional.
# We _require_ data dir as a positional argument
if [ -z "$1" ]; then
  echo "Error: Required positional argument (data dir) is missing." >&2
  exit 1
fi
DATA_DIR=$1

# Pass your conda environment name here
echo "$(date) Initializing conda"
eval "$(conda shell.bash hook)"
conda activate ScoreCRL || exit

python generate_data.py -d "$DATA_DIR" && \
  python run_score_diff.py -d "$DATA_DIR" --device "$DEVICE" && \
  python run_gscalei.py -d "$DATA_DIR" --device "$DEVICE"
