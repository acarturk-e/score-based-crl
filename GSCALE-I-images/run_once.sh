#!/bin/bash

NUM_SAMPLES=10000

if [ -z "$1" ]; then
  echo "Error: Required positional argument (number of balls) is missing." >&2
  exit 1
fi
NUM_BALLS=$1

if [ -z "$2" ]; then
  echo "Error: Required positional argument (data dir) is missing." >&2
  exit 1
fi
DATADIR=$2

if [ -z "$3" ]; then
  echo "Error: Required positional argument (device) is missing. Provide cpu or cuda:#" >&2
  exit 1
fi
DEVICE=$3

eval "$(conda shell.bash hook)" | exit
conda activate ScoreCRL | exit
python generate_data.py $NUM_BALLS $NUM_SAMPLES $DATADIR | exit
python train_ldr.py --device $DEVICE $DATADIR | exit
python train_reconstruct.py --device $DEVICE $DATADIR | exit
python x_to_y.py --device $DEVICE $DATADIR | exit
python train_disentangle.py --device $DEVICE $DATADIR | exit
echo "Finished datadir $DATADIR on $DEVICE" | exit
