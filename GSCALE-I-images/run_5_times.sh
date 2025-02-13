#!/bin/bash -i

NUM_BALLS=3
mkdir -p "data"
mkdir -p "data/${NUM_BALLS}b"
for i in {1..5}; do
  ./run_once.sh $NUM_BALLS "data/${NUM_BALLS}b/$i" "cuda:0" || exit
done
