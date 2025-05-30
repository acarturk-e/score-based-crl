#!/bin/bash -i

# set the number of runs and the data directory
for i in {1..5}; do
  ./run_once.sh "data/mlp_$i" || exit
done
