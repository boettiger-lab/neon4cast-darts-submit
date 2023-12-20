#!/bin/bash

models=("BlockRNN" "Transformer" "NLinear" "DLinear" "NBEATS" "TCN")

for model in "${models[@]}"; do
  ./jobs/train_across_sites.bash $model $1 $2 &
done