#!/bin/bash

models=("RNN" "TFT")

for model in "${models[@]}"; do
  ./jobs/train_across_sites_nocovs.bash $model $1 $2 &
done