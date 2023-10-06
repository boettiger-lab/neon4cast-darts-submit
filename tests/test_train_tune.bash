#!/bin/bash

models=("BlockRNN" "TCN" "Transformer" "NLinear" "DLinear" "NBEATS"
        "XGB" "Linear")

# Iterating over the models listed above
for model in "${models[@]}"; do
  python train_tune.py --model "$model" --target temperature --site POSE \
    --date 2023-03-09 --epochs 1
done

models=("RNN" "TFT")

# Iterating over the models listed above
for model in "${models[@]}"; do
  python train_tune.py --model "$model" --target temperature --site POSE \
    --date 2023-03-09 --epochs 1 --nocovs
done