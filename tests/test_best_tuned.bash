#!/bin/bash

models=("BlockRNN" "TCN" "Transformer" "NLinear" "DLinear" "NBEATS"
        "XGB" "Linear")

# Iterating over the models listed above
for model in "${models[@]}"; do
  python train_tune.py --model "$model" --target oxygen --site POSE \
    --date 2023-03-08 --epochs 1 --test_tuned
done

# Need to treat RNN and TFT separately as they don't accept past covariates
models=("RNN" "TFT")

for model in "${models[@]}"; do
  python train_tune.py --model "$model" --target oxygen --site POSE \
    --date 2023-03-08 --epochs 1 --nocovs --test_tuned
done