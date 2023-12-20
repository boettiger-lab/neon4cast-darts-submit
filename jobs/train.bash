#!/bin/bash

models=("BlockRNN" "Transformer" "NLinear" "DLinear" "NBEATS" "TCN")
# Deleted Linear and XGB because these were too computationally intensive

# Command line arguments: target site epochs

# Iterating over the models listed above
for model in "${models[@]}"; do
  > "logs/$2/$1/train_${model}.log"
  python -u train_tune.py --model "$model" --target "$1" --site "$2" \
    --epochs "$3" &> "logs/$2/$1/train_${model}.log" &
done

# Need to treat RNN and TFT separately as they don't accept past covariates
models=("RNN" "TFT")

for model in "${models[@]}"; do
  > "logs/$2/$1/train_${model}.log"
  python -u train_tune.py --model "$model" --target $1 --site "$2" \
    --epochs "$3" --nocovs &> "logs/$2/$1/train_${model}.log" &
done
