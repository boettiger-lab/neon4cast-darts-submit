#!/bin/bash

models=("BlockRNN" "TCN" "Transformer" "NLinear" "DLinear" "NBEATS"
        )
# Leaving out XGB and Linear

# Iterating over the models listed above
for model in "${models[@]}"; do
  > "logs/$1/tune_${model}.log"
  python -u train_tune.py --model "$model" --target "$1" --site FLNT \
    --date 2023-03-09 --epochs 200 --tune --num_trials 50 &> \
    "logs/FLNT/$1/tune_${model}.log" &
done

# Need to treat RNN and TFT separately as they don't accept past covariates
models=("RNN" "TFT")

for model in "${models[@]}"; do
  > "logs/$1/tune_${model}.log"
  python -u train_tune.py --model "$model" --target "$1" --site FLNT \
    --date 2023-03-09 --epochs 200 --nocovs --tune --num_trials 50 &> \
    "logs/FLNT/$1/tune_${model}.log" &
done
