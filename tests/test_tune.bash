#!/bin/bash
> logs/test_tune.log
models=("BlockRNN" "TCN" "Transformer" "NLinear" "DLinear" "NBEATS")
# Removed XGB and Linear as they are costly

# Iterating over the models listed above
for model in "${models[@]}"; do
  python -u train_tune.py --model "$model" --target oxygen --site ARIK \
    --date 2023-03-09 --epochs 1 --tune --test &>> logs/test_tune.log
done

# Need to treat RNN and TFT separately as they don't accept past covariates
models=("RNN" "TFT")

for model in "${models[@]}"; do
  python -u train_tune.py --model "$model" --target oxygen --site ARIK \
    --date 2023-03-09 --epochs 1 --nocovs --tune --test &>> logs/test_tune.log
done