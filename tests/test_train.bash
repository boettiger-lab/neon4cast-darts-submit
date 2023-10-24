#!/bin/bash
> logs/test_train.log
models=("BlockRNN" "TCN" "Transformer" "NLinear" "DLinear" "NBEATS"
        )

# Iterating over the models listed above
for model in "${models[@]}"; do
  python -u train_tune.py --model "$model" --target temperature --site FLNT \
    --date 2023-03-09 --epochs 1 &>> ./logs/test_train.log
done

# Need to treat RNN and TFT separately as they don't accept past covariates
models=("RNN" "TFT")

for model in "${models[@]}"; do
  python -u train_tune.py --model "$model" --target temperature --site FLNT \
    --date 2023-03-09 --epochs 1 --nocovs &>> logs/test_train.log
done