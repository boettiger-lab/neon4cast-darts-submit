from darts.utils.likelihood_models import (
                GaussianLikelihood,
                QuantileRegression
)
from residual_learning.utils import (
                BaseForecaster, 
                ResidualForecaster,
                TimeSeriesPreprocessor,
)
import argparse
import time
import os
import copy
import json
import yaml
start = time.time()

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# I want to hyperparameters from a yaml in training_hyper...
# and save best fit hypers into tuned_hyper..
for dir in ["tuned_hyperparameters/", "forecasts/"]:
    if not os.path.exists(dir):
        os.makedirs(dir)

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="BlockRNN", type=str)
parser.add_argument("--target", default="oxygen", type=str)
parser.add_argument("--site", default="BARC", type=str)
parser.add_argument("--date", default="2023-02-26", type=str)
parser.add_argument("--tune", default=False, action="store_true")
parser.add_argument("--epochs", default=500, type=int)
parser.add_argument("--nocovs", default=False, action="store_true")
args = parser.parse_args()

# Load hypers
with open(f"hyperparameters/train/{args.model}.yaml") as f:
    hyperparams_dict = yaml.safe_load(f)
# And dealing with the tricky inputs of likelihoods and lags
model_likelihood = {"QuantileRegression": 
                       {"likelihood": QuantileRegression([0.01, 0.05, 0.1, 
                                                          0.3, 0.5, 0.7, 
                                                          0.9, 0.95, 0.99])},
                    "Quantile": {"likelihood": "quantile"},
                    "Gaussian": GaussianLikelihood(),
                    "None": None}[hyperparams_dict["model_likelihood"]]
if "lags" in hyperparams_dict["model_hyperparameters"].keys():
    end_range = copy.copy(hyperparams_dict["model_hyperparameters"]["lags"])
    hyperparams_dict["model_hyperparameters"]["lags"] = [-i for i in range(1, end_range)]
    if not args.nocovs:
        hyperparams_dict["model_hyperparameters"]\
         ["lags_past_covariates"] = [-i for i in range(1, end_range)]

    
# Need to accomodate options for quantile regression vs gaussian vs dropout

# Using data as covariates besides the target series
covariates_list = ["air_tmp", "chla", "temperature", "oxygen"]
covariates_list.remove(args.target)
if args.nocovs:
    covariates_list = None

data_preprocessor = TimeSeriesPreprocessor(input_csv_name = "targets.csv.gz",
                                           load_dir_name = "preprocessed_timeseries/")
data_preprocessor.load()

# Handling csv names and directories for the final forecast
if not os.path.exists(f"forecasts/{args.site}/{args.target}/"):
    os.makedirs(f"forecasts/{args.site}/{args.target}/")
output_csv_name = f"forecasts/{args.site}/{args.target}/{args.model}"
if args.tune:
    output_csv_name += "_tuned"

# Instantiating the model
forecaster = BaseForecaster(model=args.model,
                    target_variable_column_name=args.target,
                    data_preprocessor=data_preprocessor,
                    covariates_names=covariates_list,
                    output_csv_name=f"{output_csv_name}.csv",
                    validation_split_date=args.date,
                    model_hyperparameters=hyperparams_dict["model_hyperparameters"],
                    model_likelihood=model_likelihood,
                    site_id=args.site,
                    epochs=args.epochs,)

# Handling tuning. Hmm, might be better to tuck this away, potentially in yaml
if args.tune:
    if args.model == "BlockRNN":
        forecaster.tune({
            "input_chunk_length": [60, 180, 360, 540],
            "hidden_dim": [16, 32, 64, 126],
            "model": ["RNN", "GRU", "LSTM"],
            "n_rnn_layers": [2, 3, 4, 5],
            "add_encoders": ["past"],
        })
    elif args.model == "TCN":
        forecaster.tune({
            "input_chunk_length": [60, 180, 360, 540],
            "kernel_size": [2, 3, 4, 5],
            "num_filters": [2, 3, 4, 5],
            "add_encoders": ["past"],
        })
    elif args.model == "RNN":
        forecaster.tune({
            "input_chunk_length": [60, 180, 360, 540],
            "hidden_dim": [16, 32, 64, 126],
            "model": ["RNN", "GRU", "LSTM"],
            "n_rnn_layers": [2, 3, 4, 5],
            "add_encoders": ["future"],
        })
    elif args.model == "Transformer":
        forecaster.tune({
            "input_chunk_length": [60, 180, 360, 540],
            "nhead": [2, 4, 8, 10],
            "num_encoder_layers": [2, 3, 4],
            "num_decoder_layers": [2, 3, 4],
            "add_encoders": ["past"],
        })
    elif args.model == "NLinear":
        forecaster.tune({
            "input_chunk_length": [60, 180, 360, 540],
            "const_init": [True, False],
            "normalize": [True, False],
            "batch_size": [16, 32, 64, 128],
            "add_encoders": ["past_and_future"],
        })
    elif args.model == "DLinear":
        forecaster.tune({
            "input_chunk_length": [60, 180, 360, 540],
            "kernel_size": [2, 25, 51, 101],
            "const_init": [True, False],
            "batch_size": [16, 32, 64, 128],
            "add_encoders": ["past_and_future"],
        })
    elif args.model == "XGB":
        forecaster.tune({
            "lags" : [[-i for i in range(1, 60)],
                      [-i for i in range(1, 180)],
                      [-i for i in range(1, 360)],
                      [-i for i in range(1, 540)]],
            "lags_past_covariates" : [[-i for i in range(1, 60)],
                                      [-i for i in range(1, 180)],
                                      [-i for i in range(1, 360)],
                                      [-i for i in range(1, 540)]],
            "lags_future_covariates" : [[-i for i in range(1, 60)],
                                        [-i for i in range(1, 180)],
                                        [-i for i in range(1, 360)],
                                        [-i for i in range(1, 540)]],
            "add_encoders": ["past_and_future"],
        })
    elif args.model == "NBeats":
        forecaster.tune({
            "input_chunk_length": [60, 180, 360, 540],
            "generic_architecture": [True, False],
            "num_stacks": [1, 2, 3, 4],
            "num_layers": [1, 2, 4, 8],
            "add_encoders": ["past"],
        })
    elif args.model == "Linear":
        forecaster.tune({
            "lags" : [[-i for i in range(1, 60)],
                      [-i for i in range(1, 180)],
                      [-i for i in range(1, 360)],
                      [-i for i in range(1, 540)]],
            "lags_past_covariates" : [[-i for i in range(1, 60)],
                                      [-i for i in range(1, 180)],
                                      [-i for i in range(1, 360)],
                                      [-i for i in range(1, 540)]],
            "lags_future_covariates" : [[-i for i in range(1, 60)],
                                        [-i for i in range(1, 180)],
                                        [-i for i in range(1, 360)],
                                        [-i for i in range(1, 540)]],
            "add_encoders": ["past_and_future"],
        })
    elif args.model == "TFT":
        forecaster.tune({
            "input_chunk_length": [60, 180, 360, 540],
            "hidden_size": [16, 64, [64, 64], [64, 64, 64]],
            "full_attention": [True, False],
            "lstm_layers": [1, 2, 3, 4],
            "add_encoders": ["past_and_future"],
        })

# Read and write to script to keep track of hyperparameters
forecaster.make_forecasts()

# Need to do work here on putting tuned hyperparameters in a yaml


print(f"Runtime: {time.time() - start:.0f} seconds")