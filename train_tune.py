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
parser.add_argument("--num_trials", default=1, type=int)
parser.add_argument("--nocovs", default=False, action="store_true")
parser.add_argument("--test_tuned", default=False, action="store_true")
parser.add_argument("--verbose", default=False, action="store_true")
args = parser.parse_args()

# Load hypers
hyperparams_loc = f"hyperparameters/train/{args.model}"
if args.test_tuned:
    hyperparams_loc = f"hyperparameters/tuned/{args.model}"
with open(f"{hyperparams_loc}.yaml") as f:
    hyperparams_dict = yaml.safe_load(f)
# And dealing with the tricky inputs of likelihoods and lags
model_likelihood = {"QuantileRegression": 
                       {"likelihood": QuantileRegression([0.01, 0.05, 0.1, 
                                                          0.3, 0.5, 0.7, 
                                                          0.9, 0.95, 0.99])},
                    "Quantile": {"likelihood": "quantile"},
                    "Gaussian": GaussianLikelihood(),
                    "None": None}[hyperparams_dict["model_likelihood"]]
    
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
extras = {"epochs": args.epochs,
          "verbose": args.verbose,}
    
forecaster = BaseForecaster(model=args.model,
                    target_variable_column_name=args.target,
                    data_preprocessor=data_preprocessor,
                    covariates_names=covariates_list,
                    output_csv_name=f"{output_csv_name}.csv",
                    validation_split_date=args.date,
                    model_hyperparameters=hyperparams_dict["model_hyperparameters"],
                    model_likelihood=model_likelihood,
                    site_id=args.site,
                    num_trials=args.num_trials,
                    **extras)

# Handling tuning. Hmm, might be better to tuck this away, potentially in yaml
# TO DOS: Add learning rate option
if args.tune:
    if args.model == "BlockRNN":
        forecaster.tune({
            "input_chunk_length": [60, 180, 360, 540],
            "hidden_dim": [16, 32, 64, 126],
            "model": ["RNN", "GRU", "LSTM"],
            "n_rnn_layers": [2, 3, 4, 5],
            "add_encoders": ["past"],
            "lr": [1e-3, 1e-4, 1e-5],
        })
    elif args.model == "TCN":
        forecaster.tune({
            "input_chunk_length": [60, 180, 360, 540],
            "kernel_size": [2, 3, 4, 5],
            "num_filters": [2, 3, 4, 5],
            "add_encoders": ["past"],
            "lr": [1e-3, 1e-4, 1e-5],
        })
    elif args.model == "RNN":
        forecaster.tune({
            "input_chunk_length": [60, 180, 360, 540],
            "hidden_dim": [16, 32, 64, 126],
            "model": ["RNN", "GRU", "LSTM"],
            "n_rnn_layers": [2, 3, 4, 5],
            "add_encoders": ["future"],
            "lr": [1e-3, 1e-4, 1e-5],
        })
    elif args.model == "Transformer":
        forecaster.tune({
            "input_chunk_length": [60, 180, 360, 540],
            "nhead": [2, 4, 8, 10],
            "num_encoder_layers": [2, 3, 4],
            "num_decoder_layers": [2, 3, 4],
            "add_encoders": ["past"],
            "lr": [1e-3, 1e-4, 1e-5],
        })
    elif args.model == "NLinear":
        forecaster.tune({
            "input_chunk_length": [60, 180, 360, 540],
            "const_init": [True, False],
            "batch_size": [16, 32, 64, 128],
            "add_encoders": ["past_and_future"],
            "lr": [1e-3, 1e-4, 1e-5],
        })
    elif args.model == "DLinear":
        forecaster.tune({
            "input_chunk_length": [60, 180, 360, 540],
            "kernel_size": [2, 25, 51, 101],
            "const_init": [True, False],
            "batch_size": [16, 32, 64, 128],
            "add_encoders": ["past_and_future"],
            "lr": [1e-3, 1e-4, 1e-5],
        })
    elif args.model == "XGB":
        forecaster.tune({
            "lags" : [60, 180, 360, 540],
            "lags_past_covariates" : [60, 180, 360, 540],
            "add_encoders": ["past"],
        })
    elif args.model == "NBeats":
        forecaster.tune({
            "input_chunk_length": [60, 180, 360, 540],
            "generic_architecture": [True, False],
            "num_stacks": [1, 2, 3, 4],
            "num_layers": [1, 2, 4, 8],
            "add_encoders": ["past"],
            "lr": [1e-3, 1e-4, 1e-5],
        })
    elif args.model == "Linear":
        forecaster.tune({
            "lags" : [60, 180, 360, 540],
            "lags_past_covariates" : [60, 180, 360, 540],
            "add_encoders": ["past"],
        })
    elif args.model == "TFT":
        forecaster.tune({
            "input_chunk_length": [60, 180, 360, 540],
            "hidden_size": [16, 64, 128, 256],
            "full_attention": [True, False],
            "lstm_layers": [1, 2, 3, 4],
            "add_encoders": ["past_and_future"],
            "lr": [1e-3, 1e-4, 1e-5],
        })

# Adding hyperparameters to a yaml file to use later
if args.tune:
    with open(f"hyperparameters/tuned/{args.model}.yaml", 'w') as file:
        tuned_hyperparams = {"model_hyperparameters": forecaster.hyperparams, 
                             "model_likelihood": hyperparams_dict["model_likelihood"]}
        
        yaml.dump(tuned_hyperparams, file, default_flow_style=False)
        
forecaster.make_forecasts()

print(f"Runtime for {args.model} on {args.target} at {args.site}: {time.time() - start:.0f} seconds")