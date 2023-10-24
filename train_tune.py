from darts.utils.likelihood_models import (
                GaussianLikelihood,
                QuantileRegression
)
from utils import (
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


parser = argparse.ArgumentParser()
parser.add_argument("--model", default="BlockRNN", type=str,
                    help="Specify which Darts model to train with.")
parser.add_argument("--target", default="oxygen", type=str,
                    help="Specify which target time series to train on"+\
                    "[oxygen, temperature, chla].")
parser.add_argument("--site", default="BARC", type=str,
                    help="Denotes which site to use.")
parser.add_argument("--date", default="2023-03-09", type=str,
                    help="Flags for the validation split date, "+\
                    "n.b. that this should align with last date " +\
                    "of the preprocessed time series.")
parser.add_argument("--tune", default=False, action="store_true",
                    help="Will run tuning on the selected model and "+\
                    "time series. Options for tuning are specified in " +\
                    "this python script.")
parser.add_argument("--epochs", default=200, type=int, 
                    help="The number of epochs to train a model for.")
parser.add_argument("--num_trials", default=1, type=int,
                    help="The number of trials for tuning.")
parser.add_argument("--nocovs", default=False, action="store_true",
                    help="This nullifies the use of the other target time series "+\
                    "at that site for covariates.")
parser.add_argument("--test_tuned", default=False, action="store_true",
                    help="This selects the hyperparameters saved from "+\
                    "the previous tuning run for that target series "+\
                    "and Darts model.")
parser.add_argument("--verbose", default=False, action="store_true",
                    help="An option to use if more verbose output is desired "+\
                    "while training.")
parser.add_argument("--test", default=True, action="store_false",
                    help="This boolean flag if called will stop hyperparameters "+\
                    "from being saved.")
parser.add_argument("--device", default=0, type=int,
                    help="Specify which GPU device to use [0,1].")
parser.add_argument("--suffix", default=None, type=str,
                    help="Suffix to append to the output csv of the forecast.")
args = parser.parse_args()

# For non-quantile regression, add 2 CL flags, one to store true another
# to say which non-quantile regression to use, also need to save these differently

# Need to flag to say forecast didn't use covariates; also need to be careful with
# time axis encoder here, need to save these differently
if __name__ == "__main__":
    # Selecting the device
    os.environ["CUDA_VISIBLE_DEVICES"] = "1" if args.device else "0"
    
    # Loading hyperparameters
    hyperparams_loc = f"hyperparameters/train/{args.target}/{args.model}"
    if args.test_tuned:
        hyperparams_loc = f"hyperparameters/tuned/{args.target}/{args.model}"
    with open(f"{hyperparams_loc}.yaml") as f:
        hyperparams_dict = yaml.safe_load(f)
    # Dealing with the tricky inputs of likelihoods, also would need to return
    # to how dropout is treated here especially downstream for .predict() 
    # if this gets to the docket of things to explore.
    model_likelihood = {"QuantileRegression": 
                           {"likelihood": QuantileRegression([0.01, 0.05, 0.1, 
                                                              0.3, 0.5, 0.7, 
                                                              0.9, 0.95, 0.99])},
                        "Quantile": {"likelihood": "quantile"},
                        "Gaussian": {"likelihood": GaussianLikelihood()},
                        "None": {"dropout": 0.1}}[hyperparams_dict["model_likelihood"]]
    
    # Using data as covariates besides the target series
    covariates_list = ["air_tmp", "chla", "temperature", "oxygen"]
    covariates_list.remove(args.target)
    if args.nocovs:
        covariates_list = None
    
    data_preprocessor = TimeSeriesPreprocessor(input_csv_name = "targets.csv.gz",
                                               load_dir_name = "preprocessed_timeseries/")
    data_preprocessor.load(args.site)
    
    # Handling csv names and directories for the final forecast
    if not os.path.exists(f"forecasts/{args.site}/{args.target}/"):
        os.makedirs(f"forecasts/{args.site}/{args.target}/")
    output_csv_name = f"forecasts/{args.site}/{args.target}/{args.model}"
    if args.tune:
        output_csv_name += "_tuned"
    if args.suffix is not None:
        output_csv_name += f"_{args.suffix}"
    
    # Instantiating the model
    extras = {"epochs": args.epochs,
              "verbose": args.verbose,}
    forecaster = BaseForecaster(model=args.model,
                        target_variable=args.target,
                        data_preprocessor=data_preprocessor,
                        covariates_names=covariates_list,
                        output_csv_name=f"{output_csv_name}.csv",
                        validation_split_date=args.date,
                        model_hyperparameters=hyperparams_dict["model_hyperparameters"],
                        model_likelihood=model_likelihood,
                        site_id=args.site,
                        num_trials=args.num_trials,
                        **extras)
    
    # 
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
                "nhead": [1],
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
        elif args.model == "NBEATS":
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
    if args.tune and args.test:
        # Making sure that there is a directory to save hypers in
        if not os.path.exists(f"hyperparameters/tuned/{args.target}"):
            os.makedirs(f"hyperparameters/tuned/{args.target}")
        # Saving the hyperparameters in a yaml file that can be accessed with this script
        with open(f"hyperparameters/tuned/{args.target}/{args.model}.yaml", 'w') as file:
            tuned_hyperparams = {"model_hyperparameters": forecaster.hyperparams, 
                                 "model_likelihood": hyperparams_dict["model_likelihood"]}
            
            yaml.dump(tuned_hyperparams, file, default_flow_style=False)
            
    forecaster.make_forecasts()
    
    print(f"Runtime for {args.model} on {args.target} at {args.site}: {(time.time() - start)/60:.2f} minutes")