from darts.utils.likelihood_models import QuantileRegression
from residual_learning.utils import (
                BaseForecaster, 
                ResidualForecasterDarts,
                TimeSeriesPreprocessor,
)
import argparse
import time
import os
import copy

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

start = time.time()
parser = argparse.ArgumentParser()
parser.add_argument("--model", default="BlockRNN", type=str)
parser.add_argument("--target", default="oxygen", type=str)
parser.add_argument("--site", default="BARC", type=str)
parser.add_argument("--tune", default=False, action="store_true")
args = parser.parse_args()

data_preprocessor = TimeSeriesPreprocessor(input_csv_name = "../targets.csv.gz",
                                           load_dir_name = "../preprocessed_timeseries/")
data_preprocessor.load()

output_csv = copy.copy(args.model)
if args.tune:
    output_csv += "_tuned"

forecaster = BaseForecaster(model=args.model,
                    target_variable_column_name=args.target,
                    data_preprocessor=data_preprocessor,
                    datetime_column_name="datetime",
                    covariates_names=["air_tmp", "chla", "temperature"],
                    output_csv_name=f"forecasts/{output_csv}.csv",
                    validation_split_date="2023-02-26",
                    model_hyperparameters={'input_chunk_length': 180},
                    model_likelihood={"likelihood": QuantileRegression([0.05, 0.1, 0.5, 0.9, 0.95])},
                    forecast_horizon=30,
                    site_id=args.site)

if args.tune:
    if args.model == "BlockRNN":
        forecaster.tune({
            "input_chunk_length": [60, 180, 360, 540],
            "hidden_dim": [16, 32, 64, 126],
            "model": ["RNN", "GRU", "LSTM"],
            "n_rnn_layers": [2, 3, 4, 5],
        })
    elif self.model == "TCN":
        forecaster.tune({
            "" : [],
        })
    elif self.model == "RNN":
        forecaster.tune({
            "" : [],
        })
    elif self.model == "Transformer":
        forecaster.tune({
            "" : [],
        })
    elif self.model == "NLinear":
        forecaster.tune({
            "" : [],
        })
    elif self.model == "DLinear":
        forecaster.tune({
            "" : [],
        })
    elif self.model == "XGB":
        forecaster.tune({
            "" : [],
        })
    elif self.model == "NBeats":
        forecaster.tune({
            "" : [],
        })
    elif self.model == "Linear":
        forecaster.tune({
            "" : [],
        })

forecaster.make_forecasts()

print(time.time() - start)