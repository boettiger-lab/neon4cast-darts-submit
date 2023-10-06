from residual_learning.utils import (
    BaseForecaster, 
    ResidualForecaster,
    TimeSeriesPreprocessor,
    HistoricalForecaster,
)
from darts.utils.likelihood_models import QuantileRegression
import os
import pytest
import copy

@pytest.fixture
def data_preprocessor_():
    data_preprocessor = TimeSeriesPreprocessor(
                            input_csv_name="targets.csv.gz",
                            load_dir_name="preprocessed_timeseries/")
    data_preprocessor.load()
    return data_preprocessor

def test_TimeseriesPreprocessor(data_preprocessor_):
    assert data_preprocessor_ != None

def test_HistoricalForecaster(data_preprocessor_):
    data_preprocessor = copy.copy(data_preprocessor_)
    
    model = HistoricalForecaster(
                        data_preprocessor=data_preprocessor,
                        validation_split_date="2023-03-02",
                        site_id="BARC",
                        target_variable="oxygen",
                        output_csv_name=None,
    )
    model.make_forecasts()


def test_BaseForecaster(data_preprocessor_):
    data_preprocessor = copy.copy(data_preprocessor_)

    # Models that accept what I deem as a standard set of keyword arguments
    std_models = ["BlockRNN", "TCN", "Transformer", "NLinear", \
                         "DLinear", "NBEATS"]
    for model_name in std_models:
        model = BaseForecaster(
                        model=model_name,
                        data_preprocessor=data_preprocessor,
                        target_variable_column_name="oxygen",
                        datetime_column_name="datetime",
                        covariates_names=["chla", "temperature", "air_tmp"],
                        output_csv_name=None,
                        validation_split_date="2023-03-02",
                        model_hyperparameters={"input_chunk_length": 31},
                        model_likelihood={"likelihood": QuantileRegression([0.05, 0.1, 0.5, 0.9, 0.95])},
                        site_id="BARC",
                        epochs=1,
        )
        model.make_forecasts()

    # Testing models that require lags
    lag_models = ["XGB", "Linear"]
    for model_name in lag_models:
        model = BaseForecaster(
                    model=model_name,
                    data_preprocessor=data_preprocessor,
                    target_variable_column_name="oxygen",
                    datetime_column_name="datetime",
                    output_csv_name=None,
                    validation_split_date="2023-03-02",
                    model_likelihood={"likelihood": "quantile"},
                    site_id="BARC",
                    epochs=1,
                    model_hyperparameters={"lags": [-i for i in range(1, 2)]}
                )
        model.make_forecasts()
    
    # Lastly testing TFT separately to avoid conflicts with models that don't accept past covariates
    fut_models = ["RNN", "TFT"]
    for model_name in fut_models:
        model = BaseForecaster(
                        model=model_name,
                        data_preprocessor=data_preprocessor,
                        target_variable_column_name="oxygen",
                        datetime_column_name="datetime",
                        output_csv_name=None,
                        validation_split_date="2023-03-02",
                        model_likelihood={"likelihood": QuantileRegression([0.05, 0.1, 0.5, 0.9, 0.95])},
                        model_hyperparameters={"input_chunk_length": 31,
                                               "add_encoders": {'datetime_attribute': {'future': ['dayofyear']}}},
                        site_id="BARC",
                        epochs=1,
        )
        model.make_forecasts()
    
def test_HistoricalandResidualForecaster(data_preprocessor_):
    data_preprocessor = copy.copy(data_preprocessor_)

    model = HistoricalForecaster(
                        data_preprocessor=data_preprocessor,
                        validation_split_date="2023-03-02",
                        site_id="BARC",
                        target_variable="oxygen",
                        output_csv_name=None,
    )
    model.get_residuals()

    residual_model = ResidualForecaster(
        residuals=model.residuals,
        validation_split_date="2023-02-25",
        output_csv_name = None,
    )
    residual_model.make_residual_forecasts()