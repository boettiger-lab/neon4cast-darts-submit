from darts import TimeSeries
from typing import Optional
import pandas as pd
import matplotlib.pyplot as plt
from optuna.integration import PyTorchLightningPruningCallback
import pandas as pd
from darts.models import GaussianProcessFilter
from darts import TimeSeries
from sklearn.gaussian_process.kernels import RBF
from darts.models import (
                          BlockRNNModel, 
                          TCNModel, 
                          RNNModel, 
                          TransformerModel, 
                          NLinearModel, 
                          NBEATSModel
                         )
from darts.utils.likelihood_models import QuantileRegression
from darts.dataprocessing.transformers import Scaler
from darts.metrics import smape
import ray
import os
import optuna
import argparse
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

class BaseForecaster():
    def __init__(self,
                 model: Optional[str] = None,
                 input_csv_name: Optional[str] = None,
                 target_variable_column_name: Optional[str] = None,
                 datetime_column_name: Optional[str] = None,
                 covariates_names: Optional[list] = None,
                 output_csv_name: Optional[str] = "residual_forecaster_output.csv",
                 validation_split_date: Optional[str] = None, #YYYY-MM-DD
                 model_hyperparameters: Optional[dict] = None,
                 forecast_horizon: Optional[int] = 30,
                 site_id: Optional[str] = None,
                 ):
        self.model_ = {"BlockRNN": BlockRNNModel, 
                       "TCN": TCNModel, 
                       "RNN": RNNModel, 
                       "Transformer": TransformerModel,
                       "NLinear": NLinearModel,
                       "NBEATS": NBEATSModel}[model]
        self.input_csv_name = input_csv_name
        self.df = pd.read_csv(self.input_csv_name)
        self. df = self.df.loc[self.df.site_id == site_id]
        self.target_variable_column_name = target_variable_column_name
        self.datetime_column_name = datetime_column_name
        self.covariates_names = covariates_names
        self.output_csv_name = output_csv_name
        self.validation_split_date = validation_split_date
        self.forecast_horizon = forecast_horizon
        if model_hyperparameters == None:
            self.hyperparams = {"input_chunk_length" : 180}
        else:
            self.hyperparams = model_hyperparameters

        self._preprocess_data()
    
    def make_stitched_series(self, variable_tseries):
        """
        Returns a time series where the gaps have been filled in via
        Gaussian Process Filters
        """
        kernel = RBF()
        
        gpf_missing = GaussianProcessFilter(kernel=kernel, 
                                            alpha=0.001, 
                                            n_restarts_optimizer=100)
        
        gpf_missing_big_gaps = GaussianProcessFilter(kernel=kernel, 
                                                     alpha=2, 
                                                     n_restarts_optimizer=10)
        stitched_series = {}
    
        # Filtering the TimeSeries
        try:
            filtered = gpf_missing.filter(variable_tseries, num_samples=500)
            filtered_big_gaps = gpf_missing_big_gaps.filter(variable_tseries, 
                                                            num_samples=500)
        except:
            return None
    
        # If there is a gap over 7 indices, use big gap filter
        gap_series = variable_tseries.gaps()
        stitched_df = filtered.pd_dataframe()
        replacement_df = filtered_big_gaps.pd_dataframe()
        
        for index, row in gap_series.iterrows():
            if row["gap_size"] > 7:
                for date in pd.date_range(row["gap_start"], row["gap_end"]):
                    stitched_df.loc[date] = replacement_df.loc[date]
        
        stitched_series = TimeSeries.from_times_and_values(
                                    stitched_df.index, 
                                    stitched_df.values.reshape(
                                                len(stitched_df), 
                                                1, 
                                                -1))
        
        return stitched_series
        
    def _preprocess_data(self):
        """
        Performs gap filling and processing of data into format that
        Darts models will accept
        """
        times = pd.to_datetime(self.df[self.datetime_column_name])
        times = pd.DatetimeIndex(times)
        variable_list = self.covariates_names + [self.target_variable_column_name]
        
        var_series_dict = {var: TimeSeries.from_times_and_values(times, 
                                                                 self.df[var], 
                                                                 fill_missing_dates=True,
                                                                 freq="D") 
                                                        for var in variable_list}

        stitched_series_dict = {var: self.make_stitched_series(var_series_dict[var])
                                                    for var in variable_list}
        self.inputs = stitched_series_dict[self.target_variable_column_name]

        # Some time series can't be fitted with GP, often cause there's no data
        for key, value in stitched_series_dict.copy().items():
            if value == None:
                print(f"Failed to make a GP fit on {key}")
                del stitched_series_dict[key]
                self.covariates_names.remove(key)
            
        # Initializing covariates list then concatenating in for loop
        self.covariates = stitched_series_dict[self.covariates_names[0]]
        for cov_var in self.covariates_names[1:]:
            self.covariates = self.covariates.concatenate(stitched_series_dict[cov_var], 
                                                          axis=1, 
                                                          ignore_time_axis=True)

        # Splitting training and validation set
        year = int(self.validation_split_date[:4])
        month = int(self.validation_split_date[5:7])
        day = int(self.validation_split_date[8:])
        split_date = pd.Timestamp(year=year, month=month, day=day)
        self.training_set, self.validation_set = self.inputs.split_before(split_date)



    def tune(self,
             hyperparameter_dict: Optional[dict]
            ):
        """
        Sets up Optuna trial to perform hyperparameter tuning
        Input dictionary will be of the form {"hyperparamter": [values to be tested]}
        """
        # Setting up an optuna Trial
        def objective(trial):
            callback = [PyTorchLightningPruningCallback(trial, monitor="val_loss")]
            hyperparams = {key: trial.suggest_categorical(key, value) 
                                               for key, value in hyperparameter_dict.items()}
        
            model = self.model_(**hyperparams,
                                output_chunk_length=self.forecast_horizon,
                                likelihood=QuantileRegression([0.05, 0.1, 0.5, 0.9, 0.95]))
            
            self.scaler = Scaler()
            training_set, covariates = self.scaler.fit_transform([self.training_set,
                                                                  self.covariates])
            model.fit(training_set,
                      past_covariates=covariates,
                      epochs=400, 
                      verbose=False)
        
            predictions = model.predict(n=len(self.validation_set[:self.forecast_horizon]), 
                                            past_covariates=covariates, 
                                            num_samples=50)
            predictions = self.scaler.inverse_transform(predictions)
            smapes = smape(self.validation_set[:self.forecast_horizon], 
                           predictions, 
                           n_jobs=-1, 
                           verbose=False)
            
            smape_val = np.mean(smapes)
            return smape_val if smape_val != np.nan else float("inf")

        study = optuna.create_study(direction="minimize")
        
        study.optimize(objective, n_trials=3) # Note 10 trials pretty meaningless here
        
        self.hyperparams = study.best_trial.params

    def make_forecasts(self):
        """
        This function fits a Darts model to the training_set
        """
        print(self.hyperparams)
        self.model = self.model_(**self.hyperparams,
                                 output_chunk_length=self.forecast_horizon,
                                 likelihood=QuantileRegression([0.05, 0.1, 0.5, 0.9, 0.95]),
                                 random_state=0)
        self.scaler = Scaler()
        training_set, covariates = self.scaler.fit_transform([self.training_set,
                                                              self.covariates])
        self.model.fit(training_set,
                       past_covariates=covariates,
                       epochs=500, 
                       verbose=False)

        predictions = self.model.predict(n=self.forecast_horizon,
                                         past_covariates=covariates, 
                                         num_samples=500)
        predictions = self.scaler.inverse_transform(predictions)

        predictions.pd_dataframe().to_csv(self.output_csv_name)

    def get_historicals_and_residuals(self):
        """
        This function creates a historical forecast along with their residual errors 
        """
        # This presumes that the scaler will not have been modified in interim 
        # from calling `make_forecasts`
        training_set, covariates = self.scaler.transform([self.training_set,
                                                              self.covariates])
        historical_forecasts = self.model.historical_forecasts(
                                            series=training_set,
                                            past_covariates=covariates,
                                            num_samples=500,
                                            forecast_horizon=self.forecast_horizon,
                                            stride=self.forecast_horizon,
                                            retrain=False,
                                            last_points_only=False
                                            )
        historical_forecasts = [self.scaler.inverse_transform(historical_forecast) for
                                                historical_forecast in historical_forecasts]
        # Getting the target time series slice for the historical forecast
        self.historical_ground_truth = self.training_set.slice(
                                            historical_forecasts[0].time_index[0], 
                                            historical_forecasts[-1].time_index[-1])

        # Now concatenating the historical forecasts which were returned
        # as a list above
        self.historical_forecasts = historical_forecasts[0]
        for time_series in historical_forecasts[1:]:
            self.historical_forecasts = self.historical_forecasts.concatenate(time_series, 
                                                                axis=0, 
                                                                ignore_time_axis=True)

        self.residuals = self.historical_ground_truth - self.historical_forecasts

    def make_residuals_csv(self):
        covariates_df = self.covariates.pd_dataframe()
        forecast_df = self.historical_forecasts.pd_dataframe()
        observed_df = self.historical_ground_truth.pd_dataframe()
        residuals_df = self.residuals.pd_dataframe()

        # Creating a folder if it doesn't exist already
        if not os.path.exists(f"{self.model}_residuals/"):
            os.makedirs(f"{self.model}_residuals/")
        # Saving csv's in the **model name**_test directory
        df_dict = {"covariates": block_rnn_forecaster.covariates.pd_dataframe(),
                   "forecast": block_rnn_forecaster.historical_forecasts.pd_dataframe(),
                   "observed": block_rnn_forecaster.historical_ground_truth.pd_dataframe()}
        for variable, df in df_dict.items():
            df.to_csv(f"{self.model}_test/{variable}")
        
    

class ResidualForecasterDarts():
    def __init__(self,
                 historical_forecasts: Optional[TimeSeries] = None,
                 historical_ground_truth: Optional[TimeSeries] = None,
                 covariates: Optional[TimeSeries] = None,
                 output_csv_name: Optional[str] = "residual_forecaster_output.csv",
                 validation_split_date: Optional[str] = None, #YYYY-MM-DD
                 tune_model: Optional[bool] = False,
                 model_hyperparameters: Optional[dict] = None,
                 forecast_horizon: Optional[int] = 30
                 ):

        self.historical_forecasts = historical_forecasts
        self.historical_ground_truth = historical_ground_truth
        self.residuals = self.historical_ground_truth - self.historical_forecasts
        self.covariates = covariates
        self.output_csv_name = output_csv_name
        self.validation_split_date = validation_split_date
        self.forecast_horizon = forecast_horizon
        if model_hyperparameters == None:
            self.hyperparams = {"input_chunk_length": 180}
        else:
            self.hyperparams = model_hyperparameters
        self._preprocess_data()
 
        
    def _preprocess_data(self):
        """
        Divides input time series into training and validation sets
        """
        # When Concatenating time series, they have to be the same lengths
        # so here I get the slice dates that will work across the different time series
        start_date = max(self.historical_forecasts.time_index[0],
                         self.historical_ground_truth.time_index[0],
                         self.covariates.time_index[0])
        end_date = min(self.historical_forecasts.end_time(),
                       self.historical_ground_truth.end_time(),
                       self.covariates.end_time())
        
        self.covariates = self.covariates.slice(start_date, end_date)
        self.historical_forecasts = self.historical_forecasts.slice(start_date, end_date)
        self.historical_ground_truth = self.historical_ground_truth.slice(start_date, end_date)
        # Adding the historical forecast and observed data to the covariates
        for time_series in [self.historical_forecasts, self.historical_ground_truth]:
            self.covariates = self.covariates.concatenate(time_series, axis=1, ignore_time_axis=True)

        # Getting the date so that we can create the training and test set
        year = int(self.validation_split_date[:4])
        month = int(self.validation_split_date[5:7])
        day = int(self.validation_split_date[8:])
        split_date = pd.Timestamp(year=year, month=month, day=day)
        self.residuals = self.residuals.slice(start_date, end_date)
        self.training_set, self.validation_set = self.residuals.split_before(split_date)

    
    def tune(self,
             input_chunk_length: Optional[list] = [31, 60, 180, 356],
             kernel_size: Optional[list] = [2, 3, 5],
             num_filters: Optional[list] = [1, 3, 5],
             num_layers: Optional[list] = [None, 1, 2, 3],
             dilation_base: Optional[list] = [1, 2, 3],
             dropout: Optional[list] = [0.1, 0.2, 0.3]):
        """
        Sets up Optuna trial to perform hyperparameter tuning
        """
        # The objective function will be called in the optuna loop
        def objective(trial):
            callback = [PyTorchLightningPruningCallback(trial, monitor="val_loss")]
        
            # Selecting the hyperparameters
            input_chunk_length_ = trial.suggest_categorical("input_chunk_length", 
                                                               input_chunk_length)
            kernel_size_ = trial.suggest_categorical("kernel_size", kernel_size)
            num_filters_ = trial.suggest_categorical("num_filters", num_filters)
            num_layers_ = trial.suggest_categorical("num_layers", num_layers)
            dilation_base_ = trial.suggest_categorical("dilation_base", dilation_base)
            dropout_ = trial.suggest_categorical("dropout", dropout)

            tcn_model = TCNModel(input_chunk_length=input_chunk_length_,
                            kernel_size=kernel_size_,
                            num_filters=num_filters_,
                            dilation_base=dilation_base_,
                            output_chunk_length=self.forecast_horizon,
                            likelihood=QuantileRegression([0.05, 0.1, 0.5, 0.9, 0.95]))

            # Normalizing data to 0, 1 scale and fitting the model
            self.scaler = Scaler()
            training_set, covariates = self.scaler.fit_transform([self.training_set,
                                                                  self.covariates])
            tcn_model.fit(training_set,
                          past_covariates=covariates,
                          epochs=400, 
                          verbose=False)

            predictions = tcn_model.predict(n=len(self.validation_set[:self.forecast_horizon]), 
                                            past_covariates=covariates, 
                                            num_samples=50)
            predictions = self.scaler.inverse_transform(predictions)
            
            smapes = smape(self.validation_set[:self.forecast_horizon], 
                           predictions, 
                           n_jobs=-1, 
                           verbose=False)
            
            smape_val = np.mean(smapes)
            return smape_val if smape_val != np.nan else float("inf")


        study = optuna.create_study(direction="minimize")
        
        study.optimize(objective, n_trials=3) # INCREASE NUMBER OF TRIALS LATER ON
        
        # I save the best hyperparameters to the object self.hyperparameter so that these
        # hyperparameters can be easily referend in the future
        self.hyperparams = study.best_trial.params

    def make_residual_forecasts(self):
        """
        This function fits a TCN model to the residual error
        """
        # For clarity, I print the hyperparameters. Otherwise, following the standard
        # model fitting steps in Darts
        print(self.hyperparams)
        tcn = TCNModel(**self.hyperparams,
               output_chunk_length=self.forecast_horizon,
               likelihood=QuantileRegression([0.05, 0.1, 0.5, 0.9, 0.95]),
               random_state=0)
        self.scaler = Scaler()
        training_set, covariates = self.scaler.fit_transform([self.training_set,
                                                              self.covariates])
        tcn.fit(training_set,
                past_covariates=covariates,
                epochs=500, 
                verbose=False)

        predictions = tcn.predict(n=self.forecast_horizon,
                                  past_covariates=covariates, 
                                  num_samples=500)
        self.predictions = self.scaler.inverse_transform(predictions)

        self.predictions.pd_dataframe().to_csv(self.output_csv_name)
    