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
                          DLinearModel,
                          NBEATSModel,
                          XGBModel,
                          LinearRegressionModel,
                          TFTModel,
                         )
from darts.utils.likelihood_models import QuantileRegression
from darts.dataprocessing.transformers import Scaler
from darts.metrics import smape
from datetime import datetime, timedelta
import ray
import CRPS.CRPS as forecastscore
import os
import optuna
import argparse
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def crps(forecast, observed):
    """
    Returns an array of CRPS scores 
    """
    forecast_array = forecast.pd_dataframe().values
    observed_array = observed.median().pd_series().values

    crps_scores = []
    for i in range(len(forecast_array)):
        # Note forecastscore is CRPS.CRPS
        crps, _, __ = forecastscore(forecast_array[i], observed_array[i]).compute()
        crps_scores.append(crps)

    crps_scores = TimeSeries.from_times_and_values(forecast.time_index, 
                                     crps_scores, 
                                     fill_missing_dates=True, freq="D")
    return crps_scores


class HistoricalForecaster():
    def __init__(self,
                 data_preprocessor: Optional = None,
                 output_csv_name: Optional[str] = "historical_forecaster_output.csv",
                 validation_split_date: Optional[str] = None, #YYYY-MM-DD
                 forecast_horizon: Optional[int] = 30,
                 site_id: Optional[str] = None,
                 target_variable: Optional[str] = None,
                 ):
        self.data_preprocessor = data_preprocessor
        self.output_csv_name = output_csv_name
        self.validation_split_date = validation_split_date
        self.forecast_horizon = forecast_horizon
        self.site_id = site_id
        self.target_variable = target_variable
        self._preprocess_data()
        # Using the medians of the GP filter
        median_df = self.training_set.median().pd_dataframe()
        median_df["timestamp"] = pd.to_datetime(median_df.index)
        median_df["day_of_year"] = median_df["timestamp"].dt.dayofyear
        
        # Computing average and std for doy's 
        self.doy_df = median_df.groupby(['day_of_year'])['0'].agg(['mean', 'std'])

    def _preprocess_data(self):
        stitched_series_dict = self.data_preprocessor.sites_dict[self.site_id]
        # If there was failure when doing the GP fit then we can't do preprocessing
        if self.target_variable in \
                self.data_preprocessor.sites_dict_null[self.site_id]:
            return "Cannot fit this target time series as no GP fit was performed."
        self.inputs = stitched_series_dict[self.target_variable]

        # Splitting training and validation set
        self.year = int(self.validation_split_date[:4])
        month = int(self.validation_split_date[5:7])
        day = int(self.validation_split_date[8:])
        split_date = pd.Timestamp(year=self.year, month=month, day=day)
        self.training_set, self.validation_set = self.inputs.split_before(split_date)

    def make_forecasts(self):
        """
        This function finds the historical mean and var, and uses these statistics for
        the forecast
        """
        # Filtering the previously computed averages and std for our dates of interest
        forecast_doys = pd.date_range(start=self.validation_split_date, 
                                      periods=self.forecast_horizon, 
                                      freq='D').dayofyear
        forecast_df = self.doy_df.loc[forecast_doys]

        # Function to give date from the numerical doy
        def day_of_year_to_date(year, day_of_year):
            base_date = datetime(year, 1, 1)
            target_date = base_date + timedelta(days=day_of_year - 1)
            return target_date

        samples = np.array([np.random.normal(self.doy_df.loc[self.doy_df.index == doy]["mean"],
                                    self.doy_df.loc[self.doy_df.index == doy]["std"]/2,
                                    size=(1, 500)) for doy in forecast_df.index])

        # Now creating an index going from doy to date
        date_index = [day_of_year_to_date(self.year, day) for day in forecast_df.index]
        forecast_df.index = date_index

        
        self.forecast_df = forecast_df
        self.forecast_ts = TimeSeries.from_times_and_values(forecast_df.index, samples)

    def plot(self):
        fig, ax1 = plt.subplots()
        ax1.plot(self.forecast_df.index, 
                 self.forecast_df["mean"], 
                 label="historical",
                 linewidth=2)
        ax1.fill_between(self.forecast_df.index, 
                         self.forecast_df["mean"] - self.forecast_df["std"], 
                         self.forecast_df["mean"] + self.forecast_df["std"],
                         alpha=0.2,)
        fig.autofmt_xdate()
        plt.legend()

    def get_residuals(self):
        residual_list = []
        # Going through each date and finding the difference between the doy historical mean and
        # the observed value
        for date in self.training_set.time_index:
            doy = date.dayofyear
            observed = self.training_set.slice_n_points_after(date, 1).median().values()[0][0]
            historical_mean = self.doy_df.loc[doy]["mean"]
            residual = observed - historical_mean
            residual_list.append(residual)

        self.residuals = TimeSeries.from_times_and_values(self.training_set.time_index, residual_list) 
        

class TimeSeriesPreprocessor():
    def __init__(self,
                 input_csv_name = "targets.csv.gz",
                 load_dir_name: Optional[str] = "preprocessed_timeseries/",
                 datetime_column_name: Optional[str] = "datetime",
                 covariates_names: Optional[list] = None,
                 ):
        self.input_csv_name = input_csv_name
        self.load_dir_name = load_dir_name
        self.df = pd.read_csv(self.input_csv_name)
        self.datetime_column_name = datetime_column_name
        self.sites_dict = {}
        self.sites_dict_null = {}
    
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

    def preprocess_data(self, site):
        """
        Performs gap filling and processing of data into format that
        Darts models will accept
        """
        site_df = self.df.loc[self.df.site_id == site]
        times = pd.to_datetime(site_df[self.datetime_column_name])
        times = pd.DatetimeIndex(times)
        variable_list = ["chla", "oxygen", "temperature", "air_tmp"]
        var_series_dict = {var: TimeSeries.from_times_and_values(times, 
                                                                 site_df[var], 
                                                                 fill_missing_dates=True,
                                                                 freq="D") 
                                                        for var in variable_list}

        stitched_series_dict = {var: self.make_stitched_series(var_series_dict[var])
                                                    for var in variable_list}

        # Deleting keys with none values
        keys_to_remove = [key for key, value in stitched_series_dict.items() if value == None]
        for key in keys_to_remove:
            del stitched_series_dict[key]

        self.sites_dict[site] = stitched_series_dict
        self.sites_dict_null[site] = keys_to_remove

    def save(self):
        # Check if there's a dir already
        if not os.path.exists(self.load_dir_name):
            os.makedirs(self.load_dir_name)

        # Saving each TimeSeries
        for site in self.sites_dict.keys():
            for variable in self.sites_dict[site]:
                self.sites_dict[site][variable].pd_dataframe()\
                    .to_csv(f"{self.load_dir_name}{site}-{variable}.csv")

    def load(self):
        # Need to check what are the possible variables that there could be in null, 
        # and when you load a series need to log which ones aren't added
        variables = {"chla", "oxygen", "temperature", "air_tmp"}
        sites_dict_present = {}
        
        # Need to fill sites_dict and sites_dict_null
        files = os.listdir(self.load_dir_name)
        for file in files:
            if file.endswith(".csv"):
                # Reading in file name; this is bad practice here, I should redo 
                # naming convention
                try:
                    site, variable = file.replace(".csv", "").split("_") # Change this to split at "-"
                except:
                    site, var1, var2 = file.replace(".csv", "").split("_")
                    variable = var1 + "_" + var2
                file_path = os.path.join(self.load_dir_name, file)
                df = pd.read_csv(file_path)
    
                # To make a time series, need to isolate time index and values
                times = pd.to_datetime(df["datetime"])
                times = pd.DatetimeIndex(times)
                values = df.loc[:, df.columns!="datetime"].to_numpy()\
                        .reshape((-1, 1, 500))
                time_series = TimeSeries.from_times_and_values(times, 
                                                               values, 
                                              fill_missing_dates=True, 
                                                             freq="D")
    
                # Initialize the site dict entry if one doesn't exist already
                if site not in self.sites_dict.keys():
                    self.sites_dict[site] = {}
                self.sites_dict[site][variable] = time_series
    
                # Need to keep track of the variables over different csv's
                # for each site
                if site not in sites_dict_present.keys():
                    sites_dict_present[site] = [variable]
                else:
                    sites_dict_present[site].append(variable)
    
    
        for site in sites_dict_present.keys():
            variables_present = set(sites_dict_present[site])
            absent_variables = list(variables - variables_present)
            self.sites_dict_null[site] = absent_variables
    
    def plot_by_site(self, site):
        for key in self.sites_dict[site].keys():
            plt.clf()
            self.sites_dict[site][key].plot(color="blue", label=f"{key} @ {site}")
            plt.show()

#@ray.remote
class BaseForecaster():
    def __init__(self,
                 model: Optional[str] = None,
                 data_preprocessor: Optional = None,
                 target_variable_column_name: Optional[str] = None,
                 datetime_column_name: Optional[str] = None,
                 covariates_names: Optional[list] = None,
                 output_csv_name: Optional[str] = "residual_forecaster_output.csv",
                 validation_split_date: Optional[str] = None, #YYYY-MM-DD
                 model_hyperparameters: Optional[dict] = None,
                 model_likelihood: Optional[dict] = None,
                 forecast_horizon: Optional[int] = 30,
                 site_id: Optional[str] = None,
                 epochs: Optional[int] = 1,
                 ):
        self.model_ = {"BlockRNN": BlockRNNModel, 
                       "TCN": TCNModel, 
                       "RNN": RNNModel, 
                       "Transformer": TransformerModel,
                       "NLinear": NLinearModel,
                       "DLinear": DLinearModel,
                       "XGB": XGBModel,
                       "NBEATS": NBEATSModel,
                       "Linear": LinearRegressionModel,
                       "TFT": TFTModel}[model]
        self.data_preprocessor = data_preprocessor
        self.target_variable_column_name = target_variable_column_name
        self.datetime_column_name = datetime_column_name
        self.covariates_names = covariates_names
        self.covariates = None
        self.output_csv_name = output_csv_name
        self.validation_split_date = validation_split_date
        self.forecast_horizon = forecast_horizon
        self.site_id = site_id
        self.epochs = epochs
        if model_hyperparameters == None:
            self.hyperparams = {"input_chunk_length" : 180}
        else:
            self.hyperparams = model_hyperparameters
        self.model_likelihood = model_likelihood

        self._preprocess_data()
        
    def _preprocess_data(self):
        """
        Performs gap filling and processing of data into format that
        Darts models will accept
        """
        stitched_series_dict = self.data_preprocessor.sites_dict[self.site_id]

        # If there was failure when doing the GP fit then we can't do preprocessing
        if self.target_variable_column_name in \
                self.data_preprocessor.sites_dict_null[self.site_id]:
            return "Cannot fit this target time series as no GP fit was performed."
        self.inputs = stitched_series_dict[self.target_variable_column_name]

        if self.covariates_names is not None:
            # And not using the covariates that did not yield GP fits beforehand
            for null_variable in self.data_preprocessor.sites_dict_null[self.site_id]:
                self.covariates_names.remove(null_variable)
    
            # Initializing covariates list then concatenating in for loop
            self.covariates = stitched_series_dict[self.covariates_names[0]]
            for cov_var in self.covariates_names[1:]:
                self.covariates = self.covariates.concatenate(stitched_series_dict[cov_var], 
                                                              axis=1, 
                                                              ignore_time_axis=True)
            self.covariates = self.covariates.median()
            

        # Splitting training and validation set
        year = int(self.validation_split_date[:4])
        month = int(self.validation_split_date[5:7])
        day = int(self.validation_split_date[8:])
        split_date = pd.Timestamp(year=year, month=month, day=day)
        # Taking the median now to accomodate using doy covariates
        self.training_set, self.validation_set = self.inputs.median().split_before(split_date)

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

            # Need a work around for Encoders need to map "past", "future", "past_and_future"
            if "add_encoders" in hyperparams.keys():
                if hyperparams["add_encoders"] == "past":
                    hyperparams["add_encoders"] = {'datetime_attribute': {'past': ['dayofyear']}}
                elif hyperparams["add_encoders"] == "future":
                    hyperparams["add_encoders"] = {'datetime_attribute': {'future': ['dayofyear']}}
                elif hyperparams["add_encoders"] == "past_and_future":
                    hyperparams["add_encoders"] = {'datetime_attribute': {'past': ['dayofyear'], 
                                                                       'future': ['dayofyear']}}
        
            model = self.model_(**hyperparams,
                                output_chunk_length=self.forecast_horizon,
                                **self.model_likelihood)

            extras = {"verbose": False,
                      "epochs": self.epochs}
            predict_kws = {"n": len(self.validation_set[:self.forecast_horizon]),
                           "num_samples": 100}
            self.scaler = Scaler()
            if self.covariates is not None:
                training_set, covariates = self.scaler.fit_transform([self.training_set.median(),
                                                                      self.covariates.median()])
                extras["past_covariates"] = covariates
                predict_kws["past_covariates": covariates]
            else:
                training_set = self.scaler.fit_transform([self.training_set.median()])
                
        
            model.fit(training_set, **extras)
            
            predictions = model.predict(**predict_kws)
            predictions = self.scaler.inverse_transform(predictions)

            crps_ = crps(predictions, 
                        self.validation_set[:self.forecast_horizon],
                        )
            
            crps_mean = crps_.mean(axis=0).values()[0][0]
            return crps_mean if crps_mean != np.nan else float("inf")

        study = optuna.create_study(direction="minimize")
        
        study.optimize(objective, n_trials=50) # Note 10 trials pretty meaningless here
        
        self.hyperparams = study.best_trial.params

    def make_forecasts(self):
        """
        This function fits a Darts model to the training_set
        """
        print(self.hyperparams)
        self.model = self.model_(**self.hyperparams,
                                 output_chunk_length=self.forecast_horizon,
                                 **self.model_likelihood,
                                 random_state=0)
        self.scaler = Scaler()
        extras = {"verbose": False,
                  "epochs": self.epochs}
        predict_kws = {"n": self.forecast_horizon,
                       "num_samples": 500}
        if self.covariates is not None:
            training_set, covariates = self.scaler.fit_transform([self.training_set.median(),
                                                                  self.covariates.median()])
            extras["past_covariates"] = covariates
            predict_kws["past_covariates"] = covariates
        else:
            training_set = self.scaler.fit_transform([self.training_set.median()])
        
        if self.model_ == XGBModel or self.model_ == LinearRegressionModel:
            del extras["epochs"]
            del extras["verbose"]
    
        self.model.fit(training_set,
                       **extras)

        predictions = self.model.predict(**predict_kws)
        predictions = self.scaler.inverse_transform(predictions)

        predictions.pd_dataframe().to_csv(self.output_csv_name)

    def get_historicals_and_residuals(self):
        """
        This function creates a historical forecast along with their residual errors 
        """
        # This presumes that the scaler will not have been modified in interim 
        # from calling `make_forecasts`
        historical_forecast_kws = {"num_samples": 500,
                                   "forecast_horizon": self.forecast_horizon,
                                   "retrain": False,
                                   "last_points_only": False,
                                  }
        if self.covariates is not None:
            training_set, covariates = self.scaler.transform([self.training_set,
                                                              self.covariates])
            historical_forecast_kws["past_covariates"] = covariates
        else:
            training_set = self.scaler.transform([self.training_set])
        historical_forecast_kws["series": training_set]
        
        covariates, _ = covariates.split_after(training_set.time_index[-1])
        historical_forecasts = self.model.historical_forecasts(
                                                    **historical_forecast_kws
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


        self.residuals = self.historical_ground_truth - self.historical_forecasts.median()
# This needs to be reviewed, not sure what I was doing here
   #def make_residuals_csv(self):
   #    
   #    forecast_df = self.historical_forecasts.pd_dataframe()
   #    observed_df = self.historical_ground_truth.pd_dataframe()
   #    residuals_df = self.residuals.pd_dataframe()
#
   #    # Creating a folder if it doesn't exist already
   #    if not os.path.exists(f"{self.model}_residuals/"):
   #        os.makedirs(f"{self.model}_residuals/")
   #    # Saving csv's in the **model name**_test directory
   #    df_dict = {"covariates": cov,
   #               "forecast": forecast_df,
   #               "observed": self.historical_ground_truth.pd_dataframe()}
   #    if self.covariates is not None:
   #        covariates_df = self.covariates.pd_dataframe()
   #        df_dict["covariates"] = 
   #    for variable, df in df_dict.items():
   #        df.to_csv(f"{self.model}_test/{variable}")

    def plot_by_site(self, site):
        for key in self.sites_dict[site].keys():
            plt.clf()
            self.sites_dict[site][key].plot(color="blue", label=f"{key} @ {site}")
            plt.show()
    
#@ray.remote
class ResidualForecaster():
    def __init__(self,
                 residuals: Optional[TimeSeries] = None,
                 output_csv_name: Optional[str] = "residual_forecaster_output.csv",
                 validation_split_date: Optional[str] = None, #YYYY-MM-DD
                 tune_model: Optional[bool] = False,
                 model_hyperparameters: Optional[dict] = None,
                 forecast_horizon: Optional[int] = 30,
                 epochs: Optional[int]=1,
                 ):
        self.residuals = residuals
        self.output_csv_name = output_csv_name
        self.validation_split_date = validation_split_date
        self.forecast_horizon = forecast_horizon
        self.epochs = epochs
        if model_hyperparameters == None:
            self.hyperparams = {"input_chunk_length": 540,
                                "lstm_layers": 4,
                                "add_encoders": {'datetime_attribute': {'future': ['dayofyear']}}
                                }
        else:
            self.hyperparams = model_hyperparameters
        self._preprocess_data()
 
        
    def _preprocess_data(self):
        """
        Divides input time series into training and validation sets
        """
        # Getting the date so that we can create the training and test set
        year = int(self.validation_split_date[:4])
        month = int(self.validation_split_date[5:7])
        day = int(self.validation_split_date[8:])
        split_date = pd.Timestamp(year=year, month=month, day=day)
        self.training_set, self.validation_set = self.residuals.split_before(split_date)

    
    def tune(self,
             input_chunk_length: Optional[list] = [31, 60, 180, 356],
             hidden_size: Optional[list] = [2, 3, 5],
             num_attention_heads: Optional[list] = [2, 4, 8, 16],
             lstm_layers: Optional[list] = [1, 2, 3],
             dropout: Optional[list] = [0, 0.1, 0.2, 0.3],
             ):
        """
        Sets up Optuna trial to perform hyperparameter tuning
        """
        # The objective function will be called in the optuna loop
        def objective(trial):
            callback = [PyTorchLightningPruningCallback(trial, monitor="val_loss")]
        
            # Selecting the hyperparameters
            input_chunk_length_ = trial.suggest_categorical("input_chunk_length", 
                                                               input_chunk_length)
            hidden_size_ = trial.suggest_categorical("hidden_size", hidden_size)
            num_attention_heads_ = trial.suggest_categorical("num_attention_heads", num_attention_heads)
            lstm_layers_ = trial.suggest_categorical("lstm_layers", lstm_layers)
            dropout_ = trial.suggest_categorical("dropout", dropout)

            tft_model = TFTModel(input_chunk_length=input_chunk_length_,
                            hidden_size=hidden_size_,
                            num_attention_heads=num_attention_heads_,
                            lstm_layers=lstm_layers_,
                            output_chunk_length=self.forecast_horizon,
                            likelihood=QuantileRegression([0.05, 0.1, 0.5, 0.9, 0.95]),
                            add_encoders={'datetime_attribute': {'future': ['dayofyear']}})

            # Normalizing data to 0, 1 scale and fitting the model
            self.scaler = Scaler()
            training_set = self.scaler.fit_transform(self.training_set)
            tft_model.fit(training_set,
                          epochs=self.epochs, 
                          verbose=False)

            predictions = tft_model.predict(n=len(self.validation_set[:self.forecast_horizon]),  
                                            num_samples=50)
            predictions = self.scaler.inverse_transform(predictions)
            
            smapes = smape(self.validation_set[:self.forecast_horizon], 
                           predictions, 
                           n_jobs=-1, 
                           verbose=False)
            
            smape_val = np.mean(smapes)
            return smape_val if smape_val != np.nan else float("inf")


        study = optuna.create_study(direction="minimize")
        
        study.optimize(objective, n_trials=50) # INCREASE NUMBER OF TRIALS LATER ON
        
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
        tft = TFTModel(**self.hyperparams,
               output_chunk_length=self.forecast_horizon,
               likelihood=QuantileRegression([0.05, 0.1, 0.5, 0.9, 0.95]),
               random_state=0)
        self.scaler = Scaler()
        training_set = self.scaler.fit_transform(self.training_set)
        tft.fit(training_set,
                epochs=self.epochs, 
                verbose=False)

        predictions = tft.predict(n=self.forecast_horizon,
                                  num_samples=500)
        self.predictions = self.scaler.inverse_transform(predictions)

        self.predictions.pd_dataframe().to_csv(self.output_csv_name)

