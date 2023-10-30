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
import pdb
import argparse
import copy
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def crps(forecast, observed):
    """
    Returns an array of CRPS scores 
    """
    forecast_array = forecast.pd_dataframe().values

    crps_scores = []
    for i in range(len(forecast_array)):
        # Note forecastscore is CRPS.CRPS
        crps, _, __ = forecastscore(forecast_array[i], observed[i]).compute()
        crps_scores.append(crps)

    crps_scores = TimeSeries.from_times_and_values(forecast.time_index, 
                                     crps_scores, 
                                     fill_missing_dates=True, freq="D")
    return crps_scores


class HistoricalForecaster():
    def __init__(self,
                 targets: Optional = None,
                 site_id: Optional[str] = None,
                 target_variable: Optional[str] = "oxygen",
                 output_csv_name: Optional[str] = "historical_forecaster_output.csv",
                 validation_split_date: Optional[str] = "2023-03-09", #YYYY-MM-DD
                 forecast_horizon: Optional[int] = 30,
                 datetime_column_name: Optional[str] = "datetime",
                 ):
        self.targets = targets
        # Changing the date from a string to a datetime64 object
        self.targets['datetime'] = pd.to_datetime(self.targets.datetime)
        self.target_variable = target_variable
        self.output_csv_name = output_csv_name
        self.validation_split_date = validation_split_date
        self.forecast_horizon = forecast_horizon
        self.datetime_column_name = datetime_column_name
        self.site_id = site_id
        self._preprocess_data()

    def _preprocess_data(self):
        # Doing some basic filtering and tidying
        site_df = self.targets.loc[self.targets.site_id == self.site_id]
        tidy_df = pd.melt(site_df, 
                          id_vars=['datetime', 'site_id'], 
                          var_name='variable', 
                          value_name='observation')
        variable_df = tidy_df.loc[tidy_df.variable == self.target_variable]
        # Cutting off before the validation split date
        split_date = pd.to_datetime(self.validation_split_date)
        self.year = split_date.year
        variable_df = variable_df[variable_df["datetime"] < split_date]
        # Now finding the mean and std according to day of the year
        variable_df["day_of_year"] = variable_df["datetime"].dt.dayofyear
        self.doy_df = variable_df.groupby(['day_of_year'])['observation'].agg(['mean', 'std'])
        global_mean = variable_df["observation"].mean()
        global_std = variable_df["observation"].std()
        # Confirm that there are 365 days, if there aren't fill in with na
        for doy in range(1, 366):
            try:
                self.doy_df.loc[doy]
            except:
                self.doy_df.loc[doy] = [np.nan, np.nan]
        # If there are persistent gaps in the data, fill in with global mean and std
        for index, row in self.doy_df.iterrows():
            if np.isnan(row["mean"]):
                self.doy_df.loc[index]["mean"] = global_mean
            if np.isnan(row["std"]):
                self.doy_df.loc[index]["std"] = global_std
    

    def make_forecasts(self):
        """
        This function finds the historical mean and var, and uses these statistics for
        the forecast
        """
        # Getting the doys for the forecast window
        forecast_doys = pd.date_range(start=self.validation_split_date, 
                                      periods=self.forecast_horizon, 
                                      freq='D').dayofyear
        forecast_df = self.doy_df.loc[forecast_doys]


        # Drawing samples from a gaussian centered at historical mean and std
        samples = np.array([np.random.normal(self.doy_df.loc[self.doy_df.index == doy]["mean"],
                                    self.doy_df.loc[self.doy_df.index == doy]["std"],
                                    size=(1, 500)) for doy in forecast_df.index])

        # Function to give date from the numerical doy
        def day_of_year_to_date(year, day_of_year):
            base_date = datetime(year, 1, 1)
            target_date = base_date + timedelta(days=day_of_year - 1)
            return target_date
            
        # Catching case where there is no sensor data at all for that site
        if not np.isnan(samples.mean()):
            # Now creating an index going from doy to date
            date_index = [day_of_year_to_date(self.year, day) for day in forecast_df.index]
            forecast_df.index = date_index
    
            # Putting together the forecast timeseries
            self.forecast_df = forecast_df
            self.forecast_ts = TimeSeries.from_times_and_values(forecast_df.index, samples)
            
        else:
            self.forecast_df = None
            self.forecast_ts = None


    def get_residuals(self):
        # This needs to be re-examined!!!
        residual_list = []
        # Going through each date and finding the difference between the doy historical mean and
        # the observed value
        for date in self.training_set.time_index:
            doy = date.dayofyear
            observed = self.training_set.slice_n_points_after(date, 
                                                              1).median().values()[0][0]
            historical_mean = self.doy_df.loc[doy]["mean"]
            residual = observed - historical_mean
            residual_list.append(residual)

        self.residuals = TimeSeries.from_times_and_values(self.training_set.time_index, 
                                                          residual_list)  
        

class TimeSeriesPreprocessor():
    def __init__(self,
                 input_csv_name = "targets.csv.gz",
                 load_dir_name: Optional[str] = "preprocessed_timeseries/",
                 datetime_column_name: Optional[str] = "datetime",
                 covariates_names: Optional[list] = None,
                 validation_split_date: Optional[str] = "2023-03-09",
                 filter_kw_args: Optional[dict] = {"alpha_0": 0.001,
                                                   "n_restarts_0": 100,
                                                   "num_samples": 500,},
                 ):
        self.input_csv_name = input_csv_name
        self.load_dir_name = load_dir_name
        self.datetime_column_name = datetime_column_name
        self.filter_kw_args = filter_kw_args
        self.sites_dict = {}

        self.year = int(validation_split_date[:4])
        month = int(validation_split_date[5:7])
        day = int(validation_split_date[8:])
        self.split_date = pd.Timestamp(year=self.year, month=month, day=day)
        self.df = pd.read_csv(self.input_csv_name)
        self.df['datetime'] = pd.to_datetime(self.df.datetime)
        self.df = self.df[self.df.datetime <= self.split_date]
    
    def make_stitched_series(self, var):
        """
        Returns a time series where the gaps have been filled in via
        Gaussian Process Filters
        """
        kernel = RBF()
        
        gpf_missing = GaussianProcessFilter(kernel=kernel, 
                        alpha=self.filter_kw_args["alpha_0"], 
                        n_restarts_optimizer=self.filter_kw_args["n_restarts_0"])
        
        stitched_series = {}
    
        # Filtering the TimeSeries
        try:
            filtered = gpf_missing.filter(self.var_tseries_dict[var], 
                                          num_samples=self.filter_kw_args["num_samples"])
        except:
            return None
    
        # If there is a gap over 7 indices, use big gap filter
        gap_series = self.var_tseries_dict[var].gaps()
        stitched_df = filtered.pd_dataframe()

        # For these big gaps, I replace with samples centered on historical mean and std
        for index, row in gap_series.iterrows():
            if row["gap_size"] > 7:
                for date in pd.date_range(row["gap_start"], row["gap_end"]):
                    # Finding the mean and std from the doy dictionary
                    # and avoiding leap year errors
                    try:
                        mean, std = self.doy_dict[var].loc[min(date.dayofyear, 365)]
                        if np.isnan(mean):
                            mean = self.doy_dict[var]['mean'].median()
                        if np.isnan(std):
                            std = self.doy_dict[var]['std'].median()
                    except:
                        # If there is an issue, use the global mean and std
                        mean = self.doy_dict[var]['mean'].median()
                        std = self.doy_dict[var]['std'].median()
                    stitched_df.loc[date] = np.random.normal(mean, std, size=(500,))
        
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
        # Preparing a dataframe
        site_df = self.df.loc[self.df.site_id == site]
        times = pd.to_datetime(site_df[self.datetime_column_name])
        times = pd.DatetimeIndex(times)

        # Dealing with no data being included up until splitting date
        if times[-1] != self.split_date:
            new_row = pd.DataFrame({'datetime': [self.split_date], 
                                    'site_id': [site], 
                                    'chla': [np.nan], 
                                    'oxygen': [np.nan], 
                                    'temperature': [np.nan], 
                                    'air_tmp': [np.nan]})
            site_df = pd.concat([site_df, new_row], 
                                ignore_index=True).reset_index(drop=True)
            times = pd.to_datetime(site_df[self.datetime_column_name])
            times = pd.DatetimeIndex(times)

        self.make_doy_dict(site_df)
        variable_list = ["chla", "oxygen", "temperature", "air_tmp"]
        self.var_tseries_dict = {var: TimeSeries.from_times_and_values(times, 
                                                                 site_df[[var]], 
                                                                 fill_missing_dates=True,
                                                                 freq="D") 
                                                        for var in variable_list}

        stitched_series_dict = {var: self.make_stitched_series(var)
                                                    for var in variable_list}

        # Deleting keys with none values
        keys_to_remove = [key for key, value in stitched_series_dict.items() if value == None]
        for key in keys_to_remove:
            del stitched_series_dict[key]

        # Checking that last date of stitched series is the validation split date
        for var in stitched_series_dict.keys():
            last_date = stitched_series_dict[var].time_index[-1]
            if last_date != self.split_date:
                raise Exception("Error with dates between" +\
                                " split date and the last observation in" +\
                                " the stitched series.")

        self.sites_dict[site] = stitched_series_dict
        self.sites_dict_null[site] = keys_to_remove

    def make_doy_dict(self, site_df):
        tidy_df = pd.melt(site_df, 
                          id_vars=['datetime', 'site_id'], 
                          var_name='variable', 
                          value_name='observation')
        # Now finding the mean and std according to day of the year
        tidy_df["day_of_year"] = tidy_df["datetime"].dt.dayofyear
        self.doy_dict = {}
        # Now loop over variables to make a dictionary of doy_df's
        for variable in ["chla", "oxygen", "temperature", "air_tmp"]:
            tidy_variable_df = tidy_df.loc[tidy_df.variable == variable]
            doy_df = tidy_variable_df.groupby(['day_of_year'])['observation'].agg(['mean', 'std'])
            self.doy_dict[variable] = doy_df
    
    def save(self):
        # Check if there's a dir already
        if not os.path.exists(self.load_dir_name):
            os.makedirs(self.load_dir_name)

        # Saving each TimeSeries
        for site in self.sites_dict.keys():
            for variable in self.sites_dict[site]:
                self.sites_dict[site][variable].pd_dataframe()\
                    .to_csv(f"{self.load_dir_name}{site}-{variable}.csv")

    def load(self, site):
        # Need to check what are the possible variables that there could be in null, 
        # and when you load a series need to log which ones aren't added
        variables = {"chla", "oxygen", "temperature", "air_tmp"}
        variables_present = []
        
        # Need to fill sites_dict and sites_dict_null
        files = os.listdir(self.load_dir_name)
        for file in files:
            if file.startswith(site):
                # Reading in file name
                site, variable = file.replace(".csv", "").split("-") 
                file_path = os.path.join(self.load_dir_name, file)
                df = pd.read_csv(file_path)
    
                # To make a time series, need to isolate time index and values
                times = pd.to_datetime(df["datetime"])
                times = pd.DatetimeIndex(times)
                values = df.loc[:, df.columns!="datetime"].to_numpy()\
                        .reshape((-1, 1, self.filter_kw_args["num_samples"]))
                time_series = TimeSeries.from_times_and_values(times, 
                                                               values, 
                                              fill_missing_dates=True, 
                                                             freq="D")
    
                # Initialize the site dict entry if one doesn't exist already
                if site not in self.sites_dict.keys():
                    self.sites_dict[site] = {}
                self.sites_dict[site][variable] = time_series
                variables_present.append(variable)
    
        # And finding the 
        self.site_missing_variables = list(variables - set(variables_present))
    
    def plot_by_site(self, site):
        for key in self.sites_dict[site].keys():
            plt.clf()
            self.sites_dict[site][key].plot(color="blue", label=f"{key} @ {site}")
            plt.show()

class BaseForecaster():
    def __init__(self,
                 model: Optional[str] = None,
                 data_preprocessor: Optional = None,
                 target_variable: Optional[str] = None,
                 datetime_column_name: Optional[str] = "datetime",
                 covariates_names: Optional[list] = None,
                 output_csv_name: Optional[str] = "residual_forecaster_output.csv",
                 validation_split_date: Optional[str] = "2023-03-09", #YYYY-MM-DD n.b. this is inclusive
                 model_hyperparameters: Optional[dict] = None,
                 model_likelihood: Optional[dict] = None,
                 forecast_horizon: Optional[int] = 30,
                 site_id: Optional[str] = None,
                 epochs: Optional[int] = 1,
                 num_samples: Optional[int] = 500,
                 num_trials: Optional[int] = 50,
                 seed: Optional[int] = 0,
                 verbose: Optional[bool] = False,
                 targets_csv: Optional[str] = "targets.csv.gz",
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
        self.target_variable = target_variable
        self.datetime_column_name = datetime_column_name
        self.covariates_names = covariates_names
        self.covariates = None
        self.output_csv_name = output_csv_name
        self.split_date = pd.to_datetime(validation_split_date)
        self.forecast_horizon = forecast_horizon
        self.site_id = site_id
        self.epochs = epochs
        self.num_samples = num_samples
        self.num_trials = num_trials
        self.seed = seed
        self.tuned = False
        self.verbose = verbose
        self.dropout = None
        self.targets_df = pd.read_csv(targets_csv)
        if model_hyperparameters == None:
            self.hyperparams = {"input_chunk_length" : 180}
        else:
            self.hyperparams = model_hyperparameters
        self.model_likelihood = model_likelihood
        # Try to get validation set from targets and not preprocessor

        self._preprocess_data()
        
    def _preprocess_data(self):
        """
        Performs gap filling and processing of data into format that
        Darts models will accept
        """
        stitched_series_dict = self.data_preprocessor.sites_dict[self.site_id]

        # If there was failure when doing the GP fit then we can't do preprocessing
        if self.target_variable in \
                self.data_preprocessor.site_missing_variables:
            return "Cannot fit this target time series as no GP fit was performed."
        self.inputs = stitched_series_dict[self.target_variable]

        if self.covariates_names is not None:
            # And not using the covariates that did not yield GP fits beforehand
            for null_variable in self.data_preprocessor.site_missing_variables:
                self.covariates_names.remove(null_variable)
    
            # Initializing covariates list then concatenating in for loop
            self.covariates = stitched_series_dict[self.covariates_names[0]]
            for cov_var in self.covariates_names[1:]:
                self.covariates = self.covariates.concatenate(stitched_series_dict[cov_var], 
                                                              axis=1, 
                                                              ignore_time_axis=True)
            self.covariates = self.covariates.median()
            
        # Taking the median now to accomodate using doy covariates
        self.training_set = self.inputs.median()

    def tune(self,
             hyperparameter_dict: Optional[dict]
            ):
        """
        Sets up Optuna trial to perform hyperparameter tuning
        Input dictionary will be of the form {"hyperparamter": [values to be tested]}
        """
        self.validation_set = self.get_validation_set()
        if len(self.validation_set) != self.forecast_horizon:
            raise Exception(f"There is missing data in the specified forecast window at {self.site_id}. " + \
                            "This site will not be used for evaluating hyperparameters.")
        # Setting up an optuna Trial
        def objective(trial):
            callback = [PyTorchLightningPruningCallback(trial, monitor="val_loss")]
            hyperparams = {key: trial.suggest_categorical(key, value) 
                                               for key, value in hyperparameter_dict.items()}

            # Need to handle lags and time axis encoders
            hyperparams = self.prepare_hyperparams(hyperparams)

            model = self.model_(**hyperparams,
                                output_chunk_length=self.forecast_horizon,
                                **self.model_likelihood,
                                random_state=self.seed)

            extras = {"verbose": False,
                      "epochs": self.epochs}
            predict_kws = {"n": self.forecast_horizon,
                           "num_samples": self.num_samples}
            self.scaler = Scaler()
            # Need to treat transformation differently if models use covariates or not
            if self.covariates is not None:
                training_set, covariates = self.scaler.fit_transform([self.training_set.median(),
                                                                      self.covariates.median()])
                extras["past_covariates"] = covariates
                predict_kws["past_covariates"] = covariates
            else:
                training_set = self.scaler.fit_transform(self.training_set.median())

            # Need to delete epochs which aren't used in the regression models
            if self.model_ == XGBModel or self.model_ == LinearRegressionModel:
                del extras["epochs"]
                del extras["verbose"]

            assert training_set.time_index[-1] == self.split_date, "There is a" +\
             " misalignment between the training set and the specified validation" +\
             " split date. Note that the validation split date is defined to" +\
             " include the last date of the training set."
            
            model.fit(training_set, **extras)
            
            predictions = model.predict(**predict_kws)
            predictions = self.scaler.inverse_transform(predictions)

            crps_ = crps(predictions, 
                         self.validation_set,
                        )
            
            crps_mean = crps_.mean(axis=0).values()[0][0]
            return crps_mean if crps_mean != np.nan else float("inf")

        study = optuna.create_study(direction="minimize")
        
        study.optimize(objective, n_trials=self.num_trials)
        
        self.hyperparams = study.best_trial.params
        self.tuned = True

    def get_validation_set(self):
        # take validation_split_date and add number of dates
        date_range = pd.date_range(self.split_date + pd.DateOffset(days=1), 
                                   periods=self.forecast_horizon, 
                                   freq='D')
        # Filter targets df for site and variable
        site_df = self.targets_df[self.targets_df["site_id"] == self.site_id]
        site_var_df_ = site_df[["datetime", self.target_variable]]
        site_var_df = site_var_df_.copy()
        site_var_df["datetime"] = pd.to_datetime(site_var_df_["datetime"])
        validation_df = pd.DataFrame()
        # Now creating a new dataframe of observed series from the forecast
        # window
        for date in date_range:
            entry = site_var_df[site_var_df.datetime == date]
            validation_df = pd.concat([validation_df, entry], 
                                      axis=0).reset_index(drop=True)
        
        return validation_df[self.target_variable]
        
    def make_forecasts(self):
        """
        This function fits a Darts model to the training_set
        """
        print(self.hyperparams, self.model_likelihood)
        
        # Need to handle lags and time axis encoders
        self.hyperparams = self.prepare_hyperparams(self.hyperparams)

        self.model = self.model_(**self.hyperparams,
                                 output_chunk_length=self.forecast_horizon,
                                 **self.model_likelihood,
                                 random_state=self.seed)
        self.scaler = Scaler()
        extras = {"verbose": self.verbose,
                  "epochs": self.epochs}
        predict_kws = {"n": self.forecast_horizon,
                       "num_samples": self.num_samples}

        # Need to account for models that don't use past covariates
        if self.covariates is not None:
            # Changing to median so that I can use the time axis encoder.
            # Darts does not allow inputs of mixed dimension.
            training_set, covariates = self.scaler.fit_transform([self.training_set.median(),
                                                                  self.covariates.median()])
            extras["past_covariates"] = covariates
            predict_kws["past_covariates"] = covariates
        else:
            training_set = self.scaler.fit_transform(self.training_set.median())

        # Regression models don't accept these key word arguments for .fit()
        if self.model_ == XGBModel or self.model_ == LinearRegressionModel:
            del extras["epochs"]
            del extras["verbose"]

        assert training_set.time_index[-1] == self.split_date, "There is a" +\
         " misalignment between the training set and the specified validation split" +\
         " date. Note that the validation split date is defined to include the last" +\
         " date of the training set."
        
        self.model.fit(training_set,
                       **extras)

        # Accounting for if there is dropout
        if "dropout" in list(self.model_likelihood.keys()):
            predict_kws["mc_dropout"] = True
            
        predictions = self.model.predict(**predict_kws)
        predictions = self.scaler.inverse_transform(predictions)

        predictions.pd_dataframe().to_csv(self.output_csv_name)

    def get_historicals_and_residuals(self):
        """
        This function creates a historical forecast along with their residual errors 
        """
        # This presumes that the scaler will not have been modified in interim 
        # from calling `make_forecasts`
        historical_forecast_kws = {"num_samples": self.num_samples,
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


        # Defining residual as difference between the median and ground truth
        self.residuals = self.historical_ground_truth - self.historical_forecasts.median()

    def prepare_hyperparams(self, hyperparams_dict):
        if "add_encoders" in hyperparams_dict.keys():
            if hyperparams_dict["add_encoders"] == "past":
                hyperparams_dict["add_encoders"] = {'datetime_attribute': {'past': ['dayofyear']}}
            elif hyperparams_dict["add_encoders"] == "future":
                hyperparams_dict["add_encoders"] = {'datetime_attribute': {'future': ['dayofyear']}}
            elif hyperparams_dict["add_encoders"] == "past_and_future":
                hyperparams_dict["add_encoders"] = {'datetime_attribute': {'past': ['dayofyear'], 
                                                                   'future': ['dayofyear']}}
            elif hyperparams_dict["add_encoders"] == "none":
                del hyperparams_dict["add_encoders"]
                
        if "lr" in hyperparams_dict.keys():
            hyperparams_dict["optimizer_kwargs"] = {"lr": hyperparams_dict["lr"]}
            del hyperparams_dict["lr"]

        return hyperparams_dict
        
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
    
class ResidualForecaster():
    def __init__(self,
                 residuals: Optional[TimeSeries] = None,
                 output_csv_name: Optional[str] = "residual_forecaster_output.csv",
                 validation_split_date: Optional[str] = None, #YYYY-MM-DD
                 tune_model: Optional[bool] = False,
                 model_hyperparameters: Optional[dict] = None,
                 forecast_horizon: Optional[int] = 30,
                 epochs: Optional[int]=1,
                 num_samples: Optional[int] = 500,
                 num_trials: Optional[int] = 50,
                 seed: Optional[int] = 0,
                 ):
        self.residuals = residuals
        self.output_csv_name = output_csv_name
        self.validation_split_date = validation_split_date
        self.forecast_horizon = forecast_horizon
        self.epochs = epochs
        self.num_samples = num_samples
        self.num_trials = num_trials
        self.seed = seed
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
                                            num_samples=self.num_samples)
            predictions = self.scaler.inverse_transform(predictions)
            
            smapes = smape(self.validation_set[:self.forecast_horizon], 
                           predictions, 
                           n_jobs=-1, 
                           verbose=False)
            
            smape_val = np.mean(smapes)
            return smape_val if smape_val != np.nan else float("inf")


        study = optuna.create_study(direction="minimize")
        
        study.optimize(objective, n_trials=self.num_trials)
        
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
               random_state=self.seed)
        self.scaler = Scaler()
        training_set = self.scaler.fit_transform(self.training_set)
        tft.fit(training_set,
                epochs=self.epochs, 
                verbose=False)

        predictions = tft.predict(n=self.forecast_horizon,
                                  num_samples=self.num_samples)
        self.predictions = self.scaler.inverse_transform(predictions)

        self.predictions.pd_dataframe().to_csv(self.output_csv_name)

