from darts import TimeSeries
from typing import Optional
import pandas as pd
import matplotlib.pyplot as plt
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
from torchmetrics import SymmetricMeanAbsolutePercentageError
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import warnings

def crps(forecast, observed, observed_is_ts=False):
    """
    Returns an array of CRPS scores; assumes forecast 
    """
    forecast_array = forecast.pd_dataframe(suppress_warnings=True).values
    if observed_is_ts:
        observed = observed.pd_dataframe(suppress_warnings=True).values.reshape((-1,))
    crps_scores = []
    for i in range(len(forecast_array)):
        # Note forecastscore is CRPS.CRPS
        crps, _, __ = forecastscore(forecast_array[i], observed[i]).compute()
        crps_scores.append(crps)

    crps_scores = TimeSeries.from_times_and_values(
        forecast.time_index, 
        crps_scores, 
        fill_missing_dates=True, 
        freq="D"
    )
    
    return crps_scores

class NaivePersistenceForecaster():
    def __init__(self,
                 targets: Optional = None,
                 site_id: Optional[str] = None,
                 target_variable: Optional[str] = "oxygen",
                 output_csv_name: Optional[str] = "historical_forecaster_output.csv", # This is not used
                 forecast_horizon: Optional[int] = 30,
                 validation_split_date: Optional[str] = "2023-03-09",
                 ):
        self.targets = targets.loc[targets.site_id == site_id]
        # Changing the date from a string to a datetime64 object
        column_name = 'datetime'
        column_index = self.targets.columns.get_loc(column_name)
        self.targets[self.targets.columns[column_index]] = pd.to_datetime(
            self.targets[column_name]
        )
        self.target_variable = target_variable
        self.output_csv_name = output_csv_name
        self.forecast_horizon = forecast_horizon
        self.validation_split_date = validation_split_date
        self.site_id = site_id

    def make_forecasts(self):
        forecast_doys = pd.date_range(
            start=self.validation_split_date, 
            periods=self.forecast_horizon, 
            freq='D',
        )
        # Filter the targets to only look at timestamps before the split date
        date = pd.to_datetime(self.validation_split_date)
        filtered_targets = self.targets.loc[self.targets.datetime < date]
        # Select the last observed target value is selected
        last_row = (
            filtered_targets[filtered_targets[self.target_variable].notna()]
            .iloc[-1]
        )
        last_target_value = last_row[self.target_variable]
        # Create a TimeSeries with this value for each day of the forecast window
        values = np.array([last_target_value for doy in forecast_doys])
        self.forecast_ts = TimeSeries.from_times_and_values(forecast_doys, values)

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
        tidy_df = pd.melt(
            site_df, 
            id_vars=['datetime', 'site_id'], 
            var_name='variable', 
            value_name='observation'
        )
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
        forecast_doys = pd.date_range(
            start=self.validation_split_date, 
            periods=self.forecast_horizon, 
            freq='D',
        ).dayofyear
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
            # Now creating an index going from doy to date, being careful of year
            date_index = []
            index_year = self.year
            for day in forecast_df.index:
                date_index.append(day_of_year_to_date(index_year, day))
                if day == 365:
                    index_year += 1
                    
            forecast_df.index = date_index
    
            # Putting together the forecast timeseries
            self.forecast_df = forecast_df
            # fix dates here
            self.forecast_ts = TimeSeries.from_times_and_values(
                forecast_df.index, 
                samples,
                fill_missing_dates=True,
                freq='D'
            )
            
        else:
            self.forecast_df = None
            self.forecast_ts = None


    def get_residuals(self):
        # This needs to be re-examined!!!
        residual_list = []
        # Finding the difference between the doy historical mean and
        # the observed value
        for date in self.training_set.time_index:
            doy = date.dayofyear
            observed = self.training_set.slice_n_points_after(
                date, 
                1
            ).median().values()[0][0]
            historical_mean = self.doy_df.loc[doy]["mean"]
            residual = observed - historical_mean
            residual_list.append(residual)

        self.residuals = TimeSeries.from_times_and_values(
            self.training_set.time_index, 
            residual_list
        )
        
def month_doy_range(year, month):
    # Get the first day of the month
    first_day = datetime(year, month, 1)

    # Calculating the last day of the month
    if month == 12:
        last_day = datetime(year + 1, 1, 1) - timedelta(days=1)
    else:
        last_day = datetime(year, month + 1, 1) - timedelta(days=1)

    # Finding the day of year for the first and last day
    doy_first_day = first_day.timetuple().tm_yday
    doy_last_day = last_day.timetuple().tm_yday

    return doy_first_day, doy_last_day

def season_doy_range(year, month, day):
    # Given date
    given_date = datetime(year, month, day)

    # Arbitrarily definine the start and end dates for each season
    spring_start = datetime(year, 3, 1)
    summer_start = datetime(year, 6, 1)
    fall_start = datetime(year, 9, 1)
    winter_start = datetime(year, 12, 1)

    # Determine the season based on the given date
    if given_date < spring_start or given_date >= datetime(year + 1, 3, 1):
        season = "winter"
        start_date = winter_start
        end_date = datetime(year + 1, 2, 28)  # Assuming non-leap year
    elif given_date < summer_start:
        season = "spring"
        start_date = spring_start
        end_date = summer_start - timedelta(days=1)
    elif given_date < fall_start:
        season = "summer"
        start_date = summer_start
        end_date = fall_start - timedelta(days=1)
    elif given_date < winter_start:
        season = "fall"
        start_date = fall_start
        end_date = winter_start - timedelta(days=1)

    # Calculate the DOY range for the determined season
    doy_start = start_date.timetuple().tm_yday
    doy_end = end_date.timetuple().tm_yday

    return doy_start, doy_end

class TimeSeriesPreprocessor():
    def __init__(self,
                 input_csv_name = "targets.csv.gz",
                 load_dir_name: Optional[str] = "preprocessed_timeseries/",
                 datetime_column_name: Optional[str] = "datetime",
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
        
        gpf_missing = GaussianProcessFilter(
            kernel=kernel, 
            alpha=self.filter_kw_args["alpha_0"], 
            n_restarts_optimizer=self.filter_kw_args["n_restarts_0"]
        )
        
        stitched_series = {}
    
        # Filtering the TimeSeries
        try:
            filtered = gpf_missing.filter(
                self.var_tseries_dict[var], 
                num_samples=self.filter_kw_args["num_samples"]
            )
        except:
            return None
    
        # If there is a gap over 7 indices, use big gap filter
        gap_series = self.var_tseries_dict[var].gaps()
        stitched_df = filtered.pd_dataframe(suppress_warnings=True)
        
        # Ignoring runtime warnings in this function only
        # This is because I allow means to be found of empty arrays,
        # yielding NaNs, which is certainly not elegant
        warnings.filterwarnings("ignore", category=RuntimeWarning)

        # For these big gaps, I replace with samples centered on historical mean and std
        for index, row in gap_series.iterrows():
            if row["gap_size"] > 7:
                for date in pd.date_range(row["gap_start"], row["gap_end"]):
                    # Finding the mean and std from the doy dictionary
                    # and avoiding leap year errors
                    try:
                        mean, std = self.doy_dict[var].loc[min(date.dayofyear, 365)]
                        # If there is an issue, use the median daily historical data
                        # over that month
                        if np.isnan(mean):
                            month_range = month_doy_range(date.year, date.month)
                            mean = (
                                self.doy_dict[var]
                                .loc[month_range[0]:month_range[1]]['mean']
                                .median()
                            )
                            if np.isnan(mean):
                                # And if this is still NaN, aggregate over the season
                                season_range = season_doy_range(
                                    date.year, date.month, date.day
                                )
                                mean = (
                                    self.doy_dict[var]
                                    .loc[season_range[0]:season_range[1]]['mean']
                                    .median()
                                )
                        if np.isnan(std):
                            # Not exactly sure why but filtering looks way better
                            # if I use the std over a season; this ends up not mattering
                            # as the model only uses the median.
                            season_range = season_doy_range(
                                    date.year, date.month, date.day
                            )
                            std = (
                                self.doy_dict[var]
                                .loc[season_range[0]:season_range[1]]['std']
                                .median()
                            )
                        if np.isnan(std) or np.isnan(mean):
                            raise ValueError
                    except:
                        # If above conditions fail use the previous date's samples, and
                        # if there is an issue with accessing a previous date, 
                        # use global
                       try:
                           mean = stitched_df.loc[previous_date].median()
                           std = stitched_df.loc[previous_date].std()
                       except:
                           mean = self.doy_dict[var]['mean'].median()
                           std = self.doy_dict[var]['std'].max()

                    stitched_df.loc[date] = np.random.normal(mean, std, size=(500,))
                    previous_date = date
        
        stitched_series = TimeSeries.from_times_and_values(
                              stitched_df.index, 
                              stitched_df.values.reshape(
                                  len(stitched_df), 
                                  1, 
                                  -1,
                              )
        )
        
        return stitched_series

    def preprocess_data(self, site):
        """
        Performs gap filling and processing of data into format that
        Darts models will accept
        """
        self.sites_dict_null = {}
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
                self.sites_dict[site][variable].pd_dataframe(suppress_warnings=True)\
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
                 train_preprocessor: Optional = None,
                 validate_preprocessor: Optional = None,
                 target_variable: Optional[str] = None,
                 datetime_column_name: Optional[str] = "datetime",
                 covariates_names: Optional[list] = None,
                 output_csv_name: Optional[str] = "default",
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
        self.validate_preprocessor = validate_preprocessor
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

        self.training_set, self.covs_train = self._preprocess_data(train_preprocessor)
        self.inputs, self.covariates = self._preprocess_data(validate_preprocessor,
                                                             train_set=False)
        
    def _preprocess_data(self, data_preprocessor, train_set=True):
        """
        Performs gap filling and processing of data into format that
        Darts models will accept; train_set flag is to 
        """
        stitched_series_dict = data_preprocessor.sites_dict[self.site_id]

        # If there was failure when filtering then we can't do preprocessing
        if self.target_variable in \
                data_preprocessor.site_missing_variables:
            return "Cannot fit this target time series as no GP fit was performed."
        inputs = stitched_series_dict[self.target_variable]

        if self.covariates_names is not None:
            # And not using the covariates that did not yield GP fits beforehand
            for null_variable in data_preprocessor.site_missing_variables:
                if null_variable in self.covariates_names:
                    self.covariates_names.remove(null_variable)
    
            # Initializing covariates list then concatenating in for loop
            covariates = stitched_series_dict[self.covariates_names[0]]
            for cov_var in self.covariates_names[1:]:
                covariates = covariates.concatenate(stitched_series_dict[cov_var], 
                                                              axis=1, 
                                                              ignore_time_axis=True)
            covariates = covariates.median()

            covs_train, _ = (
                covariates
                .median()
                .split_after(self.split_date)
            )
        else:
            covs_train = None
            covariates = None
            
        # Taking the median now to accomodate using doy covariates
        training_set, validation_set = (
            inputs
            .median()
            .split_after(self.split_date)
        )

        if train_set:
            return training_set, covs_train
        else:
            return inputs.median(), covariates
            
    def get_validation_set(self, scaler, input_chunk_length):
        # This function creates a sliding window across some preprocessed data
        # so we can see how the model performs at different times of the year
        interval = pd.Timedelta(days=self.forecast_horizon)
        dates = pd.date_range(self.split_date + interval, periods=12, freq=interval)
        val_set_list = []
        for date in dates:
            val_set_list.append(
                scaler.transform(
                    self.inputs.slice_n_points_before(
                        date, 
                        self.forecast_horizon + input_chunk_length
                    )
                )
            )
        return val_set_list

    def get_predict_set(self, scaler, input_chunk_length):
        # Similar to get_validation_set, except here I want to create 
        # a window that just has the data to use as an input, nothing to validate
        # a prediction
        interval = pd.Timedelta(days=self.forecast_horizon)
        dates = pd.date_range(self.split_date, periods=12, freq=interval)
        predict_set_list = []
        for date in dates:
            predict_set_list.append(
                scaler.transform(
                    self.inputs.slice_n_points_before(
                        date, 
                        input_chunk_length
                    )
                )
            )
        return predict_set_list

    def get_check_set(self, scaler, input_chunk_length):
        # Here I am doing the complementary half of get_predict_set,
        # which is getting a "ground truth" for the history provided
        # by get_predict
        interval = pd.Timedelta(days=self.forecast_horizon)
        dates = pd.date_range(self.split_date, periods=12, freq=interval)
        check_set_list = []
        for date in dates:
            check_set_list.append(
                scaler.transform(
                    self.inputs.slice_n_points_after(
                        date + pd.Timedelta(1), 
                        self.forecast_horizon
                    )
                )
            )
        return check_set_list 
        
    def tune(self,
             hyperparameter_dict: Optional[dict]
            ):
        """
        Sets up Optuna trial to perform hyperparameter tuning
        Input dictionary will be of the form {"hyperparamter": [values to be tested]}
        """
        # Setting up an optuna Trial
        def objective(trial):
            # Selecting hyperparameters for the trial
            hyperparams = {key: trial.suggest_categorical(key, value) 
                                               for key, value in hyperparameter_dict.items()}

            # Will stop training when validation loss does not decrease more than 0.05 (`min_delta`) over
            # a period of 5 epochs (`patience`)
            my_stopper = EarlyStopping(
                monitor="val_loss",
                patience=10,
                min_delta=0.05,
                mode='min',
            )
            pl_trainer_kwargs={"callbacks": [my_stopper]}
            # And watching SMAPE
            torch_metrics = SymmetricMeanAbsolutePercentageError()

            # Need to format some hypers
            hyperparams = self.prepare_hyperparams(hyperparams)

            model = self.model_(
                **hyperparams,
                output_chunk_length=self.forecast_horizon,
                **self.model_likelihood,
                random_state=self.seed,
                pl_trainer_kwargs=pl_trainer_kwargs,
                torch_metrics=torch_metrics
            )

            extras = {"verbose": False,
                      "epochs": self.epochs}
            predict_kws = {"n": self.forecast_horizon,
                           "num_samples": self.num_samples}
            self.scaler = Scaler()
            self.scaler_cov = Scaler()
            # Some models don't use past covariates, so the inputs
            # for these classes of models need to be handled differently
            if self.covariates is not None:
                training_set = self.scaler.fit_transform(self.training_set)
                covariates = self.scaler_cov.fit_transform(self.covs_train)
                extras["past_covariates"] = covariates
                # Handling training and validation series + covariates differently
                # because they were preprocessed separately
                extras['val_series'] = self.get_validation_set(
                    self.scaler,
                    hyperparams['input_chunk_length']
                )
                extras["val_past_covariates"] = [
                    self.scaler_cov.transform(self.covariates) \
                      for i in range(len(extras['val_series']))
                ]
            else:
                training_set = self.scaler.fit_transform(self.training_set)
                validation_set = self.get_validation_set(
                    self.scaler,
                    hyperparams['input_chunk_length']
                )
                extras["val_series"] = validation_set

            # Need to delete epochs which aren't used in the regression models
            if self.model_ == XGBModel or self.model_ == LinearRegressionModel:
                del extras["epochs"]
                del extras["verbose"]

            assert training_set.time_index[-1] == self.split_date, "There is a" +\
             " misalignment between the training set and the specified validation" +\
             " split date. Note that the validation split date is defined to" +\
             " include the last date of the training set."

            model.fit(training_set, **extras)

            # Preparing the series and covariates to input to the predict call
            predict_series = self.get_predict_set(
                self.scaler,
                hyperparams['input_chunk_length']
            )
            
            predict_kws = {'num_samples': 500}
            if self.covariates is not None:
                predict_kws['past_covariates'] = extras['val_past_covariates']

            predictions = model.predict(
                n=self.forecast_horizon, 
                series=predict_series, 
                **predict_kws
            )
            
            check_predictions = self.get_check_set(
                self.scaler,
                hyperparams['input_chunk_length']
            )

            # Making sure that predictions and their solutions align in dates
            assert (
                predictions[0].time_index[0] == check_predictions[0].time_index[0]
            ), "Error with date alignment on tuning check"

            # Computing CRPS for each prediction window from the validation set
            # and taking the mean of these values.
            crps_list = []
            for i in range(len(predictions)):
                crps_list.append(
                    crps(
                        predictions[i],
                        check_predictions[i],
                        observed_is_ts=True
                    ).pd_dataframe(suppress_warnings=True).values
                ) 
            crps_mean = np.array(crps_list).mean()

            # Tuning will provide the score with the lowest mean CRPS over
            # the validation set which includes 11 evenly-spaced windows
            # over the year
            return crps_mean if crps_mean != np.nan else float("inf")

        study = optuna.create_study(direction="minimize")
        
        study.optimize(objective, n_trials=self.num_trials)
        
        self.hyperparams = study.best_trial.params
        self.tuned = True
        
    def make_forecasts(self):
        """
        This function fits a Darts model to the training_set
        """
        print(self.hyperparams, self.model_likelihood)

        # Stopping early
        my_stopper = EarlyStopping(
                monitor="val_loss",
                patience=10,
                min_delta=0.02,
                mode='min',
        )
        pl_trainer_kwargs={"callbacks": [my_stopper]}
        # And watching SMAPE
        torch_metrics = SymmetricMeanAbsolutePercentageError()
        
        # Need to handle lags and time axis encoders
        self.hyperparams = self.prepare_hyperparams(self.hyperparams)

        self.model = self.model_(
            **self.hyperparams,
            output_chunk_length=self.forecast_horizon,
            **self.model_likelihood,
            random_state=self.seed,
            pl_trainer_kwargs=pl_trainer_kwargs,
            torch_metrics=torch_metrics,
        )
        
        extras = {
            "verbose": self.verbose,
            "epochs": self.epochs
        }
        predict_kws = {
            "n": self.forecast_horizon,
            "num_samples": self.num_samples
        }

        # Need to account for models that don't use past covariates
        self.scaler = Scaler()
        if self.covariates is not None:
            self.scaler_cov = Scaler()
            training_set = self.scaler.fit_transform(self.training_set)
            covariates = self.scaler_cov.fit_transform(self.covs_train)
            extras["past_covariates"] = covariates
            # Handling training and validation series + covariates differently
            # because they were preprocessed separately
            extras['val_series'] = self.get_validation_set(
                self.scaler,
                self.hyperparams['input_chunk_length']
            )
            extras["val_past_covariates"] = [
                self.scaler_cov.transform(self.covariates) \
                  for i in range(len(extras['val_series']))
            ]
        else:
            training_set = self.scaler.fit_transform(self.training_set)
            validation_set = self.get_validation_set(
                self.scaler,
                self.hyperparams['input_chunk_length']
            )
            extras["val_series"] = validation_set

        # Regression models don't accept these key word arguments for .fit()
        if self.model_ == XGBModel or self.model_ == LinearRegressionModel:
            del extras["epochs"]
            del extras["verbose"]

        assert training_set.time_index[-1] == self.split_date, "There is a" +\
         " misalignment between the training set and the specified validation split" +\
         " date. Note that the validation split date is defined to include the last" +\
         " date of the training set."
        
        self.model.fit(training_set, **extras)

        # Preparing input series and covariates for the predictions
        predict_series = self.get_predict_set(
                self.scaler,
                self.hyperparams['input_chunk_length']
        )
        if self.covariates is not None:
            predict_kws['past_covariates'] = [
                    self.scaler_cov.transform(self.covariates) \
                      for i in range(len(predict_series))
            ]

        # Accounting for if there is dropout; Might delete this as this wasn't an interesting
        # excursion
        if "dropout" in list(self.model_likelihood.keys()):
            predict_kws["mc_dropout"] = True

        predictions = self.model.predict( 
            series=predict_series, 
            **predict_kws
        )
        for prediction in predictions:
            prediction = self.scaler.inverse_transform(prediction)
            csv_name = self.output_csv_name + "_" + \
                           prediction.time_index[0].strftime('%Y_%m_%d.csv')
            prediction.pd_dataframe(suppress_warnings=True).to_csv(csv_name)
            


    def get_historicals_and_residuals(self):
        """
        This function creates a historical forecast along with their residual errors 
        """
        # This presumes that the scaler will not have been modified in interim 
        # from calling `make_forecasts`
        historical_forecast_kws = {
            "num_samples": self.num_samples,
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
            historical_forecasts[-1].time_index[-1]
        )

        # Now concatenating the historical forecasts which were returned
        # as a list above
        self.historical_forecasts = historical_forecasts[0]
        for time_series in historical_forecasts[1:]:
            self.historical_forecasts = self.historical_forecasts.concatenate(
                time_series, 
                axis=0, 
                ignore_time_axis=True
            )


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

        self.predictions.pd_dataframe(suppress_warnings=True).to_csv(self.output_csv_name)

