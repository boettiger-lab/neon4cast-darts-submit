from utils import (
                BaseForecaster, 
                ResidualForecaster,
                TimeSeriesPreprocessor,
                crps,
                HistoricalForecaster
)
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import os
from darts import TimeSeries
import numpy as np
import CRPS.CRPS as forecastscore
from darts.metrics import rmse
import matplotlib as mpl


def get_validation_series(targets_df, site_id, target_variable, date, forecast_horizon):
        # Being careful here with the date, note that I am matching the forecast,
        # so I don't need to advance.
        date_range = pd.date_range(date, 
                                   periods=forecast_horizon, 
                                   freq='D')
        # Filter targets df for site and variable
        site_df = targets_df[targets_df["site_id"] == site_id]
        site_var_df_ = site_df[["datetime", target_variable]]
        site_var_df = site_var_df_.copy()
        site_var_df["datetime"] = pd.to_datetime(site_var_df_["datetime"])
        validation_df = pd.DataFrame()
        # Now creating a new dataframe of observed series from the forecast
        # window
        for date in date_range:
            entry = site_var_df[site_var_df.datetime == date]
            validation_df = pd.concat([validation_df, entry], 
                                      axis=0).reset_index(drop=True)

        times = pd.to_datetime(validation_df.datetime)
        times = pd.DatetimeIndex(times)
        validation_series = TimeSeries.from_times_and_values(times,
                                                             validation_df[[target_variable]],
                                                             fill_missing_dates=True,)
        
        return validation_series

def make_plots(models_list, targets_df, site_id, target_variable, plot_name=None):
    cmap = mpl.colormaps["tab10"]
    colors = cmap.colors
    # Loading the forecast csv and creating a time series
    for i, model_name in enumerate(models_list):
        csv_name = f"forecasts/{site_id}/{target_variable}/{model_name}"
        df = pd.read_csv(f"{csv_name}.csv")
        times = pd.to_datetime(df["datetime"])
        times = pd.DatetimeIndex(times)
        values = df.loc[:, df.columns!="datetime"].to_numpy().reshape((len(times), 1, -1))
        model_forecast = TimeSeries.from_times_and_values(times, 
                                                          values, 
                                                          fill_missing_dates=True, freq="D")
        model_forecast.plot(label=f"{model_name}", color=colors[i])

    # Getting the validation series directly from the targets csv
    date = model_forecast.time_index[0]
    forecast_horizon = len(model_forecast)
    validation_series = get_validation_series(targets_df, 
                                              site_id, 
                                              target_variable, 
                                              date, 
                                              forecast_horizon)

    # Now, making the forecast based off of historical mean and std
    historical_model = HistoricalForecaster(targets=targets_df,
                          site_id=site_id,
                          target_variable=target_variable,
                          output_csv_name="historical_forecaster_output.csv",
                          validation_split_date=str(model_forecast.time_index[0])[:10],
                          forecast_horizon=len(model_forecast),)
    historical_model.make_forecasts()
    i += 1
    historical_model.forecast_ts.plot(label="Historical", color=colors[i])
    validation_series.plot(label="Truth", color=colors[i+1])
    x = plt.xlabel("date")
    y = plt.ylabel(target_variable)
    ax = plt.gca()
    ax.spines["left"].set_visible(True)
    ax.spines["bottom"].set_visible(True)
    plt.grid(False)

    # Saving the plot if desired
    if plot_name != None:
        if not os.path.exists(f"plots/{site_id}/{target_variable}/"):
            os.makedirs(f"plots/{site_id}/{target_variable}/")
        plt.savefig(f"plots/{site_id}/{target_variable}/{plot_name}")

def make_crps_seq_plot(models_list, targets_df, site_id, target_variable, plot_name=None):
    cmap = mpl.colormaps["tab20"]
    colors = cmap.colors
    # Loading the forecast csv and creating a time series
    for i, model_name in enumerate(models_list):
        csv_name = f"forecasts/{site_id}/{target_variable}/{model_name}"
        df = pd.read_csv(f"{csv_name}.csv")
        times = pd.to_datetime(df["datetime"])
        times = pd.DatetimeIndex(times)
        values = df.loc[:, df.columns!="datetime"].to_numpy().reshape((len(times), 1, -1))
        model_forecast = TimeSeries.from_times_and_values(times, 
                                                          values, 
                                                          fill_missing_dates=True, freq="D")

        # Getting the validation set from targets
        if i == 0:
            date = model_forecast.time_index[0]
            forecast_horizon = len(model_forecast)
            validation_series = get_validation_series(targets_df, 
                                                      site_id, 
                                                      target_variable, 
                                                      date, 
                                                      forecast_horizon,
                                                     ).pd_series().values
        # Computing CRPS and plotting it as well as its mean (dashed)
        scores = crps(model_forecast, validation_series)
        scores.plot(label=f"{model_name}", color=colors[i])
        plt.axhline(y=scores.mean(axis=0).values()[0][0], linestyle='--',
                    color=colors[i])
            
    # Now, making the forecast based off of historical mean and std
    historical_model = HistoricalForecaster(targets=targets_df,
                          site_id=site_id,
                          target_variable=target_variable,
                          output_csv_name="historical_forecaster_output.csv",
                          validation_split_date=str(model_forecast.time_index[0])[:10],
                          forecast_horizon=len(model_forecast),)
    # Computing CRPS of historical forecast and plotting
    historical_model.make_forecasts()
    i += 1
    scores = crps(historical_model.forecast_ts, validation_series)
    scores.plot(label=f"Historical", color=colors[i])
    plt.axhline(y = scores.mean(axis=0).values()[0][0], linestyle = '--', color=colors[i])
    x = plt.xlabel("date")
    y = plt.ylabel("crps")
    ax = plt.gca()
    ax.spines["left"].set_visible(True)
    ax.spines["bottom"].set_visible(True)
    plt.grid(False)

    # Saving the plot if desired
    if plot_name != None:
        if not os.path.exists(f"plots/{site_id}/{target_variable}/"):
            os.makedirs(f"plots/{site_id}/{target_variable}/")
        plt.savefig(f"plots/{site_id}/{target_variable}/{plot_name}")

def make_crps_strip_plot(models_list, targets_df, site_id, target_variable, plot_name=None):
    cmap = mpl.colormaps["tab20"]
    colors = cmap.colors
    score_dict = {}
    # Loading the forecast csv and creating a time series
    for i, model_name in enumerate(models_list):
        csv_name = f"forecasts/{site_id}/{target_variable}/{model_name}"
        df = pd.read_csv(f"{csv_name}.csv")
        times = pd.to_datetime(df["datetime"])
        times = pd.DatetimeIndex(times)
        values = df.loc[:, df.columns!="datetime"].to_numpy().reshape((len(times), 1, -1))
        model_forecast = TimeSeries.from_times_and_values(times, 
                                                          values, 
                                                          fill_missing_dates=True, freq="D")

        # Getting the validation set from targets
        if i == 0:
            date = model_forecast.time_index[0]
            forecast_horizon = len(model_forecast)
            validation_series = get_validation_series(targets_df, 
                                                      site_id, 
                                                      target_variable, 
                                                      date, 
                                                      forecast_horizon,
                                                     ).pd_series().values
        # Computing CRPS and plotting it as well as its mean (dashed)
        scores = crps(model_forecast, validation_series)
        score_dict[model_name] = scores.pd_dataframe().values[:, 0]
            
    # Now, making the forecast based off of historical mean and std
    historical_model = HistoricalForecaster(targets=targets_df,
                          site_id=site_id,
                          target_variable=target_variable,
                          output_csv_name="historical_forecaster_output.csv",
                          validation_split_date=str(model_forecast.time_index[0])[:10],
                          forecast_horizon=len(model_forecast),)
    # Computing CRPS of historical forecast and plotting
    historical_model.make_forecasts()
    
    scores = crps(historical_model.forecast_ts, validation_series)
    score_dict["historical"] = scores.pd_dataframe().values[:, 0]

    # Now creating the plot
    p = sns.stripplot(score_dict, jitter=0.05, color='0.5')

    # plot the mean line
    sns.boxplot(showmeans=False,
                meanline=False,
                meanprops={'color': 'k', 'ls': '-', 'lw': 2},
                medianprops={'visible': True, 'lw':1.75},
                whiskerprops={'visible': False},
                zorder=10,
                data=score_dict,
                showfliers=False,
                showbox=False,
                showcaps=False,
                ax=p)
    plt.grid(False)
    plt.ylabel("crps")
    ax = plt.gca()
    ax.spines["left"].set_visible(True)
    ax.spines["bottom"].set_visible(True)
    plt.xticks(rotation=30)
    
    # Saving the plot if desired
    if plot_name != None:
        if not os.path.exists(f"plots/{site_id}/{target_variable}/"):
            os.makedirs(f"plots/{site_id}/{target_variable}/")
        plt.savefig(f"plots/{site_id}/{target_variable}/{plot_name}")

def make_rmse_strip_plot(models_list, targets_df, site_id, target_variable, plot_name=None):
    cmap = mpl.colormaps["tab20"]
    colors = cmap.colors
    score_dict = {}
    # Loading the forecast csv and creating a time series
    for i, model_name in enumerate(models_list):
        csv_name = f"forecasts/{site_id}/{target_variable}/{model_name}"
        df = pd.read_csv(f"{csv_name}.csv")
        times = pd.to_datetime(df["datetime"])
        times = pd.DatetimeIndex(times)
        values = df.loc[:, df.columns!="datetime"].to_numpy().reshape((len(times), 1, -1))
        model_forecast = TimeSeries.from_times_and_values(times, 
                                                          values, 
                                                          fill_missing_dates=True, freq="D")

        # Getting the validation set from targets
        if i == 0:
            date = model_forecast.time_index[0]
            forecast_horizon = len(model_forecast)
            validation_series = get_validation_series(targets_df, 
                                                      site_id, 
                                                      target_variable, 
                                                      date, 
                                                      forecast_horizon,
                                                     )
        # Computing RMSE and plotting it as well as its mean (dashed)
        model_score = rmse(validation_series, model_forecast)
        import pdb; pdb.set_trace()
        score_dict[model_name] = model_score
            
    # Now, making the forecast based off of historical mean and std
    historical_model = HistoricalForecaster(targets=targets_df,
                          site_id=site_id,
                          target_variable=target_variable,
                          output_csv_name="historical_forecaster_output.csv",
                          validation_split_date=str(model_forecast.time_index[0])[:10],
                          forecast_horizon=len(model_forecast),)
    # Computing RMSE of historical forecast and plotting
    historical_model.make_forecasts()
    
    hist_score = rmse(validation_series, historical_model.forecast_ts)
    score_dict["historical"] = hist_score
    #import pdb; pdb.set_trace()
    # Now creating the plot
    p = sns.pointplot(score_dict, linestyle="", color="0.5")
    plt.grid(False)
    plt.ylabel("rmse")
    ax = plt.gca()
    ax.spines["left"].set_visible(True)
    ax.spines["bottom"].set_visible(True)
    plt.xticks(rotation=30)
    
    # Saving the plot if desired
    if plot_name != None:
        if not os.path.exists(f"plots/{site_id}/{target_variable}/"):
            os.makedirs(f"plots/{site_id}/{target_variable}/")
        plt.savefig(f"plots/{site_id}/{target_variable}/{plot_name}")
    
    