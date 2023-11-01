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
        validation_df = pd.concat(
            [validation_df, entry], 
            axis=0
        ).reset_index(drop=True)
        
    times = pd.to_datetime(validation_df.datetime)
    times = pd.DatetimeIndex(times)
    validation_series = TimeSeries.from_times_and_values(
        times,
        validation_df[[target_variable]],
        fill_missing_dates=True,
    )
    
    return validation_series

def filter_forecast_df(forecast_df, validation_series):
    """
    Assumes validation series is a TimeSeries
    and forecast_df has an datetime index
    """
    gaps = validation_series.gaps()
    # Filtering forecast df to only include dates in the validation series
    if len(gaps) > 0:
        for i in range(len(gaps)):
            gap_start = gaps.iloc[i].gap_start
            gap_end = gaps.iloc[i].gap_end
            forecast_df = forecast_df[(forecast_df.index < gap_start) \
                                      | (forecast_df.index > gap_end)]

    times = forecast_df.index
    validation_series = validation_series.pd_series().dropna()
    # Checking that the dates indices are the same, i.e. that filtering worked properly
    assert (validation_series.index == forecast_df.index).all()

    values = forecast_df.loc[:, forecast_df.columns!="datetime"].to_numpy().reshape(
                                                                (len(times), 1, -1))
    filtered_forecast_ts = TimeSeries.from_times_and_values(
        times, 
        values, 
        fill_missing_dates=False
    )

    return filtered_forecast_ts, validation_series

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
        forecast_df = pd.read_csv(f"{csv_name}.csv")
        forecast_df["datetime"] = pd.to_datetime(forecast_df["datetime"])
        times = pd.DatetimeIndex(forecast_df["datetime"])
        forecast_df = forecast_df.set_index("datetime")

        # Getting the validation set from targets
        if i == 0:
            date = forecast_df.index[0]
            forecast_horizon = len(forecast_df)
            validation_series = get_validation_series(
                targets_df, 
                site_id, 
                target_variable, 
                date, 
                forecast_horizon,
            )
        filtered_model_forecast, filtered_validation_series = filter_forecast_df(
            forecast_df,
            validation_series
        )
        # Computing CRPS and plotting it as well as its mean (dashed)
        scores = crps(
            filtered_model_forecast, 
            filtered_validation_series
        )
        score_dict[model_name] = scores.pd_dataframe().values[:, 0]
            
    # Now, making the forecast based off of historical mean and std
    historical_model = HistoricalForecaster(targets=targets_df,
                          site_id=site_id,
                          target_variable=target_variable,
                          output_csv_name="historical_forecaster_output.csv",
                          validation_split_date=str(forecast_df.index[0])[:10],
                          forecast_horizon=len(forecast_df),)
    # Computing CRPS of historical forecast and plotting
    historical_model.make_forecasts()
    historical_forecast_df = historical_model.forecast_ts.pd_dataframe(
        suppress_warnings=True
    )
    filtered_historical_forecast, filtered_validation_series = filter_forecast_df(
        historical_forecast_df,
        validation_series
    )
    scores = crps(filtered_historical_forecast, filtered_validation_series)
    score_dict["historical"] = scores.pd_dataframe().values[:, 0]

    # Now creating the plot
    p = sns.stripplot(score_dict, jitter=0.05, color='0.5')

    # plot the mean line
    sns.boxplot(
        showmeans=False,
        meanline=False,
        meanprops={'color': 'k', 'ls': '-', 'lw': 2},
        medianprops={'visible': True, 'lw':1.75},
        whiskerprops={'visible': False},
        zorder=10,
        data=score_dict,
        showfliers=False,
        showbox=False,
        showcaps=False,
        ax=p
    )
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
    # Selecting a colormap that I prefer
    cmap = mpl.colormaps["tab20"]
    colors = cmap.colors
    score_dict = {}
    # Loading the forecast csv from each model inputted in model_list and scoring
    for i, model_name in enumerate(models_list):
        csv_name = f"forecasts/{site_id}/{target_variable}/{model_name}"
        forecast_df = pd.read_csv(f"{csv_name}.csv")
        forecast_df["datetime"] = pd.to_datetime(forecast_df["datetime"])
        times = pd.DatetimeIndex(forecast_df["datetime"])
        forecast_df = forecast_df.set_index("datetime")

        # Getting the validation set from targets
        if i == 0:
            date = forecast_df.index[0]
            forecast_horizon = len(forecast_df)
            validation_series = get_validation_series(
                targets_df, 
                site_id, 
                target_variable, 
                date, 
                forecast_horizon,
            )
        # Filtering the forecast to exclude dates with observed NaN's 
        filtered_model_forecast, filtered_validation_series = filter_forecast_df(
            forecast_df,
            validation_series
        )
        # Computing RMSE and plotting it
        filtered_validation_ts = TimeSeries.from_times_and_values(
            filtered_validation_series.index, 
            filtered_validation_series.values, 
            fill_missing_dates=False
        )
        model_score = rmse(filtered_validation_ts, filtered_model_forecast)
        score_dict[model_name] = model_score
            
    # Now, making the forecast based off of historical mean and std
    historical_model = HistoricalForecaster(targets=targets_df,
                          site_id=site_id,
                          target_variable=target_variable,
                          output_csv_name="historical_forecaster_output.csv",
                          validation_split_date=str(forecast_df.index[0])[:10],
                          forecast_horizon=len(forecast_df),)
    historical_model.make_forecasts()
    # Filtering, computing and plottings the RMSE of historical forecast
    historical_forecast_df = historical_model.forecast_ts.pd_dataframe(
        suppress_warnings=True
    )
    filtered_historical_forecast, filtered_validation_series = filter_forecast_df(
        historical_forecast_df,
        validation_series
    )
    hist_score = rmse(filtered_validation_ts, filtered_historical_forecast)
    score_dict["historical"] = hist_score

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

def make_crps_strip_plot_bysite(model, targets_df, target_variable, suffix="", plot_name=None):

    plt.figure(figsize=(16, 12))
    score_dict = {}
    
    for site_id in targets_df.site_id.unique():
        score_dict[site_id] = []
        file = model + suffix
        csv_name = f"forecasts/{site_id}/{target_variable}/{file}"
        # If there is no forecast at the site skip
        try:
            forecast_df = pd.read_csv(f"{csv_name}.csv")
        except:
            del score_dict[site_id]
            continue
        forecast_df["datetime"] = pd.to_datetime(forecast_df["datetime"])
        times = pd.DatetimeIndex(forecast_df["datetime"])
        forecast_df = forecast_df.set_index("datetime")
        
        # Getting the validation set from targets
        forecast_horizon = len(forecast_df)
        validation_series = get_validation_series(
            targets_df, 
            site_id, 
            target_variable, 
            times[0], 
            forecast_horizon,
        )

        # If there is no validation set at the site skip
        if len(validation_series) == 0:
            del score_dict[site_id]
            continue
        
        filtered_model_forecast, filtered_validation_series = filter_forecast_df(
            forecast_df, 
            validation_series
        )
    
        # Computing CRPS and recording
        scores = crps(filtered_model_forecast, filtered_validation_series)
        score_dict[site_id] = {"forecast_crps": scores.pd_dataframe().values[:, 0]}
        
        # Now, making the forecast based off of historical mean and std
        historical_model = HistoricalForecaster(
            targets=targets_df,
            site_id=site_id,
            target_variable=target_variable,
            output_csv_name=None,
            validation_split_date=str(filtered_model_forecast.time_index[0])[:10],
            forecast_horizon=forecast_horizon,
        )
        # Computing CRPS of historical forecast and plotting
        historical_model.make_forecasts()
        historical_forecast_df = historical_model.forecast_ts.pd_dataframe(suppress_warnings=True)

        filtered_historical_forecast, filtered_validation_series = filter_forecast_df(
            historical_forecast_df, 
            validation_series
        )
        
        scores = crps(filtered_historical_forecast, filtered_validation_series)
        score_dict[site_id]["historical_crps"] = scores.pd_dataframe().values[:, 0]
        
    score_df = pd.DataFrame([(site_id, data_dict['forecast_crps'][i], data_dict['historical_crps'][i]) \
                                 for site_id, data_dict in score_dict.items() \
                                 for i in range(len(data_dict['forecast_crps']))],
                            columns=["site_id", f'{model}', 'historical'])
    score_df = pd.melt(score_df, id_vars=["site_id"], var_name="model_type", value_name="crps")

    # Now creating the plot
    p = sns.stripplot(score_df, x="site_id", y="crps", hue="model_type", dodge=True, palette="tab20")

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
    
    