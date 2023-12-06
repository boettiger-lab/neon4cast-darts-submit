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

#def create_rmse_df_from_dict(data_dict):
#    # Initialize empty lists to store the data
#    keys = []
#    historical_rmse = []
#    forecast_rmse = []
#
#    # Iterate through the dictionary and extract the data
#    for key, value in data_dict.items():
#        keys.append(key)
#        historical_rmse.append(value['rmse_historical'])
#        forecast_rmse.append(value['rmse_forecast'])
#
#    # Create a DataFrame from the extracted data
#    df = pd.DataFrame({'site_id': keys, 'rmse_historical': historical_rmse, 'rmse_forecast': forecast_rmse})
#
#    return df

def plot_forecast(date, targets_df, site_id, target_variable, model_dir, plot_name=None):
    cmap = mpl.colormaps["tab10"]
    colors = cmap.colors
    # Loading the forecast csv and creating a time series
    csv_name = f"forecasts/{site_id}/{target_variable}/{model_dir}/forecast_{date}.csv"
    df = pd.read_csv(csv_name)
    times = pd.to_datetime(df["datetime"])
    times = pd.DatetimeIndex(times)
    values = df.loc[:, df.columns!="datetime"].to_numpy().reshape((len(times), 1, -1))
    model_forecast = TimeSeries.from_times_and_values(times, 
                                                      values, 
                                                      fill_missing_dates=True, freq="D")
    model_forecast.plot(label=f"{model_dir}", color=colors[0])

    # Getting the validation series directly from the targets csv
    date = model_forecast.time_index[0]
    forecast_horizon = len(model_forecast)
    validation_series = get_validation_series(
        targets_df, 
        site_id, 
        target_variable, 
        date, 
        forecast_horizon
    )

    # Now, making the forecast based off of historical mean and std
    historical_model = HistoricalForecaster(
        targets=targets_df,
        site_id=site_id,
        target_variable=target_variable,
        output_csv_name="historical_forecaster_output.csv",
        validation_split_date=str(model_forecast.time_index[0])[:10],
        forecast_horizon=len(model_forecast),
    )
    historical_model.make_forecasts()
    historical_model.forecast_ts.plot(label="Historical", color=colors[1])
    validation_series.plot(label="Truth", color=colors[2])
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

def get_validation_series(targets_df, site_id, target_variable, date, forecast_horizon):
    # Being careful here with the date, note that I am matching the forecast,
    # so I don't need to advance.
    date_range = pd.date_range(
        date, 
        periods=forecast_horizon, 
        freq='D',
    )
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
        if len(entry) == 0:
            entry = pd.DataFrame({'datetime': [date], f'{target_variable}': [np.nan]})
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
        freq="D",
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
        (len(times), 1, -1)
    )

    # Issue is occurring here, why oh why TimeSeries so annoying
    filtered_forecast_ts = TimeSeries.from_times_and_values(
        times, 
        values,
        fill_missing_dates=True,
        freq='D'
    )

    return filtered_forecast_ts, validation_series

def modify_score_dict(csv, targets_df, target_variable, site_id, suffix, score_dict, score_rmse=False):
    try:
        forecast_df = pd.read_csv(csv)
    except:
        return score_dict
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
        return score_dict

    try:
        filtered_model_forecast, filtered_validation_series = filter_forecast_df(
            forecast_df, 
            validation_series
        )
    except:
        return score_dict

    # Initialize in case site id is empty at the site
    time_str = times[0].strftime('%Y_%m_%d')
    if time_str not in score_dict:
        score_dict[time_str] = {}
        
    # Computing CRPS and recording
    filtered_validation_ts = TimeSeries.from_times_and_values(
        filtered_validation_series.index, 
        filtered_validation_series.values, 
        fill_missing_dates=True,
        freq='D'
    )
    if score_rmse:
        rmse_score = rmse(filtered_validation_ts, filtered_model_forecast)
        score_dict[time_str]["rmse_forecast"] = rmse_score
    else:
        crps_scores = crps(
            filtered_model_forecast, 
            filtered_validation_ts,
            observed_is_ts=True,
        )
        score_dict[time_str]["crps_forecast"] = crps_scores.pd_dataframe().values[:, 0]
        
    # Now, making the forecast based off of historical mean and std
    historical_model = HistoricalForecaster(
        targets=targets_df,
        site_id=site_id,
        target_variable=target_variable,
        output_csv_name=None,
        validation_split_date=str(times[0])[:10],
        forecast_horizon=forecast_horizon,
    )
    # Computing CRPS of historical forecast and plotting
    # If issue making historical forecasts, then we'll skip.
    try:
        historical_model.make_forecasts()
    except:
        del score_dict[time_str]
        return score_dict
    historical_forecast_df = historical_model.forecast_ts.pd_dataframe(
        suppress_warnings=True
    )
    
    filtered_historical_forecast, filtered_validation_series = filter_forecast_df(
        historical_forecast_df, 
        validation_series
    )

    if score_rmse:
        rmse_score = rmse(filtered_validation_ts, filtered_historical_forecast)
        score_dict[time_str]["rmse_historical"] = rmse_score
    else:
        crps_scores = crps(
            filtered_historical_forecast, 
            filtered_validation_ts,
            observed_is_ts=True,
        )
        score_dict[time_str]["crps_historical"] = crps_scores.pd_dataframe().values[:, 0]
    return score_dict

def plot_forecasts(models_list, targets_df, site_id, target_variable, plot_name=None):
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

def plot_crps_bydate(glob_prefix, targets_df, site_id, target_variable, suffix="", plot_name=None):

    plt.figure(figsize=(16, 12))
    score_dict = {}
    
    csv_list = sorted(glob.glob(glob_prefix))
    
    for csv in csv_list:
        score_dict = modify_score_dict(
            csv, 
            targets_df, 
            target_variable, 
            site_id, 
            suffix, 
            score_dict
        )

    score_df = pd.DataFrame([(site_id, data_dict['crps_forecast'][i], data_dict['crps_historical'][i]) \
                                 for site_id, data_dict in score_dict.items() \
                                 for i in range(len(data_dict['crps_forecast']))],
                            columns=["date", 'forecast', 'historical'])
    score_df = pd.melt(score_df, id_vars=["date"], var_name="model_type", value_name="crps")

    # Now creating the plot
    p = sns.stripplot(score_df, x="date", y="crps", hue="model_type", dodge=True, palette="tab20")

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
        ax=p,
    )
    plt.grid(False)
    plt.ylabel("crps")
    ax = plt.gca()
    ax.spines["left"].set_visible(True)
    ax.spines["bottom"].set_visible(True)
    plt.xticks(rotation=30)
    plt.title(f'{target_variable} @ {site_id}')
    
    # Saving the plot if desired
    if plot_name != None:
        if not os.path.exists(f"plots/{site_id}/{target_variable}/"):
            os.makedirs(f"plots/{site_id}/{target_variable}/")
        plt.savefig(f"plots/{site_id}/{target_variable}/{plot_name}")

    
    
#def plot_score_improvement_bysite(model, targets_df, target_variable, suffix="", plot_name=None):
#
#    plt.figure(figsize=(10, 8))
#    score_dict = {}
#    
#    for site_id in targets_df.site_id.unique():
#        score_dict = modify_score_dict(
#            model, 
#            targets_df, 
#            target_variable, 
#            site_id, 
#            suffix, 
#            score_dict
#        )
#        score_dict = modify_score_dict(
#            model, 
#            targets_df, 
#            target_variable, 
#            site_id, 
#            suffix, 
#            score_dict,
#            score_rmse=True
#        )
#
#    rmse_df = create_rmse_df_from_dict(score_dict)
#    crps_df = pd.DataFrame([(site_id, data_dict['crps_forecast'][i], data_dict['crps_historical'][i]) \
#        for site_id, data_dict in score_dict.items() \
#        for i in range(len(data_dict['crps_forecast']))],
#        columns=["site_id", 'forecast', 'historical']
#    )
#    crps_df = pd.melt(crps_df, id_vars=["site_id"], var_name="model_type", value_name="crps")
#    crps_means = crps_df.groupby(["site_id", "model_type"])['crps'].mean().reset_index()
#    model_crps_scores = crps_means[crps_means['model_type'] == "forecast"]
#    historical_crps_scores = crps_means[crps_means['model_type'] == "historical"]
#    merged_df =  model_crps_scores.merge(
#        historical_crps_scores, 
#        on='site_id', 
#        suffixes=('_forecast', '_historical'),
#    )
#    merged_df = merged_df.merge(rmse_df, on="site_id")
#    merged_df.drop(
#        ['model_type_forecast', 'model_type_historical'], 
#        axis=1, 
#        inplace=True
#    )
#    melted_df = pd.melt(
#        merged_df, 
#        id_vars=['site_id'], 
#        var_name='crps/rmse', 
#        value_name='value'
#    )
#    melted_df[['score_type', 'model_type']] = melted_df['crps/rmse'].str.split(
#        '_', 
#        expand=True
#    )
#    pivoted_df = melted_df.pivot(
#        index='site_id', 
#        columns=['score_type', 'model_type'], 
#        values='value')
#    pivoted_df.reset_index(inplace=True)
#    custom_palette = {True: 'tab:blue', False: 'indianred'}
#    marker_key = {'crps': 'D', 'rmse': '*'}
#    for metric in ['crps', 'rmse']:
#        pivoted_df[f"{metric}_percent_improvement"] = (
#            (pivoted_df[metric]['historical'] - pivoted_df[metric]['forecast']) /
#            pivoted_df[metric]['historical']
#        ) * 100
#        pivoted_df[f"{metric}_positive"] = pivoted_df[f"{metric}_percent_improvement"] > 0
#        sns.pointplot(pivoted_df, 
#            x="site_id", 
#            y=f"{metric}_percent_improvement", 
#            linestyle="", 
#            marker=marker_key[metric],
#            hue=f"{metric}_positive",
#            palette=custom_palette,
#        )
#        
#    plt.grid(False)
#    plt.ylabel("% Improvement")
#    ax = plt.gca()
#    ax.spines["left"].set_visible(True)
#    ax.spines["bottom"].set_visible(True)
#    plt.xticks(rotation=30)
#    plt.legend(labels=[])
#    plt.title(f"{model+suffix} @ {target_variable}")
#    
#    # Saving the plot if desired
#    if plot_name != None:
#        if not os.path.exists(f"plots/{site_id}/{target_variable}/"):
#            os.makedirs(f"plots/{site_id}/{target_variable}/")
#        plt.savefig(f"plots/{site_id}/{target_variable}/{plot_name}")
#
#def plot_sitewide_comparison(model, target_df, suffix, plot_type='percentage_improvement'):
#    plot_function = {
#        'strip': plot_crps_bysite, 
#        'percentage_improvement': plot_score_improvement_bysite,
#    }[plot_type]
#    for target_variable in ['oxygen', 'temperature', 'chla']:
#        plot_function(model, target_df, target_variable, suffix)
#
#def count_score_improvement_bysite(model, targets_df, target_variable, suffix="", count_dict={}):
#    score_dict = {}
#    
#    for site_id in targets_df.site_id.unique():
#        score_dict = modify_score_dict(
#            model, 
#            targets_df, 
#            target_variable, 
#            site_id, 
#            suffix, 
#            score_dict
#        )
#        score_dict = modify_score_dict(
#            model, 
#            targets_df, 
#            target_variable, 
#            site_id, 
#            suffix, 
#            score_dict,
#            score_rmse=True
#        )
#
#    rmse_df = create_rmse_df_from_dict(score_dict)
#    crps_df = pd.DataFrame([(site_id, data_dict['crps_forecast'][i], data_dict['crps_historical'][i]) \
#        for site_id, data_dict in score_dict.items() \
#        for i in range(len(data_dict['crps_forecast']))],
#        columns=["site_id", 'forecast', 'historical']
#    )
#    crps_df = pd.melt(crps_df, id_vars=["site_id"], var_name="model_type", value_name="crps")
#    crps_means = crps_df.groupby(["site_id", "model_type"])['crps'].mean().reset_index()
#    model_crps_scores = crps_means[crps_means['model_type'] == "forecast"]
#    historical_crps_scores = crps_means[crps_means['model_type'] == "historical"]
#    merged_df =  model_crps_scores.merge(
#        historical_crps_scores, 
#        on='site_id', 
#        suffixes=('_forecast', '_historical'),
#    )
#    merged_df = merged_df.merge(rmse_df, on="site_id")
#    merged_df.drop(
#        ['model_type_forecast', 'model_type_historical'], 
#        axis=1, 
#        inplace=True
#    )
#    melted_df = pd.melt(
#        merged_df, 
#        id_vars=['site_id'], 
#        var_name='crps/rmse', 
#        value_name='value'
#    )
#    melted_df[['score_type', 'model_type']] = melted_df['crps/rmse'].str.split(
#        '_', 
#        expand=True
#    )
#    pivoted_df = melted_df.pivot(
#        index='site_id', 
#        columns=['score_type', 'model_type'], 
#        values='value')
#    pivoted_df.reset_index(inplace=True)
#    custom_palette = {True: 'tab:blue', False: 'indianred'}
#    marker_key = {'crps': 'D', 'rmse': '*'}
#    for metric in ['crps', 'rmse']:
#        pivoted_df[f"{metric}_percent_improvement"] = (
#            (pivoted_df[metric]['historical'] - pivoted_df[metric]['forecast']) /
#            pivoted_df[metric]['historical']
#        ) * 100
#        pivoted_df[f"{metric}_positive"] = pivoted_df[f"{metric}_percent_improvement"] > 0
#        
#    pivoted_df['combined_improvement'] = pivoted_df['crps_positive'] & pivoted_df['rmse_positive']
#    model_name = model + suffix
#    if model_name not in count_dict.keys():
#        count_dict[model_name] = {}
#    count_dict[model_name][target_variable] = f"{pivoted_df['combined_improvement'].sum()}" +\
#        f"/{len(pivoted_df['combined_improvement'])}"
#
#    return count_dict