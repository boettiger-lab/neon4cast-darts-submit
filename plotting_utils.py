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
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import os
from darts import TimeSeries
import glob
import numpy as np
import CRPS.CRPS as forecastscore
from darts.metrics import rmse
import matplotlib as mpl

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

def make_df_from_score_dict(score_dict):
    # Create lists to store the data
    site_id_list = []
    date_list = []
    metric_list = []
    model_list = []
    value_list = []
    
    # Iterate through the dictionary and extract data
    for site_id, dates in score_dict.items():
        for date, values in dates.items():
            crps_forecast_array = values['crps_forecast']
            crps_historical_array = values['crps_historical']
            rmse_forecast = values['rmse_forecast']
            rmse_historical = values['rmse_historical']
    
            # Add entries for 'crps', and 'forecast' and 'historical'
            for historical_crps_val, forecast_crps_val in \
        zip(crps_historical_array, crps_forecast_array):
                site_id_list.append(site_id)
                date_list.append(date)
                metric_list.append('crps')
                model_list.append('forecast')
                value_list.append(forecast_crps_val)
    
                site_id_list.append(site_id)
                date_list.append(date)
                metric_list.append('crps')
                model_list.append('historical')
                value_list.append(historical_crps_val)
    
            # Add entries for 'rmse' and 'forecast'
            site_id_list.append(site_id)
            date_list.append(date)
            metric_list.append('rmse')
            model_list.append('forecast')
            value_list.append(rmse_forecast)
    
            # Add entries for 'rmse' and 'historical'
            site_id_list.append(site_id)
            date_list.append(date)
            metric_list.append('rmse')
            model_list.append('historical')
            value_list.append(rmse_historical)
    
    # Create a DataFrame
    df = pd.DataFrame({
        'site_id': site_id_list,
        'date': date_list,
        'metric': metric_list,
        'model': model_list,
        'value': value_list
    })

    return df

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

def plot_crps_bydate(glob_prefix, targets_df, site_id, target_variable, suffix="", plot_name=None):

    plt.figure(figsize=(12, 8))
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

    
def score_improvement_bysite(model, targets_df, target_variable, suffix="", plot_name=None):
    score_dict = {}
    # For each site, score CRPS and RMSE individually and add to score_dict
    for site_id in targets_df.site_id.unique():
        site_dict = {}
        glob_prefix = f'forecasts/{site_id}/{target_variable}/{model}_{suffix}/forecast*'
        csv_list = sorted(glob.glob(glob_prefix))
        for csv in csv_list:
            site_dict = modify_score_dict(
                csv, 
                targets_df, 
                target_variable, 
                site_id, 
                suffix, 
                site_dict
            )
            site_dict = modify_score_dict(
                csv, 
                targets_df, 
                target_variable, 
                site_id, 
                suffix, 
                site_dict,
                score_rmse=True,
            )
        score_dict[site_id] = site_dict
    df = make_df_from_score_dict(score_dict)

    df = df.groupby(['site_id', 'date', 'metric', 'model']).mean().reset_index()

    # Filtering
    forecast_df = df[df['model'] == 'forecast']
    historical_df = df[df['model'] == 'historical']

    # Merge the two DataFrames on site_id, date, and metric
    merged_df = pd.merge(
        forecast_df, 
        historical_df, 
        on=['site_id', 'date', 'metric'], 
        suffixes=('_forecast', '_historical')
    )

    # Calculate percent improvement for each metric
    merged_df['percent_improvement'] = ((merged_df['value_historical'] - merged_df['value_forecast']) / merged_df['value_historical']) * 100

    # Finding the amount of windows where the mean crps or rmse has a positive improvement
    merged_df['positive_improvement'] = merged_df['percent_improvement'] > 0

    # Pivoting the table so that we can easily count forecast windows where both rmse and crps
    # had a positive improvement
    pivot_df = merged_df.pivot_table(
        index=['site_id', 'date'], 
        columns='metric', 
        values='positive_improvement'
    )
    pivot_df.columns = ['{}_positive_improvement'.format(col) for col in pivot_df.columns]
    pivot_df.reset_index(inplace=True)

    # Count times where both RMSE and CRPS % improvement were positive
    pivot_df['combined_improvement'] = (
        (pivot_df['crps_positive_improvement'] + pivot_df['rmse_positive_improvement']) == 2
    )

    pivot_df['model'] = model
    return pivot_df

def plot_improvement_bysite(score_df, metadata_df, title_name):
    ## Find the percentage of forecast windows during which the ML model excelled 
    ## the historical forecaster
    score_df = score_df[['site_id', 'combined_improvement']].groupby(['site_id']).mean() * 100
    score_df.reset_index(inplace=True)
    ## Rename the 'combined_improvement' column to 'combined_improvement_percentage'
    score_df.rename(columns={'combined_improvement': 'combined_improvement_percentage'}, inplace=True)

    ## Marking the sites at which 
    score_df['above_50'] = score_df['combined_improvement_percentage'] > 50

    # Combining df's to include metadata
    df = pd.merge(
        score_df, 
        metadata_df, 
        right_on='field_site_id', 
        left_on='site_id'
    ).drop(columns=['field_site_id'])
    
    plt.figure(figsize=(12, 8))
    custom_palette = {True: 'tab:blue', False: 'indianred'}
    markers = {'Wadeable Stream': 's', 'Lake': 'o', 'Non-wadeable River': '^'}

    for site_type in ['Wadeable Stream', 'Lake', 'Non-wadeable River']:
        
        sns.pointplot(
            data=df.loc[df.field_site_subtype == site_type],
            x='site_id',
            y='combined_improvement_percentage',
            linestyle='',
            hue='above_50',
            palette=custom_palette,
            markers=markers[site_type],
        )

    plt.grid(False)
    plt.ylabel("% of CRPS and RMSE Improvement over Historical")
    ax = plt.gca()
    ax.spines["left"].set_visible(True)
    ax.spines["bottom"].set_visible(True)
    plt.xticks(rotation=30)
    plt.legend(labels=[])
    plt.title(title_name)

def plot_global_percentages(df_, title_name):
    plt.figure(figsize=(12, 8))
    custom_palette = {True: 'tab:blue', False: 'indianred'}
    
    global_percentages = df_[['model', 'combined_improvement']].groupby(['model']).mean() * 100
    global_percentages.reset_index(inplace=True)
    global_percentages.rename(
        columns={'combined_improvement': 'combined_improvement_percentage'}, 
        inplace=True
    )

    # Marking the sites at which 
    global_percentages['above_50'] = global_percentages['combined_improvement_percentage'] > 50
    sns.pointplot(
        data=global_percentages,
        x='model',
        y='combined_improvement_percentage',
        linestyle='',
        hue='above_50',
        palette=custom_palette,
    )

    plt.grid(False)
    plt.ylabel("% of CRPS and RMSE Improvement over Historical")
    ax = plt.gca()
    ax.spines["left"].set_visible(True)
    ax.spines["bottom"].set_visible(True)
    plt.xticks(rotation=30)
    plt.legend(labels=[])
    plt.title(title_name)

def plot_site_type_percentages_global(df_, metadata_df, title_name):
    plt.figure(figsize=(12, 8))
    custom_palette = {True: 'tab:blue', False: 'indianred'}

    # Combining df's to include metadata
    df = pd.merge(
        df_, 
        metadata_df, 
        right_on='field_site_id', 
        left_on='site_id'
    ).drop(columns=['field_site_id'])
    
    results = df[['combined_improvement', 'field_site_subtype']].groupby(['field_site_subtype']).mean() * 100
    results.reset_index(inplace=True)
    results.rename(
        columns={'combined_improvement': 'combined_improvement_percentage'}, 
        inplace=True
    )

    # Marking the sites at which 
    results['above_50'] = results['combined_improvement_percentage'] > 50
    sns.pointplot(
        data=results,
        x='field_site_subtype',
        y='combined_improvement_percentage',
        linestyle='',
        hue='above_50',
        palette=custom_palette,
    )

    plt.grid(False)
    plt.ylabel("% of CRPS and RMSE Improvement over Historical")
    ax = plt.gca()
    ax.spines["left"].set_visible(True)
    ax.spines["bottom"].set_visible(True)
    plt.xticks(rotation=30)
    plt.legend(labels=[])
    plt.title(title_name)

def plot_site_type_percentages_bymodel(df_, metadata_df, title_name):
    plt.figure(figsize=(12, 8))
    custom_palette = {True: 'tab:blue', False: 'indianred'}
    markers = {'Wadeable Stream': 's', 'Lake': 'o', 'Non-wadeable River': '^'}

    # Combining df's to include metadata
    df = pd.merge(
        df_, 
        metadata_df, 
        right_on='field_site_id', 
        left_on='site_id'
    ).drop(columns=['field_site_id'])
    
    results = df[[
        'combined_improvement', 
        'field_site_subtype', 
        'model'
    ]].groupby(['field_site_subtype', 'model']).mean() * 100
    
    results.reset_index(inplace=True)
    results.rename(
        columns={'combined_improvement': 'combined_improvement_percentage'}, 
        inplace=True
    )

    # Marking the sites at which 
    results['above_50'] = results['combined_improvement_percentage'] > 50

    for site_type in ['Wadeable Stream', 'Lake', 'Non-wadeable River']:
        sns.pointplot(
            data=results.loc[results.field_site_subtype == site_type],
            x='model',
            y='combined_improvement_percentage',
            linestyle='',
            hue='above_50',
            palette=custom_palette,
            markers=markers[site_type],
        )

    plt.grid(False)
    plt.ylabel("% of CRPS and RMSE Improvement over Historical")
    ax = plt.gca()
    ax.spines["left"].set_visible(True)
    ax.spines["bottom"].set_visible(True)
    plt.xticks(rotation=30)
    plt.title(title_name)
    # Create your custom legend elements
    triangle = Line2D([0], [0], marker='^', color='w', markerfacecolor='grey', markersize=8, label='Non-wadeable River')
    circle = Line2D([0], [0], marker='o', color='w', markerfacecolor='grey', markersize=8, label='Lake')
    square = Line2D([0], [0], marker='s', color='w', markerfacecolor='grey', markersize=8, label='Wadeable Stream')
    
    # Create a custom legend
    legend_elements = [triangle, circle, square]
    plt.legend(labels=[])
    ax.legend(handles=legend_elements, loc='upper right')

def plot_window_and_sitetype_performance(model_df, metadata_df, title_name):
    plt.figure(figsize=(12, 8))
    custom_palette = {True: 'tab:blue', False: 'indianred'}
    markers = {'Wadeable Stream': 's', 'Lake': 'o', 'Non-wadeable River': '^'}

    # Combining df's to include metadata
    df = pd.merge(
        model_df, 
        metadata_df, 
        right_on='field_site_id', 
        left_on='site_id'
    ).drop(columns=['field_site_id'])
    
    results = df[[
        'combined_improvement', 
        'field_site_subtype',
        'date'
    ]].groupby(['field_site_subtype', 'date']).mean() * 100
    
    results.reset_index(inplace=True)
    results.rename(
        columns={'combined_improvement': 'combined_improvement_percentage'}, 
        inplace=True
    )

    # Marking the sites at which 
    results['above_50'] = results['combined_improvement_percentage'] > 50

    for site_type in ['Wadeable Stream', 'Lake', 'Non-wadeable River']:
        sns.pointplot(
            data=results.loc[results.field_site_subtype == site_type],
            x='date',
            y='combined_improvement_percentage',
            linestyle='',
            hue='above_50',
            palette=custom_palette,
            markers=markers[site_type],
        )

    plt.grid(False)
    plt.ylabel("% of CRPS and RMSE Improvement over Historical")
    ax = plt.gca()
    ax.spines["left"].set_visible(True)
    ax.spines["bottom"].set_visible(True)
    plt.xticks(rotation=30)
    plt.title(title_name)
    # Create your custom legend elements
    triangle = Line2D([0], [0], marker='^', color='w', markerfacecolor='grey', markersize=8, label='Non-wadeable River')
    circle = Line2D([0], [0], marker='o', color='w', markerfacecolor='grey', markersize=8, label='Lake')
    square = Line2D([0], [0], marker='s', color='w', markerfacecolor='grey', markersize=8, label='Wadeable Stream')
    
    # Create a custom legend
    legend_elements = [triangle, circle, square]
    plt.legend(labels=[])
    ax.legend(handles=legend_elements, loc='upper right')

def plot_region_percentages(df_, metadata_df, title_name):
    plt.figure(figsize=(12, 8))
    custom_palette = {True: 'tab:blue', False: 'indianred'}
    markers = {'Wadeable Stream': 's', 'Lake': 'o', 'Non-wadeable River': '^'}

    # Combining df's to include metadata
    df = pd.merge(
        df_, 
        metadata_df, 
        right_on='field_site_id', 
        left_on='site_id'
    ).drop(columns=['field_site_id'])
    
    results = df[[
        'combined_improvement', 
        'region',
        'field_site_subtype'
    ]].groupby(['field_site_subtype', 'region']).mean() * 100
    
    results.reset_index(inplace=True)
    results.rename(
        columns={'combined_improvement': 'combined_improvement_percentage'}, 
        inplace=True
    )

    # Marking the sites at which 
    results['above_50'] = results['combined_improvement_percentage'] > 50

    for site_type in ['Wadeable Stream', 'Lake', 'Non-wadeable River']:
        sns.pointplot(
            data=results.loc[results.field_site_subtype == site_type],
            x='region',
            y='combined_improvement_percentage',
            linestyle='',
            hue='above_50',
            palette=custom_palette,
            markers=markers[site_type],
        )

    plt.grid(False)
    plt.ylabel("% of CRPS and RMSE Improvement over Historical")
    ax = plt.gca()
    ax.spines["left"].set_visible(True)
    ax.spines["bottom"].set_visible(True)
    plt.xticks(rotation=30)
    plt.title(title_name)
    # Create your custom legend elements
    triangle = Line2D([0], [0], marker='^', color='w', markerfacecolor='grey', markersize=8, label='Non-wadeable River')
    circle = Line2D([0], [0], marker='o', color='w', markerfacecolor='grey', markersize=8, label='Lake')
    square = Line2D([0], [0], marker='s', color='w', markerfacecolor='grey', markersize=8, label='Wadeable Stream')
    
    # Create a custom legend
    legend_elements = [triangle, circle, square]
    plt.legend(labels=[])
    ax.legend(handles=legend_elements, loc='upper right')

