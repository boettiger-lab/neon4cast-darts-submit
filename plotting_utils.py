from utils import (
    BaseForecaster, 
    ResidualForecaster,
    TimeSeriesPreprocessor,
    crps,
    HistoricalForecaster,
    NaivePersistenceForecaster,
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
from sklearn.cluster import KMeans

pd.options.mode.chained_assignment = None

def generate_metadata_df():
    '''
    Reads the metadata csv and performs K-means clustering to generate 
    geographical groupings. Returns a dataframe with water body type, 
    geographical coordinates and cluster for each site id.
    '''
    metadata = pd.read_csv('NEON_Field_Site_Metadata_20220412.csv')
    metadata = metadata.loc[metadata.aquatics == 1][
        ['field_site_id', 'field_site_subtype', 'field_latitude', 'field_longitude']
    ]
    
    # Performing K-Means clustering
    num_clusters = 5
    
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    metadata['cluster'] = kmeans.fit_predict(
        metadata[['field_latitude', 'field_longitude']]
    )
    region_mapping = {
        0: 'East',
        1: 'Alaska',
        2: 'West',
        3: 'Mid',
        4: 'Puerto Rico'
    }
    metadata['region'] = metadata['cluster'].map(region_mapping)

    return metadata

def get_validation_series(targets_df, site_id, target_variable, date, forecast_horizon):
    '''
    Returns a TimeSeries of the forecast window from `targets_df`
    '''
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
    '''
    Returns a dataframe with the forecast scores and other details in `score_dict`
    '''
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
            rmse_naive = values['rmse_naive']
    
            entries = [
                (site_id, date, 'crps', 'forecast', forecast_crps_val)
                for forecast_crps_val in crps_forecast_array
            ] + [
                (site_id, date, 'crps', 'historical', historical_crps_val)
                for historical_crps_val in crps_historical_array
            ] + [
                (site_id, date, 'rmse', 'forecast', rmse_forecast),
                (site_id, date, 'rmse', 'historical', rmse_historical),
                (site_id, date, 'rmse', 'naive', rmse_naive)
            ]
    
            # Extend the lists with the generated entries
            site_id_list.extend([entry[0] for entry in entries])
            date_list.extend([entry[1] for entry in entries])
            metric_list.extend([entry[2] for entry in entries])
            model_list.extend([entry[3] for entry in entries])
            value_list.extend([entry[4] for entry in entries])
    
    # Create a DataFrame
    df = pd.DataFrame({
        'site_id': site_id_list,
        'date': date_list,
        'metric': metric_list,
        'model': model_list,
        'value': value_list
    })
    
    return df

def modify_score_dict(csv, targets_df, target_variable, site_id, suffix, score_dict):
    '''
    Returns a dictionary with the CRPS and RMSE scores for the ML model (whose forecast
    is provided in `csv`) as well as the historical and naive persistence model.
    '''
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
        # This removes entries from the forecast that do not have validation points
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

    rmse_score = rmse(filtered_validation_ts, filtered_model_forecast)
    score_dict[time_str]["rmse_forecast"] = rmse_score

    crps_scores = crps(
        filtered_model_forecast, 
        filtered_validation_ts,
        observed_is_ts=True,
    )
    score_dict[time_str]["crps_forecast"] = crps_scores.pd_dataframe().values[:, 0]

    # Instantiating the null models which includes a daily historical and a naive
    # persistence model
    input_dict = {
        'targets': targets_df,
        'site_id': site_id,
        'target_variable': target_variable,
        'output_csv_name': None,
        'validation_split_date': str(times[0])[:10],
        'forecast_horizon': forecast_horizon,
    }
    

    null_models = [
        HistoricalForecaster(**input_dict), 
        NaivePersistenceForecaster(**input_dict)
    ]
    
    # If issue making historical forecasts, then we'll skip.
    try:
        [model.make_forecasts() for model in null_models]
    except:
        del score_dict[time_str]
        return score_dict

    forecast_dfs = [
        model.forecast_ts.pd_dataframe(suppress_warnings=True) \
        for model in null_models
    ]

    # Note that the filter_forecast outputs a tuple with the filtered
    # forecast and the validation series
    filtered_forecasts = [
        filter_forecast_df(forecast_dfs[0], validation_series),
        filter_forecast_df(forecast_dfs[1], validation_series)
    ]

    rmse_scores = [
        rmse(filtered_validation_ts, filtered_forecasts[0][0]),
        rmse(filtered_validation_ts, filtered_forecasts[1][0])
    ]
    
    crps_scores = crps(
            filtered_forecasts[0][0],
            filtered_validation_ts,
            observed_is_ts=True,
    )
    
    score_dict[time_str]["rmse_historical"] = rmse_scores[0]
    score_dict[time_str]["rmse_naive"] = rmse_scores[1]
    score_dict[time_str]["crps_historical"] = crps_scores.pd_dataframe().values[:, 0]
    
    return score_dict

def score_improvement_bysite(model, targets_df, target_variable, suffix="", plot_name=None):
    '''
    This function collects the forecast scores for the specifed model and target variable.
    Then it returns a dataframe with columns for the difference in CRPS and RMSE
    compared to the historical and naive persistence null model (note that the naive will only be RMSE).
    '''
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
        score_dict[site_id] = site_dict

    # Producing a dataframe from the score dictionary, as df's are easier
    # to manipulate
    df = make_df_from_score_dict(score_dict)

    # Using the mean CRPS score over the forecast horizon
    df = df.groupby(['site_id', 'date', 'metric', 'model']).mean().reset_index()

    # Creating a CRPS and RMSE dataframe separately which is definitely
    # not the most elegant solution here
    crps_df = df[df['metric'] == 'crps']
    rmse_df = df[df['metric'] == 'rmse']
    
    forecast_dfs = [df_[df_['model'] == 'forecast'] for df_ in [crps_df, rmse_df]]
    historical_dfs = [df_[df_['model'] == 'historical'] for df_ in [crps_df, rmse_df]]
    naive_df = df[df['model'] == 'naive']
    naive_df = naive_df.rename(columns={'value': 'value_naive'})

    # Merge the two DataFrames on site_id, date, and metric
    crps_merged = pd.merge(
        forecast_dfs[0], 
        historical_dfs[0], 
        on=['site_id', 'date', 'metric'], 
        suffixes=('_forecast', '_historical')
    )

    rmse_merged = pd.merge(
        forecast_dfs[1], 
        historical_dfs[1], 
        on=['site_id', 'date', 'metric'], 
        suffixes=('_forecast', '_historical')
    )

    rmse_merged = pd.merge(
        rmse_merged, 
        naive_df, 
        on=['site_id', 'date', 'metric'], 
    )
    # Calculate percent improvement for each metric
    crps_merged['difference_historical_ml_crps'] = -(
        crps_merged['value_historical'] - crps_merged['value_forecast']
    )
    
    rmse_merged['difference_historical_ml_rmse'] = -(
        rmse_merged['value_historical'] - rmse_merged['value_forecast']
    ) 
    
    rmse_merged['difference_naive_ml_rmse'] = -(
        rmse_merged['value_naive'] - rmse_merged['value_forecast']
    )

    rmse_merged['difference_naive_historical_rmse'] = -(
        rmse_merged['value_naive'] - rmse_merged['value_historical']
    )

    # Deleting unnecessary columns
    rmse_merged = rmse_merged.drop(rmse_merged.filter(like='model').columns, axis=1)
    rmse_merged = rmse_merged.drop(rmse_merged.filter(like='value').columns, axis=1)
    crps_merged = crps_merged.drop(crps_merged.filter(like='model').columns, axis=1)
    crps_merged = crps_merged.drop(crps_merged.filter(like='value').columns, axis=1)

    # Joining the two df's along site id and date then adding a combined improvement column
    # for comparison against the climatology model
    merged_df = pd.merge(crps_merged, rmse_merged, on=['site_id', 'date'], how='inner')
    merged_df = merged_df.drop(merged_df.filter(like='metric').columns, axis=1)
    merged_df['model'] = model

    return merged_df
    
def plot_forecast(date, targets_df, site_id, target_variable, model_dir, plot_name=None):
    '''
    Returns a plot of the forecast specified by the date and model directory
    in addition to the observed values, the climatology forecast and the naive persistence
    forecast.
    '''
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

    # And the naive forecaster
    naive_model = NaivePersistenceForecaster(
        targets=targets_df,
        site_id=site_id,
        target_variable=target_variable,
        validation_split_date=str(model_forecast.time_index[0])[:10],
        forecast_horizon=len(model_forecast),
    )
    naive_model.make_forecasts()
    naive_model.forecast_ts.plot(label='Naive Persistence', color=colors[3])
    
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
    '''
    Returns a strip plot of the crps scores for the inputted ML model and the climatology model at
    each forecast window
    '''
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

def plot_improvement_bysite(score_df, metadata_df, title_name, historical=True):
    '''
    Returns a plot of the scoring metric difference vs. the site id;
    site type is encoded by color.
    '''
    ## Find the percentage of forecast windows during which the ML model excelled 
    ## the historical forecaster
    column = (
        'difference_historical_ml_crps' if historical \
         else 'difference_naive_ml_rmse'
    )

    # Combining df's to include metadata
    df = pd.merge(
        score_df, 
        metadata_df, 
        right_on='field_site_id', 
        left_on='site_id'
    ).drop(columns=['field_site_id'])
    
    plt.figure(figsize=(12, 8))
    color_dict = {
        'Wadeable Stream': 'tab:blue', 
        'Lake': 'indianred', 
        'Non-wadeable River': 'plum'
    }

    for site_type in ['Wadeable Stream', 'Lake', 'Non-wadeable River']:
        sns.boxplot(
            data=df.loc[df.field_site_subtype == site_type],
            x='site_id',
            y=column,
            color=color_dict[site_type],
            showfliers=False,
        )

    plt.grid(False)
    plt.axhline(y=0, color='black', linestyle='dashed', linewidth=1)
    if historical:
        plt.ylabel(f"CRPS(Model) - CRPS(Climatology) ")
    else:
        plt.ylabel(f"RMSE(Model) - RMSE(Naive Persistence) ")
    ax = plt.gca()
    ax.spines["left"].set_visible(True)
    ax.spines["bottom"].set_visible(True)
    plt.xticks(rotation=30)
    legend_handles = [Patch(facecolor=color, edgecolor='black') for color in color_dict.values()]
    legend_labels = list(color_dict.keys())
    plt.legend(legend_handles, legend_labels, title='Site Type', loc='upper right')
    plt.tight_layout()
    plt.title(title_name)

def plot_global_percentages(df_, title_name, historical=True):
    '''
    Returns a plot of the scoring metric difference vs. ML model type
    '''
    plt.figure(figsize=(12, 8))
    column = (
        'difference_historical_ml_crps' if historical \
         else 'difference_naive_ml_rmse'
    ) 

    sns.boxplot(
        data=df_,
        x='model',
        y=column,
        showfliers=False,
        color='tab:blue'
    )

    plt.grid(False)
    plt.axhline(y=0, color='black', linestyle='dashed', linewidth=1)
    if historical:
        plt.ylabel(f"CRPS(Model) - CRPS(Climatology) ")
    else:
        plt.ylabel(f"RMSE(Model) - RMSE(Naive Persistence) ")
    ax = plt.gca()
    ax.spines["left"].set_visible(True)
    ax.spines["bottom"].set_visible(True)
    plt.xticks(rotation=30)
    plt.legend(labels=[])
    plt.title(title_name)

def plot_site_type_percentages_global(df_, metadata_df, title_name, historical=True):
    '''
    Returns a plot of the scoring metric difference vs. water body type.
    '''
    plt.figure(figsize=(12, 8))
    color_dict = {
        'Wadeable Stream': 'tab:blue', 
        'Lake': 'indianred', 
        'Non-wadeable River': 'plum'
    }

    # Combining df's to include metadata
    df = pd.merge(
        df_, 
        metadata_df, 
        right_on='field_site_id', 
        left_on='site_id'
    ).drop(columns=['field_site_id'])

    column = (
        'difference_historical_ml_crps' if historical \
         else 'difference_naive_ml_rmse'
    ) 

    sns.boxplot(
        data=df,
        x='field_site_subtype',
        hue='field_site_subtype',
        y=column,
        showfliers=False,
        palette=color_dict,
    )

    plt.grid(False)
    plt.axhline(y=0, color='black', linestyle='dashed', linewidth=1)
    if historical:
        plt.ylabel(f"CRPS(ML Model) - CRPS(Climatology Model) ")
    else:
        plt.ylabel(f"RMSE(ML Model) - RMSE(Naive Persistence) ")
    ax = plt.gca()
    ax.spines["left"].set_visible(True)
    ax.spines["bottom"].set_visible(True)
    plt.xticks(rotation=30)
    plt.legend(labels=[])
    plt.title(title_name)

def plot_site_type_percentages_bymodel(df_, metadata_df, title_name, historical=True):
    '''
    Returns a plot of the scoring metric difference vs. model type;
    site type is encoded by color
    '''
    plt.figure(figsize=(12, 8))
    color_dict = {
        'Wadeable Stream': 'tab:blue', 
        'Lake': 'indianred', 
        'Non-wadeable River': 'plum'
    }

    # Combining df's to include metadata
    df = pd.merge(
        df_, 
        metadata_df, 
        right_on='field_site_id', 
        left_on='site_id'
    ).drop(columns=['field_site_id'])

    column = (
        'difference_historical_ml_crps' if historical \
         else 'difference_naive_ml_rmse'
    ) 

    sns.boxplot(
        data=df,
        x='model',
        hue='field_site_subtype',
        y=column,
        showfliers=False,
        dodge=True,
        palette=color_dict,
    )

    plt.grid(False)
    plt.axhline(y=0, color='black', linestyle='dashed', linewidth=1)
    if historical:
        plt.ylabel(f"CRPS(Model) - CRPS(Climatology) ")
    else:
        plt.ylabel(f"RMSE(Model) - RMSE(Naive Persistence) ")
    ax = plt.gca()
    ax.spines["left"].set_visible(True)
    ax.spines["bottom"].set_visible(True)
    plt.xticks(rotation=30)
    legend_handles = [Patch(facecolor=color, edgecolor='black') for color in color_dict.values()]
    legend_labels = list(color_dict.keys())
    plt.legend(legend_handles, legend_labels, title='Site Type', loc='upper right')
    plt.tight_layout()
    plt.title(title_name)

def plot_window_and_sitetype_performance(model_df, metadata_df, title_name, historical=True):
    '''
    Returns a plot of the difference in scoring metric vs. forecast windows;
    site type is encoded by color
    '''
    plt.figure(figsize=(12, 8))
    color_dict = {
        'Wadeable Stream': 'tab:blue', 
        'Lake': 'indianred', 
        'Non-wadeable River': 'plum'
    }

    # Combining df's to include metadata
    df = pd.merge(
        model_df, 
        metadata_df, 
        right_on='field_site_id', 
        left_on='site_id'
    ).drop(columns=['field_site_id'])

    column = (
        'difference_historical_ml_crps' if historical \
         else 'difference_naive_ml_rmse'
    ) 

    sns.boxplot(
        data=df,
        x='date',
        y=column,
        hue='field_site_subtype',
        palette=color_dict,
        dodge=True,
        showfliers=False,
    )

    plt.grid(False)
    plt.axhline(y=0, color='black', linestyle='dashed', linewidth=1)
    if historical:
        plt.ylabel(f"CRPS(Model) - CRPS(Climatology) ")
    else:
        plt.ylabel(f"RMSE(Model) - RMSE(Naive Persistence) ")
    ax = plt.gca()
    ax.spines["left"].set_visible(True)
    ax.spines["bottom"].set_visible(True)
    plt.xticks(rotation=30)
    legend_handles = [
        Patch(facecolor=color, edgecolor='black') for color in color_dict.values()
    ]
    legend_labels = list(color_dict.keys())
    plt.legend(legend_handles, legend_labels, title='Site Type', loc='upper right')
    plt.tight_layout()
    plt.title(title_name)

def plot_region_percentages(df_, metadata_df, title_name, historical=True):
    '''
    Returns a plot of the difference in scoring metric vs. the geographical regions;
    site type is encoded by color
    '''
    plt.figure(figsize=(12, 8))
    color_dict = {
        'Wadeable Stream': 'tab:blue', 
        'Lake': 'indianred', 
        'Non-wadeable River': 'plum'
    }

    # Combining df's to include metadata
    df = pd.merge(
        df_, 
        metadata_df, 
        right_on='field_site_id', 
        left_on='site_id'
    ).drop(columns=['field_site_id'])

    column = (
        'difference_historical_ml_crps' if historical \
         else 'difference_naive_ml_rmse'
    ) 
    
    sns.boxplot(
            data=df,
            x='region',
            y=column,
            hue='field_site_subtype',
            palette=color_dict,
            showfliers=False,
            dodge=True,
    )

    plt.grid(False)
    plt.axhline(y=0, color='black', linestyle='dashed', linewidth=1)
    if historical:
        plt.ylabel(f"CRPS(Model) - CRPS(Climatology) ")
    else:
        plt.ylabel(f"RMSE(Model) - RMSE(Naive Persistence) ")
    ax = plt.gca()
    ax.spines["left"].set_visible(True)
    ax.spines["bottom"].set_visible(True)
    plt.xticks(rotation=30)
    legend_handles = [
        Patch(facecolor=color, edgecolor='black') for color in color_dict.values()
    ]
    legend_labels = list(color_dict.keys())
    plt.legend(legend_handles, legend_labels, title='Site Type', loc='upper right')
    plt.tight_layout()
    plt.title(title_name)

