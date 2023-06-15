import pandas as pd
from darts.models import GaussianProcessFilter
from darts import TimeSeries
from sklearn.gaussian_process.kernels import RBF
from darts.models import BlockRNNModel
from darts.utils.likelihood_models import LaplaceLikelihood
from darts.dataprocessing.transformers import Scaler
import ray

# Make all the TimeSeries have the same length
def get_variable_dataframes(site, targets):
    """
    Returns a dictionary of {variable: dataframes with a common time index
    over site}
    """
    first_dates = []
    last_dates = []
    variable_df_dict = {}
    for variable in ["chla", "oxygen", "temperature"]:
        variable_df_dict[variable] = targets[targets.site_id == site][
                                             targets.variable == variable]
        times = pd.to_datetime(variable_df_dict[variable]["datetime"])
        times = pd.DatetimeIndex(times)
        first_date = times.sort_values()[0]
        last_date = times.sort_values()[-1]

        first_dates.append(first_date)
        last_dates.append(last_date)

    full_index = pd.date_range(min(first_dates), max(last_dates))

    for variable in ["chla", "oxygen", "temperature"]:
        variable_df = variable_df_dict[variable].set_index(["datetime"])
        variable_df.index = pd.to_datetime(variable_df.index)
        idx = variable_df.index.union(full_index)
        variable_df = variable_df.reindex(idx)
        variable_df["site_id"] = site
        variable_df["variable"] = variable
        variable_df_dict[variable] = variable_df

    return variable_df_dict

@ray.remote
def make_stitched_series(site_dict):
    """
    Returns a dictionary {"variable": stitched time series of variable}
    """
    variables = ["chla", "oxygen", "temperature"]
    kernel = RBF()
    
    gpf_missing = GaussianProcessFilter(kernel=kernel, 
                                        alpha=0.001, 
                                        n_restarts_optimizer=100)
    
    gpf_missing_big_gaps = GaussianProcessFilter(kernel=kernel, 
                                                 alpha=0.2, 
                                                 n_restarts_optimizer=100)
    stitched_series = {}
    
    for variable in variables:
        variable_df = site_dict[variable]
        variable_tseries = TimeSeries.from_times_and_values(variable_df.index, 
                                            variable_df["observation"], 
                                            fill_missing_dates=True, freq="D")

        # Filtering the TimeSeries
        try:
            filtered = gpf_missing.filter(variable_tseries, num_samples=500)
            filtered_big_gaps = gpf_missing_big_gaps.filter(variable_tseries, 
                                                            num_samples=500)
        except:
            continue
    
        #if there is a gap over 7 indices, use big gap filter
        gap_series = variable_tseries.gaps()
        stitched_df = filtered.pd_dataframe()
        replacement_df = filtered_big_gaps.pd_dataframe()
        
        for index, row in gap_series.iterrows():
            if row["gap_size"] > 7:
                for date in pd.date_range(row["gap_start"], row["gap_end"]):
                    stitched_df.loc[date] = replacement_df.loc[date]
        
        stitched_series[variable] = TimeSeries.from_times_and_values(
                                    stitched_df.index, 
                                    stitched_df.values.reshape(
                                                len(stitched_df), 
                                                1, 
                                                -1))

    return stitched_series


def train_models(site_id, targets):
    """
    Returns a dictionary {site_id: [ml_model, scaled validation inputs, scaled validation covariates, scaler]}
    """
    #Note site_stitched is a dictionary {"var": tseries}
    site_stitched = make_stitched_series(site_id, targets)

    models = {} 

    for var in site_stitched.keys():
        rnn = BlockRNNModel(model="LSTM",
                            hidden_dim=32,
                            batch_size=8,
                            input_chunk_length=15,
                            output_chunk_length=34,
                            likelihood=LaplaceLikelihood(),
                            optimizer_kwargs={"lr": 1e-4},
                            n_rnn_layers=3,
                            random_state=0)
        
        # Set a date to split between train and validation set
        val_split = pd.Timestamp(year=2023, month=1, day=1)
        train_set, val_set = site_stitched[var].split_before(val_split)

        scaler = Scaler()
        train_scaled = scaler.fit_transform(train_set)
        val_scaled = scaler.transform(val_set)
        # Dealing with covariates
        covariate_list = list(site_stitched.keys())
        covariate_list.remove(var)
        
        if len(covariate_list) == 0:
            train_covariates = None
            val_covariates = None
        elif len(covariate_list) == 1:
            covariates = site_stitched[covariate_list[0]]
        else:
            scaled_cov0 = scaler.transform(site_stitched[covariate_list[0]])
            scaled_cov1 = scaler.transform(site_stitched[covariate_list[1]])
            covariates = scaled_cov0.stack(scaled_cov1)

        # Now training model
        rnn.fit(train_scaled,
                past_covariates=covariates,
                epochs=250, 
                verbose=False)
    
        models[var] = [rnn, val_scaled, covariates, scaler]

    return models