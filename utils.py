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

def global_model_preprocess(targets, stitched_dictionary):
    # Creating a dictionary of dictionaries indexed by site with
    # nested value being the stitched time series
    named_ts_dict = {}
    site_names = targets["site_id"].unique()
    for index, item in enumerate(time_series):
        name = site_names[index]
        named_ts_dict[name] = item

    # Now 
    inputs_covs_dict = {}
    inputs_training = []
    covs_training = []
    for name in site_names:
        # Accessing the dictionary of every site, then going through each
        # time series recorded at that site
        site_dict = named_ts_dict[name]
        site_variables = list(site_dict.keys())
        variables_main = ["oxygen", "chla", "temperature"]
        variable_dict = {}
        for index, variable in enumerate(variables_main):
            variables_sub = ["oxygen", "chla", "temperature"]
            # Checking to see if the site has data for a variable
            if variable in site_variables:
                # If so I remove that variable on this dummy list
                # and then try to create covariate TS
                variables_sub.pop(index) 
                covs = []
                # Here I find covariates that were recorded at the
                # site
                for cov_variable in variables_sub:
                    if cov_variable in site_variables:
                        covs.append(cov_variable)
                variables_sub.append(variable)
                # Then if there is more than one covariate, I stack them
                if len(covs) == 2:
                    cov_sup = site_dict[covs[0]]
                    cov_sub = site_dict[covs[1]]
                    cov = cov_sub.concatenate(cov_sup, axis=1, ignore_time_axis=True)
                    variable_value = [site_dict[variable], cov]
                elif len(covs) == 1:
                    cov = site_dict[covs[0]]
                    # I add a dummy covariate so that all covariates have same shape
                    shape = (cov.n_timesteps, cov.n_components, cov.n_samples)
                    null_array = np.zeros(shape)
                    dummy_cov = TimeSeries.from_times_and_values(cov.time_index, null_array)
                    cov = cov.concatenate(dummy_cov, axis=1, ignore_time_axis=True)
                    variable_value = [site_dict[variable], cov]
                # Making the decision that if there are no covariates, don't add to the dictionary
                # Finally, we put the entry together for the dictionary
                variable_dict[variable] = variable_value
                inputs_training.append(site_dict[variable])
                covs_training.append(cov)
        inputs_covs_dict[name] = variable_dict

    # Splitting training set and validation set
    val_split = pd.Timestamp(year=2023, month=1, day=1)
    
    val_set = []
    inputs = []
    cov_set = []
    for index in range(len(inputs_training)):
        input = inputs_training[index]
        cov = covs_training[index]
        try:
            train_inp, val_inp = input.split_before(val_split)    
        except:
            continue
        inputs.append(train_inp)
        val_set.append(val_inp)
        cov_set.append(cov)

    return inputs, val_set, cov_set

@ray.remote(num_gpus=1)
def train_global_model(inputs, cov_set):
    rnn = BlockRNNModel(model="LSTM",
                                hidden_dim=256,
                                batch_size=8,
                                input_chunk_length=15,
                                output_chunk_length=34,
                                likelihood=LaplaceLikelihood(),
                                optimizer_kwargs={"lr": 1e-4},
                                n_rnn_layers=3,
                                random_state=0)
    
    rnn.fit(inputs,
            past_covariates=cov_set,
            epochs=250, 
            verbose=False)

    return rnn

def train_local_model(site_id, site_stitched_dict, site_models_dict):
    """
    Returns a dictionary {site_id: [ml_model, 
                                    scaled validation inputs, 
                                    scaled validation covariates, 
                                    scaler]}
    """
    site_stitched = site_stitched_dict[site_id]

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
        try:
            train_set, val_set = site_stitched[var].split_before(val_split)
        except:
            continue
        
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
            covariates = scaled_cov0.concatenate(scaled_cov1, 
                                                 axis=1, 
                                                 ignore_time_axis=True)

        # Now training model
        rnn.fit(train_scaled,
                past_covariates=covariates,
                epochs=250, 
                verbose=False)
    
        site_models_dict[site_id].update({var: [rnn, val_scaled, covariates, scaler]})

    return

def make_plot(model, inputs, input_num, cov_set, val_set):
    plt.clf()
    preds = model.predict(series=inputs[input_num], 
                               past_covariates=cov_set[input_num], 
                               n=30,#len(val_set[input_num]),
                               num_samples=50)
    preds.plot(label="forecast")
    val_set[input_num][:30].plot(label="truth")
    plt.show()

def make_plot_long_horizon(model, inputs, input_num, cov_set, val_set):
    plt.clf()
    preds = model.predict(series=inputs[input_num], 
                               past_covariates=cov_set[input_num], 
                               n=len(val_set[input_num]),
                               num_samples=50)
    preds.plot(label="forecast")
    val_set[input_num].plot(label="truth")
    plt.show()

def make_plot_local(site, variable, site_models_dict):
    plt.clf()
    site_dict = site_models_dict[site]
    model_ = site_dict[variable][0]
    val_set_ = site_dict[variable][1]
    covs_ = site_dict[variable][2]
    scaler = site_dict[variable][-1]
    predictions_ = model_.predict(n=len(val_set_), past_covariates=covs_, num_samples=50)
    preds = scaler.inverse_transform(predictions_)
    preds.plot()
    val_set = scaler.inverse_transform(val_set_)
    val_set.plot(label="truth")
    plt.show()