import pandas as pd
from darts.models import GaussianProcessFilter
from darts import TimeSeries
from sklearn.gaussian_process.kernels import RBF
from darts.models import BlockRNNModel
from darts.utils.likelihood_models import LaplaceLikelihood
from darts.dataprocessing.transformers import Scaler
import ray

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