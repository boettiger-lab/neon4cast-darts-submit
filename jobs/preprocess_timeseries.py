from utils import (
                TimeSeriesPreprocessor,
)
import pandas as pd
import os
from darts import TimeSeries
import numpy as np
import time

if __name__=="__main__":
    start = time.time()
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    targets = pd.read_csv("targets.csv.gz")

    # For the training set
    data_preprocessor = TimeSeriesPreprocessor(
        validation_split_date='2022-07-19',
        load_dir_name='preprocessed_train/',
    )
    
    _ = [data_preprocessor.preprocess_data(site) for site in targets.site_id.unique()]
    
    data_preprocessor.save()

    # For the validation set
    data_preprocessor = TimeSeriesPreprocessor(
        validation_split_date='2023-07-19',
        load_dir_name='preprocessed_validate/',
    )
    
    _ = [data_preprocessor.preprocess_data(site) for site in targets.site_id.unique()]
    
    data_preprocessor.save()

    
    print ("Runtime to preprocess the time series: ", time.time()-start)