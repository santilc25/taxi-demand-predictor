from pathlib import Path
from datetime import datetime,timedelta
from typing import Optional, List

import numpy as np
import pandas as pd
import requests
from tqdm import tqdm

from src.paths import RAW_DATA_DIR, TRANSFORMED_DATA_DIR


def dowload_one_file_of_raw_data(year:int, month: int) -> Path:
    URL = f'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year}-{month:02d}.parquet'
    response = requests.get(URL)
    
    if response.status_code == 200:
        path = RAW_DATA_DIR / f'rides_{year}-{month:02d}.parquet'
        open(path,'wb').write(response.content)
        return path
    else:
        raise Exception(f'{URL} is not avaliable')
    
def validate_raw_data(rides:pd.DataFrame,
                      year:int,
                      month:int,) -> pd.DataFrame:
    """
    remove rows with pickup_datetimes outside their valid range
    """
    this_month_start = f'{year}-{month:02d}-01'
    next_month_start = f'{year}-{month+1:02d}-01' if month < 12 else f'{year+1}-01-01'
    rides = rides[rides.pickup_datetime >= this_month_start]
    rides = rides[rides.pickup_datetime < next_month_start]
    
    return rides

def load_raw_data(year: int, months:Optional[List[int]] = None) -> pd.DataFrame:
    rides = pd.DataFrame()
    
    if months is None:
        # download data for the entire year (all months)
        months = list(range(1,13))
    elif isinstance(months,int):
        #download data only for the months specified by months
        months = [months]
        
    for month in months:
        local_file = RAW_DATA_DIR / f'rides_{year}-{month:02d}.parquet'
        if not local_file.exists():
            try:
                # downliad the file from the NYC website
                print(f'Downloading file {year}-{month:02d}')
                dowload_one_file_of_raw_data(year,month)
            except:
                print(f'{year}-{month:02d} file is not avaliable')
                continue
        else:
            print(f'File {year}-{month:02d} was already in local storage')
            
        # Load the file into pandas
        rides_one_month = pd.read_parquet(local_file)
        
        # rename columns
        rides_one_month = rides_one_month[['tpep_pickup_datetime','PULocationID']]
        rides_one_month.rename(columns={'tpep_pickup_datetime':'pickup_datetime',
                                        'PULocationID':'pickup_location_id'},
                            inplace=True) 
        
        # Validate the file
        rides_one_month = validate_raw_data(rides_one_month,year,month)
        
        # append to existing data
        rides = pd.concat([rides,rides_one_month])
        
    # keep only time and origin of the ride
    rides = rides[['pickup_datetime','pickup_location_id']]
    
    return rides

def add_missing_slots(agg_rides: pd.DataFrame) -> pd.DataFrame:
    location_ids = agg_rides['pickup_location_id'].unique()
    full_range = pd.date_range(agg_rides['pickup_hour'].min(),
                               agg_rides['pickup_hour'].max(),
                               freq='H')
    output = pd.DataFrame()
    
    for location_id in tqdm(location_ids):
        # Get only dates with rides
        agg_rides_i =  agg_rides.loc[agg_rides['pickup_location_id'] == location_id,['pickup_hour','rides']]
        
        # Add missing dates with 0
        agg_rides_i.set_index('pickup_hour',inplace=True)
        agg_rides_i.index = pd.DatetimeIndex(agg_rides_i.index)
        agg_rides_i = agg_rides_i.reindex(full_range,fill_value=0)
        
        # Add location ids to dates and concatenate
        agg_rides_i['pickup_location_id'] = location_id
        output = pd.concat([output,agg_rides_i])
    
    output = output.reset_index().rename(columns={'index':'pickup_hour'})
    
    return output

def transform_raw_data_into_ts_data(rides:pd.DataFrame) -> pd.DataFrame:
    # Sum rides per location and pickup hour
    rides['pickup_hour'] = rides['pickup_datetime'].dt.floor('H')
    agg_rides = rides.groupby(['pickup_hour','pickup_location_id']).size().reset_index()
    agg_rides.rename(columns={0:'rides'},inplace=True)
    
    # add rows for (locations, pickup hours) with 0 rides
    agg_rides_all_slots = add_missing_slots(agg_rides)
    
    return agg_rides_all_slots

def transform_ts_data_into_features_and_target (ts_data:pd.DataFrame, input_seq_len:int, step_size:int) -> pd.DataFrame:
    
    assert set(ts_data.columns) == {'pickup_hour','rides','pickup_location_id'}
    
    location_ids = ts_data['pickup_location_id'].unique()
    features = pd.DataFrame() 
    targets = pd.DataFrame()
    
    for location_id in tqdm(location_ids):
        # Get only one location id
        ts_data_one_location = ts_data.loc[ts_data.pickup_location_id == location_id,['pickup_hour','rides']]
        
        # Get indices
        indices = get_cutoff_indices(ts_data_one_location,input_seq_len,step_size)
        
        # Convert ts-data to tabular data (features-target)
        n_examples = len(indices)
        x = np.ndarray(shape=(n_examples,input_seq_len),dtype=np.float32)
        y = np.ndarray(shape=(n_examples))
        pickup_hours = []
        for i, idx in enumerate(indices):
            x[i,:] = ts_data_one_location.iloc[idx[0]:idx[1]]['rides'].values
            y[i] = ts_data_one_location.iloc[idx[1]:idx[2]]['rides'].values
            pickup_hours.append(ts_data_one_location.iloc[idx[1]]['pickup_hour'])
            
        # Numpy to pandas
        features_one_location = pd.DataFrame(x, 
                                             columns=[f'rides_previous_{i+1}_hour' for i in reversed(range(input_seq_len))]
                                             )
        features_one_location['pickup_hour'] = pickup_hours
        features_one_location['pickup_location_id'] = location_id
        
        targets_one_location = pd.DataFrame(y,columns=[f'target_rides_next_hour'])
        
        # Concatenate results
        features = pd.concat([features,features_one_location])
        targets = pd.concat([targets, targets_one_location])
    
    features.reset_index(inplace=True,drop=True)
    targets.reset_index(inplace=True,drop=True)
    
    return features, targets['target_rides_next_hour']


def get_cutoff_indices(data:pd.DataFrame, n_features:int, step_size:int):
    stop_position = len(data) - 1
    
    #Extraer indices
    subseq_first_idx = 0
    subseq_mid_idx = n_features
    subseq_last_idx = n_features + 1
    indices = []
    
    while subseq_last_idx <= stop_position:
        indices.append((subseq_first_idx,subseq_mid_idx,subseq_last_idx))
        
        subseq_first_idx += step_size
        subseq_mid_idx += step_size
        subseq_last_idx += step_size
        
    return indices