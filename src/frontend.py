import zipfile
from datetime import datetime

import requests
import numpy as np
import pandas as pd

# plotting libraries
import streamlit as st
import geopandas as gpd
import pydeck as pdk

from src.inference import (
    load_batch_of_features_from_store,
    load_model_from_registry,
    get_model_predictions
)

from src.paths import DATA_DIR
from src.plot import plot_one_sample


st.set_page_config(layout="wide")

# title
# current date = datetime.strptime('2023-04-18, %Y-%m-%d %H:%M:%S')
current_date = pd.to_datetime(datetime.now()).floor('H')
st.title(f'Taxi demand prediction ')
st.header(f'{current_date}')

progress_bar = st.sidebar.header('o Working Progress')
progress_bar = st.sidebar.progress(0)
N_STEPS = 7

# load coordinates

def load_shape_date_file():
    
    # download file
    URL = 'https://d37ci6vzurychx.cloudfront.net/misc/taxi_zones.zip'
    response = requests.get(URL)
    path = DATA_DIR / f'taxi_zones.zip'
    if response.status_code == 200:
        open(path,'wb').write(response.content)
    else:
        raise Exception(f'{URL} is not avaliable')
    
    # unzip file
    with zipfile.ZipFile(path,'r') as zip_ref:
        zip_ref.extractall(DATA_DIR / 'taxi_zones')
        
    # load and return shape file
    return gpd.read_file(DATA_DIR / 'taxi_zones/taxi_zones.shp').to_crs('epsg:4326')


with st.spinner(text='Downloading shape file to plot taxi zones'):
    geo_df = load_shape_date_file()
    st.sidebar.write('✅ Shape file was downloaded')
    progress_bar.progress(1/N_STEPS)
    
with st.spinner(text="Fetching batch of features used in the last run"):
    features = load_batch_of_features_from_store(current_date)
    st.sidebar.write('✅ Inference features fetched from the store')
    progress_bar.progress(2/N_STEPS)
    print(f'{features}')