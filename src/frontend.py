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
current_date = pd.to_datetime(datetime.utcnow()).floor('H')
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
    
with st.spinner(text='Loading ML model from the registry'):
    model = load_model_from_registry()
    st.sidebar.write('✅ ML model was load from the registry')
    progress_bar.progress(3/N_STEPS)
      
with st.spinner(text='Computing model predictions'):
    results = get_model_predictions(model,features)
    st.sidebar.write('✅ Model predictions arrived')
    progress_bar.progress(4/N_STEPS)
    
with st.spinner(text='Preparing data to plot'):
    
    def pseudocolor(val, minval, maxval, startcolor, stopcolor):
        """
        Convert value in the range minval...maxval to a color in the range
        startcolor to stopcolor. The colors passed and the the one returned are
        composed of a sequence of N component values.
        Credits to https://stackoverflow.com/a/10907855
        """
        f = float(val-minval) / (maxval-minval)
        return tuple(f*(b-a)+a for (a, b) in zip(startcolor, stopcolor))
        
    df = pd.merge(geo_df, results,right_on='pickup_location_id',left_on='LocationID',how='inner')
    
    BLACK, GREEN = (0, 0, 0), (0, 255, 0)
    print(f"COLUMNS {list(df.columns)}")
    df['color_scaling'] = df['predicted_demand']
    max_pred, min_pred = df['color_scaling'].max(), df['color_scaling'].min()
    df['fill_color'] = df['color_scaling'].apply(lambda x: pseudocolor(x, min_pred, max_pred, BLACK, GREEN))
    progress_bar.progress(5/N_STEPS)
    
with st.spinner(text="Generating NYC Map"):

    INITIAL_VIEW_STATE = pdk.ViewState(
        latitude=40.7831,
        longitude=-73.9712,
        zoom=11,
        max_zoom=16,
        pitch=45,
        bearing=0
    )

    geojson = pdk.Layer(
        "GeoJsonLayer",
        df,
        opacity=0.25,
        stroked=False,
        filled=True,
        extruded=False,
        wireframe=True,
        get_elevation=10,
        get_fill_color="fill_color",
        get_line_color=[255, 255, 255],
        auto_highlight=True,
        pickable=True,
    )

    tooltip = {"html": "<b>Zone:</b> [{LocationID}]{zone} <br /> <b>Predicted rides:</b> {predicted_demand}"}

    r = pdk.Deck(
        layers=[geojson],
        initial_view_state=INITIAL_VIEW_STATE,
        tooltip=tooltip
    )

    st.pydeck_chart(r)
    progress_bar.progress(6/N_STEPS)
    
    
with st.spinner(text="Plotting time-serie data"):
    
    row_indices = np.argsort(results['predicted_demand'].values)[::-1]
    n_to_plot = 10
    
    # plot each time-series with prediction
    for row_id in row_indices[:n_to_plot]:
        fig = plot_one_sample(
            features=features,
            targets=results['predicted_demand'],
            example_id=row_id,
            predictions=pd.Series(results['predicted_demand'])
        )
        st.plotly_chart(fig,theme='streamlit',use_container_width=True,width=1000)

    progress_bar.progress(7/N_STEPS)