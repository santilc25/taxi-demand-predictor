import os
from dotenv import load_dotenv
from src.paths import PARENT_DIR

# load ley-value pairs from .env file locate d in the parent directory
load_dotenv(PARENT_DIR / '.env')

HOPSWORKS_PROJECT_NAME = 'taxi_demand_project'

try:
    HOPSWORKS_API_KEY = os.environ['HOPSWORKS_API_KEY']
except:
    raise Exception('Create an .env file on the project root with the HOPSWORKS_API_KEY')

FEATURE_GROUP_NAME = 'time_series_hourly_feature_group'
FEATURE_GROUP_VERSION = 1
FEATURE_VIEW_NAME = 'time_series_hourly_feature_view'
FEATURE_VIEW_VERSION = 1
MODEL_NAME = "taxi_demand_predictor_next_hour"
MODEL_VERSION = 1

N_FEATURES = 24 * 28