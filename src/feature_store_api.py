from typing import Optional
import hsfs
import hopsworks

import src.config as config

def get_feature_store() -> hsfs.feature_store.FeatureStore:
    """Connects to Hopsworks and returns a pointer to the feature store

    Returns:
        hsfs.feature_store.FeatureStore: pointer to the feature store
    """
    project = hopsworks.login(
        project=config.HOPSWORKS_PROJECT_NAME,
        api_key_value=config.HOPSWORKS_API_KEY
    )
    return project.get_feature_store()


