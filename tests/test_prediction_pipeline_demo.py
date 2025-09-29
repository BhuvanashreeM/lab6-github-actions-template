import pytest
import pandas as pd
import numpy as np
from prediction_pipeline_demo import data_preparation, data_split, train_model, eval_model

@pytest.fixture
def housing_data_sample():
    # Tiny deterministic sample
    return pd.DataFrame(
        {
            "price": [13_300_000, 12_250_000],
            "area": [7420, 8960],
            "bedrooms": [4, 4],
            "bathrooms": [2, 4],
            "stories": [3, 4],
            "mainroad": ["yes", "yes"],
            "guestroom": ["no", "no"],
            "basement": ["no", "no"],
            "hotwaterheating": ["no", "no"],
            "airconditioning": ["yes", "yes"],
            "parking": [2, 3],
            "prefarea": ["yes", "no"],
            "furnishingstatus": ["furnished", "unfurnished"],
        }
    )

def test_data_preparation(housing_data_sample):
    feature_df, target_series = data_preparation(housing_data_sample)
    # Target and datapoints have same length
    assert feature_df.shape[0] == len(target_series)
    # Features are numeric after one-hot
    assert feature_df.shape[1] == feature_df.select_dtypes(include=(np.number, np.bool_)).shape[1]

@pytest.fixture
def feature_target_sample(housing_data_sample):
    feature_df, target_series = data_preparation(housing_data_sample)
    return (feature_df, target_series)

def test_data_split_returns_four_parts(feature_target_sample):
    parts = data_split(*feature_target_sample)
    # TODO(1): assert the tuple has exactly 4 elements

def test_end_to_end_train_and_eval(feature_target_sample):
    X_train, X_test, y_train, y_test = data_split(*feature_target_sample)
    model = train_model(X_train, y_train)
    score = eval_model(X_test, y_test, model)
    # TODO(2): Ensure eval_model produces a float score and that it is finite
