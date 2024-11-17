# Importing usual libraries
import os
import sys
import joblib
from ucimlrepo import fetch_ucirepo
from sklearn.preprocessing import power_transform
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
import yaml
import pandas as pd
import numpy as np
import pytest

#Dependencies for loading, refactoring, classes, preprocessing and testing
from classes import DataExplorer, OnlineShares
from dict_schema_test import dict_schema

#Initial process

def getting_data():
    df = fetch_ucirepo(id=332)  
    print('Loading data--------')
    X = df.data.features 
    y = df.data.targets  
    data = pd.concat([X, y], axis = 1) 
    model = OnlineShares()
    model.load_data(data)
    X_train_scaled, X_test_scaled, y_train, y_test = model.preprocess_data()
    print('Preprocessing done----------')
    model_reg = model.train_model(X_train_scaled, y_train)
    print('Training------------')

    return X_test_scaled, model_reg

@pytest.fixture
def pipeline():
    df = fetch_ucirepo(id=332)  # fetch dataset
    X = df.data.features # data (as pandas dataframes)
    y = df.data.targets  # data (as pandas dataframes)
    data = pd.concat([X, y], axis = 1) # Concatenamos ambos datasets (feature y target)
    print('---------------------------------LOADING DATA SUCESSFUL--------------')
    model = OnlineShares()
    model.load_data(data)
    X_train_scaled, X_test_scaled, y_train, y_test = model.preprocess_data()
    print('---------------------------------PREPROCESSING DATA SUCESSFUL--------------')

    return model

def test_input_data_ranges(pipeline):
    # Getting the maximum and minimum values for each column
    max_values = pipeline.X_train.max()
    min_values = pipeline.X_train.min()
    
    # Ensuring that the maximum and minimum values fall into the expected range
    for feature in list(pipeline.X_train.columns):
        assert max_values[feature] <= dict_schema[feature]['range']['max']
        print('------------MAX VALUE TEST PASSED--------------')
        assert min_values[feature] >= dict_schema[feature]['range']['min']
        print('------------MIN VALUE TEST PASSED--------------')


def test_input_data_types(pipeline):
    # Getting the data types from each column
    data_types = pipeline.X_train.dtypes
    
    # Testing compatibility between data types
    for feature in list(pipeline.X_train.columns):
        assert data_types[feature] == dict_schema[feature]['dtype']
        print('------------ VALUE TYPES TEST PASSED--------------')
        print('---------------------------------TESTING SUCESSFUL--------------')






    
