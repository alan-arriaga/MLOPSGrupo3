import pandas as pd
import sys
import random
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import joblib

RANDOM_SEED = 42

def train_model(X_train_path, y_train_path):
    X_train = pd.read_csv(X_train_path)
    y_train = pd.read_csv(y_train_path)
    
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED) 

    model = RandomForestRegressor(n_estimators=160, max_depth=16, max_features='sqrt', min_samples_leaf=2, min_samples_split=2, random_state=RANDOM_SEED)
    model.fit(X_train, y_train.values.ravel())
    return model

if __name__ == '__main__':
    X_train_path = sys.argv[1]
    y_train_path = sys.argv[2]
    model_path = sys.argv[3]

    model = train_model(X_train_path, y_train_path)
    joblib.dump(model, model_path)