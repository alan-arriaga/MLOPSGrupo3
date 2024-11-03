import pandas as pd
import numpy as np
import sys
import random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PowerTransformer

RANDOM_SEED = 42 

def preprocess_data(data_path):
    random.seed = (RANDOM_SEED)
    np.random.seed = (RANDOM_SEED)

    data = pd.read_csv(data_path)
    X = data.drop('shares', axis=1)
    y = data['shares']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_SEEDcd)

    # Transformaciones a columnas en X_train
    X_train['n_tokens_content'] = np.log1p(X_train['n_tokens_content'])
    X_train['num_hrefs'] = np.log1p(X_train['num_hrefs'])
    X_train['kw_avg_max'] = np.sqrt(X_train['kw_avg_max'])
    X_train['global_subjectivity'] = np.log1p(X_train['global_subjectivity'])
    X_train['global_rate_positive_words'] = np.sqrt(X_train['global_rate_positive_words'])

    # Columnas con Yeo-Johnson en X_train
    pt = PowerTransformer(method='yeo-johnson')

    X_train[['global_rate_negative_words', 'rate_positive_words', 'rate_negative_words']] = pt.fit_transform(
    X_train[['global_rate_negative_words', 'rate_positive_words', 'rate_negative_words']]
    )
    X_train['avg_positive_polarity'] = np.log1p(X_train['avg_positive_polarity'])

    # Transformaciones en X_test
    X_test['n_tokens_content'] = np.log1p(X_test['n_tokens_content'])
    X_test['num_hrefs'] = np.log1p(X_test['num_hrefs'])
    X_test['kw_avg_max'] = np.sqrt(X_test['kw_avg_max'])
    X_test['global_subjectivity'] = np.log1p(X_test['global_subjectivity'])
    X_test['global_rate_positive_words'] = np.sqrt(X_test['global_rate_positive_words'])

    # Aplicar Yeo-Johnson en X_test
    X_test[['global_rate_negative_words', 'rate_positive_words', 'rate_negative_words']] = pt.transform(
    X_test[['global_rate_negative_words', 'rate_positive_words', 'rate_negative_words']]
    )
    X_test['avg_positive_polarity'] = np.log1p(X_test['avg_positive_polarity'])

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test

if __name__ == '__main__':
    data_path = sys.argv[1]
    output_train_features = sys.argv[2]
    output_test_features = sys.argv[3]
    output_train_target = sys.argv[4]
    output_test_target = sys.argv[5]

    X_train, X_test, y_train, y_test = preprocess_data(data_path)
    pd.DataFrame(X_train).to_csv(output_train_features, index=False)
    pd.DataFrame(X_test).to_csv(output_test_features, index=False)
    pd.DataFrame(y_train).to_csv(output_train_target, index=False)
    pd.DataFrame(y_test).to_csv(output_test_target, index=False)