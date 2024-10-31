#Importing dependencies
import mlflow
from mlflow.models import infer_signature
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


################################################################################
# Load the Online News dataset from our processed folder
################################################################################

df = pd.read_csv(r'C:/Users/balde/Desktop/MAESTRIA MNA/Contribs/MLOPSGrupo3/data/processed/online_news_popularity_clean.csv')
X = df.drop('shares', axis=1)
y = df['shares']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Define the model hyperparameters
params = {
    "n_estimators": 50,
    "max_depth": 10,
    "max_features": "sqrt",
    "min_samples_leaf": 6,
    'min_samples_split':6,
    'random_state':42
}

# Train the model
reg = RandomForestRegressor(**params)
reg.fit(X_train, y_train)

# Predict on the test set
y_pred = reg.predict(X_test)

# Calculate metrics
score = mean_squared_error(y_test, y_pred)


################################################################################
# Set our tracking server uri for logging
################################################################################

mlflow.set_tracking_uri(uri="http://127.0.0.1:5000")

# Create a new MLflow Experiment
mlflow.set_experiment("Online News Regression Exps")

# Start an MLflow run
with mlflow.start_run():
    # Log the hyperparameters
    mlflow.log_params(params)

    # Log the loss metric
    mlflow.log_metric("score", score)

    # Set a tag that we can use to remind ourselves what this run was for
    mlflow.set_tag("Training Info", "Basic Random Forest Regressor model for online news dataset")

    # Infer the model signature
    signature = infer_signature(X_train, reg.predict(X_train))

    # Log the model
    model_info = mlflow.sklearn.log_model(
        sk_model=reg,
        artifact_path="online_news_shares",
        signature=signature,
        input_example=X_train,
        registered_model_name="model-v1",
    )

################################################################################
# Load the model back for predictions as a generic Python Function model
################################################################################



