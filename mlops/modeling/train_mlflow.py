# train.py
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import joblib
import mlflow

def train_model(train_file, model_output):
    # Cargar datos
    train_data = pd.read_csv(train_file)
    X = train_data.drop(columns=['shares'])
    y = train_data['shares']

    # Configurar experimento en MLflow
    mlflow.set_experiment("Popularidad de Artículos")

    with mlflow.start_run() as run:
        # Configurar hiperparámetros
        model = LinearRegression()
        mlflow.log_param("model_type", "LinearRegression")

        # Entrenar el modelo
        model.fit(X, y)

        # Predecir y calcular métricas
        predictions = model.predict(X)
        mse = mean_squared_error(y, predictions)
        mlflow.log_metric("mse", mse)

        # Guardar y registrar el modelo entrenado
        joblib.dump(model, model_output)
        mlflow.log_artifact(model_output)

        print(f"Modelo guardado y registrado en MLflow. Run ID: {run.info.run_id}")

# Ejecutar el entrenamiento
if __name__ == "__main__":
    train_model("data/processed/train_processed.csv", "models/linear_regression.joblib")
