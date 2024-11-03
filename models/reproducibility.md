# Reproducibilidad con modelos ML mediante MLflow: Caso Mashable

## Contenido

- [Introducción](#introducción)
- [Estructura](#estructura)
- [Versionamiento de Datos](#versionamiento-de-datos)
- [Prerrequisitos](#prerrequisitos)
- [Seguimiento de experimentos](#seguimiento-de-experimentos)
- [Paso a paso](#paso-a-paso)
- [Tips](#tips)


---

## Introducción

Este proyecto tiene como objetivo predecir la popularidad (medida por las veces que se comparte en las redes sociales) de los artículos de noticias publicados por Mashable durante dos años, utilizando una variedad de características de los artículos. Garantizar la reproducibilidad de los experimentos permite que otros validen los resultados, los utilicen como base y mantengan la coherencia en diferentes entornos informáticos.

## Estructura

Una estructura de archivos consistente facilita la reproducibilidad y mejora la organización del proyecto:

- `data/`: contiene el conjunto de datos (datos de entrada, datos procesados ​​y cualquier división para entrenamiento/prueba).

- `src/`: secuencias de comandos para preprocesamiento de datos, ingeniería de características, entrenamiento de modelos y evaluación.

- `models/`: modelos guardados y puntos de control.

- `results/`: registros y métricas para cada experimento.

- `configs/`: archivos de configuración que especifican hiperparámetros y otras configuraciones del experimento.

- `notebooks/`: cuadernos Jupyter para análisis de datos exploratorios (EDA) y creación de prototipos.

- `Reproducibility.md`: esta guía de reproducibilidad.

## Versionamiento de Datos

**Conjunto de datos**

El conjunto de datos utilizado es el conjunto de datos de popularidad de noticias en línea Mashable, donado el 30/5/2015.

**Control de versiones**

- Herramienta: Control de versiones de datos (DVC).

- Configuración: Seguimiento de cambios en datos sin procesar, divisiones y datos procesados ​​para garantizar la coherencia entre diferentes ejecuciones.

- Uso: dvc `add data/` online_news_popularity.csv para realizar un seguimiento de los datos sin procesar y `dvc push` para enviarlos a un repositorio remoto para que el equipo pueda acceder a ellos.


## Prerrequisitos

Para garantizar entornos consistentes, las dependencias se administran en un archivo de "requisitos (requirements.txt)" o en un archivo de entorno de Conda.

- **Entorno de Python**: usamos requirements.txt o environment.yml:

```
# Create a virtual environment

python -m venv venv
source venv/bin/activate  # or .\venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt
```

Algunas de las librerías importantes:

- Scikit-Learn
- Pandas
- NumPy
- Matplotlib / Seaborn
- MLflow 


## Seguimiento de experimentos

El seguimiento de experimentos es esencial para registrar parámetros, métricas y resultados.

**Herramienta**: MLflow

**Uso**:

**Seguimiento**: cada ejecución de entrenamiento del modelo registra hiperparámetros, métricas y versiones del modelo.

**Registro**: realizar un seguimiento de métricas como la precisión, el error absoluto medio y el R cuadrado para la evaluación del modelo.

**Sugerencia de reproducibilidad**: asignar una identificación única a cada ejecución y almacenar las configuraciones en `configs/` para volver a ejecutar los experimentos.

**Control de versiones de modelos y datos**

Para garantizar que todos los modelos y datos se puedan revisar en análisis futuros debermos contar con:

**Control de versiones de modelos**: guardando los modelos entrenados en `models/` con nombres de archivo descriptivos (`model_v1.0.pkl`) y almacenar metadatos (como la fecha de entrenamiento y la configuración utilizada).

**Control de versiones de datos**: realizar un seguimiento de los cambios en el preprocesamiento de datos (las canalizaciones de DVC ayudan a administrar las etapas).

**Cómo garantizar la coherencia de la aleatoriedad**

Controlando la aleatoriedad para lograr reproducibilidad en las divisiones de entrenamiento y prueba:

```bash
import numpy as np
import random
import tensorflow as tf

random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)  # For TensorFlow, if applicable
```

## Paso a paso


1. **Crear un entorno virtual** (opcional pero recomendado, en lugar de entornos virtuales para Anaconda):

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

2. **Instalar paquetes necesarios**:

   ```bash
   pip install mlflow scikit-learn pandas numpy matplotlib
   ```

### 1. Configuración del entorno

- **Instalar MLflow** (si aún no está instalado):

  ```bash
  pip install mlflow
  ```

- **Importar bibliotecas necesarias**:

  ```python
  pip install -r requirements.txt
  ```

### 2. Cargar y explorar el conjunto de datos

- **Cargar los datos**:

  ```python
  data = load_wine()
  X = pd.DataFrame(data.data, columns=data.feature_names)
  y = pd.Series(data.target)
  ```

### 3. Inicializar el seguimiento de MLflow

- **Establecer el experimento**:

  ```python
  mlflow.set_experiment("Wine_Classification")
  ```

### 4. Definir la función de entrenamiento y registro

Los *hiperparámetros* son un componente clave para la reproducibilidad en el aprendizaje automático, por este motivo estos valores deben almacenarse durante el entrenamiento de cada modelo.

```python
def train_and_log_model(model, model_name, X_train, X_test, y_train, y_test, params):
    with mlflow.start_run(run_name=model_name):
        # Train the model
        model.fit(X_train, y_train)
        # Make predictions
        y_pred = model.predict(X_test)
        # Calculate metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='weighted')
        rec = recall_score(y_test, y_pred, average='weighted')
        # Log parameters and metrics
        mlflow.log_params(params)
        mlflow.log_metrics({"accuracy": acc, "precision": prec, "recall": rec})
        # Log the model
        mlflow.sklearn.log_model(model, artifact_path="models")
```

### 5. Ejecutar experimentos con diferentes modelos

#### a. Decision Tree Classifier

- **Establecer parámetros**:

  ```python
  params_dt = {"max_depth": 5, "criterion": "entropy", "random_state": 42}
  model_dt = DecisionTreeClassifier(**params_dt)
  ```

- **Entrenamiento y registro**:

  ```python
  train_and_log_model(
      model=model_dt,
      model_name="Decision_Tree",
      X_train=X_train,
      X_test=X_test,
      y_train=y_train,
      y_test=y_test,
      params=params_dt
  )
  ```

#### b. Random Forest Classifier

- **Establecer parámetros**:

  ```python
  params_rf = {"n_estimators": 100, "max_depth": 5, "random_state": 42}
  model_rf = RandomForestClassifier(**params_rf)
  ```

- **Entrenamiento y registro**:

  ```python
  train_and_log_model(
      model=model_rf,
      model_name="Random_Forest",
      X_train=X_train,
      X_test=X_test,
      y_train=y_train,
      y_test=y_test,
      params=params_rf
  )
  ```

### 6. Comparar resultados de experimentos

- **Iniciar la interfaz de usuario de MLflow**:

  En tu terminal, ejecuta:

  ```bash
  mlflow ui
  ```

- **Acceder la interfaz UI**:

 Abra su navegador web y navegue hasta `http://localhost:5000`.

- **Comparar ejecuciones**:

    - Seleccionando nuestro experimiento (Mashable).
    - Comparar las métricas y los parámetros de diferentes ejecuciones.
    - Utilizar las funciones de clasificación y comparación para analizar el rendimiento.
    - Identificar el modelo con mejor rendimiento en función de la precisión, exactitud y recuperación.

### 7. Reproducir un experimento

- **Seleccionamos una ejecución**:

  Teniendo en cuenta el `run_id` del modelo con mejor rendimiento de la interfaz de usuario de MLflow.

- **Cargamos el modelo**:

  ```python
  run_id = "<your-selected-run-id>"
  logged_model = f'runs:/{run_id}/models'

  # Load model as a sklearn model.
  loaded_model = mlflow.sklearn.load_model(logged_model)
  ```

- **Reevaluamos el modelo**:

  ```python
  y_pred_loaded = loaded_model.predict(X_test_scaled)
  acc_loaded = accuracy_score(y_test, y_pred_loaded)
  print(f"Reproduced Accuracy: {acc_loaded}")
  ```

- **Verificar reproducibilidad**:

Asegúrese de que la precisión reproducida coincida con la precisión registrada en MLflow.

### 8. Documentar hallazgos

- **Creamos un informe o cuaderno**:

  - Resumiendo el rendimiento de diferentes modelos.

  - Analizando la importancia de la reproducibilidad en el aprendizaje automático.
  
## Tips

- **Paquetes instalados y sus versiones exactas**:

Ejecutando `pip freeze > requirements.txt` en el directorio del proyecto, se generará un archivo `requirements.txt` con todos los paquetes instalados y sus versiones exactas. 

Este archivo es esencial para garantizar que todos los que trabajan en el proyecto instalen las mismas dependencias y mantengan la coherencia en los diferentes entornos.


- **Estados aleatorios consistentes**:

Usando `random_state=42` (o cualquier número fijo) en sus modelos y en la división de datos para garantizar la coherencia en todas las ejecuciones.

- **Registro de métricas personalizadas**:

Si calcula métricas adicionales (por ejemplo, puntuación F1, ROC-AUC), asegúrese de registrarlas en MLflow mediante `mlflow.log_metric()`.

- **Gestión de artefactos**:

Puede registrar artefactos como gráficos, el objeto escalador o matrices de confusión mediante `mlflow.log_artifact()`.

  ```python
  # Example of logging an artifact
  import matplotlib.pyplot as plt
  from sklearn.metrics import plot_confusion_matrix

  fig, ax = plt.subplots()
  plot_confusion_matrix(model, X_test_scaled, y_test, ax=ax)
  plt.savefig("confusion_matrix.png")
  mlflow.log_artifact("confusion_matrix.png")
  ```
