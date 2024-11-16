from typing import Union
from pipeline import getting_data
from dict_schema_test import dict_schema
from fastapi import FastAPI
import joblib
import numpy as np
import subprocess
from pydantic import BaseModel
import os

###################Subprocess to execute test
try:
    print('-----------------Starting subprocess-----------------------')
    root = os.getcwd()
    path = os.path.join(root, 'api')
    os.chdir(path)
    subprocess.run(['pytest', 'pipeline.py'])
    print('-----------------Subprocess for testing ran correctly------')
except:
    print('----Something was wrong with the test, check out the paths!')


###############Getting model and X_train to create new model
X_train, model_reg = getting_data()
joblib.dump(model_reg, 'regression_model.joblib')
model = joblib.load('regression_model.joblib')

# Definir el esquema para la entrada de datos (esperamos 58 características)
print('----------------STARTING API PROCESS----------------')
app = FastAPI()
class PredictionRequest(BaseModel):
    feature_1: float
    feature_2: float
    feature_3: float
    feature_4: float
    feature_5: float
    feature_6: float
    feature_7: float
    feature_8: float
    feature_9: float
    feature_10: float
    feature_11: float
    feature_12: float
    feature_13: float
    feature_14: float
    feature_15: float
    feature_16: float
    feature_17: float
    feature_18: float
    feature_19: float
    feature_20: float
    feature_21: float
    feature_22: float
    feature_23: float
    feature_24: float
    feature_25: float
    feature_26: float
    feature_27: float
    feature_28: float
    feature_29: float
    feature_30: float
    feature_31: float
    feature_32: float
    feature_33: float
    feature_34: float
    feature_35: float
    feature_36: float
    feature_37: float
    feature_38: float
    feature_39: float
    feature_40: float
    feature_41: float
    feature_42: float
    feature_43: float
    feature_44: float
    feature_45: float
    feature_46: float
    feature_47: float
    feature_48: float
    feature_49: float
    feature_50: float
    feature_51: float
    feature_52: float
    feature_53: float
    feature_54: float
    feature_55: float
    feature_56: float
    feature_57: float
    feature_58: float

# Endpoint para realizar la predicción
@app.post("/predict")
def predict(request: PredictionRequest):
    # Convertir el cuerpo de la solicitud a un array de numpy (reshape para asegurar que sea una sola fila)
    features = np.array([[
        request.feature_1, request.feature_2, request.feature_3, request.feature_4, request.feature_5,
        request.feature_6, request.feature_7, request.feature_8, request.feature_9, request.feature_10,
        request.feature_11, request.feature_12, request.feature_13, request.feature_14, request.feature_15,
        request.feature_16, request.feature_17, request.feature_18, request.feature_19, request.feature_20,
        request.feature_21, request.feature_22, request.feature_23, request.feature_24, request.feature_25,
        request.feature_26, request.feature_27, request.feature_28, request.feature_29, request.feature_30,
        request.feature_31, request.feature_32, request.feature_33, request.feature_34, request.feature_35,
        request.feature_36, request.feature_37, request.feature_38, request.feature_39, request.feature_40,
        request.feature_41, request.feature_42, request.feature_43, request.feature_44, request.feature_45,
        request.feature_46, request.feature_47, request.feature_48, request.feature_49, request.feature_50,
        request.feature_51, request.feature_52, request.feature_53, request.feature_54, request.feature_55,
        request.feature_56, request.feature_57, request.feature_58
    ]])

    # Realizar la predicción
    prediction = model.predict(features)

    # Devolver la predicción en un formato JSON
    return {"prediction": prediction[0]}
