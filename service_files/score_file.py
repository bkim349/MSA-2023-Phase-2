import json
import joblib
import numpy as np
import os

import numpy as np
import requests, json, os
from sklearn.linear_model import LinearRegression

def init():
    global model
    model =  LinearRegression()
    model.load_model(os.path.join(os.getenv("AZUREML_MODEL_DIR"), "lr_model.pkl"))
    model = joblib.load(model_path)

def run(request):
    data = json.loads(request)
    data = np.array(data["data"])
    response = model.predict(data)
    return response

