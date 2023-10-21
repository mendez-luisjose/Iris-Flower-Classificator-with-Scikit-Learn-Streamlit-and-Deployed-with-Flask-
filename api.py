#API made with Flask

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
from flask import Flask, request, jsonify
import pickle

MODEL_PATH = f'./model/iris_model.pkl'
SCALER_PATH = f'./model/scaler.pkl'

#Function to load the Model and the Scaler
def load_pkl(fname):
    with open(fname, 'rb') as f:
        obj = pickle.load(f)
    return obj

model = load_pkl(MODEL_PATH)
scaler = load_pkl(SCALER_PATH)

#Prediction Function
def predict(input_array):
    input_array_scaled = scaler.transform(input_array)
    result = model.predict(input_array_scaled)
    prob_setosa = round(model.predict_proba(input_array_scaled)[0][0], 2)
    prob_versicolor = round(model.predict_proba(input_array_scaled)[0][1], 2)
    prob_virginica = round(model.predict_proba(input_array_scaled)[0][2], 2)

    print(result[0], prob_setosa, prob_versicolor, prob_virginica)
    
    results = {
        "result": int(result[0]),
        "prob_setosa" : float(prob_setosa),
        "prob_versicolor": float(prob_versicolor),
        "prob_virginica": float(prob_virginica)
    }

    return results

app = Flask(__name__)

#Flask App
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        data = request.get_json()
        np_array = np.array(data['array'])
        try:
            results = predict(np_array)
            print('Success!', 200)
            return jsonify({"Results": results})

    
        except Exception as e:
            return jsonify({"error": str(e)})

    return "OK"


if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=False)
