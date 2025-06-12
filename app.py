import numpy as np
import pandas as pd
from flask import Flask, request, render_template
import joblib
from keras.models import load_model

app = Flask(__name__)

# Load models and scaler
autoencoder = load_model('models/autoencoder_model.keras', compile=False)

xgb_model = joblib.load('models/xgboost_model.json')
scaler = joblib.load('models/scaler.pkl')

# Feature list
features = [f"V{i}" for i in range(1, 29)] + ["Amount"]

@app.route('/')
def home():
    features = [f"V{i}" for i in range(1, 29)] + ["Amount"]
    return render_template('index.html', features=features)


@app.route('/predict', methods=['POST'])
@app.route('/predict', methods=['POST'])
def predict():
    try:
        features = [float(request.form[f'V{i}']) for i in range(1, 29)]
        amount = float(request.form['Amount'])
        input_data = np.array(features + [amount]).reshape(1, -1)
        
        # Prediction using autoencoder and XGBoost (example):
        reconstruction = autoencoder.predict(input_data)
        mse = np.mean(np.power(input_data - reconstruction, 2), axis=1)
        threshold = 0.001  # use your actual threshold
        prediction = "Fraud" if mse > threshold else "Not Fraud"

        return render_template('index.html', prediction=prediction)
    except Exception as e:
        return f"Error: {e}", 400


if __name__ == '__main__':
    app.run(debug=True)