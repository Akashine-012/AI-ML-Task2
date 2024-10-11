from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np

app = Flask(__name__)
CORS(app)  # This will allow cross-origin requests

# Load the model and scaler
model = joblib.load('random_forest_model.pkl')
scaler = joblib.load('scaler.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    # Extract features from the incoming JSON
    features = [
        data['MarketingSpend'],
        data['CompetitorPrice'],
        data['AveragePrice'],
        data['EconomicIndex'],
        data['DemographicFactor'],
        data['PrescriptionRate'],
        data['Season_Spring'],
        data['Season_Summer'],
        data['Season_Winter'],
    ]

    # Scale the features
    features_scaled = scaler.transform([features])

    # Make prediction
    prediction = model.predict(features_scaled)

    return jsonify({'prediction': prediction[0]})

if __name__ == '__main__':
    app.run(debug=False)
