from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model
with open('knn_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Home route
@app.route('/')
def home():
    return "Stress Prediction API is Running!"

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)

        # Example expected input (JSON):
        # {
        #   "snoring_rate": 96.288,
        #   "respiration_rate": 26.288,
        #   "body_temperature": 85.36,
        #   "limb_movement": 17.144,
        #   "blood_oxygen": 82.432,
        #   "eye_movement": 100.36,
        #   "sleeping_hours": 0,
        #   "heart_rate": 75.72
        # }

        required_features = [
            "snoring_rate",
            "respiration_rate",
            "body_temperature",
            "limb_movement",
            "blood_oxygen",
            "eye_movement",
            "sleeping_hours",
            "heart_rate"
        ]

        input_data = [data[feature] for feature in required_features]
        prediction = model.predict([input_data])

        return jsonify({"prediction": int(prediction[0])})
    
    except Exception as e:
        return jsonify({"error": str(e)})

# Run app on Replit (port 3000)
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3000)
