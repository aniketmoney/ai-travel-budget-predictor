from flask import Flask, request, render_template
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.metrics import MeanSquaredError
import joblib  # ✅ For loading the trained scaler

# Disable GPU for Render compatibility
tf.config.set_visible_devices([], 'GPU')

# Load the trained model
try:
    model = load_model('model.h5', custom_objects={'mse': MeanSquaredError()})
    print("✅ Model loaded successfully")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None

# Load the trained scaler (use joblib to load the one saved from training)
try:
    scaler = joblib.load('scaler.pkl')  # ✅ Make sure this file is in the same directory
    print("✅ Scaler loaded successfully")
except Exception as e:
    print(f"❌ Error loading scaler: {e}")
    scaler = None

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None or scaler is None:
        return "Model or scaler not loaded properly. Check server logs."

    try:
        # Get user input
        destination = request.form['destination']
        travel_mode = request.form['travel_mode']
        people = int(request.form['people'])
        duration = int(request.form['duration'])

        # New input fields
        accommodation_cost = float(request.form['accommodation_cost'] or 0)
        food_cost = float(request.form['food_cost'] or 0)
        travel_cost = float(request.form['travel_cost'] or 0)

        print(f"🟡 Input: {destination}, {travel_mode}, People: {people}, Duration: {duration}, "
              f"Accommodation Cost: {accommodation_cost}, Food Cost: {food_cost}, Travel Cost: {travel_cost}")

        # Preprocess and scale
        input_data = preprocess(destination, travel_mode, people, duration, accommodation_cost, food_cost, travel_cost)
        print(f"🟢 Preprocessed input: {input_data}")

        input_scaled = scaler.transform([input_data])  # ✅ Use trained scaler
        print(f"🔵 Scaled input: {input_scaled}")

        # Predict
        prediction = model.predict(input_scaled)
        print(f"🔮 Predicted Budget: {prediction[0][0]}")

        return render_template('index.html', budget=round(prediction[0][0], 2))

    except Exception as e:
        print(f"❌ Prediction error: {e}")
        return render_template('index.html', budget="Error: Check your input or model setup")

def preprocess(destination, travel_mode, people, duration, accommodation_cost, food_cost, travel_cost):
    # Mapping for categorical variables
    dest_map = {"Chintpurni": 0, "Agra": 1, "Srinagar": 2, "Manali": 3, "Ooty": 4, "Rishikesh": 5}
    mode_map = {"Motorcycle": 0, "Flight": 1, "Bus": 2, "Train": 3, "Personal Car": 4}
    
    # Return processed features: 7 total features now
    return [
        dest_map.get(destination, 0),         # Destination: mapped to an integer
        mode_map.get(travel_mode, 0),        # Travel mode: mapped to an integer
        people,                             # Number of people
        duration,                           # Duration of travel in days
        accommodation_cost,                 # Accommodation cost
        food_cost,                          # Food cost
        travel_cost                         # Travel cost
    ]

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
