from flask import Flask, request, render_template
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.metrics import MeanSquaredError
from sklearn.preprocessing import StandardScaler

# Disable GPU if not available (for Render compatibility)
tf.config.set_visible_devices([], 'GPU')

# Load model with custom objects (like mse metric)
try:
    model = load_model('model.h5', custom_objects={'mse': MeanSquaredError()})
    print("Model loaded successfully")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# Load the scaler used for training
scaler = StandardScaler()

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        print("Model not loaded properly.")
        return "Model not loaded properly. Please check the server logs."

    try:
        # Get form data from user
        destination = request.form['destination']
        travel_mode = request.form['travel_mode']
        people = int(request.form['people'])
        duration = int(request.form['duration'])

        # Log the received input
        print(f"Received input: destination={destination}, travel_mode={travel_mode}, people={people}, duration={duration}")

        # Preprocess input data
        input_data = preprocess(destination, travel_mode, people, duration)
        print(f"Preprocessed input: {input_data}")

        # Scale the input data (this was done during model training)
        input_data_scaled = scaler.transform([input_data])
        print(f"Scaled input: {input_data_scaled}")

        # Predict the budget
        prediction = model.predict(input_data_scaled)
        print(f"Prediction result: {prediction}")

        # Render the result
        return render_template('index.html', budget=round(prediction[0][0], 2))

    except Exception as e:
        print(f"Prediction error: {e}")
        return render_template('index.html', budget="Error: Invalid input or model")

def preprocess(destination, travel_mode, people, duration):
    # Map categorical variables to numeric values (these are the same mappings used during training)
    dest_map = {"Chintpurni": 0, "Agra": 1, "Srinagar": 2, "Manali": 3, "Ooty": 4, "Rishikesh": 5}
    mode_map = {"Motorcycle": 0, "Flight": 1, "Bus": 2, "Train": 3, "Personal Car": 4}

    # Return a list of processed features
    return [
        dest_map.get(destination, 0),  # Default to 0 if destination is not recognized
        mode_map.get(travel_mode, 0),  # Default to 0 if travel mode is not recognized
        people,  # Number of people
        duration  # Duration in days
    ]

if __name__ == '__main__':
    # Ensure the correct port for deployment on Render
    port = int(os.environ.get("PORT", 5000))  # Render uses PORT environment variable
    app.run(host='0.0.0.0', port=port)
