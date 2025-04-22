from flask import Flask, request, render_template
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.metrics import MeanSquaredError

# Disable GPU if not available (for Render compatibility)
tf.config.set_visible_devices([], 'GPU')

# Try loading the model with custom_objects handling
try:
    model = load_model('model.h5', custom_objects={'mse': MeanSquaredError()})
except Exception as e:
    print(f"Error loading model: {e}")
    model = None  # So we can check later in routes

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return "Model not loaded properly. Please check the server logs."

    try:
        destination = request.form['destination']
        travel_mode = request.form['travel_mode']
        people = int(request.form['people'])
        duration = int(request.form['duration'])

        input_data = preprocess(destination, travel_mode, people, duration)
        prediction = model.predict(np.array([input_data]))
        return render_template('index.html', budget=round(prediction[0][0], 2))

    except Exception as e:
        print(f"Prediction error: {e}")
        return render_template('index.html', budget="Error: Invalid input or model")

def preprocess(destination, travel_mode, people, duration):
    dest_map = {"Chintpurni": 0, "Agra": 1, "Srinagar": 2, "Manali": 3}
    mode_map = {"Motorcycle": 0, "Flight": 1, "Bus": 2, "Train": 3, "Personal Car": 4}
    return [dest_map.get(destination, 0), mode_map.get(travel_mode, 0), people, duration]

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))  # Render uses PORT environment variable
    app.run(host='0.0.0.0', port=port)
