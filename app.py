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
    print("Model loaded successfully")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None  # Set model to None so you can check if it's loaded correctly

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
        # Extract form data
        destination = request.form['destination']
        travel_mode = request.form['travel_mode']
        people = int(request.form['people'])
        duration = int(request.form['duration'])

        # Log the input data for debugging
        print(f"Received input data: destination={destination}, travel_mode={travel_mode}, people={people}, duration={duration}")

        # Preprocess input data
        input_data = preprocess(destination, travel_mode, people, duration)
        print(f"Preprocessed input data: {input_data}")

        # Predict the budget
        input_data = np.array([input_data])  # Ensure the input is a 2D array
        prediction = model.predict(input_data)
        print(f"Prediction result: {prediction}")

        # Return the result to the user
        return render_template('index.html', budget=round(prediction[0][0], 2))

    except Exception as e:
        print(f"Prediction error: {e}")
        return render_template('index.html', budget="Error: Invalid input or model")

def preprocess(destination, travel_mode, people, duration):
    # Map input values to numerical values
    dest_map = {"Chintpurni": 0, "Agra": 1, "Srinagar": 2, "Manali": 3}
    mode_map = {"Motorcycle": 0, "Flight": 1, "Bus": 2, "Train": 3, "Personal Car": 4}
    
    # Return the preprocessed data as a list
    return [dest_map.get(destination, 0), mode_map.get(travel_mode, 0), people, duration]

if __name__ == '__main__':
    # Ensure the correct port for deployment on Render
    port = int(os.environ.get("PORT", 5000))  # Render uses PORT environment variable
    app.run(host='0.0.0.0', port=port)
