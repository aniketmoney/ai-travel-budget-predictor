from flask import Flask, request, render_template
import numpy as np
import tensorflow as tf
from keras.models import load_model

app = Flask(__name__)
model = load_model('your_model.h5')  # Update with your actual file

@app.route('/')
def index():
    return render_template('index.html')  # Your HTML page

@app.route('/predict', methods=['POST'])
def predict():
    # Collect form data
    destination = request.form['destination']
    travel_mode = request.form['travel_mode']
    people = int(request.form['people'])
    duration = int(request.form['duration'])

    # Process data (you might need encoding or mapping)
    input_data = preprocess(destination, travel_mode, people, duration)
    prediction = model.predict(np.array([input_data]))

    # Return the result in HTML format
    return render_template('index.html', budget=round(prediction[0][0], 2))

def preprocess(destination, travel_mode, people, duration):
    # Convert strings to numeric encodings based on your training
    dest_map = {"Chintpurni": 0, "Agra": 1, "Srinagar": 2, "Manali": 3}
    mode_map = {"Motorcycle": 0, "Flight": 1, "Bus": 2, "Train": 3, "Personal Car": 4}

    return [dest_map[destination], mode_map[travel_mode], people, duration]

if __name__ == '__main__':
    app.run(debug=True)
