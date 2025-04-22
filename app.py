
from flask import Flask, render_template, request
import pickle
import numpy as np
import tensorflow as tf

# Initialize Flask app
app = Flask(__name__)

# Load the trained AI model
model = tf.keras.models.load_model('model.h5')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get data from the form
        destination = request.form['destination']
        travel_mode = request.form['travel_mode']
        number_of_people = int(request.form['people'])
        duration_days = int(request.form['duration'])

        # Prepare the data for prediction (assuming model needs certain features)
        # You may need to preprocess or format this data according to your model's requirements
        features = np.array([[destination, travel_mode, number_of_people, duration_days]])

        # Get prediction from the model
        prediction = model.predict(features)

        # Render the result on the frontend
        return render_template('result.html', prediction=prediction[0])

if __name__ == '__main__':
    app.run(debug=True)
