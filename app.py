from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np

app = Flask(__name__)
model = tf.keras.models.load_model('model.h5', compile=False)

@app.route('/')
def home():
    return 'AI Budget Calculator Backend Running'

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    input_data = np.array([data['input']])  # expects a list of features
    prediction = model.predict(input_data)
    return jsonify({'prediction': prediction.tolist()})
