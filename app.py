from __future__ import division, print_function
import os
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify
import requests
import sys

# Allowed extensions for the uploaded file
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

# Initialize the Flask application
app = Flask(__name__)

# Path to the model
MODEL_PATH = 'model_vgg16.h5'

# Load your trained model or raise an exception
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print('Model loaded successfully.')
except OSError as e:
    print(f"Model file not found: {MODEL_PATH}")
    sys.exit(1)
except Exception as e:
    print(f"Error loading model: {e}")
    sys.exit(1)

# Function to predict the class of an image
def model_predict(img, model):
    # Preprocessing the image
    img = tf.keras.preprocessing.image.img_to_array(img)
    img = tf.image.resize(img, (224, 224))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    # Predicting the class
    classes = model.predict(img)
    return classes

# Route for the POST request
@app.route('/', methods=['POST'])
def upload():
    # Get the file from the post request
    if not request.is_json:
        return jsonify({"Error": "Request must be in JSON format"}), 400
    
    # Get the URL of the image
    url = request.json.get('url')
    if not url:
        return jsonify({'Error': 'Please provide a valid image URL'})

    if not any(url.lower().endswith(ext) for ext in ALLOWED_EXTENSIONS):
        return jsonify({'Error': 'URL has an unsupported image extension'}), 400
    
    try:
        response = requests.get(url)
        if response.status_code == 404:
            return jsonify({'Error': 'The URL is not valid or could not be found'})
        
        img = tf.keras.preprocessing.image.img_to_array(
            tf.keras.preprocessing.image.load_img(response.content, target_size=(224, 224))
        )
        
        # Make prediction
        preds = model_predict(img, model)
        
        # Prepare the response
        if preds[0][1] > 0.5:
            return jsonify({'The prediction is': 'Positive'})
        else:
            return jsonify({'The prediction is': 'Negative'})
        
    except requests.exceptions.RequestException as e:
        return jsonify({'Error': f'Request exception: {str(e)}'})
    except Exception as e:
        return jsonify({'Error': f'An unexpected error occurred: {str(e)}'})
    
    return jsonify({'Error': 'Unknown error occurred'}), 500

if __name__ == '__main__':
    app.run(debug=False)
