from __future__ import division, print_function
import os
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify
import requests
import tempfile
import sys
import urllib.request

# Allowed extensions for the uploaded file
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

# Initialize the Flask application
app = Flask(__name__, static_url_path='',
            static_folder='static',
            template_folder='templates')

# Path to the model
MODEL_PATH = 'model_vgg16.h5'

# Load your trained model or raise an exception
try:
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    model.compile()
    print('.............Model loaded. Start serving..........\n\n\n\n\n')
except OSError as e:
    print(f".............Model file not found ..........\n\n\n\n\n : {MODEL_PATH}")
    sys.exit(1)
except Exception as e:
    print(f"............Error loading model ..........\n\n\n\n\n: {e}")
    sys.exit(1)

# Function to predict the class of an image
def model_predict(img_path, model):
    # Target size must agree with what the trained model expects!!
    img = tf.keras.utils.load_img(img_path, target_size=(224, 224))
    # Preprocessing the image
    x = tf.keras.utils.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x /= 255
    # Predicting the class
    classes = model.predict(x)
    return classes

@app.errorhandler(500)
def internal_server_error(e):
    return jsonify({'Error': 'Internal Server Error'}), 500
# Route for the POST request
@app.route('/', methods=['POST'])
def upload():
    # Check if a file was uploaded
    if 'file' not in request.files:
        return jsonify({'Error': 'No file uploaded'})

    file = request.files['file']

    # Check file extension
    if file.filename.split('.')[-1].lower() not in ALLOWED_EXTENSIONS:
        return jsonify({'Error': 'Unsupported file extension'})

    try:
        # Save the uploaded file to a temporary location
        with tempfile.NamedTemporaryFile(delete=False) as f:
            file.save(f)
            temp_path = f.name

        # Make prediction
        preds = model_predict(temp_path, model)

        # Prepare the response
        str1 = 'Positive'
        str2 = 'Negative'
        if preds[0][1] > 0.5:
            return jsonify({'The prediction is': f'{str1}'})
        else:
            return jsonify({'The prediction is': f'{str2}'})

    except requests.exceptions.HTTPError as e:
            return jsonify({'Error': f'Response returned {e.response.status_code}'})
    except requests.exceptions.ConnectionError:
            return jsonify({'Error': 'Failed to establish a connection to the server.'})
    except requests.exceptions.Timeout:
            return jsonify({'Error': 'Request timed out.'})
    except requests.exceptions.RequestException:
            return jsonify({'Error': 'An unexpected error occurred.'})

if __name__ == '__main__':
    app.run(debug=False)
