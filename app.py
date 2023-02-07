#!/usr/bin/env python
# coding: utf-8

# import librairie

import io
import os
import numpy as np
import tensorflow as tf
from PIL import Image
from flask import Flask, request, render_template
import logging
logging.basicConfig(level=logging.DEBUG)


# load model 

model = tf.keras.models.load_model('retinal-oct.h5')

# prepare images 

def prepare_image(file):
    """
    prepares the image for the api call
    """
    img_bytes = file.read()
    img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
    save_img(img)
    img = img.resize((150, 150))
    img = np.array(img)
    img = np.expand_dims(img, 0)
    return img

# prediction

def predict_result(img):
    """predicts the result"""
    return str(np.argmax(model.predict(img)[0]))

def prepare_and_predict(file):
    img = prepare_image(file)
    prediction = predict_result(img)
    return prediction

def save_img(img):
    img.save("static/img.jpeg")

# initialize flask object


app = Flask(__name__)


# setting up routes and their functions


@app.route('/predict', methods=['POST'])
def infer_image():
    logging.info(str(request.files))
    
    # Catch the image file from a POST request
    if 'file' not in request.files:
        return "Please try again. The Image doesn't exist"
    
    file = request.files.get('file')
    if not file:
        return
    prediction = prepare_and_predict(file)
    return prediction

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == "POST":
        logging.info(str(request.files))
        
        # Catch the image file from a POST request
        if 'file' not in request.files:
            return "Please try again. The Image doesn't exist"
        
        file = request.files.get('file')
        if not file:
            return

        prediction = prepare_and_predict(file)
        
        return render_template('home.html', prediction=prediction)
    return render_template('home.html')

# run the API

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', use_reloader=True)



