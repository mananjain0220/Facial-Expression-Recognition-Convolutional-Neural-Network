from flask import Flask, request, jsonify, render_template,send_file,send_from_directory
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from PIL import Image
import tensorflow as tf
from keras.preprocessing import image
from keras.models import load_model
from tensorflow.keras import models
from werkzeug.utils import secure_filename

app = Flask(__name__)
model1 = load_model('my_model.h5')
app.config['UPLOAD_PATH'] = 'uploads'
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}

label_dict = {0 : 'Angry', 1 : 'Disgust', 2 : 'Fear', 3 : 'Happiness', 4 : 'Sad', 5 : 'Surprise', 6 : 'Neutral'}
	



@app.route('/')
def hello():
	return render_template("index.html")


@app.route('/submit', methods = ["POST"])
def submit_data():
	if request.method == "POST":
		img = request.files["userfile"]
		image_paath ="static/" + img.filename
		img.save(image_paath)
		i = image.load_img(image_paath,grayscale=True, target_size=(48,48))
		x = tf.keras.preprocessing.image.img_to_array(i)
		x = np.expand_dims(x, axis=0)
		x.shape
		p = np.argmax(model1.predict(x))
		p = label_dict[p]
		
	return render_template("result.html", prediction = p, img_path = image_paath)

	

if __name__ == '__main__':
	app.run(debug= True)
