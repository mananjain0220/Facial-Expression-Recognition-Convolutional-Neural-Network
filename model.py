import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
import numpy as np
import joblib

loaded_model = joblib.load('model_joblib')


#img = np.array(Image.open("download.jpeg"))
#img_string = ' '.join(map(str, img.flatten().tolist()))
an_image = PIL.Image.open("download.jpeg")
image_sequence = an_image.getdata()
image_array = np.array(image_sequence)
image_array.tofile('data.csv')
result = loaded_model.predict('data.csv')