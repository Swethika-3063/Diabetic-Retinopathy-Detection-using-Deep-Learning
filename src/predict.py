# predict.py

import numpy as np
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

model = load_model("models/dr_model.keras")

def preprocess_image(path):
    img = cv2.imread(path)
    img = cv2.resize(img, (224,224))
    img = preprocess_input(img)
    return np.expand_dims(img, axis=0)

img = preprocess_image("sample.jpg")

pred = model.predict(img)

print("DR Detected" if pred > 0.5 else "No DR")