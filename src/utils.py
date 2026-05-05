# utils.py

import cv2
import os
import numpy as np
import pandas as pd
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

def load_data(csv_path, image_folder):

    df = pd.read_csv(csv_path)
    df["binary_label"] = df["Retinopathy grade"].apply(lambda x: 0 if x == 0 else 1)

    images = []
    labels = []

    for _, row in df.iterrows():
        img_path = os.path.join(image_folder, row["Image name"] + ".jpg")

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224, 224))

        images.append(img)
        labels.append(row["binary_label"])

    images = preprocess_input(np.array(images).astype("float32"))
    labels = np.array(labels)

    return images, labels