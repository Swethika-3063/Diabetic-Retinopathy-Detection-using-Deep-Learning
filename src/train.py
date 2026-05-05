# train.py

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.utils import class_weight

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam


#load the data
from utils import load_data
import os
print("🔥 TRAIN.PY STARTED")
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

csv_path = os.path.join(BASE_DIR, "data/test/labels.csv")
img_path = os.path.join(BASE_DIR, "data/test/images")
print("CSV PATH:", csv_path)
print("IMG PATH:", img_path)

import os
print("CSV exists:", os.path.exists(csv_path))
print("IMG folder exists:", os.path.exists(img_path))
images, labels = load_data(csv_path, img_path)
# SPLIT

# LOAD TRAIN DATA
X_train, y_train = load_data(
    os.path.join(BASE_DIR, "data/train/labels.csv"),
    os.path.join(BASE_DIR, "data/train/images")
)

# LOAD TEST DATA
X_test, y_test = load_data(
    os.path.join(BASE_DIR, "data/test/labels.csv"),
    os.path.join(BASE_DIR, "data/test/images")
)

print("Train:", np.bincount(y_train))
print("Test:", np.bincount(y_test))

print("Train:", np.bincount(y_train))
print("Test:", np.bincount(y_test))

# AUGMENTATION
datagen = ImageDataGenerator(
    rotation_range=25,
    zoom_range=0.2,
    brightness_range=[0.7, 1.3],
    horizontal_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1
)

# MODEL
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224,224,3))

for layer in base_model.layers:
    layer.trainable = False

x = GlobalAveragePooling2D()(base_model.output)
output = Dense(1, activation="sigmoid")(x)

model = Model(inputs=base_model.input, outputs=output)

model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# CLASS WEIGHTS
weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)

class_weights = {0: weights[0], 1: weights[1]}

# TRAIN
early_stop = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)

model.fit(
    datagen.flow(X_train, y_train, batch_size=8),
    epochs=20,
    validation_data=(X_test, y_test),
    class_weight=class_weights,
    callbacks=[early_stop]
)

# EVALUATE
loss, acc = model.evaluate(X_test, y_test)
print("Test accuracy:", acc)

# CONFUSION MATRIX
pred = (model.predict(X_test) > 0.5).astype(int)
print(confusion_matrix(y_test, pred))

# SAVE

import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_dir = os.path.join(BASE_DIR, "models")

os.makedirs(model_dir, exist_ok=True)

model_path = os.path.join(model_dir, "dr_model.keras")

model.save(model_path)

print("Model saved at:", model_path)