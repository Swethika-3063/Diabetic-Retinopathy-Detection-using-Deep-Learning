# visualize.py

import os
import numpy as np
import cv2
import matplotlib.pyplot as plt

from tensorflow.keras.models import load_model
from utils import load_data
from gradcam import make_gradcam_heatmap

# ---------------------------
# PATH SETUP
# ---------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

model_path = os.path.join(BASE_DIR, "models", "dr_model.keras")
test_csv = os.path.join(BASE_DIR, "data/test/labels.csv")
test_images = os.path.join(BASE_DIR, "data/test/images")

print("Loading model from:", model_path)
print("Model exists:", os.path.exists(model_path))

# ---------------------------
# LOAD MODEL
# ---------------------------
model = load_model(model_path)

# ---------------------------
# LOAD TEST DATA
# ---------------------------
X_test, y_test = load_data(test_csv, test_images)

print("Test samples:", len(X_test))

# ---------------------------
# PICK IMAGE
# ---------------------------
img = X_test[0]
img_array = np.expand_dims(img, axis=0)

print("Generating Grad-CAM...")

# ---------------------------
# GENERATE HEATMAP
# ---------------------------
heatmap = make_gradcam_heatmap(img_array, model, "Conv_1")

# ---------------------------
# PROCESS HEATMAP
# ---------------------------
heatmap = cv2.resize(heatmap, (224, 224))
heatmap = np.uint8(255 * heatmap)
heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

# ---------------------------
# ORIGINAL IMAGE
# ---------------------------
original = X_test[0]
original = (original - original.min()) / (original.max() - original.min())
original = np.uint8(255 * original)

# ---------------------------
# OVERLAY
# ---------------------------
superimposed = cv2.addWeighted(original, 0.6, heatmap, 0.4, 0)

# ---------------------------
# SAVE RESULT (IMPORTANT)
# ---------------------------
results_dir = os.path.join(BASE_DIR, "results")
os.makedirs(results_dir, exist_ok=True)

output_path = os.path.join(results_dir, "gradcam.jpg")
cv2.imwrite(output_path, superimposed)

print("Grad-CAM saved at:", output_path)

# ---------------------------
# DISPLAY (optional)
# ---------------------------
plt.figure(figsize=(10, 4))

plt.subplot(1, 3, 1)
plt.title("Original")
plt.imshow(original)

plt.subplot(1, 3, 2)
plt.title("Heatmap")
plt.imshow(heatmap)

plt.subplot(1, 3, 3)
plt.title("Overlay")
plt.imshow(superimposed)

plt.tight_layout()
plt.show(block=True)