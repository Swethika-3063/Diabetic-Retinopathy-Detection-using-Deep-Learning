# Diabetic Retinopathy Detection using Deep Learning

AI-powered diabetic retinopathy detection system using CNNs for early diagnosis from retinal fundus images.


## Overview

This project focuses on detecting **Diabetic Retinopathy (DR)** from retinal fundus images using a **deep learning-based Convolutional Neural Network (CNN)**.  
It aims to support early diagnosis and improve automated screening in healthcare systems.


## Model Details

- **Architecture:** CNN / Transfer Learning (MobileNet, ResNet)
- **Task:** Binary Classification (DR vs No DR)
- **Accuracy:** ~73% *(can be improved with optimization)*


## Dataset

- **Source:** Messidor / Kaggle datasets  
- *Dataset not included in the repository due to size constraints*
- Dataset can be downloaded from:  https://ieee-dataport.org/open-access/indian-diabetic-retinopathy-image-dataset-idrid



## Features

- Image preprocessing pipeline  
- CNN-based model training  
- Prediction on new retinal images  
- Performance evaluation using confusion matrix  



## Tech Stack

- Python  
- TensorFlow / Keras  
- OpenCV  
- NumPy & Pandas  

---

## How to Run

```bash
pip install -r requirements.txt
python train.py
python predict.py
