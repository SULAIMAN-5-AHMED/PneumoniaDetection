## Pneumonia Detection Web App 

## Overview

This web application uses a Convolutional Neural Network (CNN) to detect pneumonia from chest X-ray images. Built with Django for the backend and TensorFlow/Keras for model inference, it provides a clean interface for uploading images and viewing predictions.

## Features

- Upload chest X-ray images via browser
- Real-time prediction with probability scores
- Clear output labels: "Pneumonia" or "Normal"
- Dynamic feedback and reset controls
- Modular preprocessing pipeline with error handling

## Model Architectur


<img width="868" height="495" alt="image" src="https://github.com/user-attachments/assets/e641fc35-cb9a-441c-a6e8-af590e2743cc" />



Reminder: Always convert image arrays to tensors before training, and check input shapes using print(x_train[1].shape) and print(x_train.shape). Use np.expand_dims() if dimensions are missing.


## Setup Instructions

1)Clone the repo:

  git clone https://github.com/your-username/pneumonia-app.git

2)Install dependencies:

  pip install -r requirements.txt

3)Run the server:

  python manage.py runserver


## Hosting Notes

-Model file (.h5) hosted externally due to size limits

-Use Hugging Face Spaces or Google Drive for integration (https://huggingface.co/SulaimanAhmed/pneumonia_detector)

-Update load_model() path in views.py accordingly



