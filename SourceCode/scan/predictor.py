import numpy as np
import cv2 as cv
from PIL import Image
from keras._tf_keras.keras.models import load_model
# Load the trained model once
model = load_model('C:\\Users\\sulai\\Desktop\\PYTHON\\MedicalScan\\A75L4.keras')

def preprocess_image(image_file):
    # Load image and convert to RGB
    image = Image.open(image_file).convert('RGB')
    image = np.array(image)

    # Convert to grayscale
    image = cv.cvtColor(image, cv.COLOR_RGB2GRAY)

    # Resize to 200x200
    image = cv.resize(image, (200, 200))

    # Normalize and reshape
    image_array = image.astype(np.float32) / 255.0
    image_array = np.expand_dims(image_array, axis=(0, -1))  # Shape: (1, 200, 200, 1)

    return image_array

def predict_pneumonia(image_file):
    processed = preprocess_image(image_file)
    prediction = model.predict(processed)
    probability = float(prediction[0][0])
    return f"Pneumonia Probability: {probability:.2%}"