import numpy as np
import seaborn as sns
import cv2
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Dense,
    Conv2D,
    Flatten,
    MaxPooling2D,
    Dropout,
    BatchNormalization,
    Activation,
)
import tensorflow as tf
from sklearn.metrics import *
from sklearn.model_selection import train_test_split
from tensorflow.keras import  datasets 
from tqdm import tqdm
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import PIL
import streamlit as st
model = load_model("cnn_model.keras")


def load_image():
    image = st.sidebar.file_uploader("Upload Image", type=["png"], key="image")
    
    return image

image = load_image()

if image:
    img_size = (32, 32, 3)
    image = PIL.Image.open(image)
    st.write("Original Image")
    st.image(image, caption='Uploaded Image.')
    image = image.resize((32, 32))
    image = np.array(image)
    X = np.zeros((1, *img_size))
    X[0] = image
    X = X / 255

    y_pred = model.predict(X)
    y_pred = np.argmax(y_pred, axis=1)
    st.write(f"Prediction: {y_pred[0]}")
    

