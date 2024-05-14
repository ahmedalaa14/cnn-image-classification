import streamlit as st
from keras.models import load_model
from PIL import Image
import numpy as np

# Load the model
Lrdetect_model = load_model('model1.h5')

st.title('Image Classification Tool')

input_image = st.file_uploader('Upload an image', type=['jpg', 'jpeg', 'png'])

if input_image is not None:
    st.image(input_image, caption='Uploaded Image', use_column_width=True)
    st.write('')
    st.write('Classifying...')
    
    # Open the image file and convert it to a numpy array
    image = Image.open(input_image).convert('RGB')
    
    # Resize and preprocess the image
    image = image.resize((224, 224))  # assuming your model expects 224x224 images
    image = np.array(image) / 255.0  # assuming your model expects pixel values in [0, 1]
    
    # Add an extra dimension for the batch size
    image = np.expand_dims(image, axis=0)
    
    # Predict the class of the image
    prediction = Lrdetect_model.predict(image)
    
    # Get the class with the highest probability
    predicted_class = np.argmax(prediction)
    
    st.write('Predicted class:', predicted_class)