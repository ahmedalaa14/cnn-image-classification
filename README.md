# Image Classification Tool

- This is a simple web application for image classification. It uses a pre-trained model to predict the class of an uploaded image.

# Dataset
- Cifar10 (https://www.kaggle.com/competitions/cifar-10)

## Requirements

- Python 3.6 or later
- Streamlit
- Keras
- PIL
- Numpy
- Tensorflow
- Numpy
- sklearn.metrics

## Installation

1. Clone this repository.
2. Install the required packages.

## Usage
1. Run the Streamlit app.

```bash
streamlit run app.py
```

2. Open a web browser and go to http://localhost:8501.
3. Upload an image using the file uploader.
4. The app will classify the image and display the predicted class Model.

## CNN Model

The model used for this app is a pre-trained CNN model saved in the `model1.h5` file. The model was trained to classify images into specific classes (please specify the classes here). The model architecture includes several convolutional layers, pooling layers, and dense layers.

## app.py

`app.py` is the main script that runs the web application. It uses Streamlit to create the user interface where users can upload images. The uploaded images are preprocessed and passed to the CNN model for prediction.

## Contributing

Contributions are welcome. Please open an issue or submit a pull request.


