import streamlit as st
from  tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import numpy as np
import os

#set title of the app
st.title("Cat or Dog  Classifer")

# Add  a description
st.write("Upload image  of cat or dog, the model will predict the class.")

# Function the load the model
# @st.cache(allow_output_mutation=True)
@st.cache_data
def load_trained_model():
    model_path = os.path.join("models", "vgg_model.h5")
    return load_model(model_path)

# Load the model
model = load_trained_model()

# Define class names
class_names = ['Cat', 'Dog']

# File uploader widget
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open and display the image
    image = Image.open(uploaded_file)
    # st.image(image, caption='Uploaded Image', use_column_width=True)
    st.image(image, caption='Uploaded Image', use_container_width=True)

    # Preprocess the image
    img = image.resize((224, 224))  # Resize to match model's expected input
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # Normalize the image

    # Make prediction
    prediction = model.predict(img_array)
    predicted_class = class_names[int(prediction[0] > 0.5)]  # 0: Cat, 1: Dog

    # Display the prediction
    st.write(f"The model predicts this image is a **{predicted_class}**.")
