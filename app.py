import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import os
from tensorflow.keras.models import load_model

# Load the saved model
model = load_model('/image-tone-claasification/my model.h5')

# Function to classify an image
def classify_image(img_path, model):
    try:
        img = image.load_img(img_path, target_size=(150, 150))
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        prediction = model.predict(img_array)
        return np.argmax(prediction)
    except Exception as e:
        st.error(f"Error classifying image: {e}")
        return None


# Class labels
class_labels = {0: "colorful", 1: "cool tone", 2: "earth tone", 3: "warm tone", 4: "night mood"}

# Streamlit app
st.title("Image Classifier App")


uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Save the uploaded file temporarily
    temp_file_path = "temp_image.jpg"
    with open(temp_file_path, "wb") as f:
      f.write(uploaded_file.getbuffer())

    # Classify the image
    predicted_class = classify_image(temp_file_path, model)

    if predicted_class is not None:
        st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

        # Display the predicted class
        st.write(f"Predicted class: {class_labels[predicted_class]}")

        #Remove the temporary file
        os.remove(temp_file_path)
