import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import os

# Load the saved model
model = tf.keras.models.load_model('/content/drive/MyDrive/my_model.h5')

# Define class labels
class_labels = {0: "colorful", 1: "cool tone", 2: "earth tone", 3: "warm tone", 4: "night mood"}


def classify_image(img_path, model):
    img = image.load_img(img_path, target_size=(150, 150))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)
    return np.argmax(prediction)

st.title("Image Tone Classifier")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Save the uploaded file temporarily
    temp_file_path = "temp_image.jpg"  # Or other suitable extensions
    with open(temp_file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Classify the image
    predicted_class = classify_image(temp_file_path, model)

    # Display the results
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
    st.write(f"Predicted Tone: {class_labels.get(predicted_class, 'Unknown')}")


    # Clean up the temporary file (optional, but good practice)
    os.remove(temp_file_path)
