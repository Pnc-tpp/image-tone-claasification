import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import os
from tensorflow.keras.models import load_model
import gdown
from tensorflow.keras.models import load_model
import os
st.set_option('deprecation.showfileUploaderEncoding', False)

@st.cache(allow_output_mutation=True)
def load_model():
	model = tf.keras.models.load_model('./model .h5')
	return model

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
st.title("Image Classification App")
st.subheader("A model that helps choosing image with harmonized tones.")
st.markdown("Upload an image to classify")


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
    else:
        # Display an error message if classification fails
        st.error("Unable to classify the image. Please try a different image.")
        # Optionally add a description or additional details for the class
        st.write(f"Description: This images are classified as a '{class_labels[predicted_class]}' tone.")

        # Display a success message
        st.success("Image classification completed successfully!")
        #Remove the temporary file
        os.remove(temp_file_path)
else:
    st.warning("Please upload an image to classify.")
