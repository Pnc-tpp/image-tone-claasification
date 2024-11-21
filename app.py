import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import gdown
from tensorflow.keras.models import load_model
import os
import joblib

url = 'https://drive.google.com/file/d/1nhc-Yir6d4YozU8oZdKYs5zqJ4mrOVZr/view?usp=drive_link'
output = 'model'
gdown.download(url, output, quiet=False)

model = joblib.load(output)
def classify_image(img_path, model):
  
    try:
        img = image.load_img(img_path, target_size=(150, 150))
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        prediction = model.predict(img_array)
        return np.argmax(prediction)
    except Exception as e:
        st.error(f"Error classifying image: {e}")
        return predicted_class

class_labels = {0: "colorful", 1: "cool tone", 2: "earth tone", 3: "warm tone", 4: "night mood"}

st.title("Images Tone Classification App")
st.subheader("A model that helps choosing image with harmonized tones.")
st.markdown("Please upload images to classify")


uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
	temp_file_path = "temp_image.jpg"
	with open(temp_file_path, "wb") as f:
		f.write(uploaded_file.getbuffer())
	from tensorflow.keras.preprocessing import image
	import numpy as np
	img = image.load_img(temp_file_path, target_size=(150, 150))  # กำหนดขนาดภาพให้ตรงกับโมเดล
	img_array =image.img_to_array(img)
	img_array = np.expand_dims(img_array, axis=0)  # เพิ่ม batch dimension
	img_array /= 255.0  # ปรับค่าพิกเซลเป็น 0-1
	
	predictions = model.predict(img_array)
	predicted_class = np.argmax(predictions, axis=1)[0]  # ดึงคลาสที่มีค่าความน่าจะเป็นสูงสุด
	print("Predicted class:", predicted_class)
	if predicted_class is not None:
	        st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
	        st.write(f"Predicted class: {class_labels[predicted_class]}")
	else:
	        st.error("Unable to classify the image. Please try a different image.")
	        st.write(f"Description: This images are classified as a '{class_labels[predicted_class]}' tone.")
	
	        st.success("Image classification completed successfully!")
	        os.remove(temp_file_path)		
else:
	st.warning("Please upload an image to classify.")
