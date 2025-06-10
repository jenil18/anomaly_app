import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load the trained model
model = tf.keras.models.load_model("model/")

# Function to preprocess and predict
def predict(image):
    img = image.resize((224, 224))  # Teachable Machine default size
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model(img_array)
    return prediction

# Streamlit interface
st.title("Anomaly Detection for Toothbrush 🪥")

uploaded_file = st.file_uploader("Upload an image of the toothbrush", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)
    
    prediction = predict(image)
    # st.write(prediction)
    confidence = prediction['sequential_3'].numpy()[0]
    
    class_names = ["Normal", "Defective"]  # Match classes in Teachable Machine
    predicted_class = class_names[np.argmax(confidence)]
    
    st.write(f"### Prediction: **{predicted_class}**")
    st.write(f"Confidence: {confidence[np.argmax(confidence)]:.2f}")
