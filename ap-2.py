import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import pickle

# Load model and class names
MODEL_PATH = "fashion_mnist_rgb_fast.h5"  # Update with your model path
CLASS_NAMES_PATH = "class_names.pkl"

@st.cache_resource  
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

@st.cache_resource
def load_class_names():
    with open(CLASS_NAMES_PATH, "rb") as f:
        return pickle.load(f)

model = load_model()
class_names = load_class_names()

# Preprocess image
def preprocess_image(image):
    image = image.resize((128, 128))  # Resize to match training input
    image_array = np.array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    return image_array

# Streamlit interface
st.title("🧥 Fashion Article Classifier")

uploaded_file = st.file_uploader("Upload an image of a fashion item", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)
    if st.button("Predict"):
        input_tensor = preprocess_image(image)
        prediction = model.predict(input_tensor)
        predicted_class = np.argmax(prediction)
        predicted_label = class_names[predicted_class]
        confidence = np.max(prediction)

        st.markdown(f"### 🔍 Prediction: `{predicted_label}`")
        st.markdown(f"### 📊 Confidence: `{confidence * 100:.2f}%`")