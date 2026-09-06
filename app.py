import streamlit as tf_streamlit
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

# Page Configuration
tf_streamlit.set_page_config(
    page_title="Industrial Surface Crack Detection",
    page_icon="🔍",
    layout="centered"
)

tf_streamlit.title("🔍 Industrial Surface Crack Detection System")
tf_streamlit.write("Upload a surface structure image (Concrete/Wall) to analyze structural integrity using Deep Learning.")

# Load the trained model
@tf_streamlit.cache_resource
def load_detection_model():
    model = load_model('crack_detection_model.h5')
    return model

with tf_streamlit.spinner('Loading AI Model... Please wait.'):
    model = load_detection_model()

tf_streamlit.success("✅ Model loaded successfully!")

# File uploader for user
uploaded_file = tf_streamlit.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    tf_streamlit.image(image, caption='Uploaded Structure Image', use_column_width=True)
    
    # Preprocess the image for the model
    image_resized = image.resize((64, 64))
    img_array = np.array(image_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    if tf_streamlit.button('Analyze Image'):
        with tf_streamlit.spinner('Analyzing surface structure...'):
            prediction = model.predict(img_array)
            score = prediction[0][0]
            
            # Classification logic based on sigmoid output
            # Assuming 0 is Normal/Clear and 1 is Crack (or vice versa depending on your training setup)
            if score > 0.5:
                result = "Crack Detected! (Maintenance Needed)"
                confidence = score * 100
                tf_streamlit.error(f"⚠️ **Result:** {result}")
            else:
                result = "No Crack Detected (Surface Clear / Safe)"
                confidence = (1 - score) * 100
                tf_streamlit.success(f"✅ **Result:** {result}")
            
            tf_streamlit.info(f"Confidence Score: {confidence:.2f}%")
