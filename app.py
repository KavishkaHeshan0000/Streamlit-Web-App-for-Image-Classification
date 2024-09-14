import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import time

st.set_page_config(page_title="Image Classification App", layout="wide")

@st.cache_resource
def load_model():
    return tf.keras.applications.MobileNetV2(weights='imagenet')

model = load_model()

def classify_image(image):
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    image = np.array(image).astype(np.float32)
    image = tf.image.resize(image, (224, 224))
    image = image / 255.0
    image = np.expand_dims(image, axis=0)
    predictions = model.predict(image)
    return tf.keras.applications.mobilenet_v2.decode_predictions(predictions, top=10)[0]

st.sidebar.title("Image Classification")
uploaded_file = st.sidebar.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])
confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.5)

col1, col2 = st.columns([1, 1])

def display_results(predictions, container, threshold):
    container.subheader("Classification Results")
    filtered_predictions = [pred for pred in predictions if pred[2] >= threshold]
    
    if not filtered_predictions:
        container.warning(f"No predictions meet the confidence threshold of {threshold:.2f}")
    else:
        for i, (imagenet_id, label, score) in enumerate(filtered_predictions):
            container.write(f"{i + 1}. {label}: {score * 100:.2f}%")
    
    container.subheader("Prediction Visualization")
    scores = [score for (_, _, score) in predictions]
    labels = [label for (_, label, _) in predictions]
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(labels, scores, color='skyblue')
    ax.set_xlim(0, 1)
    ax.set_xlabel('Confidence Score')
    ax.set_title('Top 10 Predictions')
    ax.axvline(x=threshold, color='r', linestyle='--', label='Threshold')
    ax.legend()
    container.pyplot(fig)

if 'current_image' not in st.session_state:
    st.session_state.current_image = None
    st.session_state.current_predictions = None

if uploaded_file is not None:
    st.session_state.current_image = Image.open(uploaded_file).convert('RGB')
    with st.spinner('Classifying...'):
        # Progress bar
        progress_bar = st.progress(0)
        for i in range(100):
            time.sleep(0.01)
            progress_bar.progress(i + 1)
        st.session_state.current_predictions = classify_image(st.session_state.current_image)

with col1:
    if st.session_state.current_image:
        st.image(st.session_state.current_image, caption="Current Image", use_column_width=True)
    
with col2:
    if st.session_state.current_predictions:
        display_results(st.session_state.current_predictions, col2, confidence_threshold)

st.sidebar.markdown("---")
st.sidebar.subheader("About")
st.sidebar.info("This app uses a pre-trained MobileNetV2 model to classify images. Upload an image, adjust the confidence threshold, and view the results!")
st.sidebar.markdown("---")
st.sidebar.markdown("Developed by Kavishka | [GitHub Repository](https://github.com/KavishkaHeshan0000)")
