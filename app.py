import streamlit as st
import torch
import numpy as np
import cv2
from PIL import Image

# Load the custom YOLOv11 model for drowsiness detection
@st.cache_resource
def load_model():
    model = torch.hub.load('JayOrlina/CPE-313_FinalProject', 'custom', path='Yolov11best.pt', source='github')
    return model

model = load_model()

st.title("Drowsiness Detection with YOLOv11")
st.markdown("Upload an image of a person to detect signs of drowsiness.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load image and convert to numpy
    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)

    # Run inference
    results = model(image_np)

    # Display results
    st.image(np.squeeze(results.render()), caption="Detection Results", use_column_width=True)
    st.write("Detection Details:")
    st.dataframe(results.pandas().xyxy[0])
