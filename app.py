import streamlit as st
from PIL import Image
import numpy as np
import cv2
from ultralytics import YOLO
import os
import requests

MODEL_URL = "https://drive.google.com/file/d/1Ww-Dv55SxFrSk-D59O7TYs9gLRFZP5Ls/view?usp=drive_link"
MODEL_PATH = "RTDETRbest.pt"

def download_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Downloading model..."):
            r = requests.get(MODEL_URL)
            with open(MODEL_PATH, "wb") as f:
                f.write(r.content)

download_model()

# Load YOLOv11 model
@st.cache_resource
def load_model():
    return YOLO(MODEL_PATH)

model = load_model()

# Set Streamlit title
st.title("Drowsiness Detection App 😴")
st.write("Upload an image (e.g., a video frame) to detect drowsiness.")

# Upload image
uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

class_names = {0: "Awake", 1: "Drowsy"}

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Predict
    with st.spinner("Detecting..."):
        results = model.predict(source=image, conf=0.1)
        boxes = results[0].boxes

        if boxes and len(boxes.cls) > 0:
            plotted_img = results[0].plot()
            plotted_rgb = cv2.cvtColor(plotted_img, cv2.COLOR_BGR2RGB)
            st.image(plotted_rgb, caption="Detection Result", use_container_width=True)
            detected_classes = [class_names.get(int(cls), f"Class {int(cls)}") for cls in boxes.cls]
            st.success(f"Detected: {', '.join(detected_classes)}")
        else:
            st.warning("No drowsiness detected.")
