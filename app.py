import streamlit as st
from PIL import Image
import cv2
from ultralytics import YOLO
import os

MODEL_PATH = "Yolov11best.pt"

# Check if model file exists
if not os.path.exists(MODEL_PATH):
    st.error(
        f"Model file '{MODEL_PATH}' not found.\n"
        "Please upload the model file to the app folder or provide a download URL."
    )
    st.stop()  # Stop running app until model is available

# Load YOLOv11 model with caching to speed up reloads
@st.cache_resource
def load_model():
    return YOLO(MODEL_PATH)

model = load_model()

# Set Streamlit title and description
st.title("Drowsiness Detection App 😴")
st.write("Upload an image to detect drowsiness.")

# Image uploader widget
uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

# Class labels dictionary
class_names = {0: "Awake", 1: "Drowsy"}

if uploaded_file is not None:
    # Open and display the uploaded image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Run detection
    with st.spinner("Detecting..."):
        results = model.predict(source=image, conf=0.1)
        boxes = results[0].boxes

        if boxes and len(boxes.cls) > 0:
            # Draw bounding boxes on the image
            plotted_img = results[0].plot()
            plotted_rgb = cv2.cvtColor(plotted_img, cv2.COLOR_BGR2RGB)
            st.image(plotted_rgb, caption="Detection Result", use_container_width=True)

            # Show detected classes
            detected_classes = [class_names.get(int(cls), f"Class {int(cls)}") for cls in boxes.cls]
            st.success(f"Detected: {', '.join(detected_classes)}")
        else:
            st.warning("No drowsiness detected.")
