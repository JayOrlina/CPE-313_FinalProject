import streamlit as st
from PIL import Image
import numpy as np
import cv2
from ultralytics import YOLO

# --- Streamlit UI ---
st.set_page_config(page_title="Drowsiness Detection App", layout="centered")
st.title("Drowsiness Detection App 😴")
st.write("Upload an image to detect if the student is **Awake** or **Drowsy**.")

# --- File uploader ---
uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

# --- Model loader with caching ---
@st.cache_resource
def load_model():
    try:
        model = YOLO("F:/JAY/FinalProj/Yolov11best.pt")  # Update path if needed
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_model()

# --- Class labels ---
class_names = {0: "Awake", 1: "Drowsy"}

# --- Prediction and display ---
if uploaded_file is not None and model is not None:
    try:
        # Load and display the uploaded image
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_container_width=True)

        # Predict
        with st.spinner("Detecting..."):
            results = model.predict(source=np.array(image), conf=0.1)
            boxes = results[0].boxes

            if boxes is not None and len(boxes.cls) > 0:
                # Plot results (BGR)
                plotted_img = results[0].plot()
                plotted_rgb = cv2.cvtColor(plotted_img, cv2.COLOR_BGR2RGB)
                st.image(plotted_rgb, caption="Detection Result", use_container_width=True)

                # Show detected classes
                detected_classes = [class_names.get(int(cls), f"Class {int(cls)}") for cls in boxes.cls]
                st.success(f"Detected: {', '.join(detected_classes)}")
            else:
                st.warning("No drowsiness detected.")
    except Exception as e:
        st.error(f"Prediction failed: {e}")
elif model is None:
    st.error("Model could not be loaded. Please check the model path.")
