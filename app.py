
import streamlit as st
from ultralytics import YOLO
from PIL import Image
from collections import Counter
import numpy as np

# Set up the website
st.set_page_config(
    page_title="Home Object Detection",
    layout="wide"
)

st.title("Home Object Detection")
st.write("Upload images to identify objects using AI.")

# Load the AI model
@st.cache_resource
def load_model():
    return YOLO("yolo11s.pt")

model = load_model()

# Upload images
uploaded_files = st.file_uploader(
    "Upload images (maximum 20)",
    type=["jpg", "jpeg", "png"],
    accept_multiple_files=True
)

# Confidence threshold
confidence = st.slider(
    "Confidence Threshold",
    min_value=0.10,
    max_value=0.90,
    value=0.25,
    step=0.05
)

# Detect objects
if uploaded_files:

    if len(uploaded_files) > 20:
        st.error("Please upload no more than 20 images.")
        st.stop()

    if st.button("Detect Objects"):

        progress = st.progress(0)

        for i, file in enumerate(uploaded_files):

            image = Image.open(file).convert("RGB")

            results = model.predict(
                source=np.array(image),
                conf=confidence,
                imgsz=960,
                device="cpu",
                verbose=False
            )

            result = results[0]

            # Draw bounding boxes
            annotated_image = Image.fromarray(
                result.plot()[:, :, ::-1]
            )

            # Collect object names
            object_names = []

            for box in result.boxes:
                class_id = int(box.cls.item())
                object_name = result.names[class_id]
                object_names.append(object_name)

            # Count objects
            object_counts = Counter(object_names)

            st.subheader(f"Image {i+1}: {file.name}")

            st.image(
                annotated_image,
                use_container_width=True
            )

            st.write("Detected Objects:")

            if object_counts:
                for name, count in object_counts.items():
                    st.write(f"- {name}: {count}")
            else:
                st.info("No objects detected.")

            st.divider()
            progress.progress(
                (i + 1) / len(uploaded_files)
            )

        st.success("Detection completed!")
