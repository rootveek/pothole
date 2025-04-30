import streamlit as st
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from ultralytics import YOLO
from streamlit_folium import folium_static
import folium

# Load YOLO model
model = YOLO("best.pt")

# Streamlit UI
st.title("🚧 Pothole Detection System")

# Sidebar for mode selection
mode = st.sidebar.radio("Choose Mode", ["Image Upload", "Live Webcam"])

if mode == "Image Upload":
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Convert to OpenCV format
        img_cv = np.array(image)
        img_cv = img_cv[:, :, ::-1].copy()

        # Run YOLO detection
        results = model(img_cv)

        # Process results
        detected_potholes = []
        for result in results:
            boxes = result.boxes.xyxy  # Get bounding boxes
            scores = result.boxes.conf  # Confidence scores

            for box, score in zip(boxes, scores):
                x1, y1, x2, y2 = map(int, box)
                confidence = round(float(score), 2)

                # Draw bounding box
                cv2.rectangle(img_cv, (x1, y1), (x2, y2), (0, 255, 0), 3)
                cv2.putText(img_cv, f"{confidence}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # Store detection details
                detected_potholes.append({"Image": uploaded_file.name, "Confidence": confidence})

        # Display processed image
        st.image(img_cv, caption="Detected Potholes", use_column_width=True)

        # Save results to CSV
        if detected_potholes:
            df = pd.DataFrame(detected_potholes)
            df.to_csv("pothole_detections.csv", mode="a", header=False, index=False)
            st.success("Detection saved to pothole_detections.csv ✅")

elif mode == "Live Webcam":
    st.warning("Press 'Start' to begin real-time detection.")

    # Start button
    if st.button("Start Live Detection"):
        ip_camera_url = "http://192.168.252.237:4747/video"
        cap = cv2.VideoCapture(ip_camera_url)  # Open webcam

        st_frame = st.empty()  # Streamlit frame placeholder

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = model(frame)

            # Draw bounding boxes
            for result in results:
                boxes = result.boxes.xyxy
                scores = result.boxes.conf
                for box, score in zip(boxes, scores):
                    x1, y1, x2, y2 = map(int, box)
                    confidence = round(float(score), 2)

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
                    cv2.putText(frame, f"{confidence}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Display the frame
            st_frame.image(frame, channels="BGR", use_column_width=True)

        cap.release()