
import os
import io
import streamlit as st
import pandas as pd
import numpy as np

from PIL import Image
from ultralytics import YOLO


# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="RoadVision | Pothole Detection",
    page_icon="🚧",
    layout="wide",
    initial_sidebar_state="expanded"
)

MODEL_PATH = "best.pt"


# ==========================================
# 2. CUSTOM CSS
# ==========================================

st.markdown("""
<style>
    .stApp {
        background-color: #0b1120;
        color: #e5e7eb;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1450px;
    }

    .hero {
        background: linear-gradient(120deg, #172554, #111827);
        border: 1px solid #263b65;
        padding: 28px;
        border-radius: 16px;
        margin-bottom: 25px;
    }

    .hero h1 {
        color: #ffffff;
        font-size: 34px;
        margin-bottom: 8px;
    }

    .hero p {
        color: #b8c7df;
        font-size: 15px;
        margin-bottom: 0;
    }

    .metric-card {
        background: #151f32;
        border: 1px solid #293750;
        padding: 20px;
        border-radius: 13px;
        min-height: 105px;
    }

    .metric-label {
        color: #94a3b8;
        font-size: 13px;
    }

    .metric-value {
        color: #ffffff;
        font-size: 29px;
        font-weight: 700;
        margin-top: 7px;
    }

    .section-title {
        color: #f8fafc;
        font-size: 21px;
        font-weight: 650;
        margin: 20px 0 12px 0;
    }

    div[data-testid="stFileUploader"] {
        background: #151f32;
        border: 1px dashed #475569;
        border-radius: 12px;
        padding: 15px;
    }

    .status {
        color: #86efac;
        font-size: 13px;
    }

    footer {
        visibility: hidden;
    }
</style>
""", unsafe_allow_html=True)


# ==========================================
# 3. LOAD TRAINED MODEL
# ==========================================

@st.cache_resource
def load_model(model_path):
    return YOLO(model_path)


if not os.path.exists(MODEL_PATH):
    st.error("Trained model not found. Please verify the model path.")
    st.code(MODEL_PATH)
    st.stop()

try:
    model = load_model(MODEL_PATH)
except Exception as e:
    st.error(f"Unable to load YOLO model: {e}")
    st.stop()


# ==========================================
# 4. HEADER
# ==========================================

st.markdown("""
<div class="hero">
    <h1>🚧 RoadVision</h1>
    <p>
        Intelligent Road Pothole Detection System
        <br>
        Powered by YOLOv8 and Computer Vision
    </p>
</div>
""", unsafe_allow_html=True)


# ==========================================
# 5. SIDEBAR SETTINGS
# ==========================================

with st.sidebar:

    st.markdown("## ⚙️ Detection Settings")

    confidence = st.slider(
        "Confidence Threshold",
        min_value=0.10,
        max_value=0.90,
        value=0.45,
        step=0.05,
        help="Minimum confidence required for a detection."
    )

    iou_threshold = st.slider(
        "NMS IoU Threshold",
        min_value=0.10,
        max_value=0.90,
        value=0.40,
        step=0.05,
        help="Controls suppression of overlapping detections."
    )

    image_size = st.select_slider(
        "Image Size",
        options=[320, 416, 512, 640, 768, 960],
        value=640
    )

    st.divider()

    st.markdown("### 🤖 Model Information")
    st.write("**Architecture:** YOLOv8")
    st.write("**Detection Class:** Pothole")
    st.write("**Input Size:**", image_size)

    st.markdown(
        '<p class="status">● Trained model loaded successfully</p>',
        unsafe_allow_html=True
    )


# ==========================================
# 6. IMAGE UPLOAD
# ==========================================

st.markdown(
    '<div class="section-title">📤 Upload Road Image</div>',
    unsafe_allow_html=True
)

uploaded_file = st.file_uploader(
    "Choose a road image",
    type=["jpg", "jpeg", "png"],
    help="Upload a JPG or PNG road image."
)


# ==========================================
# 7. IMAGE PROCESSING AND DETECTION
# ==========================================

if uploaded_file is None:

    st.info(
        "Upload a road image to begin pothole detection."
    )

    st.markdown("""
    ### How it works

    1. Upload a road image.
    2. Configure detection parameters.
    3. Click Detect Potholes.
    4. View bounding boxes and confidence scores.
    5. Download the annotated result.
    """)

else:

    try:
        original_image = Image.open(uploaded_file).convert("RGB")

        st.markdown(
            '<div class="section-title">🖼️ Image Preview</div>',
            unsafe_allow_html=True
        )

        st.image(
            original_image,
            caption="Uploaded Road Image",
            use_container_width=True
        )

        st.caption(
            f"Image dimensions: {original_image.width} × "
            f"{original_image.height} pixels"
        )

        detect_button = st.button(
            "🔍 Detect Potholes",
            type="primary",
            use_container_width=True
        )

        if detect_button:

            with st.spinner(
                "Analyzing road image and detecting potholes..."
            ):

                results = model.predict(
                    source=original_image,
                    conf=confidence,
                    iou=iou_threshold,
                    imgsz=image_size,
                    max_det=100,
                    verbose=False
                )

                result = results[0]

                # Convert YOLO BGR output to RGB
                annotated_bgr = result.plot(
                    line_width=3,
                    labels=True,
                    conf=True
                )

                annotated_rgb = np.ascontiguousarray(
                    annotated_bgr[:, :, ::-1]
                )

                detected_image = Image.fromarray(annotated_rgb)

                # Extract detection information
                detection_data = []

                for index, box in enumerate(result.boxes):

                    class_id = int(box.cls[0])
                    class_name = model.names[class_id]

                    score = float(box.conf[0])

                    x1, y1, x2, y2 = box.xyxy[0].tolist()

                    detection_data.append({
                        "Detection": index + 1,
                        "Class": class_name,
                        "Confidence (%)": round(score * 100, 2),
                        "X1": round(x1, 1),
                        "Y1": round(y1, 1),
                        "X2": round(x2, 1),
                        "Y2": round(y2, 1)
                    })

                detection_df = pd.DataFrame(detection_data)

                total_detections = len(detection_data)

                avg_confidence = (
                    detection_df["Confidence (%)"].mean()
                    if total_detections > 0 else 0
                )

                # Save result in session state
                st.session_state["detected_image"] = detected_image
                st.session_state["detection_df"] = detection_df
                st.session_state["total_detections"] = total_detections
                st.session_state["avg_confidence"] = avg_confidence

    except Exception as e:
        st.error(f"Image processing failed: {e}")


# ==========================================
# 8. DISPLAY RESULTS
# ==========================================

if "detected_image" in st.session_state:

    detected_image = st.session_state["detected_image"]
    detection_df = st.session_state["detection_df"]

    total_detections = st.session_state["total_detections"]
    avg_confidence = st.session_state["avg_confidence"]

    st.divider()

    st.markdown(
        '<div class="section-title">📊 Detection Dashboard</div>',
        unsafe_allow_html=True
    )

    # Metric cards
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Total Potholes",
            total_detections
        )

    with col2:
        st.metric(
            "Average Confidence",
            f"{avg_confidence:.2f}%"
        )

    with col3:
        st.metric(
            "Detection Status",
            "Detected" if total_detections else "No Detection"
        )

    # Image comparison
    st.markdown(
        '<div class="section-title">🔎 Visual Comparison</div>',
        unsafe_allow_html=True
    )

    original_col, result_col = st.columns(2)

    with original_col:
        st.markdown("**Original Image**")
        st.image(
            original_image,
            use_container_width=True
        )

    with result_col:
        st.markdown("**Detected Potholes**")
        st.image(
            detected_image,
            use_container_width=True
        )

    # Detection details
    st.markdown(
        '<div class="section-title">📋 Detection Details</div>',
        unsafe_allow_html=True
    )

    if total_detections > 0:

        st.dataframe(
            detection_df,
            use_container_width=True,
            hide_index=True
        )

        # Download annotated image
        image_buffer = io.BytesIO()

        detected_image.save(
            image_buffer,
            format="JPEG",
            quality=95
        )

        st.download_button(
            label="⬇️ Download Detected Image",
            data=image_buffer.getvalue(),
            file_name="pothole_detection_result.jpg",
            mime="image/jpeg",
            use_container_width=True
        )

    else:

        st.warning(
            "No potholes detected. Try adjusting the confidence threshold "
            "or uploading another image."
        )


# ==========================================
# 9. FOOTER
# ==========================================

st.divider()

st.caption(
    "RoadVision | YOLOv8 Computer Vision Project | "
    "Automated Road Pothole Detection"
)
