import streamlit as st
from PIL import Image
import numpy as np
import os
import cv2
import pandas as pd
import plotly.express as px

# Safe import for YOLO
try:
    from ultralytics import YOLO
except ImportError:
    st.error("Ultralytics not installed. Please check requirements.txt")
    st.stop()

# ==========================================
# PAGE CONFIG
# ==========================================
st.set_page_config(
    page_title="MULTI WEED DETECTION",
    page_icon="🐝",
    layout="wide"
)

# ==========================================
# HEADER
# ==========================================
st.title("🐝 Queen Bee Drone Analytics")
st.caption("Autonomous Weed Detection System")

# ==========================================
# LOAD MODEL
# ==========================================
model_path = "best.pt"

@st.cache_resource
def load_model():
    if not os.path.exists(model_path):
        return None
    try:
        return YOLO(model_path)
    except Exception:
        return None

model = load_model()

if model is None:
    st.error("❌ Model file 'best.pt' not found or failed to load")
    st.stop()
else:
    st.success("✅ YOLO Model Loaded")

# ==========================================
# SIDEBAR
# ==========================================
confidence = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.5)

# ==========================================
# INPUT
# ==========================================
st.subheader("📥 Upload Image")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

image = None
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

# ==========================================
# PROCESS
# ==========================================
if image is not None and st.button("🚀 Run Detection"):

    with st.spinner("Running YOLO..."):
        try:
            results = model.predict(source=image, conf=confidence)
            res = results[0]
        except Exception as e:
            st.error("Inference failed")
            st.exception(e)
            st.stop()

    # Convert image
    img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    names = model.names

    detected = []

    for box in res.boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        cls_id = int(box.cls[0].item())
        conf_score = float(box.conf[0].item())

        label = names[cls_id]

        # Draw box
        cv2.rectangle(img_cv, (int(x1), int(y1)), (int(x2), int(y2)), (0,255,0), 2)

        cv2.putText(
            img_cv,
            f"{label} {conf_score:.2f}",
            (int(x1), int(y1)-10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0,255,0),
            2
        )

        detected.append({
            "Label": label,
            "Confidence": conf_score,
            "X": float((x1+x2)/2),
            "Y": float((y1+y2)/2)
        })

    output_img = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))

    # ==========================================
    # OUTPUT
    # ==========================================
    st.subheader("📊 Detection Output")
    st.image(output_img, use_container_width=True)

    if detected:
        df = pd.DataFrame(detected)

        st.subheader("📋 Detection Table")
        st.dataframe(df, use_container_width=True)

        st.subheader("📈 Visualization")
        fig = px.scatter(
            df,
            x="X",
            y="Y",
            size="Confidence",
            color="Label"
        )
        st.plotly_chart(fig, use_container_width=True)

        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download CSV",
            csv,
            "detections.csv",
            "text/csv"
        )
    else:
        st.info("No objects detected.")