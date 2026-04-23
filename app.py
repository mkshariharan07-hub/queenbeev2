import streamlit as st
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import os
import pandas as pd
import plotly.express as px

# Safe import for YOLO
try:
    from ultralytics import YOLO
except Exception as e:
    st.error(f"❌ Failed to import Ultralytics: {e}")
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
st.title("🐝 Weed Deduction System")
st.caption("Autonomous Weed Detection System")

# ==========================================
# LOAD MODEL
# ==========================================
model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "best.pt")

@st.cache_resource
def load_model():
    if not os.path.exists(model_path):
        return None, f"File not found at: {model_path}"
    try:
        m = YOLO(model_path)
        return m, None
    except Exception as e:
        return None, str(e)

model, model_error = load_model()

if model is None:
    st.error(f"❌ Model failed to load: {model_error}")
    st.stop()
else:
    st.success("✅ YOLO Model Loaded")

# ==========================================
# SIDEBAR
# ==========================================
confidence = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.5)

# ==========================================
# IMAGE UPLOAD ONLY
# ==========================================
st.subheader("📥 Upload Image")
uploaded_file = st.file_uploader("Upload an image for weed detection", type=["jpg", "jpeg", "png"])

image = None
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

# ==========================================
# PROCESS
# ==========================================
if image is not None and st.button("🚀 Run Detection"):

    with st.spinner("Running YOLO detection..."):
        try:
            results = model.predict(source=image, conf=confidence)
            res = results[0]
        except Exception as e:
            st.error("Inference failed")
            st.exception(e)
            st.stop()

    # Draw boxes using PIL (no cv2 dependency for rendering)
    img_draw = image.copy()
    draw = ImageDraw.Draw(img_draw)
    names = model.names
    detected = []

    for box in res.boxes:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        cls_id = int(box.cls[0].item())
        conf_score = float(box.conf[0].item())
        label = names[cls_id]

        # Draw bounding box
        draw.rectangle([int(x1), int(y1), int(x2), int(y2)], outline=(0, 255, 0), width=3)
        draw.text((int(x1), int(y1) - 15), f"{label} {conf_score:.2f}", fill=(0, 255, 0))

        detected.append({
            "Label": label,
            "Confidence": round(conf_score, 3),
            "X": float((x1 + x2) / 2),
            "Y": float((y1 + y2) / 2)
        })

    # ==========================================
    # OUTPUT
    # ==========================================
    st.subheader("📊 Detection Output")
    st.image(img_draw, use_container_width=True)

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
            color="Label",
            title="Detected Weed Positions"
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
        st.info("No weeds detected in the uploaded image.")