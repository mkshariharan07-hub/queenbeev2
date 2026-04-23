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
    page_title="Queen Bee | Advanced Weed Detection",
    page_icon="🐝",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Premium Look
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    .stMetric {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
    }
    .weed-card {
        padding: 20px;
        border-radius: 15px;
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        color: white;
        margin-bottom: 20px;
    }
    .header-style {
        font-size: 42px;
        font-weight: 700;
        color: #2e7d32;
        margin-bottom: 0px;
    }
    </style>
    """, unsafe_allow_html=True)

# ==========================================
# SIDEBAR & MODEL LOAD
# ==========================================
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2312/2312218.png", width=100)
    st.title("Settings")
    
    crop_type = st.selectbox(
        "Select Crop Field",
        ["Paddy (Rice)", "Sugarcane", "Banana", "Other"]
    )
    
    confidence = st.slider("Confidence Threshold", 0.1, 1.0, 0.45)
    
    st.divider()
    st.info(f"System Optimized for: **{crop_type}**")

# Load Model
model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "best.pt")

@st.cache_resource
def load_model():
    if not os.path.exists(model_path):
        return None, f"Model file 'best.pt' not found at {model_path}"
    try:
        m = YOLO(model_path)
        return m, None
    except Exception as e:
        return None, str(e)

model, model_error = load_model()

# ==========================================
# MAIN UI
# ==========================================
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown('<p class="header-style">🐝 Queen Bee Drone Analytics</p>', unsafe_allow_html=True)
    st.markdown("### Intelligent Multi-Crop Weed Detection System")
    st.caption(f"Precision Agriculture Suite • Analyzing {crop_type} fields")

if model is None:
    st.error(f"❌ System Offline: {model_error}")
    st.stop()

st.divider()

# Image Upload
uploaded_file = st.file_uploader("📥 Upload Field Imagery (Drone/Mobile)", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    
    col_img, col_act = st.columns([3, 1])
    
    with col_img:
        st.image(image, caption="Current Field Preview", use_column_width=True)
    
    with col_act:
        st.markdown("#### Ready to Process?")
        if st.button("🚀 Run AI Diagnostic", use_container_width=True):
            
            with st.spinner(f"Scanning {crop_type} field for weed intrusion..."):
                try:
                    results = model.predict(source=image, conf=confidence)
                    res = results[0]
                except Exception as e:
                    st.error("Diagnostic failed")
                    st.exception(e)
                    st.stop()

            # Process Results
            img_draw = image.copy()
            draw = ImageDraw.Draw(img_draw)
            names = model.names
            detected = []

            # Sophisticated Drawing Palette
            colors = ["#FF3838", "#FF9D97", "#FF701F", "#FFB21D", "#CFD231", "#48F90A", "#92CC17", "#3DDB86", "#1A9334", "#00D4BB"]

            for i, box in enumerate(res.boxes):
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                cls_id = int(box.cls[0].item())
                conf_score = float(box.conf[0].item())
                label = names[cls_id]
                color = colors[cls_id % len(colors)]

                # Draw Bounding Box
                draw.rectangle([int(x1), int(y1), int(x2), int(y2)], outline=color, width=4)
                draw.text((int(x1), int(y1) - 15), f"{label} {conf_score:.2f}", fill=color)

                detected.append({
                    "ID": i + 1,
                    "Type": label,
                    "Confidence": round(conf_score, 3),
                    "Center X": float((x1 + x2) / 2),
                    "Center Y": float((y1 + y2) / 2)
                })

            # ==========================================
            # DASHBOARD OUTPUT
            # ==========================================
            st.divider()
            st.header("📊 Diagnostic Dashboard")
            
            # Metrics
            m1, m2, m3 = st.columns(3)
            m1.metric("Total Weeds Found", len(detected), delta=None)
            m2.metric("Detection Confidence", f"{confidence*100:.0f}%", delta=None)
            m3.metric("Crop Type", crop_type[:10], delta=None)

            tab1, tab2, tab3 = st.tabs(["🖼 Annotated Image", "📋 Raw Data", "📈 Statistics"])
            
            with tab1:
                st.image(img_draw, caption="AI Annotated Detection Map", use_column_width=True)
            
            with tab2:
                if detected:
                    df = pd.DataFrame(detected)
                    st.dataframe(df, use_container_width=True, hide_index=True)
                else:
                    st.info("Clear field detected. No weeds found.")

            with tab3:
                if detected:
                    df = pd.DataFrame(detected)
                    
                    # Count by label
                    count_df = df['Type'].value_counts().reset_index()
                    count_df.columns = ['Weed Type', 'Count']
                    
                    fig_bar = px.bar(count_df, x='Weed Type', y='Count', color='Weed Type', 
                                    title="Weed Distribution Count", template="plotly_white")
                    st.plotly_chart(fig_bar, use_container_width=True)
                    
                    fig_scatter = px.scatter(df, x="Center X", y="Center Y", size="Confidence", 
                                           color="Type", title="Spatial Distribution Map")
                    st.plotly_chart(fig_scatter, use_container_width=True)
                else:
                    st.success("🎉 Field is healthy and clear of weeds!")

            # Download Option
            if detected:
                csv = df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="⬇️ Export Diagnostic Report (CSV)",
                    data=csv,
                    file_name=f"weed_report_{crop_type}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
else:
    # Placeholder when no image is uploaded
    st.info("☝️ Please upload a drone or mobile image to start the weed detection analysis.")
    
    # Showcase Cards
    sc1, sc2, sc3 = st.columns(3)
    with sc1:
        st.markdown('<div class="weed-card"><b>Paddy Fields</b><br>Optimized for rice paddy weed variants.</div>', unsafe_allow_html=True)
    with sc2:
        st.markdown('<div class="weed-card"><b>Sugarcane</b><br>High-precision detection in dense foliage.</div>', unsafe_allow_html=True)
    with sc3:
        st.markdown('<div class="weed-card"><b>Banana Plantation</b><br>Large-leaf specimen analysis.</div>', unsafe_allow_html=True)