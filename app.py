import streamlit as st
from PIL import Image
import numpy as np
import os
import io
try:
    import cv2
except ImportError:
    import subprocess, sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "opencv-python-headless==4.9.0.80"])
    import cv2
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
try:
    from ultralytics import YOLO
except ImportError:
    st.error("Ultralytics library not found. Please run: pip install -r requirements.txt")

# ==========================================
# PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="MULTI WEED DETECTION",
    page_icon="🐝",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# PREMIUM UI/UX CUSTOM CSS
# ==========================================
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');
    
    :root {
        --primary-gold: #FFC107;
        --secondary-gold: #FF9800;
        --dark-bg: #0A0A0A;
        --panel-bg: #141414;
        --text-main: #E2E8F0;
        --text-muted: #94A3B8;
        --accent: #22C55E;
    }
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    .stApp {
        background-color: var(--dark-bg);
        color: var(--text-main);
    }
    
    /* Headers */
    h1, h2, h3 {
        color: var(--primary-gold) !important;
        font-weight: 700 !important;
        letter-spacing: 1px;
    }
    
    .main-title {
        background: -webkit-linear-gradient(45deg, #FFC107, #FF9800);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3rem !important;
        margin-bottom: 0px !important;
        padding-bottom: 0px !important;
    }
    
    .sub-title {
        color: var(--text-muted) !important;
        font-size: 1.2rem;
        font-weight: 300;
        margin-top: -10px;
        margin-bottom: 2rem;
    }
    
    /* Metrics & Cards */
    .metric-card {
        background: linear-gradient(145deg, #1A1A1A, #121212);
        padding: 1.5rem;
        border-radius: 16px;
        border: 1px solid rgba(255, 193, 7, 0.1);
        box-shadow: 0 10px 30px rgba(0,0,0,0.5);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        margin-bottom: 1rem;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 35px rgba(255, 193, 7, 0.15);
        border: 1px solid rgba(255, 193, 7, 0.3);
    }
    
    .metric-title {
        font-size: 1.1rem;
        color: var(--text-muted);
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 0.5rem;
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        color: var(--text-main);
    }
    
    .metric-value.gold { color: var(--primary-gold); }
    .metric-value.green { color: var(--accent); }
    
    /* Buttons */
    .stButton>button {
        background: linear-gradient(90deg, var(--secondary-gold), var(--primary-gold));
        color: #000;
        font-weight: 700;
        font-size: 1.1rem;
        border-radius: 50px;
        padding: 0.6rem 2.5rem;
        border: none;
        box-shadow: 0 4px 15px rgba(255, 193, 7, 0.4);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        width: 100%;
        margin-top: 1rem;
    }
    
    .stButton>button:hover {
        transform: scale(1.02) translateY(-2px);
        box-shadow: 0 8px 25px rgba(255, 193, 7, 0.6);
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: var(--panel-bg);
        border-right: 1px solid rgba(255, 193, 7, 0.05);
    }
    
    /* File Uploader */
    .stFileUploader label {
        color: var(--primary-gold);
        font-weight: 600;
    }
    
    div[data-testid="stFileUploadDropzone"] {
        background-color: rgba(255, 193, 7, 0.02);
        border: 2px dashed rgba(255, 193, 7, 0.3);
        border-radius: 12px;
        transition: all 0.3s ease;
    }
    
    div[data-testid="stFileUploadDropzone"]:hover {
        background-color: rgba(255, 193, 7, 0.05);
        border-color: var(--primary-gold);
    }
    
    /* Summary Tags */
    .tag-paddy { background: rgba(34, 197, 94, 0.15); color: #4ADE80; padding: 5px 10px; border-radius: 6px; border: 1px solid #22C55E; display: inline-block; margin: 4px; }
    .tag-banana { background: rgba(234, 179, 8, 0.15); color: #FACC15; padding: 5px 10px; border-radius: 6px; border: 1px solid #EAB308; display: inline-block; margin: 4px; }
    .tag-other { background: rgba(148, 163, 184, 0.15); color: #CBD5E1; padding: 5px 10px; border-radius: 6px; border: 1px solid #94A3B8; display: inline-block; margin: 4px; }
    
    .error-card {
        background: rgba(255, 69, 0, 0.15);
        border-left: 4px solid #ff4500;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
        color: #ffddd2;
    }
    .placeholder {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        height: 300px;
        color: var(--text-muted);
    }
    
    hr {
        border-color: rgba(255, 255, 255, 0.1);
    }
    
    #MainMenu {visibility: hidden;}
    header {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# ==========================================
# APP HEADER
# ==========================================
st.markdown('''
<h1 class="main-title">🐝 Queen Bee Drone Analytics</h1>
<p class="sub-title">Autonomous 3D Weed Detection &amp; Eradication System</p>
''', unsafe_allow_html=True)

# ==========================================
# DRONE CONFIGURATION (SIDEBAR)
# ==========================================
st.sidebar.markdown("""
<div style="text-align: center; margin-bottom: 20px;">
    <h2>🚁 Drone Fleet Parameters</h2>
    <hr style="border-color: #333;">
</div>
""", unsafe_allow_html=True)

model_path = os.path.join(os.getcwd(), 'best.pt')

@st.cache_resource
def load_yolo_model(path):
    if os.path.exists(path):
        try:
            return YOLO(path)
        except Exception as e:
            return None
    return None

# Load YOLO model with caching and status feedback
with st.spinner('Loading YOLO model...'):
    model = load_yolo_model(model_path)
if model:
    st.success('YOLO model loaded successfully 🎉')
else:
    st.error('⚠️ YOLO model could not be loaded. Please ensure `best.pt` is present in the app directory.')
    st.stop()

st.sidebar.markdown("**Sensor Sensitivity**")
confidence_threshold = st.sidebar.slider("AI Match Threshold", 0.1, 1.0, 0.50, 0.05, 
                                        help="Higher values reduce false positives for drone targeting.")

st.sidebar.markdown("---")
st.sidebar.info("""
**Autonomous Mission Protocols**  
Queen Bee integrates advanced YOLOv8 telemetry with 3D spatial mapping to coordinate drone strikes against **Banana Weed**, **Paddy Weed**, and **Sugarcane Weed**.
""")


# ==========================================
# MAIN WORKSPACE
# ==========================================

# File Upload Section
st.markdown("### 📥 Input Telemetry")
input_source = st.radio("Select telemetry source:", ("File Upload", "Live Camera"), horizontal=True)

image = None

if input_source == "File Upload":
    uploaded_file = st.file_uploader("Upload environmental capture (JPG, JPEG, PNG)...", type=['jpg', 'jpeg', 'png'])
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
elif input_source == "Live Camera":
    st.info("Mobile users: You may switch between front/back cameras using your device/browser's camera settings prompt.")
    camera_capture = st.camera_input("Take a photo of the environment")
    if camera_capture is not None:
        image = Image.open(camera_capture).convert("RGB")

if image is not None:
    
    # Layout Strategy
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-title">Original Capture</div>
        </div>
        """, unsafe_allow_html=True)
        st.image(image, use_container_width=True)
        
    # Execution Block
    if 'run_analysis' not in st.session_state:
        st.session_state.run_analysis = False
        
    def trigger_analysis():
        st.session_state.run_analysis = True
        
    st.button("SCAN ENVIRONMENT 🚀", on_click=trigger_analysis)
    
    if st.session_state.run_analysis:
        if model is None:
            st.error("Cannot proceed: YOLO Model `best.pt` is missing from the directory.")
        else:
            with st.spinner("Initializing Deep Neural Network..."):
                try:
                    # Run YOLO Inference
                    results = model.predict(source=image, conf=confidence_threshold)
                    res = results[0]  # First image results
                except Exception as e:
                    st.exception(e)
                    st.error('Failed to run YOLO inference. Please check the uploaded image and model.')
                    st.stop()
                
                # Draw circular bounding areas instead of boxes
                img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                names = model.names
                boxes = res.boxes
                
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    
                    x1_i, y1_i, x2_i, y2_i = int(x1), int(y1), int(x2), int(y2)
                    
                    cls_id = int(box.cls[0].item())
                    conf = box.conf[0].item()
                    label = names[cls_id]
                    
                    # Color Mapping
                    if 'banana' in label.lower():
                        color = (0, 215, 255) # Amber/Yellow in BGR
                        display_name = "Banana Weed"
                    elif 'paddy' in label.lower():
                        color = (50, 255, 50) # Green in BGR
                        display_name = "Paddy Weed"
                    else:
                        color = (255, 150, 50) # Blueish in BGR
                        display_name = "Sugarcane Weed"
                        
                    # Draw Rectangle
                    cv2.rectangle(img_cv, (x1_i, y1_i), (x2_i, y2_i), color, 4)
                    
                    # Draw Label
                    text_label = f"{display_name} {conf:.2f}"
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 0.6
                    thickness = 2
                    (text_width, text_height), baseline = cv2.getTextSize(text_label, font, font_scale, thickness)
                    
                    label_bg_y1 = y1_i - text_height - 10
                    label_bg_y2 = y1_i
                    
                    # Prevent going out of top bounds
                    if label_bg_y1 < 0:
                        label_bg_y1 = y1_i + 10
                        label_bg_y2 = label_bg_y1 + text_height + 10
                    
                    cv2.rectangle(img_cv, (x1_i, label_bg_y1), (x1_i + text_width, label_bg_y2), color, -1)
                    cv2.putText(img_cv, text_label, (x1_i, label_bg_y2 - 3), font, font_scale, (0, 0, 0), thickness)
                    
                output_image = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
                
            with col2:
                st.markdown("""
                <div class="metric-card">
                    <div class="metric-title">AI Diagnostics</div>
                </div>
                """, unsafe_allow_html=True)
                st.image(output_image, use_container_width=True)
                
            st.markdown("---")
            st.markdown("### 📊 Drone Mission Telemetry")
            
            # Map detections
            names = model.names
            detected_classes_ids = res.boxes.cls.cpu().numpy()
            detected_names = [names[int(c)].lower() for c in detected_classes_ids]
            
            # Categorize the expected outputs ("Banana weed", "Paddy weed", "Sugarcane weed")
            banana_weed_count = sum(1 for name in detected_names if 'banana' in name)
            paddy_weed_count = sum(1 for name in detected_names if 'paddy' in name)
            
            # Assuming any remaining/unclassified entities as Sugarcane Weed for the 3-output constraint
            sugarcane_weed_count = sum(1 for name in detected_names if 'sugarcane' in name)
            if sugarcane_weed_count == 0 and ('sugarcane' not in ''.join(detected_names)):
                sugarcane_weed_count = len(detected_names) - (banana_weed_count + paddy_weed_count)
            
            # Metric Row
            m1, m2, m3 = st.columns(3)
            with m1:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">Banana Weed</div>
                    <div class="metric-value gold">{banana_weed_count}</div>
                </div>
                """, unsafe_allow_html=True)
            with m2:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">Paddy Weed</div>
                    <div class="metric-value green">{paddy_weed_count}</div>
                </div>
                """, unsafe_allow_html=True)
            with m3:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-title">Sugarcane Weed</div>
                    <div class="metric-value">{sugarcane_weed_count}</div>
                </div>
                """, unsafe_allow_html=True)
                
            # Detailed Breakdown List
            if len(detected_names) > 0:
                df_data = []
                for box in res.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    cx = (x1 + x2) / 2
                    cy = (y1 + y2) / 2
                    w = x2 - x1
                    h = y2 - y1
                    area = w * h
                    cls_id = int(box.cls[0].item())
                    conf = box.conf[0].item()
                    label = names[cls_id]
                    
                    if 'banana' in label.lower():
                        display_name = "Banana Weed"
                        color_code = "#FFC107"
                    elif 'paddy' in label.lower():
                        display_name = "Paddy Weed"
                        color_code = "#22C55E"
                    else:
                        display_name = "Sugarcane Weed"
                        color_code = "#00D7FF"
                        
                    df_data.append({
                        "Threat Type": display_name,
                        "AI Confidence": float(conf),
                        "Field_X": float(cx),
                        "Field_Y": float(image.height - cy),
                        "Areal Coverage": float(area)
                    })
                    
                df = pd.DataFrame(df_data)
                
                st.markdown("### 🗺️ 3D Spatial Drone Mapping")
                st.markdown("<p style='color: var(--text-muted); margin-top: -10px;'>3D visualization of threat topography and sensor confidence gradients.</p>", unsafe_allow_html=True)
                
                fig = px.scatter_3d(df, x='Field_X', y='Field_Y', z='AI Confidence',
                                    color='Threat Type',
                                    size='Areal Coverage',
                                    hover_data=['Threat Type', 'AI Confidence'],
                                    opacity=0.8,
                                    color_discrete_map={"Banana Weed": "#FFC107", "Paddy Weed": "#22C55E", "Sugarcane Weed": "#00D7FF"})
                
                fig.update_layout(
                    margin=dict(l=0, r=0, b=0, t=0),
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='#E2E8F0'),
                    scene=dict(
                        xaxis=dict(title='Latitude X (px)', gridcolor='#333', backgroundcolor='rgba(0,0,0,0)'),
                        yaxis=dict(title='Longitude Y (px)', gridcolor='#333', backgroundcolor='rgba(0,0,0,0)'),
                        zaxis=dict(title='Identification Confidence', gridcolor='#333', backgroundcolor='rgba(10,10,10,1)')
                    ),
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="right",
                        x=1
                    )
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                st.markdown("### 📋 Mission Output Report")
                st.markdown("<p style='color: var(--text-muted); margin-top: -10px;'>Extracted intelligence data grid suitable for autonomous eradicator drone upload.</p>", unsafe_allow_html=True)
                st.dataframe(df.style.background_gradient(cmap='viridis', subset=['AI Confidence']), use_container_width=True)
                
                col_btn, _ = st.columns([1, 1])
                with col_btn:
                    csv = df.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="⬇️ Export Targeting Matrix (CSV)",
                        data=csv,
                        file_name='queen_bee_targeting_matrix.csv',
                        mime='text/csv',
                    )
            else:
                st.info("No threats recognized within sensor precision envelope.")
                
else:
    # Friendly placeholder when no image is uploaded yet
    st.markdown('''
        <div class="placeholder">
            <svg width="120" height="120" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm0 18c-4.41 0-8-3.59-8-8s3.59-8 8-8 8 3.59 8 8-3.59 8-8 8z" fill="var(--text-muted)"/>
                <path d="M13 7h-2v6h6v-2h-4z" fill="var(--accent)"/>
            </svg>
            <p>Awaiting image upload…</p>
        </div>
    ''', unsafe_allow_html=True)
    st.info('Please upload an image to begin analysis.')
