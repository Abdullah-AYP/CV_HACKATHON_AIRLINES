import streamlit as st
from ultralytics import YOLO
from PIL import Image
import os
import cv2
import numpy as np

# ================= CONFIGURATION =================
# 📍 EXACT MODEL PATH (Using 'r' to fix backslash errors)
MODEL_PATH = r"C:\Users\ZAH\Pictures\CV_HACKATHON_AIRLINES\Hack\Hackathon2_scripts\Hackathon2_scripts\runs\detect\runs\train\final_submission_model7\weights\best.pt"
# =================================================

st.set_page_config(page_title="Space Station AI", page_icon="🛰️", layout="wide")

# --- Sidebar ---
st.sidebar.title("⚙️ Control Panel")
conf_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.25, 0.05)
use_tta = st.sidebar.checkbox("Enable High-Accuracy Mode (TTA)", value=True, help="Runs the model 3x per image for maximum precision.")

st.sidebar.markdown("---")
st.sidebar.info("Model: YOLOv8 Custom\n\nResolution: 1024px\n\nStatus: 🟢 Online")

# --- Main Page ---
st.title("🛰️ Space Station Safety Monitor")
st.markdown("### Real-time compliance detection for ISS Modules")

# Check if model exists before loading
@st.cache_resource
def load_model():
    return YOLO(MODEL_PATH)

if os.path.exists(MODEL_PATH):
    try:
        model = load_model()
        st.success("✅ AI Model Loaded Successfully")
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        st.stop()
else:
    st.error(f"❌ Model not found at path:\n{MODEL_PATH}")
    st.info("Please check the path in the code.")
    st.stop()

# --- Input Section ---
col1, col2 = st.columns(2)

with col1:
    uploaded_file = st.file_uploader("📂 Upload Inspection Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Display Input
    image = Image.open(uploaded_file)
    with col1:
        st.image(image, caption="Original Image", use_container_width=True)
    
    # Run Prediction automatically or via button
    if st.button("🔍 Scan for Hazards", type="primary"):
        with st.spinner("Analyzing... (High-Res Mode Active)"):
            # Run Inference with Winning Settings
            results = model.predict(
                source=image, 
                conf=conf_threshold,
                imgsz=1024,      # 🏆 Force High Resolution
                augment=use_tta  # 🏆 Force TTA for Demo
            )
            
            # Plot Results
            res_plotted = results[0].plot()
            
            # 🎨 CRITICAL FIX: Convert BGR (OpenCV) to RGB (Streamlit)
            res_plotted_rgb = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)
            
            with col2:
                st.image(res_plotted_rgb, caption="AI Analysis Result", use_container_width=True)
            
            # Show Stats
            boxes = results[0].boxes
            if len(boxes) > 0:
                st.success(f"✅ Detection Complete: Found {len(boxes)} items.")
                
                # List unique items found with confidence
                st.write("### 📦 Detected Inventory:")
                for box in boxes:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    name = model.names[cls_id]
                    
                    if conf > 0.8:
                        st.markdown(f"- **{name}** (Confidence: {conf:.1%}) 🟢")
                    else:
                        st.markdown(f"- **{name}** (Confidence: {conf:.1%}) 🟡")

            else:
                st.warning("⚠️ No safety equipment detected! Please verify module contents.")

# --- The "Bonus Points" Section ---
st.markdown("---")
st.subheader("🚀 Duality Falcon Integration Plan")
st.markdown("""
**Objective:** Maintain 99.9% accuracy as station equipment evolves.

1.  **Digital Twin Sync:** When a new *Fire Extinguisher* model is sent to the station, we update its 3D asset in Duality Falcon.
2.  **Synthetic Generation:** Falcon generates 5,000 photorealistic images of the new extinguisher in zero-gravity lighting.
3.  **Continuous Training:** This app pulls the new `best.pt` model trained on that synthetic data via OTA (Over-the-Air) update.
""")