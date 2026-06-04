import streamlit as st
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.metrics import dice_loss, dice_coefficient, iou_metric

st.set_page_config(
    page_title="RetinaVision-AI",
    page_icon="👁️",
    layout="wide"
)

st.markdown("""
<style>
    .main-title {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1a1a2e;
        text-align: center;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.1rem;
        color: #555;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
        border: 1px solid #dee2e6;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #0066cc;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #666;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">👁️ RetinaVision-AI</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Retinal Blood Vessel Segmentation using U-Net · DRIVE Dataset</div>', unsafe_allow_html=True)

@st.cache_resource
def load_model():
    model_path = "outputs/retinavision_best_model.h5"
    if not os.path.exists(model_path):
        return None
    model = tf.keras.models.load_model(
        model_path,
        custom_objects={
            "dice_loss": dice_loss,
            "dice_coefficient": dice_coefficient,
            "iou_metric": iou_metric
        }
    )
    return model

def preprocess_image(image):
    image = np.array(image.convert("RGB"))
    image = cv2.resize(image, (512, 512), interpolation=cv2.INTER_AREA)
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge([l, a, b])
    image = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    image = image.astype(np.float32) / 255.0
    return image

def predict(model, image_array):
    batch = np.expand_dims(image_array, axis=0)
    pred = model.predict(batch, verbose=0)
    mask = pred[0, :, :, 0]
    binary = (mask > 0.5).astype(np.float32)
    return mask, binary

def create_overlay(image, binary_mask):
    overlay = image.copy()
    vessel_pixels = binary_mask > 0.5
    overlay[vessel_pixels, 0] = 0.0
    overlay[vessel_pixels, 1] = 1.0
    overlay[vessel_pixels, 2] = 0.0
    return overlay

with st.sidebar:
    st.header("About")
    st.markdown("""
    **RetinaVision-AI** segments blood vessels 
    in retinal fundus images using U-Net.
    
    **Dataset:** DRIVE (40 fundus images)
    
    **Architecture:** U-Net
    - 4 Encoder blocks
    - Bottleneck (1024 filters)
    - 4 Decoder blocks
    - Skip connections
    
    **Loss:** Dice Loss
    
    **Metrics:** Dice Coefficient, IoU
    
    **Clinical Applications:**
    - Diabetic Retinopathy screening
    - Glaucoma detection
    - Hypertension assessment
    - Cardiovascular risk analysis
    """)
    
    st.header("How to Use")
    st.markdown("""
    1. Upload a retinal fundus image
    2. Click **Segment Vessels**
    3. View the segmentation results
    """)

model = load_model()

if model is None:
    st.warning("⏳ Model not trained yet. Training is in progress. Please wait for training to complete, then refresh this page.")
    st.info("Model will be saved to: `outputs/retinavision_best_model.h5`")
    st.stop()
else:
    st.success("✅ Model loaded successfully")

uploaded_file = st.file_uploader(
    "Upload a retinal fundus image",
    type=["png", "jpg", "jpeg", "tif", "tiff"],
    help="Upload a retinal fundus photograph for vessel segmentation"
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    
    col1, col2 = st.columns([1, 1])
    with col1:
        st.subheader("Uploaded Image")
        st.image(image, use_column_width=True)
    
    if st.button("🔬 Segment Blood Vessels", type="primary", use_container_width=True):
        with st.spinner("Segmenting blood vessels..."):
            processed = preprocess_image(image)
            prob_mask, binary_mask = predict(model, processed)
            overlay = create_overlay(processed, binary_mask)
            
            vessel_pixels = np.sum(binary_mask)
            total_pixels = binary_mask.size
            vessel_density = (vessel_pixels / total_pixels) * 100

        st.markdown("---")
        st.subheader("Segmentation Results")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**Original Fundus Image**")
            st.image(processed, use_column_width=True, clamp=True)
        
        with col2:
            st.markdown("**Predicted Vessel Mask**")
            st.image(binary_mask, use_column_width=True, clamp=True)
        
        with col3:
            st.markdown("**Overlay (Green = Vessels)**")
            st.image(overlay, use_column_width=True, clamp=True)
        
        st.markdown("---")
        st.subheader("Analysis")
        
        m1, m2, m3 = st.columns(3)
        with m1:
            st.metric("Vessel Density", f"{vessel_density:.2f}%",
                     help="Percentage of image pixels classified as vessels")
        with m2:
            st.metric("Vessel Pixels", f"{int(vessel_pixels):,}",
                     help="Total number of pixels classified as blood vessels")
        with m3:
            st.metric("Image Size", "512 × 512",
                     help="Input image resized to standard U-Net input")
        
        st.markdown("---")
        st.subheader("Clinical Significance")
        
        if vessel_density < 8:
            st.warning("⚠️ Low vessel density detected. This may indicate poor image quality or potential vascular abnormalities.")
        elif vessel_density > 20:
            st.warning("⚠️ High vessel density detected. This may indicate neovascularisation — a sign of diabetic retinopathy.")
        else:
            st.success("✅ Vessel density within normal range (8–20%). No obvious gross abnormalities detected.")
        
        st.info("""
        **Note:** This tool is for research and educational purposes only. 
        Clinical diagnosis must be performed by a qualified ophthalmologist.
        """)
        
        mask_uint8 = (binary_mask * 255).astype(np.uint8)
        mask_pil = Image.fromarray(mask_uint8)
        import io
        buf = io.BytesIO()
        mask_pil.save(buf, format="PNG")
        st.download_button(
            label="⬇️ Download Segmentation Mask",
            data=buf.getvalue(),
            file_name="vessel_mask.png",
            mime="image/png"
        )

else:
    st.markdown("---")
    st.markdown("### How it works")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("**1. Input**\n\nRetinal fundus photograph uploaded by user")
    with col2:
        st.markdown("**2. Preprocess**\n\nResize to 512×512, apply CLAHE contrast enhancement")
    with col3:
        st.markdown("**3. U-Net**\n\nEncoder extracts features, decoder reconstructs vessel map")
    with col4:
        st.markdown("**4. Output**\n\nBinary mask showing vessel locations + clinical analysis")
