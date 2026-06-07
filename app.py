import streamlit as st
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf
import os
import sys
import time
import io
import base64
import streamlit.components.v1 as components

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.metrics import dice_loss, dice_coefficient, iou_metric

st.set_page_config(
    page_title="RetinaVision-AI",
    page_icon="👁️",
    layout="wide"
)

st.markdown("""
<style>
    :root {
        --rv-blue: #0b5ed7;
        --rv-blue-dark: #073b7a;
        --rv-teal: #11b7a4;
        --rv-teal-soft: #e8fbf8;
        --rv-sky: #edf7ff;
        --rv-white: #ffffff;
        --rv-text: #17324d;
        --rv-muted: #64748b;
        --rv-border: #dbeafe;
        --rv-shadow: 0 18px 45px rgba(14, 116, 144, 0.12);
    }

    .stApp {
        background:
            radial-gradient(circle at top left, rgba(17, 183, 164, 0.14), transparent 34rem),
            linear-gradient(180deg, #f7fcff 0%, #ffffff 42%, #f8fbff 100%);
        color: var(--rv-text);
    }

    section[data-testid="stSidebar"] {
        display: none;
    }

    .block-container {
        padding-top: 1.2rem;
        padding-bottom: 2rem;
        max-width: 1280px;
    }

    h1, h2, h3 {
        color: var(--rv-text);
        letter-spacing: -0.03em;
    }

    div[data-testid="stFileUploader"] {
        background: var(--rv-white);
        border: 1px dashed rgba(11, 94, 215, 0.35);
        border-radius: 20px;
        padding: 1rem;
        box-shadow: 0 12px 30px rgba(15, 23, 42, 0.05);
    }

    div[data-testid="stButton"] > button {
        border-radius: 14px;
        min-height: 3.2rem;
        font-weight: 700;
        background: linear-gradient(135deg, var(--rv-blue), var(--rv-teal));
        border: 0;
        box-shadow: 0 12px 26px rgba(11, 94, 215, 0.22);
    }

    div[data-testid="stDownloadButton"] > button {
        border-radius: 14px;
        min-height: 3rem;
        font-weight: 700;
        color: var(--rv-blue-dark);
        border-color: rgba(11, 94, 215, 0.24);
    }

    .hero {
        position: relative;
        overflow: hidden;
        padding: 1.85rem;
        border-radius: 28px;
        background:
            linear-gradient(135deg, rgba(7, 59, 122, 0.96), rgba(11, 94, 215, 0.9) 52%, rgba(17, 183, 164, 0.92)),
            url("data:image/svg+xml,%3Csvg width='120' height='120' viewBox='0 0 120 120' xmlns='http://www.w3.org/2000/svg'%3E%3Cg fill='none' stroke='%23ffffff' stroke-opacity='0.14'%3E%3Cpath d='M0 60h120M60 0v120M20 20c22 15 58 15 80 0M20 100c22-15 58-15 80 0'/%3E%3C/g%3E%3C/svg%3E");
        color: var(--rv-white);
        box-shadow: var(--rv-shadow);
        margin-bottom: 1rem;
    }

    .hero-eyebrow {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.45rem 0.85rem;
        border-radius: 999px;
        background: rgba(255, 255, 255, 0.14);
        color: #dffcff;
        font-size: 0.86rem;
        font-weight: 700;
        margin-bottom: 0.7rem;
        border: 1px solid rgba(255, 255, 255, 0.22);
    }

    .hero-title {
        font-size: clamp(2.45rem, 5vw, 4.7rem);
        line-height: 0.95;
        font-weight: 850;
        letter-spacing: -0.06em;
        margin: 0 0 0.55rem 0;
    }

    .hero-subtitle {
        max-width: 780px;
        color: rgba(255, 255, 255, 0.86);
        font-size: 1.24rem;
        line-height: 1.65;
        margin: 0 0 0.95rem 0;
    }

    .badge-row {
        display: flex;
        flex-wrap: wrap;
        gap: 0.75rem;
        margin-top: 0.85rem;
    }

    .badge {
        display: inline-flex;
        align-items: center;
        padding: 0.48rem 0.82rem;
        border-radius: 999px;
        background: rgba(255, 255, 255, 0.16);
        border: 1px solid rgba(255, 255, 255, 0.26);
        color: #ffffff;
        font-weight: 750;
        font-size: 0.92rem;
        backdrop-filter: blur(10px);
    }

    .section-heading {
        margin: 1.25rem 0 0.65rem 0;
    }

    .section-kicker {
        color: var(--rv-teal);
        font-size: 0.82rem;
        font-weight: 800;
        letter-spacing: 0.11em;
        text-transform: uppercase;
        margin-bottom: 0.35rem;
    }

    .section-title {
        color: var(--rv-text);
        font-size: 1.65rem;
        font-weight: 820;
        margin: 0;
    }

    .info-card,
    .metric-card {
        background: rgba(255, 255, 255, 0.92);
        border: 1px solid var(--rv-border);
        border-radius: 24px;
        padding: 1rem;
        box-shadow: 0 14px 34px rgba(15, 23, 42, 0.06);
        height: 100%;
    }

    .info-card {
        min-height: 150px;
    }

    .info-icon {
        width: 42px;
        height: 42px;
        display: flex;
        align-items: center;
        justify-content: center;
        border-radius: 14px;
        background: linear-gradient(135deg, var(--rv-sky), var(--rv-teal-soft));
        color: var(--rv-blue);
        font-size: 1.3rem;
        margin-bottom: 0.65rem;
    }

    .info-title {
        font-weight: 820;
        color: var(--rv-text);
        font-size: 1.04rem;
        margin-bottom: 0.45rem;
    }

    .info-body {
        color: var(--rv-muted);
        line-height: 1.6;
        font-size: 0.94rem;
        margin: 0;
    }

    .metric-card {
        position: relative;
        overflow: hidden;
    }

    .metric-card::after {
        content: "";
        position: absolute;
        top: 0;
        right: 0;
        width: 84px;
        height: 84px;
        background: radial-gradient(circle, rgba(17, 183, 164, 0.16), transparent 70%);
    }

    .metric-value {
        font-size: 1.85rem;
        font-weight: 850;
        color: var(--rv-blue-dark);
        letter-spacing: -0.04em;
        margin-bottom: 0.2rem;
    }

    .metric-label {
        font-size: 0.82rem;
        color: var(--rv-muted);
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }

    .panel {
        background: var(--rv-white);
        border: 1px solid var(--rv-border);
        border-radius: 26px;
        padding: 1rem;
        box-shadow: 0 14px 34px rgba(15, 23, 42, 0.06);
        margin-bottom: 0.7rem;
    }

    .image-caption {
        color: var(--rv-text);
        font-weight: 780;
        margin: 0 0 0.75rem 0;
    }

    .performance-card {
        background: linear-gradient(180deg, #ffffff 0%, #f4fbff 100%);
        border: 1px solid var(--rv-border);
        border-radius: 22px;
        padding: 0.95rem;
        height: 100%;
    }

    .performance-label {
        color: var(--rv-muted);
        font-size: 0.78rem;
        font-weight: 800;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        margin-bottom: 0.35rem;
    }

    .performance-value {
        color: var(--rv-blue-dark);
        font-size: 1.35rem;
        font-weight: 850;
    }

    .workflow-step {
        background: var(--rv-white);
        border: 1px solid var(--rv-border);
        border-radius: 22px;
        padding: 0.95rem;
        min-height: 132px;
        box-shadow: 0 10px 26px rgba(15, 23, 42, 0.05);
    }

    .workflow-number {
        color: var(--rv-teal);
        font-size: 0.82rem;
        font-weight: 850;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }

    .workflow-title {
        color: var(--rv-text);
        font-weight: 820;
        margin: 0.45rem 0;
    }

    .workflow-copy {
        color: var(--rv-muted);
        font-size: 0.93rem;
        line-height: 1.55;
        margin: 0;
    }

    .status-pill {
        display: inline-flex;
        align-items: center;
        gap: 0.45rem;
        padding: 0.6rem 0.9rem;
        border-radius: 999px;
        background: var(--rv-teal-soft);
        color: #087f73;
        font-weight: 800;
        margin-bottom: 1rem;
        border: 1px solid rgba(17, 183, 164, 0.24);
    }

    .clinical-note {
        border-radius: 18px;
        padding: 1rem 1.15rem;
        background: #eff6ff;
        border: 1px solid #bfdbfe;
        color: var(--rv-blue-dark);
        line-height: 1.55;
        margin-top: 0.75rem;
    }

    .sticky-nav {
        position: sticky;
        top: 0.35rem;
        z-index: 999;
        display: flex;
        flex-wrap: wrap;
        gap: 0.5rem;
        align-items: center;
        padding: 0.55rem;
        margin: 0 0 0.8rem 0;
        border: 1px solid rgba(219, 234, 254, 0.92);
        border-radius: 999px;
        background: rgba(255, 255, 255, 0.86);
        backdrop-filter: blur(16px);
        box-shadow: 0 10px 24px rgba(15, 23, 42, 0.07);
    }

    .sticky-nav a {
        color: var(--rv-blue-dark);
        text-decoration: none;
        font-size: 0.82rem;
        font-weight: 800;
        padding: 0.5rem 0.78rem;
        border-radius: 999px;
    }

    .sticky-nav a:hover {
        background: var(--rv-sky);
        color: var(--rv-blue);
    }

    .anchor-target {
        scroll-margin-top: 5rem;
    }

    .architecture-flow {
        display: grid;
        grid-template-columns: repeat(5, minmax(120px, 1fr));
        gap: 0.7rem;
        align-items: stretch;
    }

    .pipeline-node {
        position: relative;
        padding: 0.95rem;
        min-height: 130px;
        border-radius: 22px;
        background: linear-gradient(180deg, #ffffff 0%, #f2fbff 100%);
        border: 1px solid var(--rv-border);
        box-shadow: 0 12px 28px rgba(15, 23, 42, 0.06);
    }

    .pipeline-node::after {
        content: "→";
        position: absolute;
        right: -0.58rem;
        top: 43%;
        color: var(--rv-teal);
        font-weight: 900;
        background: var(--rv-white);
        border-radius: 999px;
        padding: 0.1rem 0.25rem;
    }

    .pipeline-node:last-child::after {
        display: none;
    }

    .pipeline-index {
        color: var(--rv-teal);
        font-size: 0.75rem;
        font-weight: 850;
        text-transform: uppercase;
        letter-spacing: 0.09em;
    }

    .pipeline-title {
        margin: 0.4rem 0 0.35rem;
        color: var(--rv-text);
        font-weight: 850;
    }

    .pipeline-copy {
        color: var(--rv-muted);
        font-size: 0.86rem;
        line-height: 1.45;
        margin: 0;
    }

    .detail-grid {
        display: grid;
        grid-template-columns: repeat(4, minmax(140px, 1fr));
        gap: 0.7rem;
    }

    .detail-item {
        background: var(--rv-white);
        border: 1px solid var(--rv-border);
        border-radius: 18px;
        padding: 0.9rem;
    }

    .detail-label {
        color: var(--rv-muted);
        font-size: 0.76rem;
        font-weight: 850;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 0.25rem;
    }

    .detail-value {
        color: var(--rv-blue-dark);
        font-weight: 850;
        font-size: 1.02rem;
    }

    .confidence-card {
        background: linear-gradient(135deg, var(--rv-blue-dark), var(--rv-blue), var(--rv-teal));
        border-radius: 24px;
        padding: 1rem;
        color: var(--rv-white);
        height: 100%;
        box-shadow: var(--rv-shadow);
    }

    .confidence-value {
        font-size: 2.35rem;
        line-height: 1;
        font-weight: 900;
        letter-spacing: -0.05em;
        margin: 0.5rem 0;
    }

    .confidence-copy {
        color: rgba(255, 255, 255, 0.82);
        line-height: 1.45;
        margin: 0;
        font-size: 0.9rem;
    }

    .loading-card {
        display: flex;
        align-items: center;
        gap: 0.85rem;
        padding: 1rem;
        border-radius: 20px;
        background: linear-gradient(135deg, #eff6ff, #e8fbf8);
        border: 1px solid var(--rv-border);
        color: var(--rv-blue-dark);
        font-weight: 800;
        margin: 0.85rem 0;
    }

    .loader {
        width: 26px;
        height: 26px;
        border: 3px solid rgba(11, 94, 215, 0.18);
        border-top-color: var(--rv-teal);
        border-radius: 50%;
        animation: spin 0.8s linear infinite;
    }

    @keyframes spin {
        to { transform: rotate(360deg); }
    }

    .footer {
        margin-top: 1.4rem;
        padding: 1.3rem;
        border-radius: 26px;
        background: linear-gradient(135deg, #073b7a, #0b5ed7);
        color: var(--rv-white);
        box-shadow: var(--rv-shadow);
    }

    .footer-title {
        font-size: 1.2rem;
        font-weight: 850;
        margin-bottom: 0.35rem;
    }

    .footer-copy {
        color: rgba(255, 255, 255, 0.82);
        max-width: 760px;
        line-height: 1.55;
        margin: 0 0 0.85rem 0;
    }

    .footer-links {
        display: flex;
        flex-wrap: wrap;
        gap: 0.7rem;
    }

    .footer-links a {
        color: #ffffff;
        text-decoration: none;
        font-weight: 800;
        padding: 0.5rem 0.8rem;
        border-radius: 999px;
        border: 1px solid rgba(255, 255, 255, 0.24);
        background: rgba(255, 255, 255, 0.12);
    }

    @media (max-width: 900px) {
        .block-container {
            padding-left: 1rem;
            padding-right: 1rem;
        }

        .hero {
            padding: 1.35rem;
            border-radius: 22px;
        }

        .architecture-flow,
        .detail-grid {
            grid-template-columns: 1fr;
        }

        .pipeline-node::after {
            display: none;
        }

        .sticky-nav {
            border-radius: 20px;
        }
    }
</style>
""", unsafe_allow_html=True)

def section_heading(kicker, title, anchor=None):
    anchor_html = f'<div id="{anchor}" class="anchor-target"></div>' if anchor else ""
    st.markdown(
        f"""
        {anchor_html}
        <div class="section-heading">
            <div class="section-kicker">{kicker}</div>
            <h2 class="section-title">{title}</h2>
        </div>
        """,
        unsafe_allow_html=True
    )

def info_card(icon, title, body):
    st.markdown(
        f"""
        <div class="info-card">
            <div class="info-icon">{icon}</div>
            <div class="info-title">{title}</div>
            <p class="info-body">{body}</p>
        </div>
        """,
        unsafe_allow_html=True
    )

def kpi_card(label, value):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

def performance_card(label, value):
    st.markdown(
        f"""
        <div class="performance-card">
            <div class="performance-label">{label}</div>
            <div class="performance-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

def detail_item(label, value):
    st.markdown(
        f"""
        <div class="detail-item">
            <div class="detail-label">{label}</div>
            <div class="detail-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

def pipeline_visualization():
    st.markdown("""
    <div class="architecture-flow">
        <div class="pipeline-node">
            <div class="pipeline-index">Input</div>
            <div class="pipeline-title">Fundus Image</div>
            <p class="pipeline-copy">Clinical retinal photograph uploaded by the user.</p>
        </div>
        <div class="pipeline-node">
            <div class="pipeline-index">Preprocess</div>
            <div class="pipeline-title">CLAHE + Resize</div>
            <p class="pipeline-copy">Image is normalized to 512 x 512 with contrast enhancement.</p>
        </div>
        <div class="pipeline-node">
            <div class="pipeline-index">Model</div>
            <div class="pipeline-title">U-Net</div>
            <p class="pipeline-copy">Encoder-decoder segmentation with skip connections.</p>
        </div>
        <div class="pipeline-node">
            <div class="pipeline-index">Postprocess</div>
            <div class="pipeline-title">Binary Mask</div>
            <p class="pipeline-copy">Probability map is thresholded for vessel extraction.</p>
        </div>
        <div class="pipeline-node">
            <div class="pipeline-index">Dashboard</div>
            <div class="pipeline-title">Analytics</div>
            <p class="pipeline-copy">KPIs, confidence, overlay, and morphology estimates.</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

def zhang_suen_thinning(binary_mask):
    image = (binary_mask > 0.5).astype(np.uint8)
    changing = True
    while changing:
        changing = False
        for step in range(2):
            to_remove = []
            rows, cols = image.shape
            for y in range(1, rows - 1):
                for x in range(1, cols - 1):
                    if image[y, x] != 1:
                        continue
                    p2 = image[y - 1, x]
                    p3 = image[y - 1, x + 1]
                    p4 = image[y, x + 1]
                    p5 = image[y + 1, x + 1]
                    p6 = image[y + 1, x]
                    p7 = image[y + 1, x - 1]
                    p8 = image[y, x - 1]
                    p9 = image[y - 1, x - 1]
                    neighbors = [p2, p3, p4, p5, p6, p7, p8, p9]
                    transitions = sum(
                        neighbors[i] == 0 and neighbors[(i + 1) % 8] == 1
                        for i in range(8)
                    )
                    neighbor_count = sum(neighbors)
                    if not (2 <= neighbor_count <= 6 and transitions == 1):
                        continue
                    if step == 0:
                        condition = p2 * p4 * p6 == 0 and p4 * p6 * p8 == 0
                    else:
                        condition = p2 * p4 * p8 == 0 and p2 * p6 * p8 == 0
                    if condition:
                        to_remove.append((y, x))
            if to_remove:
                changing = True
                for y, x in to_remove:
                    image[y, x] = 0
    return image

def calculate_vessel_analytics(binary_mask, prob_mask):
    vessel_pixels = int(np.sum(binary_mask))
    total_pixels = int(binary_mask.size)
    vessel_density = (vessel_pixels / total_pixels) * 100
    vessel_area_ratio = vessel_pixels / total_pixels
    confidence = np.mean(np.maximum(prob_mask, 1.0 - prob_mask)) * 100

    analysis_mask = cv2.resize(
        binary_mask.astype(np.uint8),
        (256, 256),
        interpolation=cv2.INTER_NEAREST
    )
    skeleton = zhang_suen_thinning(analysis_mask)
    padded = np.pad(skeleton, 1, mode="constant")
    neighbor_counts = np.zeros_like(skeleton, dtype=np.uint8)
    for y_offset in range(3):
        for x_offset in range(3):
            if y_offset == 1 and x_offset == 1:
                continue
            neighbor_counts += padded[
                y_offset:y_offset + skeleton.shape[0],
                x_offset:x_offset + skeleton.shape[1]
            ]

    branch_points = int(np.sum((skeleton == 1) & (neighbor_counts >= 3)))
    endpoints = (skeleton == 1) & (neighbor_counts == 1)
    component_count, labels = cv2.connectedComponents(skeleton.astype(np.uint8))
    tortuosities = []

    for component_id in range(1, component_count):
        component = labels == component_id
        length = int(np.sum(component))
        if length < 8:
            continue
        component_endpoints = np.argwhere(endpoints & component)
        if len(component_endpoints) < 2:
            continue
        distances = np.linalg.norm(
            component_endpoints[:, None, :] - component_endpoints[None, :, :],
            axis=2
        )
        chord = float(np.max(distances))
        if chord > 0:
            tortuosities.append(length / chord)

    tortuosity = float(np.mean(tortuosities)) if tortuosities else 1.0

    return {
        "vessel_pixels": vessel_pixels,
        "total_pixels": total_pixels,
        "vessel_density": vessel_density,
        "vessel_area_ratio": vessel_area_ratio,
        "branch_points": branch_points,
        "tortuosity": tortuosity,
        "confidence": confidence
    }

def create_comparison_image(original, overlay, split_percent):
    original_uint8 = np.clip(original * 255, 0, 255).astype(np.uint8)
    overlay_uint8 = np.clip(overlay * 255, 0, 255).astype(np.uint8)
    split_x = int(original_uint8.shape[1] * split_percent / 100)
    comparison = overlay_uint8.copy()
    comparison[:, :split_x] = original_uint8[:, :split_x]
    comparison[:, max(split_x - 2, 0):min(split_x + 2, comparison.shape[1])] = [255, 255, 255]
    return comparison

def image_array_to_data_uri(image_array):
    image_uint8 = np.clip(image_array * 255, 0, 255).astype(np.uint8)
    image_pil = Image.fromarray(image_uint8)
    buffer = io.BytesIO()
    image_pil.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"

def render_comparison_slider(original, overlay):
    original_uri = image_array_to_data_uri(original)
    overlay_uri = image_array_to_data_uri(overlay)

    components.html(
        f"""
        <div class="rv-comparison-shell">
            <div class="rv-comparison-header">
                <div>
                    <div class="rv-kicker">Interactive Review</div>
                    <div class="rv-title">Original Fundus Image vs Vessel Overlay</div>
                </div>
                <div class="rv-help">Drag the divider</div>
            </div>

            <div class="rv-compare" id="rvCompare">
                <img class="rv-img rv-original" src="{original_uri}" alt="Original fundus image">
                <img class="rv-img rv-overlay" id="rvOverlay" src="{overlay_uri}" alt="Vessel overlay image">

                <div class="rv-label rv-label-left">Original Fundus</div>
                <div class="rv-label rv-label-right">Vessel Overlay</div>

                <div class="rv-divider" id="rvDivider" role="slider" tabindex="0"
                     aria-label="Comparison divider" aria-valuemin="0" aria-valuemax="100"
                     aria-valuenow="50">
                    <div class="rv-handle">
                        <span></span>
                        <span></span>
                    </div>
                </div>
            </div>
        </div>

        <style>
            :root {{
                --split: 50%;
                --rv-blue: #0b5ed7;
                --rv-blue-dark: #073b7a;
                --rv-teal: #11b7a4;
                --rv-border: #dbeafe;
                --rv-muted: #64748b;
                --rv-text: #17324d;
            }}

            * {{
                box-sizing: border-box;
            }}

            body {{
                margin: 0;
                font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
                background: transparent;
            }}

            .rv-comparison-shell {{
                width: 100%;
                padding: 1rem;
                border: 1px solid var(--rv-border);
                border-radius: 26px;
                background: linear-gradient(180deg, rgba(255,255,255,0.98), rgba(244,251,255,0.98));
                box-shadow: 0 14px 34px rgba(15, 23, 42, 0.08);
            }}

            .rv-comparison-header {{
                display: flex;
                justify-content: space-between;
                gap: 1rem;
                align-items: center;
                margin-bottom: 0.85rem;
            }}

            .rv-kicker {{
                color: var(--rv-teal);
                font-size: 0.76rem;
                font-weight: 850;
                letter-spacing: 0.09em;
                text-transform: uppercase;
                margin-bottom: 0.2rem;
            }}

            .rv-title {{
                color: var(--rv-text);
                font-size: clamp(1rem, 2.3vw, 1.35rem);
                font-weight: 850;
                letter-spacing: -0.03em;
            }}

            .rv-help {{
                color: var(--rv-blue-dark);
                background: #e8fbf8;
                border: 1px solid rgba(17, 183, 164, 0.28);
                border-radius: 999px;
                padding: 0.45rem 0.75rem;
                font-size: 0.8rem;
                font-weight: 800;
                white-space: nowrap;
            }}

            .rv-compare {{
                position: relative;
                width: 100%;
                aspect-ratio: 1 / 1;
                overflow: hidden;
                border-radius: 22px;
                border: 1px solid rgba(11, 94, 215, 0.18);
                background: #071f3a;
                user-select: none;
                cursor: ew-resize;
                touch-action: none;
            }}

            .rv-img {{
                position: absolute;
                inset: 0;
                width: 100%;
                height: 100%;
                object-fit: cover;
                pointer-events: none;
            }}

            .rv-original {{
                z-index: 1;
            }}

            .rv-overlay {{
                z-index: 2;
                clip-path: inset(0 0 0 var(--split));
                will-change: clip-path;
            }}

            .rv-divider {{
                position: absolute;
                top: 0;
                bottom: 0;
                left: var(--split);
                z-index: 5;
                width: 0;
                outline: none;
                will-change: left;
            }}

            .rv-divider::before {{
                content: "";
                position: absolute;
                top: 0;
                bottom: 0;
                left: -1.5px;
                width: 3px;
                background: linear-gradient(180deg, #ffffff, #dffcff, #ffffff);
                box-shadow: 0 0 0 1px rgba(11, 94, 215, 0.22), 0 0 18px rgba(17, 183, 164, 0.65);
            }}

            .rv-handle {{
                position: absolute;
                top: 50%;
                left: 50%;
                width: 54px;
                height: 54px;
                transform: translate(-50%, -50%);
                display: flex;
                align-items: center;
                justify-content: center;
                gap: 5px;
                border-radius: 50%;
                background: linear-gradient(135deg, var(--rv-blue), var(--rv-teal));
                border: 3px solid #ffffff;
                box-shadow: 0 12px 30px rgba(7, 59, 122, 0.28);
            }}

            .rv-handle span {{
                width: 8px;
                height: 8px;
                border-left: 2px solid #ffffff;
                border-bottom: 2px solid #ffffff;
            }}

            .rv-handle span:first-child {{
                transform: rotate(45deg);
            }}

            .rv-handle span:last-child {{
                transform: rotate(225deg);
            }}

            .rv-label {{
                position: absolute;
                top: 0.85rem;
                z-index: 6;
                padding: 0.42rem 0.72rem;
                border-radius: 999px;
                color: #ffffff;
                font-size: 0.76rem;
                font-weight: 850;
                letter-spacing: 0.02em;
                background: rgba(7, 59, 122, 0.72);
                border: 1px solid rgba(255, 255, 255, 0.22);
                backdrop-filter: blur(10px);
            }}

            .rv-label-left {{
                left: 0.85rem;
            }}

            .rv-label-right {{
                right: 0.85rem;
                background: rgba(17, 183, 164, 0.78);
            }}

            @media (max-width: 680px) {{
                .rv-comparison-shell {{
                    padding: 0.75rem;
                    border-radius: 20px;
                }}

                .rv-comparison-header {{
                    align-items: flex-start;
                    flex-direction: column;
                    gap: 0.45rem;
                }}

                .rv-label {{
                    top: 0.55rem;
                    font-size: 0.68rem;
                    padding: 0.35rem 0.55rem;
                }}

                .rv-label-left {{
                    left: 0.55rem;
                }}

                .rv-label-right {{
                    right: 0.55rem;
                }}

                .rv-handle {{
                    width: 46px;
                    height: 46px;
                }}
            }}
        </style>

        <script>
            const compare = document.getElementById("rvCompare");
            const divider = document.getElementById("rvDivider");

            function clamp(value, min, max) {{
                return Math.min(Math.max(value, min), max);
            }}

            function setSplit(percent) {{
                const safePercent = clamp(percent, 0, 100);
                compare.style.setProperty("--split", safePercent + "%");
                divider.setAttribute("aria-valuenow", Math.round(safePercent));
            }}

            function updateFromClientX(clientX) {{
                const rect = compare.getBoundingClientRect();
                const percent = ((clientX - rect.left) / rect.width) * 100;
                setSplit(percent);
            }}

            compare.addEventListener("pointerdown", (event) => {{
                compare.setPointerCapture(event.pointerId);
                updateFromClientX(event.clientX);
            }});

            compare.addEventListener("pointermove", (event) => {{
                if (event.buttons === 1 || event.pressure > 0) {{
                    updateFromClientX(event.clientX);
                }}
            }});

            divider.addEventListener("keydown", (event) => {{
                const current = Number(divider.getAttribute("aria-valuenow")) || 50;
                if (event.key === "ArrowLeft") {{
                    setSplit(current - 2);
                    event.preventDefault();
                }}
                if (event.key === "ArrowRight") {{
                    setSplit(current + 2);
                    event.preventDefault();
                }}
                if (event.key === "Home") {{
                    setSplit(0);
                    event.preventDefault();
                }}
                if (event.key === "End") {{
                    setSplit(100);
                    event.preventDefault();
                }}
            }});
        </script>
        """,
        height=650,
        scrolling=False
    )

st.markdown("""
<div class="hero">
    <div class="hero-eyebrow">Clinical AI Research Dashboard</div>
    <h1 class="hero-title">RetinaVision AI</h1>
    <p class="hero-subtitle">
        AI-Powered Retinal Vessel Analysis for fundus image segmentation,
        quantitative vessel mapping, and research-grade clinical decision support.
    </p>
    <div class="badge-row">
        <span class="badge">U-Net</span>
        <span class="badge">DRIVE Dataset</span>
        <span class="badge">Dice Score 79.41%</span>
        <span class="badge">IoU 65.88%</span>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<nav class="sticky-nav">
    <a href="#overview">Overview</a>
    <a href="#architecture">Architecture</a>
    <a href="#technical">Technical Details</a>
    <a href="#analyze">Analyze</a>
    <a href="#results">Results</a>
    <a href="#footer">Project</a>
</nav>
""", unsafe_allow_html=True)

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

section_heading("Platform Overview", "Retinal Vessel Intelligence", "overview")
info_cols = st.columns(3)
with info_cols[0]:
    info_card(
        "🧠",
        "Deep Learning Segmentation",
        "A U-Net model isolates retinal vasculature from fundus photographs while preserving the existing prediction workflow."
    )
with info_cols[1]:
    info_card(
        "🩺",
        "Clinical Research Context",
        "Designed for educational exploration of vessel morphology linked to diabetic retinopathy, glaucoma, hypertension, and vascular risk."
    )
with info_cols[2]:
    info_card(
        "📊",
        "Quantitative Dashboard",
        "Results are summarized with density, vessel pixel counts, image dimensions, inference timing, masks, and overlay visualization."
    )

section_heading("Architecture", "Interactive AI Pipeline", "architecture")
pipeline_visualization()

stage = st.selectbox(
    "Inspect pipeline stage",
    ["Input", "Preprocessing", "U-Net Inference", "Postprocessing", "Analytics Dashboard"],
    help="Explore the production workflow from image upload to vessel analytics."
)
stage_details = {
    "Input": "Accepts retinal fundus images and keeps the original image available for comparison.",
    "Preprocessing": "Resizes the image to 512 x 512, converts to LAB color space, and applies CLAHE enhancement.",
    "U-Net Inference": "Runs the trained segmentation model and returns a vessel probability map.",
    "Postprocessing": "Applies the existing 0.5 threshold to generate the binary vessel mask.",
    "Analytics Dashboard": "Presents KPI summaries, morphology estimates, confidence, and overlay comparison."
}
st.info(stage_details[stage])

section_heading("Technical Details", "Model Registry and Dataset Profile", "technical")
with st.expander("View technical model card", expanded=True):
    st.markdown(f"""
    <div class="detail-grid">
        <div class="detail-item">
            <div class="detail-label">Model Version</div>
            <div class="detail-value">retinavision_best_model.h5</div>
        </div>
        <div class="detail-item">
            <div class="detail-label">Framework</div>
            <div class="detail-value">TensorFlow {tf.__version__}</div>
        </div>
        <div class="detail-item">
            <div class="detail-label">Input Size</div>
            <div class="detail-value">512 x 512 RGB</div>
        </div>
        <div class="detail-item">
            <div class="detail-label">Architecture</div>
            <div class="detail-value">U-Net</div>
        </div>
        <div class="detail-item">
            <div class="detail-label">Dice Score</div>
            <div class="detail-value">79.41%</div>
        </div>
        <div class="detail-item">
            <div class="detail-label">IoU</div>
            <div class="detail-value">65.88%</div>
        </div>
        <div class="detail-item">
            <div class="detail-label">Dataset</div>
            <div class="detail-value">DRIVE</div>
        </div>
        <div class="detail-item">
            <div class="detail-label">Dataset Statistics</div>
            <div class="detail-value">40 fundus images</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

section_heading("Model Performance", "Validated Research Metrics")
perf_cols = st.columns(4)
with perf_cols[0]:
    performance_card("Dice Score", "79.41%")
with perf_cols[1]:
    performance_card("IoU", "65.88%")
with perf_cols[2]:
    performance_card("Dataset", "DRIVE")
with perf_cols[3]:
    performance_card("Architecture", "U-Net")

model = load_model()

if model is None:
    st.warning("⏳ Model not trained yet. Training is in progress. Please wait for training to complete, then refresh this page.")
    st.info("Model will be saved to: `outputs/retinavision_best_model.h5`")
    st.stop()
else:
    st.markdown('<div class="status-pill">✅ Model loaded successfully</div>', unsafe_allow_html=True)

section_heading("Analyze Image", "Upload Retinal Fundus Image", "analyze")
uploaded_file = st.file_uploader(
    "Upload a retinal fundus image",
    type=["png", "jpg", "jpeg", "tif", "tiff"],
    help="Upload a retinal fundus photograph for vessel segmentation"
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    
    col1, col2 = st.columns([1.05, 0.95])
    with col1:
        st.markdown('<div class="panel"><p class="image-caption">Uploaded Fundus Image</p>', unsafe_allow_html=True)
        st.image(image, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="panel">
            <p class="image-caption">Analysis Console</p>
            <p class="info-body">
                Run vessel segmentation to generate the binary mask, overlay, confidence score,
                and morphology estimates. The model inference path remains unchanged.
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    if st.button("🔬 Segment Blood Vessels", type="primary", use_container_width=True):
        loading_slot = st.empty()
        loading_slot.markdown("""
        <div class="loading-card">
            <div class="loader"></div>
            <div>Running retinal vessel segmentation and preparing dashboard analytics...</div>
        </div>
        """, unsafe_allow_html=True)
        with st.spinner("Segmenting blood vessels..."):
            processed = preprocess_image(image)
            start_time = time.perf_counter()
            prob_mask, binary_mask = predict(model, processed)
            inference_time = time.perf_counter() - start_time
            overlay = create_overlay(processed, binary_mask)
            analytics = calculate_vessel_analytics(binary_mask, prob_mask)
        loading_slot.empty()

        render_comparison_slider(processed, overlay)

        section_heading("Segmentation Results", "AI Vessel Analysis", "results")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown('<div class="panel"><p class="image-caption">Original Fundus Image</p>', unsafe_allow_html=True)
            st.image(processed, use_container_width=True, clamp=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="panel"><p class="image-caption">Predicted Vessel Mask</p>', unsafe_allow_html=True)
            st.image(binary_mask, use_container_width=True, clamp=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col3:
            st.markdown('<div class="panel"><p class="image-caption">Overlay: Green Indicates Vessels</p>', unsafe_allow_html=True)
            st.image(overlay, use_container_width=True, clamp=True)
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown(
            f"""
            <div class="confidence-card">
                <div class="performance-label" style="color: rgba(255,255,255,0.74);">Prediction Confidence</div>
                <div class="confidence-value">{analytics["confidence"]:.1f}%</div>
                <p class="confidence-copy">
                    Mean pixel-level certainty derived from the vessel probability map.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        section_heading("Advanced Vessel Analytics", "Retinal Vessel Measurements")
        
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            kpi_card("Vessel Density", f"{analytics['vessel_density']:.2f}%")
        with m2:
            kpi_card("Vessel Area Ratio", f"{analytics['vessel_area_ratio']:.4f}")
        with m3:
            kpi_card("Branch Point Estimate", f"{analytics['branch_points']:,}")
        with m4:
            kpi_card("Tortuosity Estimate", f"{analytics['tortuosity']:.2f}")

        k1, k2, k3, k4 = st.columns(4)
        with k1:
            kpi_card("Vessel Pixels", f"{analytics['vessel_pixels']:,}")
        with k2:
            kpi_card("Image Size", "512 x 512")
        with k3:
            kpi_card("Inference Time", f"{inference_time:.2f}s")
        with k4:
            kpi_card("Total Pixels", f"{analytics['total_pixels']:,}")
        
        section_heading("Clinical Significance", "Research Interpretation")
        
        if analytics["vessel_density"] < 8:
            st.warning("⚠️ Low vessel density detected. This may indicate poor image quality or potential vascular abnormalities.")
        elif analytics["vessel_density"] > 20:
            st.warning("⚠️ High vessel density detected. This may indicate neovascularisation — a sign of diabetic retinopathy.")
        else:
            st.success("✅ Vessel density within normal range (8–20%). No obvious gross abnormalities detected.")
        
        st.markdown("""
        <div class="clinical-note">
            <strong>Research use only.</strong> RetinaVision AI supports educational exploration of retinal vessel segmentation.
            Clinical diagnosis must be performed by a qualified ophthalmologist.
        </div>
        """, unsafe_allow_html=True)
        
        mask_uint8 = (binary_mask * 255).astype(np.uint8)
        mask_pil = Image.fromarray(mask_uint8)
        buf = io.BytesIO()
        mask_pil.save(buf, format="PNG")
        st.download_button(
            label="⬇️ Download Segmentation Mask",
            data=buf.getvalue(),
            file_name="vessel_mask.png",
            mime="image/png"
        )

else:
    section_heading("Workflow", "How RetinaVision AI Works")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("""
        <div class="workflow-step">
            <div class="workflow-number">Step 01</div>
            <div class="workflow-title">Input</div>
            <p class="workflow-copy">Upload a retinal fundus photograph in PNG, JPG, JPEG, TIF, or TIFF format.</p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="workflow-step">
            <div class="workflow-number">Step 02</div>
            <div class="workflow-title">Preprocess</div>
            <p class="workflow-copy">Resize to 512 x 512 and apply CLAHE contrast enhancement before inference.</p>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="workflow-step">
            <div class="workflow-number">Step 03</div>
            <div class="workflow-title">U-Net Inference</div>
            <p class="workflow-copy">Encoder-decoder features reconstruct a probability map for retinal vessels.</p>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown("""
        <div class="workflow-step">
            <div class="workflow-number">Step 04</div>
            <div class="workflow-title">Output</div>
            <p class="workflow-copy">View the vessel mask, overlay, KPI measurements, and research interpretation.</p>
        </div>
        """, unsafe_allow_html=True)

st.markdown("""
<div id="footer" class="anchor-target"></div>
<footer class="footer">
    <div class="footer-title">RetinaVision AI</div>
    <p class="footer-copy">
        A production-style medical AI dashboard for retinal vessel segmentation research,
        combining U-Net inference with explainable visualization, vessel analytics, and
        clinical education workflows.
    </p>
    <div class="footer-links">
        <a href="https://github.com/Supriya0Mishra/ASRA" target="_blank">GitHub Repository</a>
        <a href="https://github.com/Supriya0Mishra/ASRA/issues" target="_blank">Issues</a>
        <a href="https://drive.grand-challenge.org/" target="_blank">DRIVE Dataset</a>
    </div>
</footer>
""", unsafe_allow_html=True)
