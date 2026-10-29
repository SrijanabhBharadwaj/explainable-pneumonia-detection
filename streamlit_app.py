from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from PIL import Image

import pneumonia_pipeline as pipeline


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "outputs"
CHECKPOINT = OUTPUT_DIR / "cpu_baseline_exploratory.keras"
METRICS_FILE = OUTPUT_DIR / "cpu_baseline_exploratory_metrics.json"
DATASET_ROOT = Path(
    r"C:\Users\USER\.cache\kagglehub\datasets\paultimothymooney\chest-xray-pneumonia\versions\2\chest_xray"
)
IMAGE_SIZE = 224

st.set_page_config(page_title="Pneumonia X-ray Research Demo", layout="wide")
st.title("Chest X-ray Research Demo")
st.caption(
    "Educational prototype only. Not a medical device; predictions are not diagnoses. "
    "Grad-CAM is a coarse indication of model sensitivity, not clinical localization."
)


@st.cache_data(show_spinner=False)
def read_dataset_counts(dataset_root: str) -> pd.DataFrame:
    root = Path(dataset_root)
    rows = []
    if not root.exists():
        return pd.DataFrame(columns=["Split", "NORMAL", "PNEUMONIA", "Total"])
    for split in pipeline.SPLITS:
        counts = {}
        for class_name in pipeline.CLASS_NAMES:
            folder = root / split / class_name
            counts[class_name] = len(pipeline.image_files(folder)) if folder.is_dir() else 0
        rows.append({"Split": split, **counts, "Total": sum(counts.values())})
    return pd.DataFrame(rows)


@st.cache_resource(show_spinner="Loading the trained checkpoint...")
def load_model(checkpoint_path: str):
    return tf.keras.models.load_model(checkpoint_path)


def decode_and_pad(image_bytes: bytes) -> np.ndarray:
    encoded = np.frombuffer(image_bytes, dtype=np.uint8)
    bgr = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("The uploaded file could not be decoded as an image.")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    height, width = rgb.shape[:2]
    scale = min(IMAGE_SIZE / width, IMAGE_SIZE / height)
    resized_width = max(1, round(width * scale))
    resized_height = max(1, round(height * scale))
    resized = cv2.resize(rgb, (resized_width, resized_height), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
    top = (IMAGE_SIZE - resized_height) // 2
    left = (IMAGE_SIZE - resized_width) // 2
    canvas[top:top + resized_height, left:left + resized_width] = resized
    return canvas


dashboard_tab, prediction_tab = st.tabs(["Project status", "Upload X-ray"])

with dashboard_tab:
    st.subheader("Dataset")
    if DATASET_ROOT.exists():
        counts = read_dataset_counts(str(DATASET_ROOT))
        st.dataframe(counts, hide_index=True, use_container_width=True)
        st.caption(f"Dataset root: {DATASET_ROOT}")
    else:
        st.info("Dataset folder not found at the configured local path.")

    st.subheader("Training checkpoint")
    if CHECKPOINT.exists():
        st.success(f"Baseline checkpoint available: {CHECKPOINT.name}")
        st.caption(f"Last modified: {pd.Timestamp(CHECKPOINT.stat().st_mtime, unit='s')}")
    else:
        st.warning("Training checkpoint is not available yet. The prediction tab will unlock after a checkpoint is saved.")

    if METRICS_FILE.exists():
        metrics = pd.read_json(METRICS_FILE, typ="series").rename("Value").to_frame()
        st.subheader("Saved test metrics")
        st.dataframe(metrics, use_container_width=True)
    else:
        st.caption("No completed test metrics have been saved for this run yet.")

    if st.button("Refresh status", icon="🔄"):
        st.cache_data.clear()
        st.rerun()

with prediction_tab:
    uploaded_file = st.file_uploader("Choose a chest X-ray image", type=["jpg", "jpeg", "png"])
    if not CHECKPOINT.exists():
        st.info("Inference will be available when the baseline training run writes its checkpoint.")
    elif uploaded_file is not None:
        try:
            image = decode_and_pad(uploaded_file.getvalue())
            model = load_model(str(CHECKPOINT))
            probability = float(model.predict(image[None].astype(np.float32), verbose=0)[0, 0])
            predicted_class = "PNEUMONIA" if probability >= 0.5 else "NORMAL"
            heatmap = pipeline.gradcam_heatmap(model, image)
            color_map = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
            color_map = cv2.cvtColor(color_map, cv2.COLOR_BGR2RGB)
            overlay = np.clip(0.55 * image + 0.45 * color_map, 0, 255).astype(np.uint8)

            left, middle, right = st.columns(3)
            left.image(image, caption="Resized input", use_container_width=True)
            middle.image(heatmap, caption="Grad-CAM heatmap", clamp=True, use_container_width=True)
            right.image(overlay, caption="Overlay", use_container_width=True)
            st.metric("Model output", predicted_class, f"P(PNEUMONIA) = {probability:.3f}")
            st.caption("Threshold = 0.5 for this research demo; it is not clinically calibrated.")
        except Exception as error:
            st.error(f"Could not process this image: {error}")