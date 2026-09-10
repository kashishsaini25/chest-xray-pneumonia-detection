import os
import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import gdown
import matplotlib
from PIL import Image

st.set_page_config(
    page_title="Pneumonia Detection",
    page_icon="🩻",
    layout="centered"
)

st.title("Chest X-Ray Pneumonia Detection")
st.write(
    "Upload a chest X-ray image and the trained ResNet50 model will predict "
    "NORMAL or PNEUMONIA, along with a Grad-CAM heatmap showing which regions "
    "influenced the prediction."
)

IMG_HEIGHT = 180
IMG_WIDTH = 180
CLASS_NAMES = ["NORMAL", "PNEUMONIA"]
GOOGLE_DRIVE_FILE_ID = "1BDsTzu6QnQZg57-0d4x52wPRo0kunDVT"
MODEL_PATH = "pneumonia_resnet_final.keras"

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Downloading model (first run only, may take a minute)..."):
            url = f"https://drive.google.com/uc?id={GOOGLE_DRIVE_FILE_ID}"
            gdown.download(url, MODEL_PATH, quiet=False)
    return tf.keras.models.load_model(MODEL_PATH)


@st.cache_resource
def build_gradcam_model(_model):
    resnet_base = _model.get_layer("resnet50")
    last_conv_layer_name = "conv5_block3_3_conv"

    grad_model = tf.keras.models.Model(
        inputs=resnet_base.input,
        outputs=[
            resnet_base.get_layer(last_conv_layer_name).output,
            resnet_base.output
        ]
    )
    classifier_layers = _model.layers[1:]
    return grad_model, classifier_layers


model = load_model()
grad_model, classifier_layers = build_gradcam_model(model)


def make_gradcam_heatmap(img_array, pred_index=None):
    with tf.GradientTape() as tape:
        conv_output, resnet_output = grad_model(img_array)
        tape.watch(conv_output)

        x = resnet_output
        for layer in classifier_layers:
            x = layer(x)
        predictions = x

        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_output = conv_output[0]
    heatmap = conv_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)

    return heatmap.numpy(), predictions.numpy()[0]


def overlay_heatmap(original_array, heatmap, alpha=0.4):
    heatmap_resized = cv2.resize(heatmap, (IMG_WIDTH, IMG_HEIGHT))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)

    jet = matplotlib.colormaps["jet"]
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = np.uint8(jet_colors[heatmap_uint8] * 255)

    superimposed = np.clip(jet_heatmap * alpha + original_array, 0, 255).astype("uint8")
    return superimposed

uploaded_file = st.file_uploader("Upload a Chest X-Ray", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Chest X-Ray", use_container_width=True)

    if st.button("Predict"):
        resized = image.resize((IMG_WIDTH, IMG_HEIGHT))
        img_array = np.array(resized).astype("float32")
        img_batch = np.expand_dims(img_array, axis=0)

        heatmap, predictions = make_gradcam_heatmap(img_batch)

        predicted_index = int(np.argmax(predictions))
        result = CLASS_NAMES[predicted_index]
        confidence = float(predictions[predicted_index] * 100)

        st.subheader("Prediction")
        if result == "PNEUMONIA":
            st.error(f"Prediction: {result}")
        else:
            st.success(f"Prediction: {result}")
        st.write(f"Model confidence: {confidence:.2f}%")

        overlay = overlay_heatmap(img_array, heatmap)

        col1, col2 = st.columns(2)
        with col1:
            st.image(resized, caption="Input (resized)", use_container_width=True)
        with col2:
            st.image(overlay, caption="Grad-CAM", use_container_width=True)

        st.warning(
            "This is an educational/research prototype and not a medical diagnosis."
        )

