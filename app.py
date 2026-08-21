import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from pathlib import Path
import json
import numpy as np
from PIL import Image
import streamlit as st
import tensorflow as tf

st.set_page_config(
    page_title="Vehicle Image Classifier",
    page_icon="🚗",
    layout="centered"
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "image_classifier_vehicles.h5"
CLASS_PATH = BASE_DIR / "image_classifier_vehicles_classes.json"

IMAGE_SIZE = (128, 128)

@st.cache_resource
def load_resources():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"ไม่พบไฟล์โมเดล: {MODEL_PATH.name}"
        )

    if not CLASS_PATH.exists():
        raise FileNotFoundError(
            f"ไม่พบไฟล์ class mapping: {CLASS_PATH.name}"
        )

    with open(CLASS_PATH, "r", encoding="utf-8") as f:
        class_names = json.load(f)

    model = tf.keras.models.load_model(
        MODEL_PATH,
        compile=False
    )

    return model, class_names

def preprocess(image):
    image = image.convert("RGB").resize(IMAGE_SIZE)
    x = np.asarray(image, dtype=np.float32) / 255.0
    return np.expand_dims(x, axis=0)

st.title("🚗 Vehicle Image Classifier")
st.caption(
    "อัปโหลดภาพรถ 1 รูป ระบบจะปรับภาพเป็น RGB ขนาด 128×128 "
    "และทำนายด้วย TensorFlow/Keras model"
)

try:
    model, class_names = load_resources()
except Exception as error:
    st.error(f"โหลดโมเดลไม่สำเร็จ: {error}")
    st.stop()

uploaded_file = st.file_uploader(
    "เลือกรูปภาพ",
    type=["jpg", "jpeg", "png", "webp"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(
        image,
        caption="รูปภาพที่อัปโหลด",
        use_container_width=True
    )

    if st.button("Predict", type="primary", use_container_width=True):
        with st.spinner("กำลังทำนาย..."):
            probabilities = model.predict(
                preprocess(image),
                verbose=0
            )[0]

        probabilities = np.asarray(probabilities, dtype=np.float32)
        best_index = int(np.argmax(probabilities))
        best_label = class_names[best_index]
        confidence = float(probabilities[best_index]) * 100

        st.success(f"ผลทำนาย: {best_label}")
        st.metric("Confidence", f"{confidence:.2f}%")

        results = {
            class_names[i]: round(float(probabilities[i]) * 100, 2)
            for i in range(len(class_names))
        }

        st.subheader("ความน่าจะเป็นของแต่ละคลาส")
        st.bar_chart(results)
else:
    st.info("เลือกรูปภาพเพื่อเริ่มทดสอบโมเดล")