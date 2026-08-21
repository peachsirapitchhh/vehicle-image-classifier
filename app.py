# import os

# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# from pathlib import Path
# import json
# import numpy as np
# from PIL import Image
# import streamlit as st
# import tensorflow as tf

# st.set_page_config(
#     page_title="Vehicle Image Classifier",
#     page_icon="🚗",
#     layout="centered"
# )

# BASE_DIR = Path(__file__).resolve().parent
# MODEL_PATH = BASE_DIR / "image_classifier_vehicles.h5"
# CLASS_PATH = BASE_DIR / "image_classifier_vehicles_classes.json"

# IMAGE_SIZE = (128, 128)

# @st.cache_resource
# def load_resources():
#     if not MODEL_PATH.exists():
#         raise FileNotFoundError(
#             f"ไม่พบไฟล์โมเดล: {MODEL_PATH.name}"
#         )

#     if not CLASS_PATH.exists():
#         raise FileNotFoundError(
#             f"ไม่พบไฟล์ class mapping: {CLASS_PATH.name}"
#         )

#     with open(CLASS_PATH, "r", encoding="utf-8") as f:
#         class_names = json.load(f)

#     model = tf.keras.models.load_model(
#         MODEL_PATH,
#         compile=False
#     )

#     return model, class_names

# def preprocess(image):
#     image = image.convert("RGB").resize(IMAGE_SIZE)
#     x = np.asarray(image, dtype=np.float32) / 255.0
#     return np.expand_dims(x, axis=0)

# st.title("🚗 Vehicle Image Classifier")
# st.caption(
#     "อัปโหลดภาพรถ 1 รูป ระบบจะปรับภาพเป็น RGB ขนาด 128×128 "
#     "และทำนายด้วย TensorFlow/Keras model"
# )

# try:
#     model, class_names = load_resources()
# except Exception as error:
#     st.error(f"โหลดโมเดลไม่สำเร็จ: {error}")
#     st.stop()

# uploaded_file = st.file_uploader(
#     "เลือกรูปภาพ",
#     type=["jpg", "jpeg", "png", "webp"]
# )

# if uploaded_file is not None:
#     image = Image.open(uploaded_file).convert("RGB")
#     st.image(
#         image,
#         caption="รูปภาพที่อัปโหลด",
#         use_container_width=True
#     )

#     if st.button("Predict", type="primary", use_container_width=True):
#         with st.spinner("กำลังทำนาย..."):
#             probabilities = model.predict(
#                 preprocess(image),
#                 verbose=0
#             )[0]

#         probabilities = np.asarray(probabilities, dtype=np.float32)
#         best_index = int(np.argmax(probabilities))
#         best_label = class_names[best_index]
#         confidence = float(probabilities[best_index]) * 100

#         st.success(f"ผลทำนาย: {best_label}")
#         st.metric("Confidence", f"{confidence:.2f}%")

#         results = {
#             class_names[i]: round(float(probabilities[i]) * 100, 2)
#             for i in range(len(class_names))
#         }

#         st.subheader("ความน่าจะเป็นของแต่ละคลาส")
#         st.bar_chart(results)
# else:
#     st.info("เลือกรูปภาพเพื่อเริ่มทดสอบโมเดล")

"""Streamlit demo UI for the KNIME-trained vehicle image classifier.

keras_compat must be imported before tensorflow: it sets TF_USE_LEGACY_KERAS
so the Keras-2 .h5 written by model/model-trainer.py stays loadable.
"""

import keras_compat  # noqa: F401  - sets env vars, must come first

import json
import os
from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

st.set_page_config(
    page_title="Vehicle Image Classifier",
    page_icon="🚗",
    layout="centered",
)

# Artifacts live in models/, next to this file, so the app also works on
# Streamlit Cloud where the KNIME output directory does not exist.
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = Path(os.environ.get("MODEL_DIR", BASE_DIR))
MODEL_PATH = MODEL_DIR / "image_classifier_vehicles.h5"
CLASS_PATH = MODEL_DIR / "image_classifier_vehicles_classes.json"


@st.cache_resource(show_spinner="Loading model...")
def load_resources():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"ไม่พบไฟล์โมเดล: {MODEL_PATH}")
    if not CLASS_PATH.exists():
        raise FileNotFoundError(f"ไม่พบไฟล์ class mapping: {CLASS_PATH}")

    with open(CLASS_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)

    # The trainer ships resize + scaling alongside the class list. Reading
    # them back is what keeps prediction preprocessing identical to training
    # - hardcoding 128/255.0 here silently destroys accuracy on a model
    # trained at 160 with mobilenet_v2 scaling.
    if isinstance(meta, list):  # legacy file: bare class-name array
        meta = {"class_names": meta, "image_size": [128, 128], "preprocess": "rescale"}

    model = keras_compat.load_model(MODEL_PATH)

    if model.output_shape[-1] != len(meta["class_names"]):
        raise ValueError(
            f"model outputs {model.output_shape[-1]} classes but class mapping has "
            f"{len(meta['class_names'])}: {MODEL_PATH} and {CLASS_PATH} are out of sync"
        )
    return model, meta


def preprocess(image, size, mode):
    image = image.convert("RGB").resize(tuple(size))
    x = np.asarray(image, dtype=np.float32)
    x = x / 127.5 - 1.0 if mode == "mobilenet_v2" else x / 255.0
    return np.expand_dims(x, axis=0)


st.title("🚗 Vehicle Image Classifier")

try:
    model, meta = load_resources()
except Exception as error:  # noqa: BLE001 - surface the reason in the UI
    st.error(f"โหลดโมเดลไม่สำเร็จ: {error}")
    st.stop()

class_names = meta["class_names"]
image_size = meta["image_size"]
preprocess_mode = meta["preprocess"]

st.caption(
    f"อัปโหลดภาพรถ 1 รูป ระบบจะปรับภาพเป็น RGB ขนาด {image_size[0]}×{image_size[1]} "
    f"(preprocess `{preprocess_mode}`) และทำนายด้วย TensorFlow/Keras model"
)
st.caption(f"tensorflow {tf.__version__} · keras {keras_compat.keras_version()}")

uploaded_file = st.file_uploader("เลือกรูปภาพ", type=["jpg", "jpeg", "png", "webp"])

if uploaded_file is None:
    st.info("เลือกรูปภาพเพื่อเริ่มทดสอบโมเดล")
    st.stop()

image = Image.open(uploaded_file).convert("RGB")
st.image(image, caption="รูปภาพที่อัปโหลด", width="stretch")

if st.button("Predict", type="primary", width="stretch"):
    with st.spinner("กำลังทำนาย..."):
        probabilities = model.predict(
            preprocess(image, image_size, preprocess_mode), verbose=0
        )[0]

    probabilities = np.asarray(probabilities, dtype=np.float32)
    best_index = int(np.argmax(probabilities))

    st.success(f"ผลทำนาย: {class_names[best_index]}")
    st.metric("Confidence", f"{float(probabilities[best_index]) * 100:.2f}%")

    st.subheader("ความน่าจะเป็นของแต่ละคลาส")
    st.bar_chart(
        {name: round(float(p) * 100, 2) for name, p in zip(class_names, probabilities)}
    )
