import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2

st.set_page_config(page_title="my first app")

st.title("Yolo Website Uncle Engineer") 
st.write("อัปโหลด แล้วรันโมเดล")

@st.cache_resource
def load_model():
    model = YOLO('yolov8n.pt')
    return model

try:
    model = load_model()
    st.success("โหลดโมเดลแล้ว เยยยยย้")
except:
    st.error(f"error")
    st.stop()


uplorded_file = st.file_uploader("Choose am image", type=["jpg","jpeg","png"])

if uplorded_file:
    image = Image.open(uplorded_file).convert("RGB")
    st.image(image, caption = "รูป", use_container_width=True)

    if st.button("รัน YOLO"):
        with st.spinner("กำลังจับภาพให้คุณ..."):

            results = model(image)

            plotted = results[0].plot()
            plotted = cv2.cvtColor(plotted, cv2.COLOR_BGR2RGB)
            st.image(plotted, use_container_width=True )

