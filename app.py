import tensorflow as tf
import streamlit as st
import numpy as np
from keras.preprocessing import image

def process(imag):
    model = tf.keras.models.load_model('humanorhorse.h5')
    img = image.load_img(imag, target_size=(300, 300))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)

    images = np.vstack([x])
    classes = model.predict(images, batch_size=10)
    st.write(classes[0])
    if classes[0]>0.5:
        st.write(" is a human")
    else:
        st.write(" is a horse")

st.header("Horse Or Human Prediction")
hor = st.file_uploader("upload horse or human",type=["jpg", "jpeg", "png","webp","gif"])
if hor is None:
        hor=  st.camera_input("Capture a photo")
if hor is not None:
    if st.button('Find'):

        process(hor)

