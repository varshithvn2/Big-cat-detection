import streamlit as st
from PIL import Image
from roboflow import Roboflow

# Initialize Roboflow
rf = Roboflow(api_key="VWJEsV25FRq11fja5SML")
project = rf.workspace().project("big-cat-classification-jpicx")
model = project.version(2).model

# Streamlit App
st.title("Big Cat Classification App")

uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image.", use_column_width=True)

    # Make prediction using Roboflow
    prediction_image = model.predict(uploaded_file).image

    # Display the annotated image
    st.image(prediction_image, caption="Prediction.", use_column_width=True)
