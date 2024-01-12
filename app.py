import streamlit as st
from PIL import Image
from roboflow import Roboflow
import io

# Initialize Roboflow
rf = Roboflow(api_key="VWJEsV25FRq11fja5SML")
project = rf.workspace().project("big-cat-classification-jpicx")
model = project.version(2).model

# Streamlit App
st.title("Big Cat Classification App")

# Allow user to upload an image
uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image.", use_column_width=True)

    # Get the file path of the uploaded image
    uploaded_file_path = uploaded_file.name

    # Make prediction using Roboflow on the image file
    prediction_image = model.predict(uploaded_file_path).image

    # Display the annotated image with predictions
    st.image(prediction_image, caption="Prediction Image.", use_column_width=True)
