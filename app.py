import streamlit as st
from PIL import Image
from roboflow import Roboflow

# Initialize Roboflow
rf = Roboflow(api_key="VWJEsV25FRq11fja5SML")
project = rf.workspace().project("big-cat-classification-jpicx")
model = project.version(2).model

# Streamlit App
st.title("Big Cat Classification App")

# Allow user to upload an image
uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    # Get the uploaded image
    image = Image.open(uploaded_file)

    # Save the uploaded image locally
    uploaded_file_path = "uploaded_image.jpg"
    image.save(uploaded_file_path)

    # Make prediction using Roboflow on the local image
    prediction = model.predict(uploaded_file_path).json()

    # Extract the highest probability class and its confidence
    highest_prob_class = prediction['predictions'][0]['predicted_classes'][0]
    confidence = round(prediction['predictions'][0]['predictions'][highest_prob_class]['confidence'])*100

    # Display the highest probability class
    st.write(f"This is a {highest_prob_class}")

    # Preview Image
    st.image(image, caption="Uploaded Image.", use_column_width=True)
