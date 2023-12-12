import streamlit as st
from PIL import Image
from fastai.vision.all import load_learner, ClassificationInterpretation
import torch

#Title
st.title("Fighterjets image classification")
st.subheader(':blue[_Created by Wouter Selis & Kieran Cornelissen & Gilles Witters_] :male-technologist:', divider='rainbow')


#image uploader to predict
allowed_types = ("jpg", "jpeg", "png")
uploaded_file = st.file_uploader("Choose an image...", type=allowed_types)

#load model
loaded_model = load_learner("models/model50Extra")


if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image.", use_column_width=True)

    #predicting image
    with st.spinner('Classifying...'):
        #predict image
        predictions=loaded_model.predict(image)
        
        st.write(f"Prediction: {predictions[0].replace('_', ' ')}")
        st.write(f"Precision: {round(float(torch.max(predictions[2])*100))}%")
    
    #Compare confusion matrix
    st.write("")
    st.subheader("Confusion matrix models: ")
    st.write("")
    

    st.write("Our model:")
    st.image('./images/CM_OurModel.png')

    st.write("Google teachable machine model:")
    st.image('./images/CM_GTM.png')

    #Compare losses
    st.write("")
    st.subheader("Losses: ")
    st.write("")
    

    st.write("Our loss per epoch:")
    st.image('./images/Loss_OurModel.png')

    st.write("Google teachable machine losses per epoch:")
    st.image('./images/Loss_GTM.png')


