import streamlit as st
from PIL import Image
from fastai.vision.all import load_learner
import torch
import pathlib
plt = platform.system()
if plt == 'Windows': pathlib.PosixPath = pathlib.WindowsPath

#Title
st.title("Fighterjets image classification")
st.subheader(':blue[_Created by Wouter Selis & Kieran Cornelissen & Gilles Witters_] :male-technologist:', divider='rainbow')

st.write("In this streamlit app you can test our classification model for fighterjets. The goal of this project is to classify 5 different fighterjets: F-1117 Nighthawk, F-16 Fighting Falcon, F-22 Raptor, F-4 Phantom, MiG-29 Fulcrum. We also compared our model with Google Teachable Machine and you can see the results below. Our model is created with fastAi, resnet50 and uses the fit_one_cycle() method for better performance in speed and accuracy. To test our model you can upload an image and view the prediction and how certain our model is.")

#image uploader to predict
allowed_types = ("jpg", "jpeg", "png")
uploaded_file = st.file_uploader("Choose an image...", type=allowed_types)

#load model
loaded_model = load_learner("./models/model50Extra")


if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image.", use_column_width=True)

    #predicting image
    with st.spinner('Classifying...'):
        #predict image
        predictions=loaded_model.predict(image)

        prediction = f"{predictions[0].replace('_', ' ')}"
        st.write(f"Our model is {round(float(torch.max(predictions[2])*100))}% certain it is a {prediction}.")
        
    st.divider()

#Compare confusion matrix
st.write("")
st.subheader("Confusion matrix models: ")
st.write("")

st.write("Here we compare the confusion matrix from our model with the confusion matrix from Google Teachable Machine. As you can see the accuracy of our model is in all cases higher. This means our model is more accurate in classifying fighterjets.")

with st.container():
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Our model:**")
        st.image('./images/CM_OurModel.png')
        st.write("Accuracy per class:")
        lst=["F-1117 Nighthawk: 96%","F-16 Fighting Falcon: 75%","F-22 Raptor: 84%","F-4 Phantom: 91%","MiG-29 Fulcrum: 97%"]
        for i in lst:
            st.markdown("- " + i)
       
    with col2:
        st.write("**Google teachable machine model:**")
        st.image('./images/CM_GTM.png')
        st.write("Accuracy per class:")
        lst_GTM=["F-1117 Nighthawk: 93%","F-16 Fighting Falcon: 58%","F-22 Raptor: 78%","F-4 Phantom: 77%","MiG-29 Fulcrum: 80%"]
        for i in lst_GTM:
            st.markdown("- " + i)


st.divider()
#Compare losses
st.write("")
st.subheader("Losses: ")
st.write("")
st.write("Here we can compare the loss of the models. The difference between our loss and the Google Teachable Machine loss is that our validation loss has a big peak and rapidly descents. This means our model was overfitting and the learning rate was changed accordly due to adam. With Google Teachable Machine the validation loss first started to increas and overfit but then it stabilized and increases slightly. We can also see that our final validation loss ends under 0.5 and Google Teachable Machine's validation loss ends on +-0.8.")

with st.container():
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Our loss per epoch:**")
        st.image('./images/Loss_OurModel.png')
    with col2:
        st.write("**Google teachable machine losses per epoch:**")
        st.image('./images/Loss_GTM.png')


