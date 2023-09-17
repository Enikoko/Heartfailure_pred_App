import pickle
import sklearn
import numpy as np
import streamlit as st
#To Display Images
from PIL import Image

#loading the saved model
loaded_model = pickle.load(open('trained_model_heartfailure.sav', 'rb'))

#creating a function for prediction

def heartfailure_prediction(input_data):
    # changing the input_data to numpy array
    input_data_as_numpy_array = np.asarray(input_data)

    # reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

    prediction = loaded_model.predict(input_data_reshaped)
    print(prediction)

    if (prediction[0] == 0):
        return 'The person would have an heart attack'
    else:
        return 'The person would not have an heart attack'


def main():
    # display image
    img = Image.open("heart.png")
    new_image = img.resize((700, 200))
    st.image(new_image)
    # let's display
    #st.image(img, width=700)

    # giving a title
    st.title('Heart Failure Prediction Web App')

    # getting the input data from the user

    age = st.text_input('How Old are you?')
    anaemia = st.text_input('Are you anaemic? : 0 for Yes and 1 for No')
    creatinine_phosphokinase = st.text_input('input your creatinine phosphokinase level')
    diabetes = st.text_input('Are you diabetic? : 0 for Yes and 1 for No')
    ejection_fraction = st.text_input('input your ejection fraction')
    high_blood_pressure = st.text_input('Do you have a high blood pressure? : 0 for Yes and 1 for No')
    platelets = st.text_input('Input your platelets value')
    serum_creatinine = st.text_input('Input your serum creatinine Value')
    serum_sodium = st.text_input('Input your serum_sodium')
    sex = st.text_input('Are you male or female? : 0 for Male and 1 for Female')
    smoking = st.text_input('Are you a smoker? : 0 for Yes and 1 for No')
    time = st.text_input("input doctor's time with patient")

    # code for Prediction
    heartfailure = ''

    # creating a button for Prediction

    if st.button('Heart Failure Test Result'):
        heartfailure = heartfailure_prediction([age, anaemia, creatinine_phosphokinase, diabetes, ejection_fraction,
                                         high_blood_pressure, platelets, serum_creatinine, serum_sodium, sex,
                                         smoking, time])

    st.success(heartfailure)


if __name__ == '__main__':
    main()
