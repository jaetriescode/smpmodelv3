import streamlit as st
import pandas as pd
from sklearn.preprocessing import  LabelEncoder
import numpy as np
import pickle
st.header("Failure Load Prediction - please input all values in millimetres")
st.text_input("Enter your Name: ", key="name")
data = pd.read_excel("smpdata.xlsx")


encoder = LabelEncoder()
encoder.classes_ = np.load('classes.npy',allow_pickle=True)
rfc_model=pickle.load(open('rfc_model.pkl'))


e1=st.number_input("Enter end distance")
e2=st.number_input("Enter edge distance")
p1=st.number_input("Enter longitudinal pitch")
p2=st.number_input("Enter transverse pitch")
d0=st.number_input("Emter bolt hole diameter")
d=st.number_input("Enter bolt diameter")
t=st.number_input("Enter plate thickness")
n=st.number_input("Enter number of bolts")
fy=st.number_input("Enter material yield strength")

if st.button('Make Prediction'):
    inputs = np.expand_dims(
        [e1, e2, p1, p2, d0, d, t, n, fy], 0)
    prediction = rfc_model.predict(inputs)
    print("final pred", np.squeeze(prediction, -1))
    st.write(f"The failure load is: {np.squeeze(prediction, -1):.2f}kN")

    


