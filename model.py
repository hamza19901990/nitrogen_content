import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
import pickle

# App Title and Description
st.write("""
# MSW Fuel Higher Heating Value Prediction
This app predicts the **Higher Heating Value (HHV)** of municipal solid waste (MSW) fuel using key composition and process parameters!
""")
st.write('---')

# Contextual image (optional: change the filename if needed)
image = Image.open('msw.jpg')  # replace with your relevant image
st.image(image, use_column_width=True)

# Load your dataset
data = pd.read_csv("sludge3.csv")  # replace with your actual CSV filename

# Display basic dataset info
st.subheader('Data Information')
st.write(data.head())
st.write("Missing values in each column:")
st.write(data.isna().sum())
st.write("Correlation Matrix:")
st.write(data.corr())

# Sidebar for input parameters
st.sidebar.header('Specify Input Parameters')

def get_input_features():
    N   = st.sidebar.slider('N (%)', 1.21, 8.85, 4.00)
    O   = st.sidebar.slider('O (%)', 10.50, 30.28, 20.00)
    Fc  = st.sidebar.slider('Fc (%)', 0.70, 18.55, 9.00)
    A   = st.sidebar.slider('A (%)', 14.96, 80.40, 45.00)
    Ht  = st.sidebar.slider('Ht (min)', 0.00, 720.00, 360.00)
    HT  = st.sidebar.slider('HT (°C)', 100.00, 380.00, 240.00)
    Nhc = st.sidebar.slider('Nhc (%)', 0.39, 9.29, 5.00)

    data_user = {
        'N (%)': N,
        'O (%)': O,
        'Fc (%)': Fc,
        'A (%)': A,
        'Ht (min)': Ht,
        'HT (oC)': HT,
        'Nhc (%)': Nhc
    }

    features = pd.DataFrame(data_user, index=[0])
    return features

# Get user input
df = get_input_features()

# Display selected parameters
st.header('Specified Input Parameters')
st.write(df)
st.write('---')

# Load the pre-trained model
load_model = pickle.load(open('gradient_boosting_model', 'rb'))  # update the filename

# Predict HHV
st.header('Predicted Higher Heating Value (MJ/kg)')
prediction = load_model.predict(df)
st.write(prediction[0])
st.write('---')
