import streamlit as st
import pickle
import numpy as np
import os

model_path = r'car_selling_price.model'
if not os.path.isfile(model_path):
    raise FileNotFoundError(f"Model file not found at {model_path}")

with open(model_path, 'rb') as model_file:
    # Load your model
    loaded_model = pickle.load(model_file)


#input
st.title("Car Price Prediction")

st.write("Enter the car features to predict the selling price:")


engine = st.number_input('Engine (in CC)', min_value=500, max_value=5000, step=1)
mileage = st.number_input('Mileage (in KMPL)', min_value=5.0, max_value=50.0, step=0.1)
max_power = st.number_input('Max Power (in BHP)', min_value=20.0, max_value=500.0, step=1.0)

if st.button('Predict Selling Price'):
    # Prepare the feature array for prediction
    features = np.array([[engine, mileage, max_power]])
    
    # Predict the selling price
    predicted_price_log = loaded_model.predict(features)
    
    # Reverse the log transformation to get the actual price
    predicted_price = np.exp(predicted_price_log)
    
    # Display the result
    st.write(f"Predicted Selling Price: {predicted_price[0]:,.2f}")
    pass

st.write("This is a prototype app for predicting car selling prices using a pre-trained model.")
