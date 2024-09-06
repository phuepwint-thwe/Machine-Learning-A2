import streamlit as st
import pickle
import numpy as np
import matplotlib as plt
from linear_regression import LinearRegression, NormalPenalty, LassoPenalty, RidgePenalty, ElasticPenalty, Normal, Lasso, Ridge, ElasticNet

# Load the old model
filename1 = './car_selling_price_a1.model'
with open(filename1, 'rb') as file1:
    model_old = pickle.load(file1)

# Load the new model
filename2 = './car_selling_price_a2.model'
with open(filename2, 'rb') as file2:
    loaded_data2 = pickle.load(file2)
model_new = loaded_data2['model']
scaler_new = loaded_data2['scaler']
year_default_new = loaded_data2['year_default']
max_power_default_new = loaded_data2['max_power_default']
mileage_default_new = loaded_data2['mileage_default']

# Prediction function for the old model (no scaling)
def prediction_old(engine, max_power, mileage):
    sample = np.array([[engine, max_power, mileage]])
    result = np.exp(model_old.predict(sample))
    return result

# Prediction function for the new model (with scaling)
def prediction_new(year, max_power, mileage):
    sample = np.array([[year, max_power, mileage]])
    sample_scaled = scaler_new.transform(sample)
    intercept = np.ones((sample_scaled.shape[0], 1))
    sample_scaled = np.concatenate((intercept, sample_scaled), axis=1)
    result = np.exp(model_new.predict(sample_scaled))
    return result

# Streamlit app
st.title('Car Price Prediction App')

# Sidebar to choose the model
model_choice = st.sidebar.selectbox('Choose a model', ('Old Model', 'New Model'))

st.write(f'You selected the **{model_choice}**.')

# Input fields for car attributes
if model_choice == 'Old Model':
    engine = st.number_input('Engine (in CC)', min_value=500, max_value=5000, step=1)
    max_power = st.number_input('Max Power (in BHP)', min_value=20.0, max_value=500.0, step=1.0)
    mileage = st.number_input('Mileage (in KMPL)', min_value=5.0, max_value=50.0, step=0.1)
    if st.button('Predict'):
        result = prediction_old(engine, max_power, mileage)
        st.write(f'Estimated price: {int(result[0])}')
else:
    year = st.number_input('Year', min_value=1980, max_value=2024, step=1, value=int(year_default_new))
    max_power = st.number_input('Max Power (in BHP)', min_value=20.0, max_value=500.0, step=1.0, value=float(max_power_default_new))
    mileage = st.number_input('Mileage (in KMPL)', min_value=5.0, max_value=50.0, step=0.1, value=float(mileage_default_new))
    if st.button('Predict'):
        result = prediction_new(year, max_power, mileage)
        st.write(f'Estimated price: {int(result[0])}')

st.write("This is a prototype app for predicting car selling prices using both an old and a new model.")