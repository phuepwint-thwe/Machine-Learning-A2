import streamlit as st
import pickle
import numpy as np
from linear_regression import LinearRegression, NormalPenalty, LassoPenalty, RidgePenalty, ElasticPenalty, Normal, Lasso, Ridge, ElasticNet

# Load the old model
filename1 = r'./car_selling_price_a1.model'
model_old = pickle.load(open(filename1, 'rb'))

# Load the new model
filename2 = r'./car_selling_price_a2.model'
loaded_data2 = pickle.load(open(filename2, 'rb'))
model_new = loaded_data2['model']
scaler_new = loaded_data2['scaler']
year_default_new = loaded_data2['year_default']
max_power_default_new = loaded_data2['max_power_default']
mileage_default_new = loaded_data2['mileage_default']

# Prediction function for the old model
def prediction_old(engine, max_power,mileage):
    sample = np.array([[engine, max_power, mileage]])
    sample_scaled = model_old.transform(sample)
    result = np.exp(model_old.predict(sample_scaled))
    return result

# Prediction function for the new model
def prediction_new(year, max_power,mileage):
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
engine = st.text_input('Enter engine size (default: will be used if left blank)')
year = st.text_input('Enter year (default: will be used if left blank)')
max_power = st.text_input('Enter max_power (default: will be used if left blank)')
mileage = st.text_input('Enter mileage (default: will be used if left blank)')

# Set defaults if inputs are empty
if model_choice == 'Old Model':
    engine = float(engine) if engine else engine_default_old
    max_power = float(max_power) if max_power else max_power_default_old
    mileage = float(mileage) if mileage else mileage_default_old
    if st.button('Predict'):
        result = prediction_old(engine, max_power,mileage)
        st.write(f'Estimated price: {int(result[0])}')
else:
    year = float(year) if engine else year_default_new
    max_power = float(max_power) if max_power else max_power_default_new
    mileage = float(mileage) if mileage else mileage_default_new
    if st.button('Predict'):
        result = prediction_new(engine, max_power, mileage)
        st.write(f'Estimated price: {int(result[0])}')