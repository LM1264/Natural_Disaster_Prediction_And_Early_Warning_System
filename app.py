import streamlit as st 
import pandas as pd 
import numpy as np 
import joblib 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Load Required models and encoders
model_1 = joblib.load("lr_model.pkl")
model_2 = joblib.load("rf_model.pkl")
model_3 = joblib.load("gb_model.pkl")
scaler = joblib.load("scaler.pkl")
label_encoders = joblib.load("label_encoders.pkl")# dictionary of encoders
results_df = joblib.load("results_df.pkl")   # model performance table

st.set_page_config(page_title="Disaster Prediction & Early Warning System", layout='wide')  
st.title("🌍 Natural Disaster Prediction & Early Warning System")  
st.write("Predicting earthquake magnitudes using Machine Learning models.") 

# Sidebar inputs 
st.sidebar.header("Input Parameters") 

def get_user_input():
    # Numerical features
    date = st.sidebar.number_input("Date", min_value=0, max_value=3000, value=200)
    time = st.sidebar.slider("Time (hour)", 0, 23, 12)
    latitude = st.sidebar.number_input("Latitude", -90.0, 90.0, 20.0)
    longitude = st.sidebar.number_input("Longitude", -180.0, 180.0, 78.0)
    depth = st.sidebar.number_input("Depth", 0.0, 700.0, 10.0)
    depth_error = st.sidebar.number_input("Depth Error", 0.0, 50.0, 0.0)
    depth_stations = st.sidebar.number_input("Depth Seismic Stations", 0, 50, 0)
    magnitude = st.sidebar.number_input("Magnitude", 0.0, 10.0, 5.0)
    magnitude_error = st.sidebar.number_input("Magnitude Error", 0.0, 2.0, 0.0)
    magnitude_stations = st.sidebar.number_input("Magnitude Seismic Stations", 0, 50, 0)
    azimuthal_gap = st.sidebar.number_input("Azimuthal Gap", 0.0, 360.0, 180.0)
    horizontal_distance = st.sidebar.number_input("Horizontal Distance", 0.0, 1000.0, 0.0)
    horizontal_error = st.sidebar.number_input("Horizontal Error", 0.0, 100.0, 0.0)
    rms = st.sidebar.number_input("Root Mean Square", 0.0, 20.0, 1.0)

    # Encoded categorical features (dropdowns from your LabelEncoder)
    typ = st.sidebar.selectbox("Type", label_encoders['Type'].classes_)
    mt = st.sidebar.selectbox("Magnitude Type", label_encoders['Magnitude Type'].classes_)
    eid = st.sidebar.text_input("ID", "abc123")  #encode manually
    src = st.sidebar.selectbox("Source", label_encoders['Source'].classes_)
    lsrc = st.sidebar.selectbox("Location Source", label_encoders['Location Source'].classes_)
    msrc = st.sidebar.selectbox("Magnitude Source", label_encoders['Magnitude Source'].classes_)
    status = st.sidebar.selectbox("Status", ['True', 'False'])
    status_final = 1 if status == "True" else 0

    # Put everything together in correct order
    data = {
        'Date': date,
        'Time': time,
        'Latitude': latitude,
        'Longitude': longitude,
        'Type': label_encoders['Type'].transform([typ])[0],
        'Depth': depth,
        'Depth Error': depth_error,
        'Depth Seismic Stations': depth_stations,
        'Magnitude': magnitude,
        'Magnitude Type': label_encoders['Magnitude Type'].transform([mt])[0],
        'Magnitude Error': magnitude_error,
        'Magnitude Seismic Stations': magnitude_stations,
        'Azimuthal Gap': azimuthal_gap,
        'Horizontal Distance': horizontal_distance,
        'Horizontal Error': horizontal_error,
        'Root Mean Square': rms,
        'ID': 0,  # if you want keep encoded as 0 for now
        'Source': label_encoders['Source'].transform([src])[0],
        'Location Source': label_encoders['Location Source'].transform([lsrc])[0],
        'Magnitude Source': label_encoders['Magnitude Source'].transform([msrc])[0],
        'Status': status_final
    }

    return pd.DataFrame([data])  

input_df = get_user_input()
# Load training feature names (saved during training)
training_columns = joblib.load("feature_columns.pkl")   # <-- Save this once during training

# Ensure input_df has the same columns & order
for col in training_columns:
    if col not in input_df.columns:
        input_df[col] = 0   # fill missing columns with default value (0)
# Reorder to match training
input_df = input_df[training_columns]

# --- Model Selection ---
model_choice = st.sidebar.radio("Choose Model:", ["Linear Regression", "Random Forest", "Gradient Boosting"])

if st.button("Predict Magnitude Type"):
    if model_choice == "Linear Regression":
        prediction = model_1.predict(input_df)
    elif model_choice == "Random Forest":
        prediction = model_2.predict(input_df)
    else:
        prediction = model_3.predict(input_df)

    st.subheader("Prediction Result") 
    st.success(f"**Predicted Magnitude Type:** {prediction[0]}")

    # --- Show Model Performance Comparison ---
    st.subheader("📊 Model Performance Comparison")
    st.dataframe(results_df)   # show results table

    # Bar Chart for R² Score
    st.subheader("📈 R² Score Comparison")
    results_df.rename(columns={"R²": "R² Score"}, inplace=True)
    st.bar_chart(results_df.set_index("Model")[["R² Score"]])

