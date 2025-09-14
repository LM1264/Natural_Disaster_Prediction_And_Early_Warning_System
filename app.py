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
cat_dict = joblib.load("cat_dict.pkl")
results_df = joblib.load("results_df.pkl")   # store model performance table

st.set_page_config(page_title="Disaster Prediction & Early Warning System", layout='wide')  
st.title("🌍 Natural Disaster Prediction & Early Warning System")  
st.write("Predicting earthquake magnitudes using Machine Learning models.") 

# Sidebar inputs 
st.sidebar.header("Input Parameters") 

def get_user_input(): 
    typ = st.sidebar.selectbox("Type", cat_dict['Type']) 
    mt = st.sidebar.selectbox("Magnitude Type", cat_dict['Magnitude Type']) 
    eid = st.sidebar.selectbox("ID", cat_dict['ID']) 
    s = st.sidebar.selectbox("Source", cat_dict['Source']) 
    ls = st.sidebar.selectbox("Location Source", cat_dict['Location Source']) 
    tme = st.sidebar.slider("Time Duration", 0, 23, 8) 
    dt = st.sidebar.slider("Date", 0, 200, 50)  
    stat = st.sidebar.selectbox("Check Status", ['True', 'False']) 
    ms = st.sidebar.number_input("Magnitude Source", 0.0, 500.0, 100.0)

    Stat_final = 1 if stat == 'True' else 0 

    data = {  
        'Type': cat_dict['Type'].tolist().index(typ),  
        'Magnitude Type': cat_dict['Magnitude Type'].tolist().index(mt),  
        'ID': cat_dict['ID'].tolist().index(eid),  
        'Source': cat_dict['Source'].tolist().index(s),  
        'Location Source': cat_dict['Location Source'].tolist().index(ls),  
        'Time': tme,  
        'Date': dt,  
        'Status': Stat_final, 
        'Magnitude Source': ms  
    }
    return pd.DataFrame([data])  

input_df = get_user_input()

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
    st.success(f"**Predicted Magnitude Type:** {results_df[prediction[0]]}")

    # --- Show Model Performance Comparison ---
    st.subheader("📊 Model Performance Comparison")
    st.dataframe(results_df)   # show results table

    # Bar Chart for R² Score
    st.subheader("📈 R² Score Comparison")
    st.bar_chart(results_df.set_index("Model")[["R² Score"]])
