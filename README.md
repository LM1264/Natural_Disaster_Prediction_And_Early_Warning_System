# 🌍 Natural Disaster Prediction & Early Warning System  

A Machine Learning project to predict **earthquake magnitudes** using regression models (Linear Regression, Random Forest Regressor, and Gradient Boosting Regressor).  
The project includes both an **exploratory Colab/Jupyter Notebook** for model training & evaluation and a **Streamlit web app** for deployment.  

---

## 📌 Project Overview  
This project leverages earthquake data from **Kaggle** (with records since 1965 for magnitudes ≥ 5.5).  
The goal is to build a **prediction and early warning system** for earthquake magnitudes.  

- **Dataset Features:** Date, Time, Location, Depth, Magnitude, Source, etc.  
- **Target:** Earthquake Magnitude.  
- **ML Models:**  
  - Linear Regression  
  - Random Forest Regressor  
  - Gradient Boosting Regressor  

---

## 🛠️ Tools & Technologies  
- **Programming Language:** Python  
- **Libraries:**  
  - `pandas`, `numpy` → data preprocessing  
  - `matplotlib`, `seaborn` → visualization  
  - `scikit-learn` → ML models & metrics  
  - `joblib` → model persistence  
  - `streamlit` → web app deployment
- **Dataset Source:** Kaggle (Earthquake dataset)
- **Environment:** Jupyter Notebook / Google Colab

---

## 📂 Project Structure  

├── PROJECT_Natural_Disaster__Prediction__&_Early__Warning_System.ipynb # Jupyter Notebook (training + evaluation)

├── app.py # Streamlit App

├── lr_model.pkl

├── rf_model.pkl

├── gb_model.pkl

├── scaler.pkl

├── label_encoders.pkl

├── feature_columns.pkl

├── results_df.pkl

├── requirements.txt

└── README.md

---

## 🚀 Running the Project  
### 🔗 Option 1: Try Online by my webapp (Recommended)  
You can directly use the deployed app here: 

👉 [Natural Disaster Prediction & Early Warning System](https://naturaldisasterpredictionandearlywarningsystem-lakshay12.streamlit.app/)  
### 💻 Option 2: Run Locally 
### 1️⃣ Clone the Repository  
git clone https://github.com/LM1264/Natural_Disaster_Prediction_And_Early_Warning_System.git

cd Natural_Disaster_Prediction_And_Early_Warning_System
### 2️⃣ Install Dependencies
pip install -r requirements.txt
### 3️⃣ Run the Colab/Jupyter Notebook
jupyter notebook PROJECT_Natural_Disaster__Prediction__&_Early__Warning_System.ipynb
### 4️⃣ Run the Streamlit App
streamlit run app.py

---

## 📊 Model Evaluation Metrics

For each model, the following metrics were calculated:
- MSE (Mean Squared Error)
- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)
- R² Score
- A comparison table & bar chart are displayed in the app.
 
---

## 📈 Streamlit App Features

- Upload custom earthquake dataset (CSV).
- Select features and target column.
- Compare model performance.
- Get prediction results in real time.

---

## 📜 Requirements

Example [requirements.txt](requirements.txt) :
- streamlit
- pandas
- numpy
- scikit-learn
- seaborn
- matplotlib
- joblib
