import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.impute import SimpleImputer

# Streamlit App
st.set_page_config(page_title="Disaster Prediction & Early Warning System", layout="wide")
st.title("🌍 Natural Disaster Prediction & Early Warning System")
st.write("Predicting earthquake magnitudes using Machine Learning models.")

# Sidebar for dataset upload
uploaded_file = st.sidebar.file_uploader("📂 Upload Earthquake Dataset (CSV)", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("📊 Dataset Preview")
    st.dataframe(df.head())

    # Basic info
    st.write(f"**Dataset Shape:** {df.shape}")
    st.write("**Columns:**", list(df.columns))

    # Select target & features
    target_col = st.sidebar.selectbox("🎯 Select Target Column", df.columns)
    feature_cols = st.sidebar.multiselect(
        "⚙️ Select Feature Columns", [col for col in df.columns if col != target_col]
    )

    if target_col and feature_cols:
        X = df[feature_cols]
        y = df[target_col]

        # Handle missing values
        imputer = SimpleImputer(strategy="mean")
        X = pd.DataFrame(imputer.fit_transform(X), columns=feature_cols)

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Models
        models = {
            "Linear Regression": LinearRegression(),
            "Random Forest": RandomForestRegressor(random_state=42),
            "Gradient Boosting": GradientBoostingRegressor(random_state=42),
        }

        results = []
        # Training & Evaluation
        for name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test, y_pred)
            results.append([name, mse, mae, rmse, r2])

        # Results Table
        results_df = pd.DataFrame(
            results, columns=["Model", "MSE", "MAE", "RMSE", "R² Score"]
        )
        st.subheader("📈 Model Performance Comparison")
        st.dataframe(results_df.style.format({"MSE": "{:.4f}", "MAE": "{:.4f}", "RMSE": "{:.4f}", "R² Score": "{:.4f}"}))

        # Chart
        st.bar_chart(results_df.set_index("Model")["R² Score"])

        # Best Model
        best_model = results_df.loc[results_df["R² Score"].idxmax()]
        st.success(f"✅ Best Model: {best_model['Model']} with R² = {best_model['R² Score']:.4f}")

else:
    st.info("👆 Upload a dataset to get started.")
