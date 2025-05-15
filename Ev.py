import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# App Title
st.title("🚗 Electric Car Performance Predictor")
st.write("Train ML models to predict Range, Top Speed, and Acceleration of electric cars.")

# Step 1: Upload CSV
uploaded_file = st.file_uploader("Upload your Electric Car dataset (.csv)", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success("Dataset loaded successfully!")
    st.write(df.head())

    # Step 2: Data Preprocessing
    df['FastCharge_KmH'] = df['FastCharge_KmH'].replace('Unknown', '0 km/h')
    df['FastCharge_KmH'] = df['FastCharge_KmH'].str.extract(r'(\d+)').astype(float)
    df['FastCharge_KmH'].fillna(df['FastCharge_KmH'].median(), inplace=True)

    categorical_cols = ['Brand', 'Model', 'RapidCharge', 'PowerTrain', 'PlugType', 'BodyStyle', 'Segment']
    for col in categorical_cols:
        if df[col].dtype == 'object':
            df[col] = LabelEncoder().fit_transform(df[col])

    targets = ['Range_Km', 'TopSpeed_KmH', 'AccelSec']
    models = {
        'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42),
        'LinearRegression': LinearRegression(),
        'GradientBoosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
    }
  # This Code is made by Anshika Singh
