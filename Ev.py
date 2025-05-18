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

 # Step 3: User selects target to predict
    selected_target = st.selectbox("Select the target variable to predict", targets)

    # Features and label
    X = df.drop(columns=targets)
    y = df[selected_target]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train and evaluate
    st.subheader(f"Model Results for Target: {selected_target}")
    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        mse = mean_squared_error(y_test, preds)
        r2 = r2_score(y_test, preds)

        st.write(f"**{name}**")
        st.write(f"- MSE: {mse:.2f}")
        st.write(f"- R² Score: {r2:.2f}")

        # Display predictions
        results_df = pd.DataFrame({
            f'{selected_target}_Actual': y_test.values,
            f'{selected_target}_Predicted_{name}': preds
        })
        st.write(results_df.head())

    # Download predictions (optional)
    st.markdown("### Download all model predictions")
    all_results = pd.concat([results_df], axis=1)
    st.download_button(
        label="Download Results as Excel",
        data=all_results.to_excel(index=False, engine='openpyxl'),
        file_name='ElectricCar_Predictions.xlsx'
    )
