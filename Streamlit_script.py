import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score

# External columns that need scaling 
EXTERNAL_COLS = ['Temperature', 'Fuel_Price', 'CPI', 'Unemployment']

# -----------------------------------------
@st.cache_data
def load_training_data():
    df = pd.read_csv(r"D:\anton\DEPI\IBM Data Scientist\0- Graduation Project\anton\clean_data_final.csv")
    return df


# -----------------------------------------
# Train model with StandardScaler on external columns only
# -----------------------------------------
@st.cache_resource
def train_model(df):
    X = df.drop("Weekly_Sales", axis=1)
    y = df["Weekly_Sales"]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale only external columns (Temperature, Fuel_Price, CPI, Unemployment)
    scaler = StandardScaler()
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    
    X_train_scaled[EXTERNAL_COLS] = scaler.fit_transform(X_train[EXTERNAL_COLS])
    X_test_scaled[EXTERNAL_COLS] = scaler.transform(X_test[EXTERNAL_COLS])

    # Train XGBoost model 
    model = XGBRegressor(
        objective="reg:squarederror",
        n_estimators=300,
        learning_rate=0.2,
        max_depth=7,
        subsample=0.8,
        random_state=42
    )
    model.fit(X_train_scaled, y_train)

    # Evaluate
    preds = model.predict(X_test_scaled)
    r2 = r2_score(y_test, preds)
    mse = mean_squared_error(y_test, preds)
    rmse = np.sqrt(mse)

    return model, scaler, X.columns.tolist(), r2, mse, rmse


# -----------------------------------------
# Preprocess uploaded CSV for prediction
# Assumes uploaded data is also clean (same format as training data)
# -----------------------------------------
def preprocess_for_prediction(df, feature_cols):
    # Ensure correct column order
    df = df[feature_cols]
    return df


# -------------------------------------------------------------
# Streamlit UI
# -------------------------------------------------------------
st.title("Walmart Weekly Sales Prediction App")

# Load clean data and train model
train_df = load_training_data()
model, scaler, feature_cols, r2, mse, rmse = train_model(train_df)

st.subheader("Model Performance")
col1, col2, col3 = st.columns(3)
col1.metric("R² Score", f"{r2:.4f}")
col2.metric("MSE", f"{mse:,.2f}")
col3.metric("RMSE", f"${rmse:,.2f}")

st.subheader("Upload CSV to Predict")
st.info("Upload a clean CSV file with the same format as training data (already preprocessed with date features and label encoded Type)")

uploaded = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded:
    input_df = pd.read_csv(uploaded)

    # Preprocess for prediction
    processed_df = preprocess_for_prediction(input_df.copy(), feature_cols)

    # Scale only external columns
    scaled_df = processed_df.copy()
    scaled_df[EXTERNAL_COLS] = scaler.transform(processed_df[EXTERNAL_COLS])
    
    # Predict
    preds = model.predict(scaled_df)

    input_df["Predicted_Weekly_Sales"] = preds

    st.write("### Predictions")
    st.write(input_df.head())

    st.download_button(
        label="Download Predictions",
        data=input_df.to_csv(index=False),
        file_name="predictions.csv",
        mime="text/csv"
    )
