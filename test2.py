import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

st.title("📈 Predictive Model (Year → 2035)")

# ------------------------------------
# Load CSV already inside Codespace
# ------------------------------------
df = pd.read_csv("salaies_cyber_clean.csv")  # your file inside Codespace
st.write("### Dataset Preview")
st.dataframe(df.head())

# ------------------------------------
# Auto-check for Year column
# ------------------------------------
if "Year" not in df.columns:
    st.error("❌ Your dataset does not contain a 'Year' column — model cannot forecast to 2035.")
    st.stop()

# ------------------------------------
# User selects features
# ------------------------------------
st.write("### Select Features (⚠ must include 'Year')")
feature_cols = st.multiselect("Feature columns:", df.columns.tolist(), default=["Year"])

if "Year" not in feature_cols:
    st.warning("⚠ You must include 'Year' in Features for prediction to work.")
    st.stop()

# ------------------------------------
# Select target
# ------------------------------------
st.write("### Select Target")
target_col = st.selectbox("Target column:", df.columns.tolist())

# Proceed only if target chosen
if feature_cols and target_col:

    X = df[feature_cols]
    y = df[target_col]

    # ------------------------------------
    # Encode categorical data
    # ------------------------------------
    cat_cols = X.select_dtypes(include=["object"]).columns
    num_cols = X.select_dtypes(exclude=["object"]).columns

    encoder = OneHotEncoder(handle_unknown="ignore")  # valid for sklearn 1.7.2

    X_cat = encoder.fit_transform(X[cat_cols]).toarray() if len(cat_cols) > 0 else np.empty((len(X), 0))
    X_num = X[num_cols].to_numpy() if len(num_cols) > 0 else np.empty((len(X), 0))

    X_final = np.concatenate([X_num, X_cat], axis=1)

    # ------------------------------------
    # Model training
    # ------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X_final, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor()
    model.fit(X_train, y_train)

    st.success("✅ Model trained successfully!")

    # ------------------------------------
    # Predict future years until 2035
    # ------------------------------------
    last_year = int(df["Year"].max())
    future_years = list(range(last_year + 1, 2036))

    future_df = pd.DataFrame({"Year": future_years})

    # Create future input using same structure
    f_cat = future_df.reindex(columns=cat_cols, fill_value="")
    f_num = future_df.reindex(columns=num_cols, fill_value=0)

    f_cat_encoded = encoder.transform(f_cat).toarray() if len(cat_cols) > 0 else np.empty((len(future_df), 0))
    f_num_values = f_num.to_numpy() if len(num_cols) > 0 else np.empty((len(future_df), 0))

    X_future = np.concatenate([f_num_values, f_cat_encoded], axis=1)
    future_pred = model.predict(X_future)

    prediction_df = pd.DataFrame({
        "Year": future_years,
        "Prediction": future_pred
    })

    st.write("### 📈 Forecast to 2035")
    st.line_chart(prediction_df.set_index("Year"))
    st.dataframe(prediction_df)
