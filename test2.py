import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

st.title("📈 Predictive Model (up to 2035)")

# ------------------------------------
# Load CSV already in Codespace
# ------------------------------------
df = pd.read_csv("salaries_cyber_clean.csv")   # <-- your file inside Codespace
st.write("### Dataset Preview")
st.dataframe(df.head())

# ------------------------------------
# Select columns
# ------------------------------------
st.write("### Select Feature Columns")
feature_cols = st.multiselect("Features:", df.columns.tolist())

st.write("### Select Target Column")
target_col = st.selectbox("Target:", df.columns.tolist())

if feature_cols and target_col:
    X = df[feature_cols]
    y = df[target_col]

    # ------------------------------------
    # OneHotEncoder (correct for sklearn 1.7.2)
    # ------------------------------------
    cat_cols = X.select_dtypes(include=["object"]).columns
    num_cols = X.select_dtypes(exclude=["object"]).columns

    encoder = OneHotEncoder(handle_unknown="ignore")   # no sparse parameter

    X_cat = encoder.fit_transform(X[cat_cols]).toarray() if len(cat_cols) > 0 else np.empty((len(X), 0))
    X_num = X[num_cols].to_numpy() if len(num_cols) > 0 else np.empty((len(X), 0))

    X_final = np.concatenate([X_num, X_cat], axis=1)

    # ------------------------------------
    # Train-test split
    # ------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X_final, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor()
    model.fit(X_train, y_train)

    st.success("Model trained successfully!")

    # ------------------------------------
    # Predict future years to 2035
    # ------------------------------------
    st.write("### Predictions up to 2035")

    if "Year" in feature_cols:
        last_year = int(df["Year"].max())
        future_years = list(range(last_year + 1, 2036))

        future_df = pd.DataFrame({"Year": future_years})

        # Build future input rows
        f_cat = future_df.select_dtypes(include=["object"]).reindex(columns=cat_cols, fill_value="")
        f_num = future_df.select_dtypes(exclude=["object"]).reindex(columns=num_cols, fill_value=0)

        f_cat_encoded = encoder.transform(f_cat).toarray() if len(cat_cols) > 0 else np.empty((len(future_df), 0))
        f_num_values = f_num.to_numpy() if len(num_cols) > 0 else np.empty((len(future_df), 0))

        X_future = np.concatenate([f_num_values, f_cat_encoded], axis=1)

        future_pred = model.predict(X_future)

        result = pd.DataFrame({
            "Year": future_years,
            "Prediction": future_pred
        })

        st.line_chart(result.set_index("Year"))
        st.write(result)
    else:
        st.warning("No 'Year' column selected — cannot generate predictions up to 2035.")
