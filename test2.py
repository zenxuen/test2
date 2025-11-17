import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
import plotly.express as px

# ---------------------------------------------------------
# Page Config
# ---------------------------------------------------------
st.set_page_config(
    page_title="Salary Prediction Dashboard",
    layout="wide",
    page_icon="💼"
)

st.title("💼 Salary Prediction Dashboard (2021–2025)")


# ---------------------------------------------------------
# Sidebar Upload
# ---------------------------------------------------------
st.sidebar.header("📤 Upload Dataset")
uploaded = st.file_uploader("C:\\Users\\user\\Downloads\\Assignment.zip\\Assignment\\salaries_cyber_clean", type=["csv"], key="csv")

# Stop unless file uploaded
if uploaded is None:
    st.info("Please upload a CSV file from the sidebar to continue.")
    st.stop()

df = pd.read_csv(uploaded)

# ---------------------------------------------------------
# Dataset preview
# ---------------------------------------------------------
st.subheader("📄 Dataset Preview")
st.dataframe(df, use_container_width=True)

# ---------------------------------------------------------
# Train Model
# ---------------------------------------------------------
X = df[["work_year", "job_title", "experience_level", "company_size"]]
y = df["salary_in_usd"]

categorical_cols = ["job_title", "experience_level", "company_size"]

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
    ],
    remainder="passthrough"
)

model = Pipeline(steps=[
    ("prep", preprocessor),
    ("reg", LinearRegression())
])

model.fit(X, y)

st.success("Model trained successfully!")

# ---------------------------------------------------------
# Forecast 2021–2025
# ---------------------------------------------------------
future_years = np.arange(2021, 2026)
sample_job = df["job_title"].mode()[0]
sample_exp = df["experience_level"].mode()[0]
sample_size = df["company_size"].mode()[0]

future_data = pd.DataFrame({
    "work_year": future_years,
    "job_title": sample_job,
    "experience_level": sample_exp,
    "company_size": sample_size
})

future_predictions = model.predict(future_data)

forecast_df = pd.DataFrame({
    "Year": future_years,
    "Predicted Salary (USD)": future_predictions
})

# ---------------------------------------------------------
# Graphical forecast (Plotly)
# ---------------------------------------------------------
st.subheader("📈 Interactive Salary Forecast (2021–2025)")

fig = px.line(
    forecast_df,
    x="Year",
    y="Predicted Salary (USD)",
    markers=True,
    title="Salary Prediction Trend (2021–2025)",
    template="plotly_white"
)

st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------
# Custom Prediction Section
# ---------------------------------------------------------
st.subheader("🔮 Predict Salary for Custom Input")

col1, col2, col3, col4 = st.columns(4)

with col1:
    input_year = st.number_input("Work Year", min_value=2020, max_value=2030, value=2023)

with col2:
    input_job = st.selectbox("Job Title", sorted(df["job_title"].unique()))

with col3:
    input_exp = st.selectbox("Experience Level", sorted(df["experience_level"].unique()))

with col4:
    input_size = st.selectbox("Company Size", sorted(df["company_size"].unique()))

user_data = pd.DataFrame({
    "work_year": [input_year],
    "job_title": [input_job],
    "experience_level": [input_exp],
    "company_size": [input_size]
})

predicted_salary = model.predict(user_data)[0]

st.metric("💰 Predicted Salary (USD)", f"${predicted_salary:,.2f}")
