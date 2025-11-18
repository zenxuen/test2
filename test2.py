import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
import plotly.express as px

st.set_page_config(page_title="Salary Prediction", layout="wide")

st.title("💼 Salary Prediction Dashboard (Improved Model)")

# ---------------------------------------------------------
# Load Dataset (already in your Codespace)
# ---------------------------------------------------------
file_path = "salaries_cyber_clean.csv"
df = pd.read_csv(file_path)

# ---------------------------------------------------------
# Train Prediction Model
# ---------------------------------------------------------
X = df[["work_year", "job_title", "experience_level", "company_size"]]
y = df["salary_in_usd"]

categorical_cols = ["job_title", "experience_level", "company_size"]

preprocessor = ColumnTransformer(
    transformers=[("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)],
    remainder="passthrough"
)

# ⭐ Use Random Forest for better prediction changes
model = Pipeline([
    ("prep", preprocessor),
    ("rf", RandomForestRegressor(n_estimators=300, random_state=42))
])

model.fit(X, y)

# ---------------------------------------------------------
# Custom selection
# ---------------------------------------------------------
st.subheader("⚙️ Customize Model Inputs")

col1, col2, col3 = st.columns(3)

with col1:
    custom_job = st.selectbox("Job Title", sorted(df["job_title"].unique()))

with col2:
    custom_exp = st.selectbox("Experience Level", sorted(df["experience_level"].unique()))

with col3:
    custom_size = st.selectbox("Company Size", sorted(df["company_size"].unique()))

# ---------------------------------------------------------
# Forecast from dataset start to 2035
# ---------------------------------------------------------
future_years = np.arange(df["work_year"].min(), 2036)

custom_future_data = pd.DataFrame({
    "work_year": future_years,
    "job_title": custom_job,
    "experience_level": custom_exp,
    "company_size": custom_size
})

future_predictions = model.predict(custom_future_data)

forecast_df = pd.DataFrame({
    "Year": future_years,
    "Predicted Salary (USD)": future_predictions
})

# ---------------------------------------------------------
# Forecast Graph
# ---------------------------------------------------------
st.subheader("📈 Salary Forecast Based on Your Selections (to 2035)")

fig = px.line(
    forecast_df,
    x="Year",
    y="Predicted Salary (USD)",
    markers=True,
    title=f"Salary Forecast for {custom_job} ({custom_exp}, {custom_size})",
    template="plotly_white"
)

fig.update_traces(line=dict(width=5), marker=dict(size=12))
fig.update_layout(xaxis=dict(dtick=1), hovermode="x unified")

st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------
# Single prediction
# ---------------------------------------------------------
st.subheader("🔮 Predict Salary for a Specific Year")

single_year = st.slider("Select Year", 2020, 2035, 2024)

single_input = pd.DataFrame({
    "work_year": [single_year],
    "job_title": [custom_job],
    "experience_level": [custom_exp],
    "company_size": [custom_size]
})

single_prediction = model.predict(single_input)[0]

st.metric("💰 Predicted Salary", f"${single_prediction:,.2f}")
