import streamlit as st
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
import plotly.express as px

# ---------------------------------------------------------
# Page Config
# ---------------------------------------------------------
st.set_page_config(
    page_title="Salary Predictor (2021 - 2025)",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Cybersecurity Salary Predictor (2021 - 2025)")

# ---------------------------------------------------------
# Load dataset (fixed path)
# ---------------------------------------------------------
file_path = r"C:\Users\user\Downloads\Assignment\Assignment\salaries_cyber_clean"
df = pd.read_csv(file_path)

# ---------------------------------------------------------
# Train model
# ---------------------------------------------------------
features = ["work_year", "experience_level", "employment_type", "job_title"]
target = "salary_in_usd"

categorical_cols = ["experience_level", "employment_type", "job_title"]

preprocess = ColumnTransformer(
    transformers=[("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)],
    remainder="passthrough"
)

model = Pipeline(steps=[
    ("preprocess", preprocess),
    ("regression", LinearRegression())
])

model.fit(df[features], df[target])

# ---------------------------------------------------------
# User Controls – Custom Prediction Inputs
# ---------------------------------------------------------
st.subheader("🎛️ Customize Prediction")

col1, col2, col3 = st.columns(3)

with col1:
    work_year = st.selectbox("Select Year", [2021, 2022, 2023, 2024, 2025])

with col2:
    experience_level = st.selectbox("Experience Level", sorted(df["experience_level"].unique()))

with col3:
    employment_type = st.selectbox("Employment Type", sorted(df["employment_type"].unique()))

job_title = st.selectbox("Job Title", sorted(df["job_title"].unique()))

# Prepare input for prediction
input_data = pd.DataFrame([{
    "work_year": work_year,
    "experience_level": experience_level,
    "employment_type": employment_type,
    "job_title": job_title
}])

predicted_salary = model.predict(input_data)[0]

st.success(f"💰 **Predicted Salary (USD):** ${predicted_salary:,.2f}")

# ---------------------------------------------------------
# Interactive Prediction Graph
# ---------------------------------------------------------
st.subheader("📊 Salary Trend Prediction (2021 - 2025)")

years = [2021, 2022, 2023, 2024, 2025]

graph_data = pd.DataFrame([{
    "work_year": y,
    "experience_level": experience_level,
    "employment_type": employment_type,
    "job_title": job_title
} for y in years])

graph_data["predicted_salary"] = model.predict(graph_data)

fig = px.line(
    graph_data,
    x="work_year",
    y="predicted_salary",
    markers=True,
    line_shape="spline",  # Smooth curve
    title="Predicted Salary Trend",
)

fig.update_traces(marker=dict(size=10), line=dict(width=4))

# Add interactive gap effect
fig.update_xaxes(dtick=1, tickangle=0)
fig.update_layout(
    xaxis=dict(
        tickmode="linear",
        tick0=2021,
    ),
    plot_bgcolor="rgba(0,0,0,0)",
    hovermode="x unified",
    margin=dict(l=40, r=40, t=50, b=50),
)

st.plotly_chart(fig, use_container_width=True)



