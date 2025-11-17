import streamlit as st
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
import plotly.express as px
import numpy as np

# -----------------------------
# Page config
# -----------------------------
st.set_page_config(page_title="Cyber Salary Predictor", layout="wide")
st.title("Cybersecurity Salary Prediction (2021-2025)")

# -----------------------------
# Load dataset
# -----------------------------
file_path = "salaries_cyber_clean.csv"
df = pd.read_csv(file_path)

# -----------------------------
# Features & target
# -----------------------------
X = df[['work_year', 'experience_level', 'employment_type', 'job_title', 'employee_residence', 'remote_ratio']]
y = df['salary_in_usd']

# -----------------------------
# Preprocessing & model
# -----------------------------
categorical_features = ['experience_level', 'employment_type', 'job_title', 'employee_residence']
encoder = ColumnTransformer(transformers=[
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
], remainder='passthrough')

model = Pipeline([
    ('encode', encoder),
    ('regressor', LinearRegression())
])

model.fit(X, y)

# -----------------------------
# Sidebar filters
# -----------------------------
st.sidebar.header("Filter Options")
selected_experience = st.sidebar.multiselect(
    "Experience Level", options=df['experience_level'].unique(), default=df['experience_level'].unique()
)
selected_type = st.sidebar.multiselect(
    "Employment Type", options=df['employment_type'].unique(), default=df['employment_type'].unique()
)
selected_years = st.sidebar.slider("Work Year Range", 2021, 2025, (2021, 2025))

# -----------------------------
# Generate predictions
# -----------------------------
future_years = np.arange(selected_years[0], selected_years[1]+1, 1)
predictions = []

for year in future_years:
    for exp in selected_experience:
        for emp_type in selected_type:
            sample = pd.DataFrame([{
                'work_year': year,
                'experience_level': exp,
                'employment_type': emp_type,
                'job_title': 'SECURITY ANALYST',  # default
                'employee_residence': 'US',       # default
                'remote_ratio': 50                # default
            }])
            pred = model.predict(sample)[0]
            predictions.append({
                'Year': year,
                'Experience': exp,
                'Employment Type': emp_type,
                'Predicted Salary': pred
            })

pred_df = pd.DataFrame(predictions)

# -----------------------------
# Interactive line plot
# -----------------------------
fig = px.line(
    pred_df,
    x='Year',
    y='Predicted Salary',
    color='Experience',
    line_dash='Employment Type',
    markers=True,
    title="Predicted Cybersecurity Salaries (USD)"
)

# Make gaps more visible
fig.update_traces(connectgaps=False, line=dict(width=3))
fig.update_layout(
    xaxis=dict(dtick=1),
    yaxis_title="Salary (USD)",
    template="plotly_white",
    hovermode="x unified"
)

st.plotly_chart(fig, use_container_width=True)
