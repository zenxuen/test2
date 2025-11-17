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
st.set_page_config(page_title="Salary Predictor", layout="wide")
st.title("📈 Cybersecurity Salary Predictor (2021 - 2025)")

# ---------------------------------------------------------
# Load CSV
# ---------------------------------------------------------
file_path = r"C:\Users\user\Downloads\Assignment\Assignment\salaries_cyber_clean.csv"  # make sure it ends with .csv
df = pd.read_csv(file_path)

st.success("Dataset loaded successfully!")

# ---------------------------------------------------------
# Features & Target
# ---------------------------------------------------------
X = df[['work_year', 'experience_level', 'employment_type', 'job_title', 'employee_residence', 'remote_ratio', 'company_location', 'company_size']]
y = df['salary_in_usd']

# ---------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------
categorical_features = ['experience_level', 'employment_type', 'job_title', 'employee_residence', 'company_location', 'company_size']
preprocessor = ColumnTransformer(transformers=[
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
])

model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

model.fit(X, y)
st.success("Model trained successfully!")

# ---------------------------------------------------------
# Interactive selection
# ---------------------------------------------------------
st.sidebar.header("Custom Input")
work_year = st.sidebar.slider("Work Year", 2021, 2025, 2022)
experience_level = st.sidebar.selectbox("Experience Level", df['experience_level'].unique())
employment_type = st.sidebar.selectbox("Employment Type", df['employment_type'].unique())
job_title = st.sidebar.selectbox("Job Title", df['job_title'].unique())
employee_residence = st.sidebar.selectbox("Employee Residence", df['employee_residence'].unique())
remote_ratio = st.sidebar.slider("Remote Ratio (%)", 0, 100, 50)
company_location = st.sidebar.selectbox("Company Location", df['company_location'].unique())
company_size = st.sidebar.selectbox("Company Size", df['company_size'].unique())

input_df = pd.DataFrame([{
    'work_year': work_year,
    'experience_level': experience_level,
    'employment_type': employment_type,
    'job_title': job_title,
    'employee_residence': employee_residence,
    'remote_ratio': remote_ratio,
    'company_location': company_location,
    'company_size': company_size
}])

predicted_salary = model.predict(input_df)[0]
st.metric("💰 Predicted Salary (USD)", f"${predicted_salary:,.2f}")

# ---------------------------------------------------------
# Plot predicted vs historical
# ---------------------------------------------------------
df_plot = df.copy()
df_plot['Predicted Salary'] = model.predict(X)

fig = px.scatter(
    df_plot,
    x='work_year',
    y='salary_in_usd',
    color='job_title',
    size='remote_ratio',
    hover_data=['experience_level', 'company_size', 'company_location'],
    title="Historical vs Predicted Salaries",
)
fig.add_scatter(x=[work_year], y=[predicted_salary], mode='markers', marker=dict(color='red', size=15), name='Your Prediction')
st.plotly_chart(fig, use_container_width=True)
