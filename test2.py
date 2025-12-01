import streamlit as st
import pandas as pd
import numpy as np

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import plotly.express as px

# ---------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------
st.set_page_config(
    page_title="ML Cybersecurity Salary Predictor",
    page_icon="💼",
    layout="wide",
)

st.markdown("<h1 style='font-size:3rem;'>💼 ML Cybersecurity Salary Prediction Dashboard</h1>", unsafe_allow_html=True)

# ---------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("salaries_cyber_clean.csv")
    return df

df = load_data()

# Keep only required columns
df = df[[
    "salary_in_usd",
    "job_title",
    "experience_level",
    "employment_type",
    "company_size"
]].dropna()

# ---------------------------------------------------------
# SIDEBAR MODEL
# ---------------------------------------------------------
st.sidebar.header("🎛️ Model Settings")

model_choice = st.sidebar.selectbox(
    "Choose ML Model",
    ["Linear Regression", "Random Forest", "Gradient Boosting"]
)

# ---------------------------------------------------------
# TRAIN MODEL
# ---------------------------------------------------------
@st.cache_resource
def train_ml(df):
    X = df[["job_title", "experience_level", "employment_type", "company_size"]]
    y = df["salary_in_usd"]

    cat_cols = ["job_title", "experience_level", "employment_type", "company_size"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
        ],
        remainder="drop"
    )

    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=150, random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(random_state=42)
    }

    trained_models = {}
    metrics = {}

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    for name, model in models.items():
        pipe = Pipeline([
            ("prep", preprocessor),
            ("model", model)
        ])

        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)

        cv = cross_val_score(pipe, X_train, y_train, cv=5, scoring="r2")

        trained_models[name] = pipe
        metrics[name] = {
            "R² Score": r2_score(y_test, y_pred),
            "MAE": mean_absolute_error(y_test, y_pred),
            "RMSE": mean_squared_error(y_test, y_pred, squared=False),
            "CV Mean": cv.mean(),
            "CV Std": cv.std()
        }

    return trained_models, metrics

models, metrics = train_ml(df)
selected_model = models[model_choice]

# ---------------------------------------------------------
# SIDEBAR METRICS
# ---------------------------------------------------------
st.sidebar.subheader("📈 Model Performance")
m = metrics[model_choice]
st.sidebar.metric("R² Score", f"{m['R² Score']:.3f}")
st.sidebar.metric("MAE", f"${m['MAE']:,.0f}")
st.sidebar.metric("RMSE", f"${m['RMSE']:,.0f}")
st.sidebar.caption(f"CV R² = {m['CV Mean']:.3f} ± {m['CV Std']:.3f}")

# ---------------------------------------------------------
# USER INPUT
# ---------------------------------------------------------
st.subheader("🔧 Input Your Profile")

col1, col2, col3, col4 = st.columns(4)

with col1:
    job = st.selectbox("👔 Job Title", sorted(df["job_title"].unique()))

with col2:
    exp = st.selectbox("📈 Experience", sorted(df["experience_level"].unique()))

with col3:
    emp = st.selectbox("📝 Employment Type", sorted(df["employment_type"].unique()))

with col4:
    size = st.selectbox("🏢 Company Size", sorted(df["company_size"].unique()))

# ---------------------------------------------------------
# PREDICT
# ---------------------------------------------------------
input_df = pd.DataFrame({
    "job_title": [job],
    "experience_level": [exp],
    "employment_type": [emp],
    "company_size": [size],
})

pred_salary = selected_model.predict(input_df)[0]

st.markdown("---")
st.markdown(f"""
<div style="padding:2rem; background:#4F46E5; color:white; border-radius:10px; text-align:center;">
    <h2>Predicted Salary</h2>
    <h1 style="font-size:3.2rem;">${pred_salary:,.0f}</h1>
    <p>{job}</p>
    <p>{exp} | {emp} | {size}</p>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# OPTIONAL: SIMPLE EDA
# ---------------------------------------------------------
st.markdown("---")
st.subheader("📊 Salary by Experience Level")

exp_avg = df.groupby("experience_level")["salary_in_usd"].mean().reset_index()

fig = px.bar(
    exp_avg,
    x="experience_level",
    y="salary_in_usd",
    title="Average Salary by Experience Level",
    color="salary_in_usd",
    text="salary_in_usd"
)

fig.update_traces(texttemplate='$%{text:,.0f}', textposition='outside')
st.plotly_chart(fig, use_container_width=True)

