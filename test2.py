import streamlit as st
import pandas as pd
import numpy as np

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline

import plotly.express as px
import plotly.graph_objects as go

# ---------------------------------------------------------
# Page Config
# ---------------------------------------------------------
st.set_page_config(
    page_title="Cybersecurity Salary Forecast (Pure ML)",
    layout="wide",
    page_icon="🛡️",
)

st.title("🛡️ Cybersecurity Salary Forecast (Pure ML)")
st.caption(
    "Target: **salary_in_usd** · Features: work_year, job_title, "
    "experience_level, employment_type, company_size"
)

# ---------------------------------------------------------
# Load data
# ---------------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("salaries_cyber_clean.csv")
    # 清理列名空格
    df.columns = [c.strip() for c in df.columns]

    # 只留 2020–2022（你的原始数据就是这个范围）
    if "work_year" not in df.columns:
        st.error("Column 'work_year' not found in CSV. Check your file.")
        return pd.DataFrame()

    df = df[df["work_year"].between(2020, 2022)]

    # 确保目标列存在
    if "salary_in_usd" not in df.columns:
        st.error("Column 'salary_in_usd' not found in CSV. Check your file.")
        return pd.DataFrame()

    df = df.dropna(subset=["salary_in_usd"])
    return df

df = load_data()

if df.empty:
    st.error("Dataset is empty or invalid after loading. Please check CSV file.")
    st.stop()

st.write("#### Quick Data Check")
st.write("Shape:", df.shape)
st.write("Columns:", list(df.columns))
st.dataframe(df.head())

# ---------------------------------------------------------
# Sidebar – Profile selection & model selection
# ---------------------------------------------------------
st.sidebar.header("Profile Selection")

# Job
job_options = sorted(df["job_title"].unique())
job_title = st.sidebar.selectbox("Job Title", job_options)

# Experience based on job
exp_options = sorted(
    df[df["job_title"] == job_title]["experience_level"].unique()
)
experience_level = st.sidebar.selectbox("Experience Level", exp_options)

# Employment type based on job + exp
emp_type_options = sorted(
    df[
        (df["job_title"] == job_title)
        & (df["experience_level"] == experience_level)
    ]["employment_type"].unique()
)
employment_type = st.sidebar.selectbox("Employment Type", emp_type_options)

# Company size based on job + exp + emp_type
size_options = sorted(
    df[
        (df["job_title"] == job_title)
        & (df["experience_level"] == experience_level)
        & (df["employment_type"] == employment_type)
    ]["company_size"].unique()
)
company_size = st.sidebar.selectbox("Company Size", size_options)

st.sidebar.markdown("---")
model_type = st.sidebar.selectbox(
    "Model",
    ["Linear Regression", "Random Forest", "Gradient Boosting"],
)

# ---------------------------------------------------------
# Train ML models
# ---------------------------------------------------------
@st.cache_resource
def train_models(data: pd.DataFrame):
    feature_cols = [
        "work_year",
        "job_title",
        "experience_level",
        "employment_type",
        "company_size",
    ]
    target_col = "salary_in_usd"

    # 检查所有 feature 列是否存在
    missing = [c for c in feature_cols if c not in data.columns]
    if missing:
        raise ValueError(f"Missing feature columns in data: {missing}")

    X = data[feature_cols]
    y = data[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    categorical_cols = [
        "job_title",
        "experience_level",
        "employment_type",
        "company_size",
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ],
        remainder="passthrough",  # keep work_year
    )

    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(
            n_estimators=200,
            random_state=42,
            max_depth=15,
            min_samples_split=4,
            min_samples_leaf=2,
            n_jobs=-1,
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=200,
            random_state=42,
            max_depth=5,
            learning_rate=0.1,
        ),
    }

    trained = {}
    metrics = {}

    for name, est in models.items():
        pipe = Pipeline(
            [
                ("prep", preprocessor),
                ("model", est),
            ]
        )
        pipe.fit(X_train, y_train)

        y_pred = pipe.predict(X_test)

        cv_scores = cross_val_score(
            pipe, X_train, y_train, cv=5, scoring="r2"
        )

        trained[name] = pipe
        metrics[name] = {
            "R2": r2_score(y_test, y_pred),
            "MAE": mean_absolute_error(y_test, y_pred),
            "RMSE": float(np.sqrt(mean_squared_error(y_test, y_pred))),
            "CV_R2_mean": float(cv_scores.mean()),
            "CV_R2_std": float(cv_scores.std()),
        }

    return trained, metrics


try:
    models, perf = train_models(df)
except Exception as e:
    st.error(f"Error while training models: {e}")
    st.stop()

model = models[model_type]
model_perf = perf[model_type]

# ---------------------------------------------------------
# Extract current profile history
# ---------------------------------------------------------
profile_df = df[
    (df["job_title"] == job_title)
    & (df["experience_level"] == experience_level)
    & (df["employment_type"] == employment_type)
    & (df["company_size"] == company_size)
].copy()

years_available = sorted(profile_df["work_year"].unique())

st.write("#### Selected Profile")
st.write(
    f"Job: **{job_title}**, Exp: **{experience_level}**, "
    f"Employment: **{employment_type}**, Size: **{company_size}**"
)
st.write("Years with actual data:", years_available)

# ---------------------------------------------------------
# Use ML to predict 2020–2030 for this profile
# ---------------------------------------------------------
forecast_years = list(range(2020, 2031))
rows = []

for year in forecast_years:
    X_row = pd.DataFrame(
        [
            {
                "work_year": year,
                "job_title": job_title,
                "experience_level": experience_level,
                "employment_type": employment_type,
                "company_size": company_size,
            }
        ]
    )
    try:
        pred = float(model.predict(X_row)[0])
    except Exception as e:
        st.error(f"Prediction error at year {year}: {e}")
        st.stop()

    pred = max(0.0, pred)

    actual_val = np.nan
    if year in years_available:
        actual_val = float(
            profile_df[profile_df["work_year"] == year]["salary_in_usd"].mean()
        )

    rows.append(
        {
            "year": year,
            "salary_pred": pred,
            "salary_actual": actual_val,
        }
    )

forecast_df = pd.DataFrame(rows)

# ---------------------------------------------------------
# Metrics (all based on ML prediction)
# ---------------------------------------------------------
start_row = forecast_df[forecast_df["year"] == 2020]
end_row = forecast_df[forecast_df["year"] == 2030]

if not start_row.empty and not end_row.empty:
    start_salary = start_row["salary_pred"].iloc[0]
    end_salary = end_row["salary_pred"].iloc[0]
    total_growth_pct = (
        (end_salary - start_salary) / start_salary * 100.0
        if start_salary > 0
        else np.nan
    )
else:
    start_salary = end_salary = total_growth_pct = np.nan

row_2025 = forecast_df[forecast_df["year"] == 2025]
salary_2025 = (
    float(row_2025["salary_pred"].iloc[0]) if not row_2025.empty else np.nan
)

col_m1, col_m2, col_m3 = st.columns(3)
with col_m1:
    st.metric("2020 Salary (ML)", f"${start_salary:,.0f}")
with col_m2:
    st.metric("2025 Salary (ML)", f"${salary_2025:,.0f}")
with col_m3:
    st.metric("Total Growth 2020–2030 (ML)", f"{total_growth_pct:,.1f}%")

# ---------------------------------------------------------
# Plot: ML prediction line + actual points
# ---------------------------------------------------------
st.write("### Salary Forecast (2020–2030)")

fig = go.Figure()

# ML prediction line
fig.add_trace(
    go.Scatter(
        x=forecast_df["year"],
        y=forecast_df["salary_pred"],
        mode="lines+markers",
        name="ML Predicted",
        line=dict(width=3, dash="dash"),
        marker=dict(size=8),
        hovertemplate="Year: %{x}<br>Pred: $%{y:,.0f}<extra></extra>",
    )
)

# Actual points
actual_points = forecast_df.dropna(subset=["salary_actual"])
if not actual_points.empty:
    fig.add_trace(
        go.Scatter(
            x=actual_points["year"],
            y=actual_points["salary_actual"],
            mode="markers",
            name="Actual (2020–2022)",
            marker=dict(size=10, symbol="circle"),
            hovertemplate="Year: %{x}<br>Actual: $%{y:,.0f}<extra></extra>",
        )
    )

fig.update_layout(
    height=450,
    xaxis_title="Year",
    yaxis_title="Salary (USD)",
    template="plotly_white",
    hovermode="x unified",
)

st.plotly_chart(fig)

# ---------------------------------------------------------
# Forecast table
# ---------------------------------------------------------
st.write("### Forecast Table")
display_df = forecast_df.copy()
display_df["Predicted Salary (USD)"] = display_df["salary_pred"].round(0).astype(int)
display_df["Actual Salary (USD)"] = display_df["salary_actual"].round(0)
display_df = display_df[["year", "Predicted Salary (USD)", "Actual Salary (USD)"]]

st.dataframe(display_df, hide_index=True)

# ---------------------------------------------------------
# Model Performance
# ---------------------------------------------------------
st.write("### Model Performance")

col_p1, col_p2, col_p3 = st.columns(3)
with col_p1:
    st.metric("R²", f"{model_perf['R2']:.3f}")
with col_p2:
    st.metric("MAE", f"${model_perf['MAE']:,.0f}")
with col_p3:
    st.metric("RMSE", f"${model_perf['RMSE']:,.0f}")

st.caption(
    f"Cross-validation R² (mean ± std): "
    f"{model_perf['CV_R2_mean']:.3f} ± {model_perf['CV_R2_std']:.3f}"
)
