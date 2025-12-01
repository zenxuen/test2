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
    page_title="Cybersecurity Salary Forecast — ML Model",
    layout="wide",
    page_icon="🛡️",
)

# ---------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.6rem;
        font-weight: 800;
        letter-spacing: 0.03em;
        margin-bottom: 0.3rem;
    }
    .sub-header {
        font-size: 0.95rem;
        color: #bbbbbb;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        padding: 1.4rem 1.6rem;
        border-radius: 0.9rem;
        background: linear-gradient(135deg, #1f2933, #111827);
        border: 1px solid rgba(148,163,184,0.4);
    }
    .metric-label {
        font-size: 0.85rem;
        color: #9ca3af;
        margin-bottom: 0.25rem;
    }
    .metric-value {
        font-size: 1.7rem;
        font-weight: 700;
    }
    .chip {
        display: inline-block;
        padding: 0.15rem 0.55rem;
        border-radius: 999px;
        background-color: rgba(55,65,81,0.9);
        font-size: 0.78rem;
        color: #e5e7eb;
        margin-right: 0.25rem;
    }
    .info-banner {
        padding: 0.75rem 1rem;
        border-radius: 0.6rem;
        font-size: 0.85rem;
        margin-bottom: 1rem;
    }
    .info-banner.green {
        background-color: rgba(16,185,129,0.1);
        border: 1px solid rgba(16,185,129,0.4);
        color: #bbf7d0;
    }
    .info-banner.red {
        background-color: rgba(248,113,113,0.1);
        border: 1px solid rgba(248,113,113,0.4);
        color: #fecaca;
    }
    .info-banner.blue {
        background-color: rgba(59,130,246,0.1);
        border: 1px solid rgba(59,130,246,0.4);
        color: #bfdbfe;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------
# Load data
# ---------------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("salaries_cyber_clean.csv")
    # 清一下列名空格
    df.columns = [c.strip() for c in df.columns]
    # 你的数据本来就只有 2020–2022，这里顺便限制一下
    df = df[df["work_year"].between(2020, 2022)]
    # 目标列不能是 NaN
    df = df.dropna(subset=["salary_in_usd"])
    return df

df = load_data()

if df.empty:
    st.error("Dataset is empty after filtering. Please check `salaries_cyber_clean.csv`.")
    st.stop()

# ---------------------------------------------------------
# Sidebar – 级联 Profile 选择 + 模型选择
# ---------------------------------------------------------
st.sidebar.header("Select Profile for Prediction")

# 第一步：Job
job_options = sorted(df["job_title"].unique())
job_title = st.sidebar.selectbox("Job Title", job_options)

# 第二步：Experience，只给这个 Job 里真的存在的
exp_options = sorted(df[df["job_title"] == job_title]["experience_level"].unique())
experience_level = st.sidebar.selectbox("Experience Level", exp_options)

# 第三步：Employment Type，基于前两个过滤
emp_type_options = sorted(
    df[
        (df["job_title"] == job_title)
        & (df["experience_level"] == experience_level)
    ]["employment_type"].unique()
)
employment_type = st.sidebar.selectbox("Employment Type", emp_type_options)

# 第四步：Company Size，基于前三个过滤
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
# Train ML models (纯 ML，完全不算 growth%)
# ---------------------------------------------------------
@st.cache_resource
def train_models(data: pd.DataFrame):
    feature_cols = ["work_year", "job_title", "experience_level", "employment_type", "company_size"]
    target_col = "salary_in_usd"

    X = data[feature_cols]
    y = data[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    categorical_cols = ["job_title", "experience_level", "employment_type", "company_size"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ],
        remainder="passthrough",  # 保留 work_year 这个数值特征
    )

    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(
            n_estimators=250,
            random_state=42,
            max_depth=18,
            min_samples_split=4,
            min_samples_leaf=2,
            n_jobs=-1,
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=220,
            random_state=42,
            max_depth=5,
            learning_rate=0.08,
        ),
    }

    trained = {}
    metrics = {}

    for name, est in models.items():
        pipe = Pipeline([
            ("prep", preprocessor),
            ("model", est),
        ])
        pipe.fit(X_train, y_train)

        y_pred = pipe.predict(X_test)

        cv_scores = cross_val_score(pipe, X_train, y_train, cv=5, scoring="r2")

        trained[name] = pipe
        metrics[name] = {
            "R2": r2_score(y_test, y_pred),
            "MAE": mean_absolute_error(y_test, y_pred),
            "RMSE": float(np.sqrt(mean_squared_error(y_test, y_pred))),
            "CV_R2_mean": float(cv_scores.mean()),
            "CV_R2_std": float(cv_scores.std()),
        }

    return trained, metrics

models, perf_metrics = train_models(df)
model = models[model_type]
model_perf = perf_metrics[model_type]

# ---------------------------------------------------------
# Header
# ---------------------------------------------------------
st.markdown(
    '<div class="main-header">🛡️ Cybersecurity Salary Forecast — ML Model (2020–2030)</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="sub-header">Target: <b>salary_in_usd</b> &nbsp;·&nbsp; Features: work_year, job_title, experience_level, employment_type, company_size</div>',
    unsafe_allow_html=True,
)

# ---------------------------------------------------------
# 当前 Profile 的真实历史数据
# ---------------------------------------------------------
profile_df = df[
    (df["job_title"] == job_title)
    & (df["experience_level"] == experience_level)
    & (df["employment_type"] == employment_type)
    & (df["company_size"] == company_size)
].copy()

years_available = sorted(profile_df["work_year"].unique())
num_years = len(years_available)

if num_years == 0:
    st.markdown(
        '<div class="info-banner red">This profile has <b>no actual data</b> in 2020–2022. (This should not happen because we filtered options.)</div>',
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        f'<div class="info-banner blue">This profile has <b>{len(profile_df)}</b> record(s) across years: <b>{", ".join(map(str, years_available))}</b>.</div>',
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------
# 用 ML 预测 2020–2030
# ---------------------------------------------------------
forecast_years = list(range(2020, 2031))
rows = []

for year in forecast_years:
    X_row = pd.DataFrame(
        [{
            "work_year": year,
            "job_title": job_title,
            "experience_level": experience_level,
            "employment_type": employment_type,
            "company_size": company_size,
        }]
    )
    pred = float(model.predict(X_row)[0])
    pred = max(0.0, pred)  # 不要负数

    # 真正有 data 的年份就记录实际平均值
    actual_val = np.nan
    if year in years_available:
        actual_val = float(profile_df[profile_df["work_year"] == year]["salary_in_usd"].mean())

    rows.append(
        {
            "year": year,
            "salary_pred": pred,
            "salary_actual": actual_val,
        }
    )

forecast_df = pd.DataFrame(rows)

# ---------------------------------------------------------
# 顶部 Metrics（全都基于 ML）
# ---------------------------------------------------------
safe_start = forecast_df[forecast_df["year"] == 2020]["salary_pred"]
safe_end = forecast_df[forecast_df["year"] == 2030]["salary_pred"]

if not safe_start.empty and not safe_end.empty:
    start_salary = safe_start.iloc[0]
    end_salary = safe_end.iloc[0]
    total_growth_pct = ((end_salary - start_salary) / start_salary * 100.0) if start_salary > 0 else np.nan
else:
    start_salary = end_salary = total_growth_pct = np.nan

year_2025_pred = forecast_df.loc[forecast_df["year"] == 2025, "salary_pred"]
salary_2025 = float(year_2025_pred.iloc[0]) if not year_2025_pred.empty else np.nan

actual_latest = np.nan
if num_years > 0:
    latest_year = years_available[-1]
    actual_latest = float(
        profile_df[profile_df["work_year"] == latest_year]["salary_in_usd"].mean()
    )

col_m1, col_m2, col_m3 = st.columns(3)

with col_m1:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">📅 2020 Salary (ML)</div>
            <div class="metric-value">${start_salary:,.0f}</div>
            <div style="font-size:0.8rem;color:#9ca3af;">Based on selected ML model</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col_m2:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">📅 2025 Salary (ML)</div>
            <div class="metric-value">${salary_2025:,.0f}</div>
            <div style="font-size:0.8rem;color:#9ca3af;">Forecasted for 2025</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col_m3:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">📈 Total Growth 2020–2030 (ML)</div>
            <div class="metric-value">{total_growth_pct:,.1f}%</div>
            <div style="font-size:0.8rem;color:#9ca3af;">Relative to 2020 prediction</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------
# 预测曲线 + 实际点
# ---------------------------------------------------------
st.markdown("### Salary Forecast (2020–2030)")

fig = go.Figure()

# ML prediction line (2020–2030)
fig.add_trace(
    go.Scatter(
        x=forecast_df["year"],
        y=forecast_df["salary_pred"],
        mode="lines+markers",
        name="ML Predicted",
        line=dict(color="#fbbf24", width=3, dash="dash"),
        marker=dict(color="#f59e0b", size=8),
        hovertemplate="Year %{x}<br>Predicted: $%{y:,.0f}<extra></extra>",
    )
)

# Actual points (only where available)
actual_points = forecast_df.dropna(subset=["salary_actual"])
if not actual_points.empty:
    fig.add_trace(
        go.Scatter(
            x=actual_points["year"],
            y=actual_points["salary_actual"],
            mode="markers",
            name="Actual (2020–2022)",
            marker=dict(color="#10b981", size=11, symbol="circle"),
            hovertemplate="Year %{x}<br>Actual: $%{y:,.0f}<extra></extra>",
        )
    )

fig.update_layout(
    height=480,
    xaxis_title="Year",
    yaxis_title="Salary (USD)",
    template="plotly_dark",
    hovermode="x unified",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
)

st.plotly_chart(fig, width="stretch")

# ---------------------------------------------------------
# Forecast 表格
# ---------------------------------------------------------
st.markdown("### Forecast Data")

display_df = forecast_df.copy()
display_df["Predicted Salary (USD)"] = display_df["salary_pred"].round(0).astype(int)
display_df["Actual Salary (USD)"] = display_df["salary_actual"].round(0)
display_df = display_df[["year", "Predicted Salary (USD)", "Actual Salary (USD)"]]

st.dataframe(
    display_df,
    width="stretch",
    hide_index=True,
)

# ---------------------------------------------------------
# 模型性能
# ---------------------------------------------------------
st.markdown("---")
st.markdown("### Model Performance")

col_p1, col_p2, col_p3 = st.columns(3)

with col_p1:
    st.metric("R² Score", f"{model_perf['R2']:.3f}")
with col_p2:
    st.metric("MAE", f"${model_perf['MAE']:,.0f}")
with col_p3:
    st.metric("RMSE", f"${model_perf['RMSE']:,.0f}")

st.caption(
    f"Cross-validation R² (mean ± std): {model_perf['CV_R2_mean']:.3f} ± {model_perf['CV_R2_std']:.3f}"
)

# Feature importance（只有树模型有）
if model_type in ["Random Forest", "Gradient Boosting"]:
    st.markdown("#### Feature Importance (by feature group)")

    model_est = model.named_steps["model"]
    prep = model.named_steps["prep"]

    if hasattr(model_est, "feature_importances_"):
        ohe = prep.named_transformers_["cat"]

        n_job = len(ohe.categories_[0])
        n_exp = len(ohe.categories_[1])
        n_emp = len(ohe.categories_[2])
        n_size = len(ohe.categories_[3])

        importances = pd.DataFrame(
            {
                "importance": model_est.feature_importances_,
                "feature_group": (
                    ["Job Title"] * n_job
                    + ["Experience Level"] * n_exp
                    + ["Employment Type"] * n_emp
                    + ["Company Size"] * n_size
                    + ["Year"]  # passthrough work_year
                ),
            }
        )

        grouped = (
            importances.groupby("feature_group")["importance"]
            .sum()
            .sort_values(ascending=False)
            .reset_index()
        )

        fig_imp = px.bar(
            grouped,
            x="importance",
            y="feature_group",
            orientation="h",
            title=f"Feature Importance – {model_type}",
            color="importance",
            color_continuous_scale="Blues",
        )
        fig_imp.update_layout(
            height=420,
            yaxis=dict(categoryorder="total ascending"),
        )

        st.plotly_chart(fig_imp, width="stretch")

st.markdown(
    """
    <div style="text-align:center;color:#6b7280;font-size:0.8rem;margin-top:1.5rem;">
        Built with Streamlit · Pure ML forecasting (no manual growth rules)<br/>
        Target: <code>salary_in_usd</code> · Features: <code>work_year</code>, <code>job_title</code>, <code>experience_level</code>, <code>employment_type</code>, <code>company_size</code>
    </div>
    """,
    unsafe_allow_html=True,
)
