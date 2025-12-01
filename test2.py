import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
import plotly.graph_objects as go
import plotly.express as px

# ---------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------
st.set_page_config(
    page_title="Salary Prediction Dashboard",
    layout="wide",
    page_icon="💼"
)

st.markdown("""
<style>
.main-header {
    font-size: 3rem;
    font-weight: bold;
    background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.5rem;
}
.metric-card {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 1.5rem;
    border-radius: 10px;
    color: white;
    text-align: center;
}
.info-box {
    background-color: #e3f2fd;
    border-left: 4px solid #2196f3;
    padding: 1rem;
    border-radius: 5px;
    margin: 1rem 0;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-header">💼 Cybersecurity Salary Prediction Dashboard</h1>', unsafe_allow_html=True)
st.markdown("*Actual Data: 2020-2022 | Predictions: 2023-2030*")
st.markdown("**Target:** salary_in_usd | **Predictors:** job_title, experience_level, company_size, work_year")

# ---------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------
@st.cache_data
def load_data():
    return pd.read_csv("salaries_cyber_clean.csv")

df = load_data()

# ---------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------
st.sidebar.header("🎛️ Model Selection")

model_type = st.sidebar.selectbox(
    "Machine Learning Model",
    ["Linear Regression", "Random Forest", "Gradient Boosting"]
)

prediction_method = st.sidebar.radio(
    "Prediction Mode",
    ["Growth-Based", "Pure ML Model", "Hybrid"]
)

st.sidebar.markdown("---")
st.sidebar.info(f"""
🔢 Records: {len(df):,}  
📅 Years: 2020–2022 (actual), 2023–2030 (predicted)  
👔 Job titles: {df['job_title'].nunique()}  
""")


# ---------------------------------------------------------
# TRAIN ML MODELS
# ---------------------------------------------------------
@st.cache_resource
def train_models(df):

    X = df[["work_year", "job_title", "experience_level", "company_size"]]
    y = df["salary_in_usd"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    cat_cols = ["job_title", "experience_level", "company_size"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
        ],
        remainder="passthrough"
    )

    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(
            n_estimators=250, max_depth=18, random_state=42
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=250, learning_rate=0.06, max_depth=6, random_state=42
        )
    }

    trained = {}
    metrics = {}

    for name, model in models.items():

        pipe = Pipeline([
            ("prep", preprocessor),
            ("model", model)
        ])

        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)

        cv = cross_val_score(pipe, X_train, y_train, cv=5, scoring="r2")

        trained[name] = pipe
        metrics[name] = {
            "R2": r2_score(y_test, pred),
            "MAE": mean_absolute_error(y_test, pred),
            "RMSE": np.sqrt(mean_squared_error(y_test, pred)),
            "CV": cv.mean()
        }

    return trained, metrics

models, model_metrics = train_models(df)
selected_model = models[model_type]

# ---------------------------------------------------------
# GROWTH CALCULATIONS (unchanged logic)
# ---------------------------------------------------------

@st.cache_data
def calc_profile_growth(job, exp, size):

    prof = df[
        (df["job_title"] == job) &
        (df["experience_level"] == exp) &
        (df["company_size"] == size)
    ].groupby("work_year")["salary_in_usd"].mean().sort_index()

    if len(prof) < 2:
        return None, None, None

    years = prof.index.values
    vals = prof.values

    growth = (vals[-1] - vals[0]) / vals[0] / (years[-1] - years[0])
    return float(np.clip(growth, -0.20, 0.20)), years, vals


@st.cache_data
def similar_growth(job, exp):

    similar = df[
        (df["job_title"] == job) &
        (df["experience_level"] == exp)
    ]

    profs = []
    for size in similar["company_size"].unique():

        grp = df[
            (df["job_title"] == job) &
            (df["experience_level"] == exp) &
            (df["company_size"] == size)
        ].groupby("work_year")["salary_in_usd"].mean().sort_index()

        if len(grp) >= 2:
            yrs = grp.index.values
            sals = grp.values
            g = (sals[-1]-sals[0]) / sals[0] / (yrs[-1]-yrs[0])
            profs.append(g)

    if len(profs) == 0:
        return 0.05, 0

    return float(np.mean(profs)), len(profs)


# ---------------------------------------------------------
# MASTER SALARY PREDICTION FUNCTION
# ---------------------------------------------------------
def get_salary(year, job, exp, size, method):

    history = df[
        (df["job_title"] == job) &
        (df["experience_level"] == exp) &
        (df["company_size"] == size)
    ]

    years_avail = sorted(history["work_year"].unique())
    n = len(years_avail)

    # ---------------------------
    # 1) Actual data
    # ---------------------------
    if year <= 2022:
        row = history[history["work_year"] == year]
        if len(row) > 0:
            return row["salary_in_usd"].mean(), "Actual"

    # ---------------------------
    # 2) ML input (for ML / Hybrid)
    # ---------------------------
    ml_input = pd.DataFrame({
        "work_year": [year],
        "job_title": [job],
        "experience_level": [exp],
        "company_size": [size]
    })
    ml_pred = selected_model.predict(ml_input)[0]

    # ---------------------------
    # 3) Growth-Based Rules
    # ---------------------------
    if method != "Pure ML Model":

        if n == 0:
            g_salary = ml_pred
            g_src = "ML Only (No History)"

        elif n == 1:
            base_year = years_avail[0]
            base_val = history["salary_in_usd"].mean()
            g, c = similar_growth(job, exp)
            yrs = year - base_year
            g_salary = base_val * ((1 + g) ** yrs)
            g_src = "Similar Profile Growth"

        else:
            g, yrs, sals = calc_profile_growth(job, exp, size)
            if g is None:
                g_salary = ml_pred
                g_src = "ML Fallback"
            else:
                base_year = yrs[-1]
                base_val = sals[-1]
                ahead = year - base_year
                g_salary = base_val * ((1 + g) ** ahead)
                g_src = "Own Growth Pattern"

    else:
        g_salary = ml_pred
        g_src = "ML Only"

    # ---------------------------
    # 4) Hybrid
    # ---------------------------
    if method == "Hybrid":
        final_salary = 0.7 * g_salary + 0.3 * ml_pred
        final_src = "Hybrid (70% Growth + 30% ML)"
    elif method == "Pure ML Model":
        final_salary = ml_pred
        final_src = "Pure ML"
    else:
        final_salary = g_salary
        final_src = g_src

    return max(0, float(final_salary)), final_src

# ---------------------------------------------------------
# TABS
# ---------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "🔮 Salary Forecast",
    "📊 Data Analysis",
    "💰 Salary Calculator",
    "🗺️ Model Insights"
])

# ---------------------------------------------------------
# TAB 1 — FORECAST
# ---------------------------------------------------------
with tab1:
    st.subheader("⚙️ Customize Your Profile")

    c1, c2, c3 = st.columns(3)
    with c1:
        t_job = st.selectbox("Job Title", sorted(df["job_title"].unique()))
    with c2:
        t_exp = st.selectbox("Experience Level", sorted(df["experience_level"].unique()))
    with c3:
        t_size = st.selectbox("Company Size", sorted(df["company_size"].unique()))

    # -----------------------------------------------------
    # FORECAST CALCULATION (2020–2030)
    # -----------------------------------------------------
    years = np.arange(2020, 2031)
    forecast_rows = []

    for yr in years:
        sal, src = get_salary(
            yr,
            t_job,
            t_exp,
            t_size,
            prediction_method
        )
        forecast_rows.append({
            "Year": yr,
            "Salary": sal,
            "Source": src
        })

    forecast_df = pd.DataFrame(forecast_rows)

    # -----------------------------------------------------
    # SPLIT ACTUAL / PREDICTED
    # -----------------------------------------------------
    actual_df = forecast_df[forecast_df["Source"].str.contains("Actual")]
    pred_df = forecast_df[~forecast_df["Source"].str.contains("Actual")]

    # -----------------------------------------------------
    # SUMMARY METRICS
    # -----------------------------------------------------
    st.markdown("---")

    cA, cB, cC = st.columns(3)

    start_salary = forecast_df.loc[forecast_df["Year"] == 2020, "Salary"].values[0]
    end_salary = forecast_df.loc[forecast_df["Year"] == 2030, "Salary"].values[0]

    with cA:
        st.metric("Salary in 2020", f"${start_salary:,.0f}")

    with cB:
        mid_year = 2025
        mid_salary = forecast_df.loc[forecast_df["Year"] == mid_year, "Salary"].values[0]
        st.metric(f"Salary in {mid_year}", f"${mid_salary:,.0f}")

    with cC:
        if start_salary > 0:
            total_growth = ((end_salary - start_salary) / start_salary) * 100
            st.metric("Total Growth (2020→2030)", f"{total_growth:.1f}%")
        else:
            st.metric("Total Growth", "N/A")

    # -----------------------------------------------------
    # LINE CHART (2020–2030)
    # -----------------------------------------------------
    fig = go.Figure()

    # Actual
    if len(actual_df) > 0:
        fig.add_trace(go.Scatter(
            x=actual_df["Year"],
            y=actual_df["Salary"],
            mode="lines+markers",
            name="Actual (2020–2022)",
            line=dict(width=4, color="#10b981"),
            marker=dict(size=10, color="#10b981"),
            hovertemplate="Year %{x}<br>Salary $%{y:,.0f}<extra></extra>"
        ))

    # Predicted
    fig.add_trace(go.Scatter(
        x=pred_df["Year"],
        y=pred_df["Salary"],
        mode="lines+markers",
        name="Predicted (2023–2030)",
        line=dict(width=4, color="#667eea", dash="dot"),
        marker=dict(size=10, color="#764ba2"),
        hovertemplate="Year %{x}<br>$%{y:,.0f}<br>%{text}<extra></extra>",
        text=pred_df["Source"]
    ))

    fig.update_layout(
        title=f"Salary Forecast — {t_job} ({t_exp}, {t_size})",
        xaxis_title="Year",
        yaxis_title="Salary (USD)",
        height=480,
        template="plotly_white",
        xaxis=dict(dtick=1),
        hovermode="x unified"
    )

    st.plotly_chart(fig, use_container_width=True)

    # -----------------------------------------------------
    # FORECAST TABLE
    # -----------------------------------------------------
    with st.expander("📄 Detailed Forecast Table"):
        tbl = forecast_df.copy()
        tbl["Salary"] = tbl["Salary"].apply(lambda x: f"${x:,.0f}")
        st.dataframe(tbl, use_container_width=True, hide_index=True)
# ---------------------------------------------------------
# TAB 2 — DATA ANALYSIS
# ---------------------------------------------------------
with tab2:
    st.subheader("📊 Salary Distribution (Actual Data 2020–2022)")

    # ------------------------ Experience ------------------------
    colA, colB = st.columns(2)

    with colA:
        exp_stats = df.groupby("experience_level")["salary_in_usd"].mean().reset_index()
        fig_exp = px.bar(
            exp_stats,
            x="experience_level",
            y="salary_in_usd",
            title="Average Salary by Experience Level",
            color="salary_in_usd",
            color_continuous_scale="Purples",
            text="salary_in_usd"
        )
        fig_exp.update_traces(texttemplate="$%{text:,.0f}", textposition="outside")
        fig_exp.update_layout(showlegend=False)
        st.plotly_chart(fig_exp, use_container_width=True)

    with colB:
        size_stats = df.groupby("company_size")["salary_in_usd"].mean().reset_index()
        fig_size = px.bar(
            size_stats,
            x="company_size",
            y="salary_in_usd",
            title="Average Salary by Company Size",
            color="salary_in_usd",
            color_continuous_scale="Blues",
            text="salary_in_usd"
        )
        fig_size.update_traces(texttemplate="$%{text:,.0f}", textposition="outside")
        fig_size.update_layout(showlegend=False)
        st.plotly_chart(fig_size, use_container_width=True)

    # ------------------------ Top Jobs ------------------------
    st.markdown("---")
    st.subheader("💎 Top 10 Highest-Paying Jobs")

    selected_year = st.slider("Select Year", 2020, 2030, 2022)

    if selected_year <= 2022:
        top_df = (
            df[df["work_year"] == selected_year]
            .groupby("job_title")["salary_in_usd"]
            .mean()
            .sort_values(ascending=False)
            .head(10)
            .reset_index()
        )
        mode_text = "Actual Data"
    else:
        all_jobs = df["job_title"].unique()
        results = []
        for j in all_jobs:
            j_df = df[df["job_title"] == j]
            most_exp = j_df["experience_level"].mode()[0]
            most_size = j_df["company_size"].mode()[0]

            s, _ = get_salary(selected_year, j, most_exp, most_size, prediction_method)
            results.append({"job_title": j, "salary_in_usd": s})

        top_df = pd.DataFrame(results).sort_values("salary_in_usd", ascending=False).head(10)
        mode_text = "Predicted (Growth/ML)"

    fig_top = px.bar(
        top_df,
        x="salary_in_usd",
        y="job_title",
        orientation="h",
        title=f"Top 10 Highest-Paying Jobs — {selected_year} ({mode_text})",
        color="salary_in_usd",
        color_continuous_scale="Turbo",
        text="salary_in_usd"
    )
    fig_top.update_traces(texttemplate="$%{text:,.0f}", textposition="outside")
    fig_top.update_layout(
        showlegend=False,
        yaxis={"categoryorder": "total ascending"},
        height=500
    )
    st.plotly_chart(fig_top, use_container_width=True)


# ---------------------------------------------------------
# TAB 3 — SALARY CALCULATOR
# ---------------------------------------------------------
with tab3:
    st.subheader("💰 Salary Calculator")

    calc_year = st.slider("Select Year", 2020, 2030, 2025)

    s_val, s_src = get_salary(calc_year, t_job, t_exp, t_size, prediction_method)

    st.markdown("---")

    # Explanation
    if "Actual" in s_src:
        st.success(f"Using: {s_src}")
    elif "ML Only" in s_src:
        st.warning(f"Prediction Source: {s_src}")
    else:
        st.info(f"Prediction Source: {s_src}")

    # Display card
    st.markdown(f"""
    <div class="metric-card">
        <h2>Salary in {calc_year}</h2>
        <h1 style="font-size: 3rem; margin: 1rem 0;">${s_val:,.0f}</h1>
        <p>{t_job} — {t_exp} — {t_size}</p>
    </div>
    """, unsafe_allow_html=True)

    # Market comparison
    st.markdown("---")
    st.subheader("📊 Market Comparison")

    job_mean_all = df[df["job_title"] == t_job]["salary_in_usd"].mean()

    if calc_year <= 2022:
        year_mean = df[(df["job_title"] == t_job) & (df["work_year"] == calc_year)]["salary_in_usd"].mean()
        if pd.notnull(year_mean):
            market_val = year_mean
            label = f"Market Avg ({calc_year})"
        else:
            market_val = job_mean_all
            label = "Overall Market Avg"
    else:
        market_val = job_mean_all
        label = "Historical Market Avg (2020–2022)"

    diff = s_val - market_val
    diff_pct = (diff / market_val * 100) if market_val > 0 else 0

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Your Salary", f"${s_val:,.0f}")
    with c2:
        st.metric(label, f"${market_val:,.0f}")
    with c3:
        st.metric("Difference", f"${diff:,.0f}", f"{diff_pct:+.1f}%")


# ---------------------------------------------------------
# TAB 4 — MODEL INSIGHTS
# ---------------------------------------------------------
with tab4:
    st.subheader("🗺️ Model Insights")

    st.markdown("""
    <div class="info-box">
        <b>Target:</b> salary_in_usd <br>
        <b>Predictors:</b> work_year, job_title, experience_level, company_size <br>
        <b>Training Data:</b> Actual salaries from 2020–2022 <br>
        <b>Forecast:</b> 2023–2030 (Growth / ML / Hybrid)
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # Model performance table
    comp = []
    for mname, m in model_perf.items():
        comp.append({
            "Model": mname,
            "R²": m["r2"],
            "MAE": m["mae"],
            "RMSE": m["rmse"]
        })
    comp_df = pd.DataFrame(comp)

    st.dataframe(comp_df, use_container_width=True, hide_index=True)

    # Feature importance (RF & GB)
    if model_type in ["Random Forest", "Gradient Boosting"]:
        st.subheader("🎯 Feature Importance")

        est = selected_model.named_steps["model"]

        if hasattr(est, "feature_importances_"):
            pre = selected_model.named_steps["prep"]
            ohe = pre.named_transformers_["cat"]

            n_job = len(ohe.categories_[0])
            n_exp = len(ohe.categories_[1])
            n_size = len(ohe.categories_[2])

            imp_list = (
                ["Job Title"] * n_job +
                ["Experience"] * n_exp +
                ["Company Size"] * n_size +
                ["Year"]
            )

            imp_df = pd.DataFrame({
                "Feature": imp_list,
                "Importance": est.feature_importances_
            })

            grp = (
                imp_df.groupby("Feature")["Importance"]
                .sum()
                .sort_values(ascending=False)
                .reset_index()
            )

            fig_imp = px.bar(
                grp,
                x="Importance",
                y="Feature",
                orientation="h",
                color="Importance",
                color_continuous_scale="Blues"
            )
            st.plotly_chart(fig_imp, use_container_width=True)


# ---------------------------------------------------------
# FOOTER
# ---------------------------------------------------------
st.markdown("---")
st.markdown(f"""
<div style="text-align: center; padding: 1rem; color: #666;">
    <p><b>Cybersecurity Salary Dashboard</b></p>
    <p>Model: {model_type} • R² = {model_perf[model_type]['r2']:.3f}</p>
</div>
""", unsafe_allow_html=True)
