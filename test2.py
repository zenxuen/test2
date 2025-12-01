import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
import plotly.express as px
import plotly.graph_objects as go

# ---------------------------------------------------------
# Page Config
# ---------------------------------------------------------
st.set_page_config(
    page_title="Salary Prediction Dashboard",
    layout="wide",
    page_icon="💼"
)

# Custom CSS
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
st.markdown("*Actual Data: 2020–2022 | Forecast Window: 2020–2030*")
st.markdown("**Target:** salary_in_usd | **Features:** work_year, job_title, experience_level, company_size")

# ---------------------------------------------------------
# Load Dataset
# ---------------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("salaries_cyber_clean.csv")
    return df

df = load_data()

# ---------------------------------------------------------
# Sidebar – Engine & Profile
# ---------------------------------------------------------
st.sidebar.header("🎛️ Prediction Settings")

engine = st.sidebar.radio(
    "Prediction Engine",
    ["Growth-Based Pattern", "Random Forest ML"],
    help="• Growth-Based: uses historical salary patterns per profile.\n"
         "• Random Forest ML: pure machine-learning model based on all data."
)

st.sidebar.markdown("---")
st.sidebar.subheader("📊 Dataset Overview")
st.sidebar.info(
    f"- Records: **{len(df):,}**\n"
    f"- Years in dataset: **{', '.join(map(str, sorted(df['work_year'].unique())))}**\n"
    f"- Job titles: **{df['job_title'].nunique()}**\n"
    f"- Avg salary (2020–2022): **${df['salary_in_usd'].mean():,.0f}**"
)

# ---------------------------------------------------------
# Train Random-Forest Model (used for RF mode + fallbacks)
# ---------------------------------------------------------
@st.cache_resource
def train_rf_model(data: pd.DataFrame):
    feature_cols = ["work_year", "job_title", "experience_level", "company_size"]
    target_col = "salary_in_usd"

    X = data[feature_cols]
    y = data[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    cat_cols = ["job_title", "experience_level", "company_size"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
        ],
        remainder="passthrough"  # keep work_year numeric
    )

    rf = RandomForestRegressor(
        n_estimators=300,
        random_state=42,
        max_depth=18,
        min_samples_split=3,
        min_samples_leaf=1,
        n_jobs=-1,
    )

    pipeline = Pipeline([
        ("prep", preprocessor),
        ("model", rf),
    ])

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)

    cv_scores = cross_val_score(
        pipeline, X_train, y_train, cv=5, scoring="r2"
    )

    metrics = dict(
        r2=r2_score(y_test, y_pred),
        mae=mean_absolute_error(y_test, y_pred),
        rmse=np.sqrt(mean_squared_error(y_test, y_pred)),
        cv_r2_mean=cv_scores.mean(),
        cv_r2_std=cv_scores.std(),
    )

    return pipeline, metrics

rf_model, rf_metrics = train_rf_model(df)

with st.sidebar.expander("📈 Random Forest Performance", expanded=True):
    st.metric("R²", f"{rf_metrics['r2']:.3f}")
    st.metric("MAE", f"${rf_metrics['mae']:,.0f}")
    st.metric("RMSE", f"${rf_metrics['rmse']:,.0f}")
    st.caption(f"CV R²: {rf_metrics['cv_r2_mean']:.3f} ± {rf_metrics['cv_r2_std']:.3f}")

# ---------------------------------------------------------
# Growth-Pattern Helpers (UNCHANGED behaviour)
# ---------------------------------------------------------
@st.cache_data
def get_mean_growth_from_similar(data, job, exp):
    # same-job + experience first
    similar = data[
        (data["job_title"] == job) &
        (data["experience_level"] == exp)
    ]

    growth_rates = []

    if len(similar) > 0:
        for size in similar["company_size"].unique():
            prof = data[
                (data["job_title"] == job) &
                (data["experience_level"] == exp) &
                (data["company_size"] == size)
            ].groupby("work_year")["salary_in_usd"].mean().sort_index()

            if len(prof) >= 2:
                years = prof.index.values
                salaries = prof.values
                g = (salaries[-1] - salaries[0]) / salaries[0] / (years[-1] - years[0])
                g = float(np.clip(g, -0.20, 0.20))
                growth_rates.append(g)

    if len(growth_rates) > 0:
        return float(np.mean(growth_rates)), len(growth_rates)

    # Fallback: job-only
    job_series = data[data["job_title"] == job].groupby("work_year")["salary_in_usd"].mean().sort_index()
    if len(job_series) >= 2:
        years = job_series.index.values
        salaries = job_series.values
        g = (salaries[-1] - salaries[0]) / salaries[0] / (years[-1] - years[0])
        g = float(np.clip(g, -0.15, 0.15))
        return g, 0

    # Fallback: experience-only
    exp_series = data[data["experience_level"] == exp].groupby("work_year")["salary_in_usd"].mean().sort_index()
    if len(exp_series) >= 2:
        years = exp_series.index.values
        salaries = exp_series.values
        g = (salaries[-1] - salaries[0]) / salaries[0] / (years[-1] - years[0])
        g = float(np.clip(g, -0.15, 0.15))
        return g, 0

    # Final fallback: 5%
    return 0.05, 0


@st.cache_data
def calculate_growth_rate(data, job, exp, size):
    profile_data = data[
        (data["job_title"] == job) &
        (data["experience_level"] == exp) &
        (data["company_size"] == size)
    ].groupby("work_year")["salary_in_usd"].mean().sort_index()

    if len(profile_data) < 2:
        return None, None, None

    years = profile_data.index.values
    salaries = profile_data.values

    first_year, last_year = years[0], years[-1]
    first_salary, last_salary = salaries[0], salaries[-1]

    if first_salary <= 0:
        return None, None, None

    growth = (last_salary - first_salary) / first_salary / (last_year - first_year)
    growth = float(np.clip(growth, -0.20, 0.20))

    return growth, years.tolist(), salaries.tolist()


def get_salary_growth(year, job, exp, size):
    """
    ORIGINAL growth-based logic (2020–2022 use actual if available).
    ML is ONLY used as fallback when there is 0 history for this profile.
    """
    profile_history = df[
        (df["job_title"] == job) &
        (df["experience_level"] == exp) &
        (df["company_size"] == size)
    ]

    years_available = sorted(profile_history["work_year"].unique())
    num_years = len(years_available)

    # 1) For 2020–2022: use actual if available
    if year <= 2022:
        actual_for_year = profile_history[profile_history["work_year"] == year]
        if len(actual_for_year) > 0:
            return actual_for_year["salary_in_usd"].mean(), "Actual"

    # Case 0: no history → RF fallback
    if num_years == 0:
        X_pred = pd.DataFrame({
            "work_year": [year],
            "job_title": [job],
            "experience_level": [exp],
            "company_size": [size],
        })
        pred = rf_model.predict(X_pred)[0]
        return max(0, pred), "Predicted (RF – No Profile History)"

    # Case 1: exactly 1 year of data
    if num_years == 1:
        base_year = years_available[0]
        base_salary = profile_history["salary_in_usd"].mean()

        if year == base_year:
            return base_salary, "Actual (Single Year)"

        avg_growth, n_profiles = get_mean_growth_from_similar(df, job, exp)
        years_ahead = year - base_year
        pred = base_salary * ((1 + avg_growth) ** years_ahead)

        if n_profiles > 0:
            src = f"Predicted (Similar Profiles Pattern, {n_profiles} profiles)"
        else:
            src = "Predicted (Job/Experience-Level Pattern)"

        return max(0, pred), src

    # Case 2: ≥2 years → own profile pattern
    if num_years >= 2:
        growth_rate, yrs, sals = calculate_growth_rate(df, job, exp, size)
        if growth_rate is None:
            # fallback to RF
            X_pred = pd.DataFrame({
                "work_year": [year],
                "job_title": [job],
                "experience_level": [exp],
                "company_size": [size],
            })
            pred = rf_model.predict(X_pred)[0]
            return max(0, pred), "Predicted (RF Fallback)"

        last_year = yrs[-1]
        last_salary = sals[-1]

        # if within historical range & we have actual
        if year in yrs:
            actual_for_year = profile_history[profile_history["work_year"] == year]
            if len(actual_for_year) > 0:
                return actual_for_year["salary_in_usd"].mean(), "Actual"

        years_ahead = year - last_year
        pred = last_salary * ((1 + growth_rate) ** years_ahead)

        return max(0, pred), "Predicted (Own Profile Pattern)"

    # safety fallback
    X_pred = pd.DataFrame({
        "work_year": [year],
        "job_title": [job],
        "experience_level": [exp],
        "company_size": [size],
    })
    pred = rf_model.predict(X_pred)[0]
    return max(0, pred), "Predicted (RF Fallback)"


def get_salary_rf(year, job, exp, size):
    """
    PURE Random-Forest mode.
    2020–2022 still show actual data when available, otherwise RF prediction.
    No growth-pattern logic is used here.
    """
    profile_history = df[
        (df["job_title"] == job) &
        (df["experience_level"] == exp) &
        (df["company_size"] == size)
    ]

    if year <= 2022:
        actual_for_year = profile_history[profile_history["work_year"] == year]
        if len(actual_for_year) > 0:
            return actual_for_year["salary_in_usd"].mean(), "Actual"

    X_pred = pd.DataFrame({
        "work_year": [year],
        "job_title": [job],
        "experience_level": [exp],
        "company_size": [size],
    })
    pred = rf_model.predict(X_pred)[0]
    return max(0, pred), "Predicted (Random Forest)"

# ---------------------------------------------------------
# MAIN PAGE – single layout
# ---------------------------------------------------------
st.subheader("⚙️ Choose Your Profile")

c1, c2, c3 = st.columns(3)
with c1:
    sel_job = st.selectbox("👔 Job Title", sorted(df["job_title"].unique()))
with c2:
    sel_exp = st.selectbox("📈 Experience Level", sorted(df["experience_level"].unique()))
with c3:
    sel_size = st.selectbox("🏢 Company Size", sorted(df["company_size"].unique()))

st.markdown("---")

# Explain data availability & growth for this profile (always based on growth logic)
profile_hist = df[
    (df["job_title"] == sel_job) &
    (df["experience_level"] == sel_exp) &
    (df["company_size"] == sel_size)
]
uniq_years = profile_hist["work_year"].nunique()

if uniq_years == 0:
    st.warning(
        "⚠️ This exact profile has **no historical records** in 2020–2022. "
        "Growth-based mode will lean on RF fallback; Random-Forest mode is fully ML-driven."
    )
elif uniq_years == 1:
    y0 = int(profile_hist["work_year"].iloc[0])
    s0 = profile_hist["salary_in_usd"].mean()
    g, nprof = get_mean_growth_from_similar(df, sel_job, sel_exp)
    msg = (
        f"ℹ️ This profile has **only 1 year** of data "
        f"({y0}: ${s0:,.0f}). Estimated average growth from similar profiles: "
        f"**{g*100:.1f}%/year**"
    )
    if g < 0:
        st.warning("📉 " + msg)
    else:
        st.info(msg)
else:
    g, yrs, sals = calculate_growth_rate(df, sel_job, sel_exp, sel_size)
    if g is not None:
        seq = " → ".join(f"${v:,.0f}" for v in sals)
        if g < 0:
            st.warning(
                f"📉 This profile has **{uniq_years} years** of data: {seq}. "
                f"Average change: **{g*100:.1f}% per year (decline)**."
            )
        else:
            st.success(
                f"✅ This profile has **{uniq_years} years** of data: {seq}. "
                f"Average growth: **{g*100:.1f}% per year**."
            )
    else:
        st.info(
            "ℹ️ Multiple years of data exist, but a stable growth rate "
            "could not be estimated. RF fallback will support predictions."
        )

st.markdown("---")

# Year slider for detailed view
year_selected = st.slider("📅 Focus Year", 2020, 2030, 2025)

# Build full 2020–2030 forecast according to selected engine
years = np.arange(2020, 2031)
rows = []

for y in years:
    if engine == "Growth-Based Pattern":
        sal, src = get_salary_growth(y, sel_job, sel_exp, sel_size)
    else:  # Random Forest ML
        sal, src = get_salary_rf(y, sel_job, sel_exp, sel_size)

    rows.append(dict(Year=int(y), Salary=float(sal), Source=src))

forecast_df = pd.DataFrame(rows)

# Split actual vs predicted (based on Source text)
actual_mask = forecast_df["Source"].str.contains("Actual", na=False)
actual_df = forecast_df[actual_mask]
pred_df = forecast_df[~actual_mask]

# Summary metrics 2020 vs 2030 (if valid)
safe = forecast_df.dropna(subset=["Salary"])
if not safe.empty:
    if 2020 in safe["Year"].values:
        s0 = float(safe.loc[safe["Year"] == 2020, "Salary"].iloc[0])
    else:
        s0 = float(safe.iloc[0]["Salary"])

    if 2030 in safe["Year"].values:
        s1 = float(safe.loc[safe["Year"] == 2030, "Salary"].iloc[0])
    else:
        s1 = float(safe.iloc[-1]["Salary"])

    if year_selected in safe["Year"].values:
        sy = float(safe.loc[safe["Year"] == year_selected, "Salary"].iloc[0])
    else:
        sy = s1

    mc1, mc2, mc3 = st.columns(3)
    with mc1:
        st.metric("📅 2020 Salary", f"${s0:,.0f}")
    with mc2:
        st.metric(f"📅 {year_selected} Salary", f"${sy:,.0f}")
    with mc3:
        if s0 > 0:
            gtot = (s1 - s0) / s0 * 100
            st.metric("📈 Total Change (2020–2030)", f"{gtot:.1f}%")
        else:
            st.metric("📈 Total Change (2020–2030)", "N/A")

st.markdown("---")

# Single line chart
fig = go.Figure()

if len(actual_df) > 0:
    fig.add_trace(go.Scatter(
        x=actual_df["Year"],
        y=actual_df["Salary"],
        mode="lines+markers",
        name="Actual (2020–2022)",
        line=dict(color="#10b981", width=4),
        marker=dict(size=10),
        hovertemplate="Year %{x}<br>Salary $%{y:,.0f}<extra></extra>",
    ))

if len(pred_df) > 0:
    fig.add_trace(go.Scatter(
        x=pred_df["Year"],
        y=pred_df["Salary"],
        mode="lines+markers",
        name=f"{engine}",
        line=dict(color="#667eea", width=4, dash="dash"),
        marker=dict(size=10),
        hovertemplate="Year %{x}<br>Salary $%{y:,.0f}<br>%{text}<extra></extra>",
        text=pred_df["Source"],
    ))

if len(actual_df) > 0 and len(pred_df) > 0:
    fig.add_vline(
        x=2022.5,
        line_dash="dot",
        line_color="red",
        line_width=2,
        annotation_text="Actual → Forecast",
        annotation_position="top",
    )

fig.update_layout(
    title=dict(
        text=f"Salary Trajectory (2020–2030)<br><sup>{sel_job} | {sel_exp} | {sel_size}</sup>",
        font=dict(size=20),
    ),
    xaxis_title="Year",
    yaxis_title="Salary (USD)",
    template="plotly_white",
    hovermode="x unified",
    height=520,
    xaxis=dict(dtick=1),
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1,
    ),
)

st.plotly_chart(fig, use_container_width=True)

# Focus-year card
focus_salary = float(forecast_df.loc[forecast_df["Year"] == year_selected, "Salary"].iloc[0])
focus_source = forecast_df.loc[forecast_df["Year"] == year_selected, "Source"].iloc[0]

st.markdown("---")
fx1, fx2, fx3 = st.columns([1, 2, 1])
with fx2:
    st.markdown(f"""
    <div class="metric-card">
        <h3>Forecast for {year_selected}</h3>
        <h1 style="font-size: 3.5rem; margin: 1rem 0;">${focus_salary:,.0f}</h1>
        <p style="font-size: 1.1rem;">{sel_job}</p>
        <p>{sel_exp} | {sel_size}</p>
        <p style="font-size: 0.9rem; opacity: 0.9;">Source: {focus_source}</p>
    </div>
    """, unsafe_allow_html=True)

# Simple comparison: this profile vs overall job-title average
st.markdown("---")
st.subheader("📊 Market Context (Historical 2020–2022)")

job_hist_avg = df[df["job_title"] == sel_job]["salary_in_usd"].mean()
diff = focus_salary - job_hist_avg
pct = (diff / job_hist_avg) * 100 if job_hist_avg > 0 else 0.0

cma, cmb, cmc = st.columns(3)
with cma:
    st.metric("This Profile (Selected Year)", f"${focus_salary:,.0f}")
with cmb:
    st.metric("Job-Title Historical Avg", f"${job_hist_avg:,.0f}")
with cmc:
    st.metric("Difference", f"${diff:,.0f}", f"{pct:+.1f}%")

st.markdown("---")
st.markdown(
    f"<div style='text-align:center; color:#666; padding:1.5rem 0;'>"
    f"Engine: <b>{engine}</b> · RF R²: {rf_metrics['r2']:.3f} · "
    f"Target: salary_in_usd · Features: work_year, job_title, experience_level, company_size"
    f"</div>",
    unsafe_allow_html=True,
)
