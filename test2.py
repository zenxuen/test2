import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

import plotly.graph_objects as go

# ---------------------------------------------------------
# Page Config & Basic Styles
# ---------------------------------------------------------
st.set_page_config(
    page_title="Cybersecurity Salary Forecast",
    page_icon="💼",
    layout="wide",
)

st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.4rem;
        font-weight: 800;
        background: linear-gradient(90deg, #6366f1 0%, #a855f7 50%, #ec4899 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        color: #6b7280;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #6366f1, #a855f7);
        padding: 1.2rem 1.4rem;
        border-radius: 12px;
        color: white;
        text-align: center;
    }
    .metric-card h1 {
        font-size: 2.4rem;
        margin: 0.4rem 0;
    }
    .soft-box {
        background-color: #eff6ff;
        border-radius: 10px;
        padding: 0.8rem 1rem;
        font-size: 0.9rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="main-header">💼 Cybersecurity Salary Forecast (2020–2030)</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="sub-header">Actual 2020–2022 data + two engines: Growth-based rules & Random Forest ML.</div>',
    unsafe_allow_html=True,
)

# ---------------------------------------------------------
# Load Data
# ---------------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("salaries_cyber_clean.csv")
    # Keep only the columns we need
    keep_cols = [
        "work_year",
        "experience_level",
        "employment_type",
        "job_title",
        "salary_in_usd",
        "remote_ratio",
        "company_location",
        "company_size",
    ]
    df = df[keep_cols].copy()
    df["work_year"] = df["work_year"].astype(int)
    return df

df = load_data()

# ---------------------------------------------------------
# Build "valid profile" list so every choice has data
# ---------------------------------------------------------
profiles = (
    df.groupby(["job_title", "experience_level", "company_size"])
      .size()
      .reset_index(name="count")
)

# ---------------------------------------------------------
# Train Random Forest ML model
#   Target: salary_in_usd
#   Features:
#     - work_year (numeric)
#     - experience_level (cat)
#     - employment_type (cat)
#     - job_title (cat)
#     - remote_ratio (numeric)
#     - company_location (cat)
#     - company_size (cat)
# ---------------------------------------------------------
@st.cache_resource
def train_ml_model(data: pd.DataFrame):
    feature_cols = [
        "work_year",
        "experience_level",
        "employment_type",
        "job_title",
        "remote_ratio",
        "company_location",
        "company_size",
    ]
    target_col = "salary_in_usd"

    X = data[feature_cols].copy()
    y = data[target_col].copy()

    cat_cols = ["experience_level", "employment_type", "job_title",
                "company_location", "company_size"]
    num_cols = ["work_year", "remote_ratio"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ],
        remainder="passthrough",  # keep numeric as is
    )

    rf = RandomForestRegressor(
        n_estimators=300,
        random_state=42,
        max_depth=18,
        min_samples_split=4,
        min_samples_leaf=2,
        n_jobs=-1,
    )

    model = Pipeline(
        steps=[
            ("prep", preprocessor),
            ("model", rf),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred) ** 0.5

    metrics = {
        "R2": r2,
        "MAE": mae,
        "RMSE": rmse,
    }

    return model, metrics, feature_cols

ml_model, ml_metrics, ml_feature_cols = train_ml_model(df)

# ---------------------------------------------------------
# Helper: Growth-based pattern for one profile
# ---------------------------------------------------------
def compute_profile_growth(data, job_title, exp_level, company_size):
    profile = (
        data[
            (data["job_title"] == job_title)
            & (data["experience_level"] == exp_level)
            & (data["company_size"] == company_size)
        ]
        .groupby("work_year")["salary_in_usd"]
        .mean()
        .sort_index()
    )

    if len(profile) < 2:
        return None, None, None

    years = profile.index.to_numpy()
    salaries = profile.to_numpy()

    first_year = years[0]
    last_year = years[-1]
    first_salary = float(salaries[0])
    last_salary = float(salaries[-1])

    if first_salary <= 0 or last_year == first_year:
        return None, None, None

    annual_growth = (last_salary - first_salary) / first_salary / (last_year - first_year)
    annual_growth = float(np.clip(annual_growth, -0.25, 0.25))  # clamp

    return annual_growth, years, salaries


def predict_with_ml(
    year: int,
    job_title: str,
    exp_level: str,
    company_size: str,
    employment_type: str,
    remote_ratio: float,
    company_location: str,
):
    sample = pd.DataFrame(
        [
            {
                "work_year": year,
                "experience_level": exp_level,
                "employment_type": employment_type,
                "job_title": job_title,
                "remote_ratio": remote_ratio,
                "company_location": company_location,
                "company_size": company_size,
            }
        ]
    )
    pred = ml_model.predict(sample)[0]
    return float(max(0.0, pred)), "Random Forest (ML)"


def predict_with_growth(
    year: int,
    job_title: str,
    exp_level: str,
    company_size: str,
    employment_type: str,
    remote_ratio: float,
    company_location: str,
):
    # Historical rows for that exact profile
    mask = (
        (df["job_title"] == job_title)
        & (df["experience_level"] == exp_level)
        & (df["company_size"] == company_size)
        & (df["employment_type"] == employment_type)
        & (df["company_location"] == company_location)
        & (df["remote_ratio"] == remote_ratio)
    )
    profile_rows = df[mask]

    # 1. 2020–2022: if actual data exists for that exact combo, use it
    if year <= 2022 and not profile_rows.empty:
        actual = profile_rows[profile_rows["work_year"] == year]
        if not actual.empty:
            return float(actual["salary_in_usd"].mean()), "Actual (dataset)"

    # 2. Growth on (job, exp, size) profile
    growth, years, salaries = compute_profile_growth(df, job_title, exp_level, company_size)

    if growth is not None:
        last_year = int(years[-1])
        last_salary = float(salaries[-1])
        years_ahead = year - last_year
        predicted = last_salary * ((1.0 + growth) ** years_ahead)
        return float(max(0.0, predicted)), "Growth-based (profile pattern)"

    # 3. Fallback: Random Forest
    ml_val, _ = predict_with_ml(
        year,
        job_title,
        exp_level,
        company_size,
        employment_type,
        remote_ratio,
        company_location,
    )
    return ml_val, "Random Forest (fallback)"


def get_salary(
    year: int,
    job_title: str,
    exp_level: str,
    company_size: str,
    employment_type: str,
    remote_ratio: float,
    company_location: str,
    engine: str,
):
    """
    engine:
      - 'Growth-based'
      - 'Random Forest (ML)'
    """
    # Always respect actual data for 2020–2022 if it exists
    base_mask = (
        (df["job_title"] == job_title)
        & (df["experience_level"] == exp_level)
        & (df["company_size"] == company_size)
        & (df["employment_type"] == employment_type)
        & (df["company_location"] == company_location)
        & (df["remote_ratio"] == remote_ratio)
        & (df["work_year"] == year)
    )
    base_rows = df[base_mask]
    if year <= 2022 and not base_rows.empty:
        return float(base_rows["salary_in_usd"].mean()), "Actual (dataset)"

    if engine == "Growth-based":
        return predict_with_growth(
            year,
            job_title,
            exp_level,
            company_size,
            employment_type,
            remote_ratio,
            company_location,
        )
    else:
        return predict_with_ml(
            year,
            job_title,
            exp_level,
            company_size,
            employment_type,
            remote_ratio,
            company_location,
        )

# ---------------------------------------------------------
# Sidebar Controls
# ---------------------------------------------------------
st.sidebar.header("⚙️ Settings")

engine = st.sidebar.radio(
    "Prediction engine",
    ["Growth-based", "Random Forest (ML)"],
    index=0,
)

st.sidebar.markdown("### 🎛 Select Profile")

# Job title
all_jobs = sorted(profiles["job_title"].unique())
selected_job = st.sidebar.selectbox("Job title", all_jobs)

# Experience options limited to this job
exp_options = sorted(
    profiles.loc[profiles["job_title"] == selected_job, "experience_level"].unique()
)
selected_exp = st.sidebar.selectbox("Experience level", exp_options)

# Company size options limited by job + exp
size_options = sorted(
    profiles.loc[
        (profiles["job_title"] == selected_job)
        & (profiles["experience_level"] == selected_exp),
        "company_size",
    ].unique()
)
selected_size = st.sidebar.selectbox("Company size", size_options)

# Subset of df for this profile (for default values)
subset = df[
    (df["job_title"] == selected_job)
    & (df["experience_level"] == selected_exp)
    & (df["company_size"] == selected_size)
]

if subset.empty:
    st.sidebar.error("No data for this profile combination. Please choose another.")
    st.stop()

default_employment = subset["employment_type"].mode()[0]
default_location = subset["company_location"].mode()[0]
default_remote = float(subset["remote_ratio"].mode()[0])

selected_employment = st.sidebar.selectbox(
    "Employment type", sorted(subset["employment_type"].unique()),
    index=list(sorted(subset["employment_type"].unique())).index(default_employment),
)
selected_location = st.sidebar.selectbox(
    "Company location", sorted(subset["company_location"].unique()),
    index=list(sorted(subset["company_location"].unique())).index(default_location),
)
selected_remote = st.sidebar.slider(
    "Remote ratio (%)",
    min_value=0,
    max_value=100,
    value=int(default_remote),
    step=25,
)

st.sidebar.markdown("---")
st.sidebar.subheader("📏 Calculator Year")
calc_year = st.sidebar.slider("Year for detailed view", 2020, 2030, 2025)

# ---------------------------------------------------------
# Tabs
# ---------------------------------------------------------
tab1, tab2 = st.tabs(["📈 Forecast & Calculator", "🧠 Model Info"])

# ---------------------------------------------------------
# TAB 1: Forecast & Calculator
# ---------------------------------------------------------
with tab1:
    col_left, col_right = st.columns([2, 1])

    with col_left:
        st.subheader("Profile forecast (2020–2030)")

        YEARS = list(range(2020, 2031))
        rows = []
        for yr in YEARS:
            sal, source = get_salary(
                yr,
                selected_job,
                selected_exp,
                selected_size,
                selected_employment,
                selected_remote,
                selected_location,
                engine,
            )
            rows.append({"year": yr, "salary": sal, "source": source})

        forecast_df = pd.DataFrame(rows)

        # Split actual vs predicted
        actual_mask = forecast_df["source"].str.contains("Actual", na=False)
        actual_df = forecast_df[actual_mask]
        pred_df = forecast_df[~actual_mask]

        fig = go.Figure()

        if not actual_df.empty:
            fig.add_trace(
                go.Scatter(
                    x=actual_df["year"],
                    y=actual_df["salary"],
                    mode="lines+markers",
                    name="Actual (2020–2022)",
                    line=dict(color="#10b981", width=4),
                )
            )

        if not pred_df.empty:
            fig.add_trace(
                go.Scatter(
                    x=pred_df["year"],
                    y=pred_df["salary"],
                    mode="lines+markers",
                    name="Forecast",
                    line=dict(color="#6366f1", width=3, dash="dash"),
                )
            )

        if not actual_df.empty and not pred_df.empty:
            fig.add_vline(
                x=2022.5,
                line_dash="dot",
                line_color="#ef4444",
                line_width=1.5,
            )

        fig.update_layout(
            height=420,
            xaxis_title="Year",
            yaxis_title="Salary (USD)",
            template="plotly_white",
            hovermode="x unified",
            legend=dict(
                orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
            ),
        )

        st.plotly_chart(fig, width="stretch")

        with st.expander("Show forecast table"):
            tmp = forecast_df.copy()
            tmp["salary"] = tmp["salary"].map(lambda v: f"${v:,.0f}")
            st.dataframe(tmp, width="stretch", hide_index=True)

    with col_right:
        st.subheader("Salary calculator")

        calc_salary, calc_source = get_salary(
            calc_year,
            selected_job,
            selected_exp,
            selected_size,
            selected_employment,
            selected_remote,
            selected_location,
            engine,
        )

        st.markdown(
            f"""
            <div class="metric-card">
                <div style="font-size:0.9rem; opacity:0.9;">Salary in {calc_year}</div>
                <h1>${calc_salary:,.0f}</h1>
                <div style="font-size:0.95rem;">{selected_job}</div>
                <div style="font-size:0.85rem; opacity:0.9;">
                    {selected_exp} · {selected_size}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("<br>", unsafe_allow_html=True)

        hist_profile = subset[subset["work_year"].between(2020, 2022, inclusive="both")]
        if not hist_profile.empty:
            min_hist = float(hist_profile["salary_in_usd"].min())
            max_hist = float(hist_profile["salary_in_usd"].max())
            st.markdown(
                f"""
                <div class="soft-box">
                    <b>Historical range (2020–2022):</b><br>
                    Min: ${min_hist:,.0f} · Max: ${max_hist:,.0f}<br>
                    Source: dataset (no prediction)
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                """
                <div class="soft-box">
                    No direct 2020–2022 history for this exact combination.
                    Values shown are based on the selected engine.
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("##### Engine used")
        if "Growth" in calc_source:
            st.info(f"Growth-based pattern · {calc_source}")
        elif "Actual" in calc_source:
            st.success("Actual dataset value (no prediction)")
        else:
            st.warning(f"Random Forest ML · {calc_source}")

# ---------------------------------------------------------
# TAB 2: Model Info
# ---------------------------------------------------------
with tab2:
    st.subheader("Random Forest model (ML engine)")

    c1, c2, c3 = st.columns(3)
    c1.metric("R² on test set", f"{ml_metrics['R2']:.3f}")
    c2.metric("MAE", f"${ml_metrics['MAE']:,.0f}")
    c3.metric("RMSE", f"${ml_metrics['RMSE']:,.0f}")

    st.markdown("---")
    st.markdown("#### Features used by the ML model")

    feat_df = pd.DataFrame(
        {
            "Feature": ml_feature_cols,
            "Type": [
                "Numeric" if col in ["work_year", "remote_ratio"] else "Categorical"
                for col in ml_feature_cols
            ],
            "Meaning": [
                "Year of the observation (captures time trend)",
                "Seniority / level of experience",
                "Employment type (FT, PT, etc.)",
                "Job title in cybersecurity",
                "How remote the job is (0–100)",
                "Country / region of the company",
                "Size class of the company",
            ],
        }
    )

    st.dataframe(feat_df, width="stretch", hide_index=True)

    st.markdown(
        """
        **How the two engines work:**
        - **Growth-based:** looks at how this profile's salary changed across years
          and extrapolates that annual growth rate into the future.  
        - **Random Forest (ML):** learns patterns across *all* rows in the dataset
          using the features above and predicts salary directly.
        """
    )
