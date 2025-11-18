# app.py — Premium Salary Prediction Dashboard
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import plotly.express as px
import plotly.graph_objects as go

# ---------------------------
# Page config
# ---------------------------
st.set_page_config(page_title="💼 Premium Salary Prediction Dashboard",
                   layout="wide",
                   page_icon="💼")

st.title("💼 Premium Salary Prediction Dashboard")

# ---------------------------
# Load CSV (try a few common paths)
# ---------------------------
possible_paths = [
    "./salaries_cyber_clean.csv",
    "salaries_cyber_clean.csv",
    "salaries_cyber_clean.csv",
    "/mount/src/test2/salaries_cyber_clean.csv"
]

df = None
for p in possible_paths:
    pth = Path(p)
    if pth.exists():
        try:
            df = pd.read_csv(pth)
            break
        except Exception:
            pass

if df is None:
    st.error("Could not find `salaries_cyber_clean.csv`. Put the CSV in the app folder or adjust the path in the script.")
    st.stop()

# Basic cleanup / preview columns expected
expected_cols = {"work_year", "experience_level", "employment_type", "job_title",
                 "salary_in_usd", "employee_residence", "remote_ratio", "company_location", "company_size"}
missing = expected_cols - set(df.columns)
if missing:
    st.warning(f"The CSV is missing columns: {', '.join(missing)}. The app will still try to run with available columns.")

# ---------------------------
# Sidebar: model & options
# ---------------------------
st.sidebar.header("Model & Forecast Options")
model_choice = st.sidebar.selectbox("Model", ["Random Forest", "Gradient Boosting", "Linear Regression"])
n_estimators = st.sidebar.slider("RF / GB estimators", 50, 500, 200, 50)
max_depth = st.sidebar.slider("RF / GB max depth (0 = None)", 0, 20, 10)
test_size = st.sidebar.slider("Test set size (%)", 10, 40, 20)
random_state = 42

# Forecast years
start_year = st.sidebar.number_input("Forecast start year", min_value=2020, max_value=2025, value=2021)
end_year = st.sidebar.number_input("Forecast end year", min_value=2025, max_value=2035, value=2035)

# UI: quick stats
st.sidebar.markdown("---")
st.sidebar.write("Data summary")
st.sidebar.write(f"Rows: {len(df):,}")
st.sidebar.write(f"Years: {int(df['work_year'].min())} – {int(df['work_year'].max())}")
st.sidebar.write(f"Unique jobs: {df['job_title'].nunique()}")

# ---------------------------
# Prepare dataset & pipeline
# ---------------------------
feature_cols = ["work_year", "job_title", "experience_level", "company_size"]
# ensure columns exist
feature_cols = [c for c in feature_cols if c in df.columns]
target_col = "salary_in_usd"

X = df[feature_cols].copy()
y = df[target_col].copy()

# Column transformer (one-hot categorical, passthrough year)
categorical_cols = [c for c in ["job_title", "experience_level", "company_size"] if c in X.columns]
preprocessor = ColumnTransformer(
    transformers=[("cat", OneHotEncoder(handle_unknown="ignore", sparse=False), categorical_cols)],
    remainder="passthrough",
    sparse_threshold=0
)

@st.cache_resource(show_spinner=False)
def build_and_train_model(model_choice, n_estimators, max_depth, X, y, test_size, random_state):
    # pick model
    if model_choice == "Random Forest":
        reg = RandomForestRegressor(n_estimators=n_estimators, max_depth=(None if max_depth==0 else max_depth),
                                    random_state=random_state)
    elif model_choice == "Gradient Boosting":
        reg = GradientBoostingRegressor(n_estimators=n_estimators, max_depth=(1 if max_depth==0 else max_depth),
                                        random_state=random_state)
    else:
        reg = LinearRegression()

    pipeline = Pipeline([("preprocessor", preprocessor), ("regressor", reg)])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size/100.0, random_state=random_state)
    pipeline.fit(X_train, y_train)
    # metrics
    y_pred = pipeline.predict(X_test)
    metrics = {
        "mae": mean_absolute_error(y_test, y_pred),
        "rmse": mean_squared_error(y_test, y_pred, squared=False),
        "r2": r2_score(y_test, y_pred),
        "train_rows": len(X_train),
        "test_rows": len(X_test)
    }
    return pipeline, metrics, (X_train, X_test, y_train, y_test)

model_pipeline, metrics, splits = build_and_train_model(model_choice, n_estimators, max_depth, X, y, test_size, random_state)
X_train, X_test, y_train, y_test = splits

# ---------------------------
# Top row metrics
# ---------------------------
col_kpi1, col_kpi2, col_kpi3, col_kpi4 = st.columns(4)
col_kpi1.metric("Rows", f"{len(df):,}")
col_kpi2.metric("Train / Test", f"{metrics['train_rows']:,} / {metrics['test_rows']:,}")
col_kpi3.metric("MAE", f"${metrics['mae']:,.0f}")
col_kpi4.metric("RMSE", f"${metrics['rmse']:,.0f}")

# ---------------------------
# Tabs: Overview, Forecast, Compare, Diagnostics
# ---------------------------
tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Forecast", "Compare", "Model Diagnostics"])

with tab1:
    st.header("Dataset Overview")
    st.write("A quick look at the data (first rows):")
    st.dataframe(df.head(10), use_container_width=True)
    st.markdown("**Salary distribution**")
    fig_hist = px.histogram(df, x=target_col, nbins=40, title="Salary distribution", template="plotly_white")
    st.plotly_chart(fig_hist, use_container_width=True)
    st.markdown("Top job titles")
    st.table(df["job_title"].value_counts().head(10))

with tab2:
    st.header("Forecast (single selection)")
    # selection for single forecast
    col1, col2, col3 = st.columns(3)
    with col1:
        job = st.selectbox("Job Title", sorted(df["job_title"].unique()))
    with col2:
        exp = st.selectbox("Experience Level", sorted(df["experience_level"].unique()))
    with col3:
        size = st.selectbox("Company Size", sorted(df["company_size"].unique()))

    # build future dataframe
    years = np.arange(start_year, end_year + 1)
    future_df = pd.DataFrame({
        "work_year": years,
        "job_title": [job] * len(years),
        "experience_level": [exp] * len(years),
        "company_size": [size] * len(years)
    })
    preds = model_pipeline.predict(future_df)

    # If RandomForest, get prediction intervals via individual estimators (approx.)
    lower, upper = None, None
    if model_choice == "Random Forest":
        try:
            rf = model_pipeline.named_steps["regressor"]
            all_preds = np.vstack([est.predict(model_pipeline.named_steps["preprocessor"].transform(future_df)) 
                                   for est in rf.estimators_])
            lower = np.percentile(all_preds, 10, axis=0)
            upper = np.percentile(all_preds, 90, axis=0)
        except Exception:
            lower = None
            upper = None

    forecast_plot_df = pd.DataFrame({"Year": years, "Predicted": preds})
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=forecast_plot_df["Year"], y=forecast_plot_df["Predicted"],
                             mode="lines+markers", name="Prediction", line=dict(width=3)))
    if lower is not None and upper is not None:
        fig.add_trace(go.Scatter(
            x=np.concatenate([years, years[::-1]]),
            y=np.concatenate([upper, lower[::-1]]),
            fill='toself',
            fillcolor='rgba(0,176,246,0.15)',
            line=dict(color='rgba(255,255,255,0)'),
            hoverinfo="skip",
            showlegend=True,
            name="80% PI"
        ))

    fig.update_layout(title=f"Forecast for {job} — {exp} — {size}",
                      xaxis=dict(dtick=1),
                      yaxis_title="Predicted Salary (USD)",
                      template="plotly_white", hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

    # Download forecast as csv
    download_df = future_df.copy()
    download_df["predicted_salary_usd"] = preds
    csv = download_df.to_csv(index=False).encode('utf-8')
    st.download_button("Download forecast CSV", data=csv, file_name=f"forecast_{job}_{exp}_{size}.csv", mime="text/csv")

with tab3:
    st.header("Compare multiple jobs")
    jobs = st.multiselect("Select job titles to compare", options=sorted(df["job_title"].unique()), default=[df["job_title"].unique()[0]])
    exp_cmp = st.selectbox("Experience Level (compare)", sorted(df["experience_level"].unique()))
    size_cmp = st.selectbox("Company Size (compare)", sorted(df["company_size"].unique()))

    years = np.arange(start_year, end_year + 1)
    compare_df = pd.DataFrame()
    for j in jobs:
        tmp = pd.DataFrame({
            "work_year": years,
            "job_title": [j] * len(years),
            "experience_level": [exp_cmp] * len(years),
            "company_size": [size_cmp] * len(years)
        })
        tmp["predicted"] = model_pipeline.predict(tmp)
        tmp["job_title"] = j
        compare_df = pd.concat([compare_df, tmp], ignore_index=True)

    # Plot with gaps visible
    fig2 = px.line(compare_df, x="work_year", y="predicted", color="job_title",
                   markers=True, line_dash="job_title", title="Job comparison (predicted)", 
                   template="plotly_white")
    fig2.update_traces(connectgaps=False)
    fig2.update_layout(xaxis=dict(dtick=1), yaxis_title="Predicted Salary (USD)", hovermode="x unified")
    st.plotly_chart(fig2, use_container_width=True)

with tab4:
    st.header("Model Diagnostics & Explainability")

    st.subheader("Train / Test performance")
    st.write(f"Model: **{model_choice}**")
    st.write(f"MAE: ${metrics['mae']:,.0f}  |  RMSE: ${metrics['rmse']:,.0f}  |  R²: {metrics['r2']:.3f}")

    # Feature importance (only for tree models)
    reg = model_pipeline.named_steps["regressor"]
    try:
        if hasattr(reg, "feature_importances_"):
            # get preprocessor feature names
            ohe = None
            for t in model_pipeline.named_steps["preprocessor"].transformers_:
                if t[0] == "cat":
                    ohe = t[1]
            # build feature names
            cat_names = []
            if ohe is not None:
                # get categories from the fitted OneHotEncoder
                ohe_obj = model_pipeline.named_steps["preprocessor"].named_transformers_["cat"]
                cat_cols = categorical_cols
                for i, cats in enumerate(ohe_obj.categories_):
                    colname = cat_cols[i]
                    for c in cats:
                        cat_names.append(f"{colname}={c}")
            remainder = [c for c in X.columns if c not in categorical_cols]
            feature_names = cat_names + remainder
            importances = reg.feature_importances_
            fi_df = pd.DataFrame({"feature": feature_names, "importance": importances})
            fi_df = fi_df.sort_values("importance", ascending=False).head(20)
            st.subheader("Feature importances (top 20)")
            st.table(fi_df.reset_index(drop=True))
        else:
            st.info("Feature importances not available for linear models.")
    except Exception:
        st.info("Could not compute feature importances.")

    st.subheader("Residual plot (test set)")
    yhat = model_pipeline.predict(X_test)
    resid_df = pd.DataFrame({"y_true": y_test, "y_pred": yhat})
    fig_res = px.scatter(resid_df, x="y_pred", y=(resid_df["y_true"] - resid_df["y_pred"]),
                         title="Residuals vs Predicted", template="plotly_white")
    fig_res.update_layout(xaxis_title="Predicted", yaxis_title="Residual (True - Pred)")
    st.plotly_chart(fig_res, use_container_width=True)

    st.subheader("Notes & guidance")
    st.markdown("""
    - This dashboard trains on your CSV and uses categorical encoding.
    - For small datasets predictions are less reliable; look at MAE/RMSE for model trust.
    - Random Forest gives non-linear behavior and prediction intervals (approx).
    - To further improve accuracy: more historical data, external market indices, and feature engineering (location, currency adjustments, company revenue).
    """)
