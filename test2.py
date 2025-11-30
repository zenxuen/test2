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
    .warning-box {
        background-color: #fff3cd;
        border-left: 4px solid #ffc107;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-header">💼 Cybersecurity Salary Prediction Dashboard</h1>', unsafe_allow_html=True)
st.markdown("*Predict salaries with advanced ML models - Target: Salary | Predictors: Year, Job, Experience, Company Size*")

# ---------------------------------------------------------
# Load Dataset
# ---------------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("salaries_cyber_clean.csv")
    return df

df = load_data()

# ---------------------------------------------------------
# Feature Engineering: Add salary growth trends
# ---------------------------------------------------------
@st.cache_data
def add_features(data):
    df_enhanced = data.copy()
    
    # Calculate average salary growth rate per job/experience combination
    df_enhanced['years_since_2020'] = df_enhanced['work_year'] - 2020
    
    # Add interaction features
    df_enhanced['exp_size_combo'] = df_enhanced['experience_level'] + '_' + df_enhanced['company_size']
    
    return df_enhanced

df_enhanced = add_features(df)

# ---------------------------------------------------------
# Sidebar - Model Selection & Info
# ---------------------------------------------------------
st.sidebar.header("🎛️ Model Configuration")
model_type = st.sidebar.selectbox(
    "Select Model",
    ["Linear Regression", "Random Forest", "Gradient Boosting"],
    help="Choose the prediction algorithm"
)

# Add prediction method selector
prediction_method = st.sidebar.radio(
    "Prediction Method for Future Years",
    ["ML Model Only", "ML Model + Growth Rate", "Conservative (Actual Data Priority)"],
    help="How to handle predictions for years beyond training data"
)

st.sidebar.markdown("---")
st.sidebar.subheader("📊 Dataset Info")
st.sidebar.info(f"""
- **Total Records:** {len(df):,}
- **Date Range:** {df['work_year'].min()} - {df['work_year'].max()}
- **Job Titles:** {df['job_title'].nunique()}
- **Avg Salary:** ${df['salary_in_usd'].mean():,.0f}
- **Salary Range:** ${df['salary_in_usd'].min():,.0f} - ${df['salary_in_usd'].max():,.0f}
""")

st.sidebar.success(f"**Active Model:** {model_type}")

# ---------------------------------------------------------
# Train Models with Enhanced Features
# ---------------------------------------------------------
@st.cache_resource
def train_models(data):
    # Use enhanced features
    X = data[["work_year", "job_title", "experience_level", "company_size", "years_since_2020"]]
    y = data["salary_in_usd"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    categorical_cols = ["job_title", "experience_level", "company_size"]
    numeric_cols = ["work_year", "years_since_2020"]
    
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
        ],
        remainder="passthrough"
    )
    
    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=200, random_state=42, max_depth=15, min_samples_split=5),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=200, random_state=42, max_depth=7, learning_rate=0.1)
    }
    
    trained_models = {}
    metrics = {}
    
    for name, estimator in models.items():
        pipeline = Pipeline([
            ("prep", preprocessor),
            ("model", estimator)
        ])
        pipeline.fit(X_train, y_train)
        
        y_pred = pipeline.predict(X_test)
        
        # Cross-validation score
        cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='r2')
        
        trained_models[name] = pipeline
        metrics[name] = {
            "R² Score": r2_score(y_test, y_pred),
            "MAE": mean_absolute_error(y_test, y_pred),
            "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
            "CV R² Mean": cv_scores.mean(),
            "CV R² Std": cv_scores.std()
        }
    
    return trained_models, metrics

models, performance_metrics = train_models(df_enhanced)
selected_model = models[model_type]

# Calculate historical growth rate
@st.cache_data
def calculate_growth_rates(data):
    growth_rates = {}
    
    for job in data['job_title'].unique():
        job_data = data[data['job_title'] == job].groupby('work_year')['salary_in_usd'].mean()
        
        if len(job_data) >= 2:
            # Calculate year-over-year growth
            years = job_data.index.values
            salaries = job_data.values
            
            if len(years) > 1:
                # Simple linear growth rate
                growth = (salaries[-1] - salaries[0]) / salaries[0] / (years[-1] - years[0])
                growth_rates[job] = max(0.02, min(growth, 0.15))  # Cap between 2% and 15%
            else:
                growth_rates[job] = 0.05  # Default 5%
        else:
            growth_rates[job] = 0.05
    
    return growth_rates

growth_rates = calculate_growth_rates(df)

# ---------------------------------------------------------
# Display Model Performance
# ---------------------------------------------------------
with st.sidebar.expander("📈 Model Performance", expanded=True):
    perf = performance_metrics[model_type]
    st.metric("R² Score", f"{perf['R² Score']:.3f}")
    st.metric("MAE", f"${perf['MAE']:,.0f}")
    st.metric("RMSE", f"${perf['RMSE']:,.0f}")
    st.metric("CV R² (Mean±Std)", f"{perf['CV R² Mean']:.3f}±{perf['CV R² Std']:.3f}")

# ---------------------------------------------------------
# Prediction Function with Multiple Methods
# ---------------------------------------------------------
def predict_salary(year, job, exp, size, method="ML Model Only"):
    years_since = year - 2020
    
    pred_input = pd.DataFrame({
        "work_year": [year],
        "job_title": [job],
        "experience_level": [exp],
        "company_size": [size],
        "years_since_2020": [years_since]
    })
    
    base_prediction = selected_model.predict(pred_input)[0]
    
    if method == "ML Model Only":
        return base_prediction
    
    elif method == "ML Model + Growth Rate":
        # Get base salary from 2022 (last training year)
        base_year_input = pd.DataFrame({
            "work_year": [2022],
            "job_title": [job],
            "experience_level": [exp],
            "company_size": [size],
            "years_since_2020": [2]
        })
        base_salary = selected_model.predict(base_year_input)[0]
        
        # Apply growth rate
        years_ahead = year - 2022
        growth_rate = growth_rates.get(job, 0.05)
        adjusted_salary = base_salary * ((1 + growth_rate) ** years_ahead)
        
        # Blend ML prediction with growth-adjusted prediction
        return 0.6 * base_prediction + 0.4 * adjusted_salary
    
    elif method == "Conservative (Actual Data Priority)":
        # For years in training data, try to get actual
        if year <= 2022:
            actual = df[(df['work_year'] == year) & 
                       (df['job_title'] == job) & 
                       (df['experience_level'] == exp) & 
                       (df['company_size'] == size)]
            
            if len(actual) > 0:
                return actual['salary_in_usd'].mean()
        
        # Otherwise use ML + growth
        base_year_input = pd.DataFrame({
            "work_year": [2022],
            "job_title": [job],
            "experience_level": [exp],
            "company_size": [size],
            "years_since_2020": [2]
        })
        base_salary = selected_model.predict(base_year_input)[0]
        years_ahead = year - 2022
        growth_rate = growth_rates.get(job, 0.05)
        return base_salary * ((1 + growth_rate) ** years_ahead)
    
    return base_prediction

# ---------------------------------------------------------
# Main Content - Tabs
# ---------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs(["🔮 Salary Forecast", "📊 Data Analysis", "💰 Salary Calculator", "🗺️ Model Insights"])

# ---------------------------------------------------------
# TAB 1: Salary Forecast
# ---------------------------------------------------------
with tab1:
    st.subheader("⚙️ Customize Your Profile")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        custom_job = st.selectbox("👔 Job Title", sorted(df["job_title"].unique()))
    with col2:
        custom_exp = st.selectbox("📈 Experience Level", sorted(df["experience_level"].unique()))
    with col3:
        custom_size = st.selectbox("🏢 Company Size", sorted(df["company_size"].unique()))
    
    # Show growth rate for selected job
    job_growth = growth_rates.get(custom_job, 0.05) * 100
    st.info(f"📊 Historical growth rate for **{custom_job}**: ~{job_growth:.1f}% per year")
    
    # Forecast 2020–2030
    all_forecast_years = np.arange(2020, 2031)
    forecast_data = []
    
    for year in all_forecast_years:
        salary = predict_salary(year, custom_job, custom_exp, custom_size, prediction_method)
        
        # Determine data source
        if year <= 2022:
            actual = df[(df['work_year'] == year) & 
                       (df['job_title'] == custom_job) & 
                       (df['experience_level'] == custom_exp) & 
                       (df['company_size'] == custom_size)]
            data_source = "Actual" if len(actual) > 0 else "Predicted"
        else:
            data_source = "Predicted"
        
        forecast_data.append({
            "Year": year,
            "Salary (USD)": salary,
            "Source": data_source
        })
    
    forecast_df = pd.DataFrame(forecast_data)
    
    # Calculate metrics
    st.markdown("---")
    
    start_salary = forecast_df.iloc[0]['Salary (USD)']
    end_salary = forecast_df.iloc[-1]['Salary (USD)']
    current_2025 = forecast_df[forecast_df["Year"] == 2025].iloc[0]['Salary (USD)']
    
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.metric("📅 Starting Salary (2020)", f"${start_salary:,.0f}")
    with col_b:
        st.metric("📅 Current Salary (2025)", f"${current_2025:,.0f}")
    with col_c:
        growth = ((end_salary - start_salary) / start_salary) * 100
        st.metric("📈 10-Year Growth", f"{growth:.1f}%")
    
    # Enhanced Visualization
    fig = go.Figure()
    
    # Actual data
    actual_df = forecast_df[forecast_df["Source"] == "Actual"]
    if len(actual_df) > 0:
        fig.add_trace(go.Scatter(
            x=actual_df["Year"],
            y=actual_df["Salary (USD)"],
            mode='lines+markers',
            name='Actual Data',
            line=dict(color='#10b981', width=4),
            marker=dict(size=12, color='#10b981', line=dict(width=2, color='white')),
            fill='tozeroy',
            fillcolor='rgba(16, 185, 129, 0.1)',
            hovertemplate='<b>Year:</b> %{x}<br><b>Salary:</b> $%{y:,.0f}<br><b>Source:</b> Actual<extra></extra>'
        ))
    
    # Predicted data
    predicted_df = forecast_df[forecast_df["Source"] == "Predicted"]
    if len(predicted_df) > 0:
        fig.add_trace(go.Scatter(
            x=predicted_df["Year"],
            y=predicted_df["Salary (USD)"],
            mode='lines+markers',
            name='Predicted',
            line=dict(color='#667eea', width=4, dash='dash'),
            marker=dict(size=12, color='#764ba2', line=dict(width=2, color='white')),
            fill='tozeroy',
            fillcolor='rgba(102, 126, 234, 0.1)',
            hovertemplate='<b>Year:</b> %{x}<br><b>Salary:</b> $%{y:,.0f}<br><b>Source:</b> Predicted<extra></extra>'
        ))
    
    fig.update_layout(
        title=f"Salary Forecast (2020-2030): {custom_job} ({custom_exp}, {custom_size})",
        xaxis_title="Year",
        yaxis_title="Salary (USD)",
        template="plotly_white",
        hovermode="x unified",
        height=500,
        xaxis=dict(dtick=1),
        showlegend=True,
        font=dict(size=12)
    )
    
    # Add vertical line at 2022
    fig.add_vline(x=2022.5, line_dash="dot", line_color="gray", 
                  annotation_text="Training Data | Predictions", 
                  annotation_position="top")
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Show forecast table
    with st.expander("📋 View Detailed Forecast Table"):
        display_df = forecast_df.copy()
        display_df['Salary (USD)'] = display_df['Salary (USD)'].apply(lambda x: f"${x:,.0f}")
        st.dataframe(display_df, use_container_width=True, hide_index=True)

# ---------------------------------------------------------
# TAB 2: Data Analysis
# ---------------------------------------------------------
with tab2:
    st.subheader("📊 Salary Distribution Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        exp_avg = df.groupby("experience_level")["salary_in_usd"].mean().sort_values(ascending=False).reset_index()
        fig_exp = px.bar(
            exp_avg,
            x="experience_level",
            y="salary_in_usd",
            title="Average Salary by Experience Level",
            color="salary_in_usd",
            color_continuous_scale="Viridis",
            text="salary_in_usd"
        )
        fig_exp.update_traces(texttemplate='$%{text:,.0f}', textposition='outside')
        fig_exp.update_layout(showlegend=False, xaxis_title="Experience Level", yaxis_title="Avg Salary (USD)")
        st.plotly_chart(fig_exp, use_container_width=True)
    
    with col2:
        size_avg = df.groupby("company_size")["salary_in_usd"].mean().sort_values(ascending=False).reset_index()
        fig_size = px.bar(
            size_avg,
            x="company_size",
            y="salary_in_usd",
            title="Average Salary by Company Size",
            color="salary_in_usd",
            color_continuous_scale="Plasma",
            text="salary_in_usd"
        )
        fig_size.update_traces(texttemplate='$%{text:,.0f}', textposition='outside')
        fig_size.update_layout(showlegend=False, xaxis_title="Company Size", yaxis_title="Avg Salary (USD)")
        st.plotly_chart(fig_size, use_container_width=True)
    
    # Top Paying Jobs
    st.markdown("---")
    st.subheader("💎 Top 10 Highest Paying Jobs (2022 Data)")
    
    top_jobs_2022 = df[df['work_year'] == 2022].groupby('job_title')['salary_in_usd'].mean().sort_values(ascending=False).head(10).reset_index()
    
    fig_top = px.bar(
        top_jobs_2022,
        x="salary_in_usd",
        y="job_title",
        orientation='h',
        title="Top 10 Highest Paying Jobs (2022)",
        color="salary_in_usd",
        color_continuous_scale="Turbo",
        text="salary_in_usd"
    )
    fig_top.update_traces(texttemplate='$%{text:,.0f}', textposition='outside')
    fig_top.update_layout(yaxis={'categoryorder':'total ascending'}, showlegend=False)
    st.plotly_chart(fig_top, use_container_width=True)

# ---------------------------------------------------------
# TAB 3: Salary Calculator
# ---------------------------------------------------------
with tab3:
    st.subheader("💰 Salary Calculator")
    
    calc_year = st.slider("📅 Select Year", 2020, 2035, 2025)
    
    salary_value = predict_salary(calc_year, custom_job, custom_exp, custom_size, prediction_method)
    
    # Display prediction
    st.markdown("---")
    
    if calc_year <= 2022:
        st.success("✅ Based on actual training data")
    else:
        st.info(f"ℹ️ Predicted using {prediction_method}")
    
    col_x, col_y, col_z = st.columns([1, 2, 1])
    
    with col_y:
        st.markdown(f"""
        <div class="metric-card">
            <h3>Predicted Salary for {calc_year}</h3>
            <h1 style="font-size: 3rem; margin: 1rem 0;">${salary_value:,.0f}</h1>
            <p>{custom_job} | {custom_exp} | {custom_size}</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Market comparison
    st.markdown("---")
    st.subheader("📊 Market Comparison")
    
    market_avg = df[df["job_title"] == custom_job]["salary_in_usd"].mean()
    diff = salary_value - market_avg
    diff_pct = (diff / market_avg) * 100 if market_avg > 0 else 0
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Your Predicted Salary", f"${salary_value:,.0f}")
    with col2:
        st.metric("Overall Market Avg", f"${market_avg:,.0f}")
    with col3:
        st.metric("Difference", f"${diff:,.0f}", f"{diff_pct:+.1f}%")

# ---------------------------------------------------------
# TAB 4: Model Insights
# ---------------------------------------------------------
with tab4:
    st.subheader("🗺️ Model Performance & Insights")
    
    # Compare all models
    st.markdown("### 📊 Model Comparison")
    
    comparison_data = []
    for model_name, metrics in performance_metrics.items():
        comparison_data.append({
            "Model": model_name,
            "R² Score": metrics["R² Score"],
            "MAE": metrics["MAE"],
            "RMSE": metrics["RMSE"]
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig_r2 = px.bar(
            comparison_df,
            x="Model",
            y="R² Score",
            title="R² Score Comparison (Higher is Better)",
            color="R² Score",
            color_continuous_scale="Greens"
        )
        st.plotly_chart(fig_r2, use_container_width=True)
    
    with col2:
        fig_mae = px.bar(
            comparison_df,
            x="Model",
            y="MAE",
            title="MAE Comparison (Lower is Better)",
            color="MAE",
            color_continuous_scale="Reds_r"
        )
        st.plotly_chart(fig_mae, use_container_width=True)
    
    # Prediction uncertainty warning
    st.markdown("---")
    st.markdown("### ⚠️ Understanding Predictions")
    
    st.markdown("""
    <div class="warning-box">
        <h4>🎯 Model Limitations:</h4>
        <ul>
            <li><b>Training Data:</b> Models trained on 2020-2022 data only</li>
            <li><b>Future Predictions:</b> Predictions beyond 2022 are extrapolations</li>
            <li><b>Uncertainty:</b> Predictions get less reliable further into the future</li>
            <li><b>Market Changes:</b> Cannot predict major market disruptions or changes</li>
        </ul>
        <p><b>Recommendation:</b> Use predictions as guidelines, not absolute values. Consider confidence intervals and market trends.</p>
    </div>
    """, unsafe_allow_html=True)

# ---------------------------------------------------------
# Footer
# ---------------------------------------------------------
st.markdown("---")
st.markdown(f"""
    <div style='text-align: center; color: #666; padding: 2rem;'>
        <p>Built with Streamlit • Powered by Machine Learning</p>
        <p style='font-size: 0.8rem;'>Model: {model_type} | R² Score: {performance_metrics[model_type]["R² Score"]:.3f} | Method: {prediction_method}</p>
    </div>
""", unsafe_allow_html=True)
