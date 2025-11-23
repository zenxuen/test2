import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
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

# Custom CSS for better styling
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
    </style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-header">💼 Cybersecurity Salary Prediction Dashboard</h1>', unsafe_allow_html=True)
st.markdown("*Predict salaries with advanced ML models and interactive visualizations*")

# ---------------------------------------------------------
# Load Dataset
# ---------------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("salaries_cyber_clean.csv")
    return df

df = load_data()

# ---------------------------------------------------------
# Sidebar - Model Selection & Info
# ---------------------------------------------------------
st.sidebar.header("🎛️ Model Configuration")
model_type = st.sidebar.selectbox(
    "Select Model",
    ["Linear Regression", "Random Forest", "Gradient Boosting"],
    help="Choose the prediction algorithm"
)

st.sidebar.markdown("---")
st.sidebar.subheader("📊 Dataset Info")
st.sidebar.info(f"""
- **Total Records:** {len(df):,}
- **Date Range:** {df['work_year'].min()} - {df['work_year'].max()}
- **Job Titles:** {df['job_title'].nunique()}
- **Avg Salary:** ${df['salary_in_usd'].mean():,.0f}
""")

# ---------------------------------------------------------
# Train Models with Performance Metrics
# ---------------------------------------------------------
@st.cache_resource
def train_models(data):
    X = data[["work_year", "job_title", "experience_level", "company_size"]]
    y = data["salary_in_usd"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    categorical_cols = ["job_title", "experience_level", "company_size"]
    preprocessor = ColumnTransformer(
        transformers=[("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)],
        remainder="passthrough"
    )
    
    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42, max_depth=5)
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
        
        trained_models[name] = pipeline
        metrics[name] = {
            "R² Score": r2_score(y_test, y_pred),
            "MAE": mean_absolute_error(y_test, y_pred),
            "RMSE": np.sqrt(mean_squared_error(y_test, y_pred))
        }
    
    return trained_models, metrics

models, performance_metrics = train_models(df)
selected_model = models[model_type]

# ---------------------------------------------------------
# Display Model Performance
# ---------------------------------------------------------
with st.sidebar.expander("📈 Model Performance"):
    perf = performance_metrics[model_type]
    st.metric("R² Score", f"{perf['R² Score']:.3f}")
    st.metric("MAE", f"${perf['MAE']:,.0f}")
    st.metric("RMSE", f"${perf['RMSE']:,.0f}")

# ---------------------------------------------------------
# Main Content - Tabs
# ---------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs(["🔮 Salary Forecast", "📊 Data Analysis", "💰 Salary Calculator", "🗺️ Insights"])

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
    
    # Forecast 2021–2030
    future_years = np.arange(2021, 2031)
    custom_future_data = pd.DataFrame({
        "work_year": future_years,
        "job_title": custom_job,
        "experience_level": custom_exp,
        "company_size": custom_size
    })
    
    future_predictions = selected_model.predict(custom_future_data)
    
    forecast_df = pd.DataFrame({
        "Year": future_years,
        "Predicted Salary (USD)": future_predictions
    })
    
    # Enhanced Graph
    st.markdown("---")
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.metric("📅 Starting Salary (2021)", f"${forecast_df.iloc[0]['Predicted Salary (USD)']:,.0f}")
    with col_b:
        st.metric("📅 Current Salary (2025)", f"${forecast_df.iloc[4]['Predicted Salary (USD)']:,.0f}")
    with col_c:
        growth = ((forecast_df.iloc[-1]['Predicted Salary (USD)'] - forecast_df.iloc[0]['Predicted Salary (USD)']) / forecast_df.iloc[0]['Predicted Salary (USD)']) * 100
        st.metric("📈 10-Year Growth", f"{growth:.1f}%")
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=forecast_df["Year"],
        y=forecast_df["Predicted Salary (USD)"],
        mode='lines+markers',
        name='Predicted Salary',
        line=dict(color='#667eea', width=4),
        marker=dict(size=12, color='#764ba2', line=dict(width=2, color='white')),
        fill='tozeroy',
        fillcolor='rgba(102, 126, 234, 0.1)'
    ))
    
    fig.update_layout(
        title=f"Salary Forecast: {custom_job} ({custom_exp}, {custom_size})",
        xaxis_title="Year",
        yaxis_title="Salary (USD)",
        template="plotly_white",
        hovermode="x unified",
        height=500,
        xaxis=dict(dtick=1)
    )
    
    st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------
# TAB 2: Data Analysis
# ---------------------------------------------------------
with tab2:
    st.subheader("📊 Salary Distribution Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Salary by Experience Level
        exp_avg = df.groupby("experience_level")["salary_in_usd"].mean().reset_index()
        fig_exp = px.bar(
            exp_avg,
            x="experience_level",
            y="salary_in_usd",
            title="Average Salary by Experience Level",
            color="salary_in_usd",
            color_continuous_scale="Viridis"
        )
        fig_exp.update_layout(showlegend=False, xaxis_title="Experience Level", yaxis_title="Avg Salary (USD)")
        st.plotly_chart(fig_exp, use_container_width=True)
    
    with col2:
        # Salary by Company Size
        size_avg = df.groupby("company_size")["salary_in_usd"].mean().reset_index()
        fig_size = px.bar(
            size_avg,
            x="company_size",
            y="salary_in_usd",
            title="Average Salary by Company Size",
            color="salary_in_usd",
            color_continuous_scale="Plasma"
        )
        fig_size.update_layout(showlegend=False, xaxis_title="Company Size", yaxis_title="Avg Salary (USD)")
        st.plotly_chart(fig_size, use_container_width=True)
    
    # Top Paying Jobs
    st.markdown("---")
    st.subheader("💎 Top 10 Highest Paying Jobs")
    top_jobs = df.groupby("job_title")["salary_in_usd"].mean().sort_values(ascending=False).head(10).reset_index()
    
    fig_top = px.bar(
        top_jobs,
        x="salary_in_usd",
        y="job_title",
        orientation='h',
        title="Top 10 Highest Paying Job Titles",
        color="salary_in_usd",
        color_continuous_scale="Turbo"
    )
    fig_top.update_layout(yaxis={'categoryorder':'total ascending'}, showlegend=False, xaxis_title="Avg Salary (USD)", yaxis_title="")
    st.plotly_chart(fig_top, use_container_width=True)

# ---------------------------------------------------------
# TAB 3: Salary Calculator
# ---------------------------------------------------------
with tab3:
    st.subheader("🔮 Predict Salary for a Specific Year")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        single_year = st.slider("📅 Select Year", 2020, 2035, 2025)
    
    single_input = pd.DataFrame({
        "work_year": [single_year],
        "job_title": [custom_job],
        "experience_level": [custom_exp],
        "company_size": [custom_size]
    })
    
    single_prediction = selected_model.predict(single_input)[0]
    
    # Display prediction with styling
    st.markdown("---")
    col_x, col_y, col_z = st.columns([1, 2, 1])
    
    with col_y:
        st.markdown(f"""
        <div class="metric-card">
            <h3>Predicted Salary for {single_year}</h3>
            <h1 style="font-size: 3rem; margin: 1rem 0;">${single_prediction:,.0f}</h1>
            <p>{custom_job} | {custom_exp} | {custom_size}</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Comparison with market
    st.markdown("---")
    st.subheader("📊 Market Comparison")
    
    market_avg = df[df["job_title"] == custom_job]["salary_in_usd"].mean()
    diff = single_prediction - market_avg
    diff_pct = (diff / market_avg) * 100
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Your Prediction", f"${single_prediction:,.0f}")
    with col2:
        st.metric("Market Average", f"${market_avg:,.0f}")
    with col3:
        st.metric("Difference", f"${diff:,.0f}", f"{diff_pct:+.1f}%")

# ---------------------------------------------------------
# TAB 4: Insights
# ---------------------------------------------------------
with tab4:
    st.subheader("🗺️ Key Insights & Trends")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📈 Yearly Salary Trends")
        yearly_avg = df.groupby("work_year")["salary_in_usd"].mean().reset_index()
        fig_yearly = px.line(
            yearly_avg,
            x="work_year",
            y="salary_in_usd",
            markers=True,
            title="Average Salary Over Years"
        )
        fig_yearly.update_traces(line=dict(color='#667eea', width=3), marker=dict(size=10))
        st.plotly_chart(fig_yearly, use_container_width=True)
    
    with col2:
        st.markdown("### 🎯 Salary Distribution")
        fig_dist = px.histogram(
            df,
            x="salary_in_usd",
            nbins=50,
            title="Overall Salary Distribution",
            color_discrete_sequence=['#764ba2']
        )
        fig_dist.update_layout(xaxis_title="Salary (USD)", yaxis_title="Count")
        st.plotly_chart(fig_dist, use_container_width=True)
    
    # Key Statistics
    st.markdown("---")
    st.markdown("### 📊 Key Statistics")
    
    stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)
    
    with stat_col1:
        st.metric("📊 Median Salary", f"${df['salary_in_usd'].median():,.0f}")
    with stat_col2:
        st.metric("💰 Max Salary", f"${df['salary_in_usd'].max():,.0f}")
    with stat_col3:
        st.metric("📉 Min Salary", f"${df['salary_in_usd'].min():,.0f}")
    with stat_col4:
        std_dev = df['salary_in_usd'].std()
        st.metric("📏 Std Deviation", f"${std_dev:,.0f}")

# ---------------------------------------------------------
# Footer
# ---------------------------------------------------------
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #666; padding: 2rem;'>
        <p>Built with Streamlit • Powered by Machine Learning</p>
        <p style='font-size: 0.8rem;'>Model: {} | R² Score: {:.3f}</p>
    </div>
""".format(model_type, performance_metrics[model_type]["R² Score"]), unsafe_allow_html=True)
