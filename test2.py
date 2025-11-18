# -------------------------
# Forecast 2021–2035 (based on custom selection)
# -------------------------
future_years = np.arange(2021, 2036)

# Create future data by repeating the custom selection for each year
custom_future_data = pd.DataFrame({
    "work_year": future_years,
    "job_title": custom_job,
    "experience_level": custom_exp,
    "company_size": custom_size
})

# Predict salaries
future_predictions = model.predict(custom_future_data)

# Add small random variation to make lines visually distinct
np.random.seed(42)
variation = np.random.normal(0, 1000, size=future_predictions.shape)
future_predictions_vis = future_predictions + variation

forecast_df = pd.DataFrame({
    "Year": future_years,
    "Predicted Salary (USD)": future_predictions_vis
})

# -------------------------
# Forecast Graph
# -------------------------
fig = px.line(
    forecast_df,
    x="Year",
    y="Predicted Salary (USD)",
    markers=True,
    title=f"Salary Forecast for {custom_job} ({custom_exp}, {custom_size})",
    template="plotly_white"
)
fig.update_traces(line=dict(width=4), marker=dict(size=10))
fig.update_layout(
    yaxis_title="Salary (USD)",
    xaxis=dict(dtick=1),
    hovermode="x unified"
)

st.plotly_chart(fig, use_container_width=True)
