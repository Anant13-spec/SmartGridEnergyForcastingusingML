
import streamlit as st
import pandas as pd
import plotly.graph_objects as go

st.set_page_config(
    page_title="Smart Grid Forecasting",
    page_icon="⚡",
    layout="wide"
)

st.title("⚡ Smart Grid Energy Load Forecasting")
st.caption("Prototype | Historical electricity demand and ML predictions")

# Load historical data and saved test predictions
df = pd.read_csv("df_final.csv")
df["DATE"] = pd.to_datetime(df["DATE"], errors="coerce")
df = df.dropna(subset=["DATE", "POWER_DEMAND"]).sort_values("DATE")

results = pd.read_csv("test_predictions.csv")
results["DATE"] = pd.to_datetime(results["DATE"], errors="coerce")

# Summary metrics
c1, c2, c3 = st.columns(3)

c1.metric(
    "Average Demand",
    f"{df['POWER_DEMAND'].mean():,.0f}"
)

c2.metric(
    "Peak Demand",
    f"{df['POWER_DEMAND'].max():,.0f}"
)

c3.metric(
    "Historical Records",
    f"{len(df):,}"
)

st.subheader("Historical Electricity Demand")

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=df["DATE"],
    y=df["POWER_DEMAND"],
    mode="lines",
    name="Actual Demand"
))
fig.update_layout(
    xaxis_title="Date",
    yaxis_title="Power Demand",
    hovermode="x unified"
)
st.plotly_chart(fig, use_container_width=True)

st.subheader("Actual vs Predicted Demand")

comparison = go.Figure()
comparison.add_trace(go.Scatter(
    x=results["DATE"],
    y=results["ACTUAL_DEMAND"],
    mode="lines",
    name="Actual"
))
comparison.add_trace(go.Scatter(
    x=results["DATE"],
    y=results["PREDICTED_DEMAND"],
    mode="lines",
    name="Predicted"
))
comparison.update_layout(
    xaxis_title="Date",
    yaxis_title="Power Demand",
    hovermode="x unified"
)
st.plotly_chart(comparison, use_container_width=True)

mae = (
    results["ACTUAL_DEMAND"] - results["PREDICTED_DEMAND"]
).abs().mean()

rmse = (
    (results["ACTUAL_DEMAND"] - results["PREDICTED_DEMAND"]) ** 2
).mean() ** 0.5

m1, m2 = st.columns(2)
m1.metric("Test MAE", f"{mae:,.2f}")
m2.metric("Test RMSE", f"{rmse:,.2f}")

st.subheader("Prediction Samples")
st.dataframe(results.head(20), use_container_width=True)
