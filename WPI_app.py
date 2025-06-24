import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
df = pd.read_excel("data/WPI_Master-dataset.xlsx", sheet_name="Sheet1", parse_dates=["Date"])

st.set_page_config(page_title="WPI Steel Forecasting Dashboard", layout="wide")
st.title("🛠️ Artson WPI Steel Analysis Dashboard")

st.markdown("""
Analyze **Wholesale Price Index (WPI)** trends, correlations, and forecasts for steel products and related indicators.
""")

# Tabs
tab1, tab2, tab3 = st.tabs(["📈 Seasonality", "📊 Correlation", "🔮 Forecasting"])

# --- Seasonality Tab ---
with tab1:
    st.header("📈 Seasonal Trends")

    steel_col = st.selectbox("Select WPI Category", 
                             ["WPI (stainless)", "WPI (mild flat)", "WPI (mild long)"])

    df_season = df[["Date", steel_col]].dropna()
    df_season.set_index("Date", inplace=True)
    df_monthly = df_season.resample("M").mean()
    df_monthly["Trend"] = df_monthly[steel_col].rolling(12, center=True).mean()
    df_monthly["Seasonality"] = df_monthly[steel_col] - df_monthly["Trend"]

    st.line_chart(df_monthly[[steel_col, "Trend"]])
    st.line_chart(df_monthly["Seasonality"])

# --- Correlation Tab ---
with tab2:
    st.header("📊 Correlation Heatmap")
    st.markdown("Understand how steel prices relate to other economic factors.")

    df_corr = df.drop(columns=["Date"]).corr()
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(df_corr, annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
    st.pyplot(fig)

# --- Forecasting Tab ---
with tab3:
    st.header("🔮 WPI Forecasting Results")
    st.markdown("See past trends and projected values for key steel categories.")

    plot_cols = ["WPI (stainless)", "WPI (mild flat)", "WPI (mild long)"]
    forecast_df = df[["Date"] + plot_cols].dropna()
    forecast_df.set_index("Date", inplace=True)

    st.line_chart(forecast_df)

    st.markdown("📘 Detailed model output is available in the Jupyter notebook `Final_WPI_Steel_Forecasting.ipynb`.")

st.markdown("---")
st.caption("Created by Pushkin Dugam | IIT Jodhpur © 2025")
