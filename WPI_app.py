import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load Excel file from current directory (no "data/" folder)
try:
    df = pd.read_excel("WPI_Master-dataset.xlsx", sheet_name="Sheet1", parse_dates=["Date"])
except FileNotFoundError:
    st.error("❌ Could not find 'WPI_Master-dataset.xlsx' in the repo. Please upload it.")
    st.stop()

st.set_page_config(page_title="WPI Steel Forecasting", layout="wide")
st.title("🛠️ Steel WPI Forecasting & Seasonality Dashboard")

st.markdown("""
Explore seasonal trends, correlations, and predictions of **Wholesale Price Index (WPI)** for steel (stainless, mild flat, mild long) and related indicators.
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

    st.subheader("📊 WPI with Trend Line")
    st.line_chart(df_monthly[[steel_col, "Trend"]])

    st.subheader("📊 Seasonality Component")
    st.line_chart(df_monthly["Seasonality"])

# --- Correlation Tab ---
with tab2:
    st.header("📊 Correlation Heatmap")
    st.markdown("Visualize correlations between steel WPI and other economic indicators.")

    df_corr = df.drop(columns=["Date"]).corr()
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(df_corr, annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
    st.pyplot(fig)

# --- Forecasting Tab ---
with tab3:
    st.header("🔮 Forecasting Output")
    st.markdown("View past trends and model-ready inputs. Predictions can be added later.")

    plot_cols = ["WPI (stainless)", "WPI (mild flat)", "WPI (mild long)"]
    forecast_df = df[["Date"] + plot_cols].dropna()
    forecast_df.set_index("Date", inplace=True)

    st.line_chart(forecast_df)

    st.info("📘 Advanced forecasting logic is available in your notebook: `Final_WPI_Steel_Forecasting.ipynb`")

# Footer
st.markdown("---")
st.caption("Developed by Pushkin Dugam | IIT Jodhpur © 2025")
