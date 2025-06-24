import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
import numpy as np

# Load data
df = pd.read_excel("WPI_Master-dataset.xlsx", sheet_name="Sheet1", parse_dates=["Date"])
df = df.rename(columns={"Date": "date"})
df.set_index("date", inplace=True)

st.set_page_config(page_title="Steel WPI Dashboard", layout="wide")
st.title("📊 Steel WPI Analysis Dashboard")

tab1, tab2, tab3 = st.tabs(["📈 Seasonal Analysis", "📊 Correlation", "🔮 Forecasting"])

# --- Tab 1: Seasonal ---
with tab1:
    st.header("📈 Seasonal Analysis")

    selected_series = st.selectbox("Select Series", 
                                   ["WPI (stainless)", "WPI (mild flat)", "WPI (mild long)"])
    
    data = df[selected_series].dropna().resample("M").mean()
    trend = data.rolling(12, center=True).mean()
    seasonality = data - trend

    st.subheader("WPI with Trend")
    fig1, ax1 = plt.subplots()
    ax1.plot(data, label='WPI')
    ax1.plot(trend, label='Trend')
    ax1.legend()
    st.pyplot(fig1)

    st.subheader("Seasonality Component")
    fig2, ax2 = plt.subplots()
    ax2.plot(seasonality, color='green')
    ax2.set_title("Seasonality = WPI - Trend")
    st.pyplot(fig2)

# --- Tab 2: Correlation ---
with tab2:
    st.header("📊 Correlation Matrix")

    numeric_df = df.select_dtypes(include=[np.number]).dropna()
    corr = numeric_df.corr()

    fig3, ax3 = plt.subplots(figsize=(12, 8))
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", ax=ax3)
    st.pyplot(fig3)

# --- Tab 3: Forecasting ---
with tab3:
    st.header("🔮 Forecasting Steel WPI using XGBoost")

    target_col = st.selectbox("Target WPI", 
                              ["WPI (stainless)", "WPI (mild flat)", "WPI (mild long)"])

    forecast_df = df[[target_col, "WPI (Coking Coal)", "WPI (Iron Ore)", "WPI (Crude Petroleum)", 
                      "Avg. USD/INR", "Inflation Rate", "Steel Production"]].dropna().copy()

    for lag in range(1, 4):
        forecast_df[f"{target_col}_lag{lag}"] = forecast_df[target_col].shift(lag)
    forecast_df.dropna(inplace=True)

    X = forecast_df.drop(columns=[target_col])
    y = forecast_df[target_col]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = xgb.XGBRegressor(n_estimators=100, random_state=42)
    model.fit(X_scaled, y)
    y_pred = model.predict(X_scaled)

    st.subheader("Actual vs Predicted WPI")
    fig4, ax4 = plt.subplots()
    ax4.plot(y.values, label="Actual")
    ax4.plot(y_pred, label="Predicted", color='red')
    ax4.set_title(f"{target_col} – Actual vs Predicted")
    ax4.legend()
    st.pyplot(fig4)

    st.subheader("🔍 Feature Importance")
    importance = model.feature_importances_
    imp_df = pd.DataFrame({"Feature": X.columns, "Importance": importance}).sort_values("Importance", ascending=False)

    fig5, ax5 = plt.subplots()
    sns.barplot(data=imp_df, x="Importance", y="Feature", ax=ax5)
    st.pyplot(fig5)

    st.info("Model: XGBoost with lag features and macroeconomic indicators")
