import streamlit as st

# Page setup
st.set_page_config(page_title="Artson Ltd – WPI Steel Dashboard", layout="centered")

# Sidebar
with st.sidebar:
    st.image(
        "https://raw.githubusercontent.com/Pushkindugam/Artson-Steel-WPI-Prediction-Seasonality/main/artson_logo.png",
        use_container_width=True,
        caption="Artson Engineering Ltd."
    )

    st.markdown("## 📘 What is WPI?")
    st.markdown("""
    WPI stands for **Wholesale Price Index**.  
    It tracks the price changes of goods at the **wholesale level**,  
    useful for procurement and forecasting in industries like **steel**.
    """)

    st.markdown("---")
    st.markdown("### 🛠️ Built by Artson SCM Team – 2025")
    st.markdown("[GitHub Repository](https://github.com/Pushkindugam/Artson-Steel-WPI-Prediction-Seasonality)")

# Main Title
st.title("Artson Ltd – A Tata Enterprise")
st.markdown("### 📈 WPI Steel Analysis & Forecasting Dashboard")

# Image URLs
img_urls = {
    "Prediction": "https://raw.githubusercontent.com/Pushkindugam/Artson-Steel-WPI-Prediction-Seasonality/main/WPI_Prediction_screenshot.png",
    "Seasonality": "https://raw.githubusercontent.com/Pushkindugam/Artson-Steel-WPI-Prediction-Seasonality/main/WPI_Seasonality_screenshot.png",
    "Correlation": "https://raw.githubusercontent.com/Pushkindugam/Artson-Steel-WPI-Prediction-Seasonality/main/artson_logo.png"
}

# Tabs
tabs = st.tabs(["Prediction", "Seasonality", "Correlation"])

with tabs[0]:
    st.subheader("📈 Forecasting WPI for Steel (2022–2026)")
    st.image(img_urls["Prediction"], use_container_width=True)

with tabs[1]:
    st.subheader("📊 Seasonal Pattern of Steel Prices")
    st.image(img_urls["Seasonality"], use_container_width=True)

with tabs[2]:
    st.subheader("📌 Correlation Matrix of WPI Indicators")
    st.image(img_urls["Correlation"], use_container_width=True)
