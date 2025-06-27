import streamlit as st

st.set_page_config(page_title="WPI Steel Dashboard", layout="wide")

# Title block
st.markdown(
    """
    <div style="text-align: center;">
        <h1 style="margin-bottom: 0;">Artson Ltd, A Tata Enterprise</h1>
        <h3 style="color: gray; margin-top: 0;">WPI Steel Analysis with Forecasting</h3>
    </div>
    """,
    unsafe_allow_html=True
)

# Image URLs (from GitHub)
img_urls = {
    "Prediction": "https://raw.githubusercontent.com/Pushkindugam/Artson-Steel-WPI-Prediction-Seasonality/main/WPI_Prediction_screenshot.png",
    "Seasonality": "https://raw.githubusercontent.com/Pushkindugam/Artson-Steel-WPI-Prediction-Seasonality/main/WPI_Seasonality_screenshot.png",
    "Correlation": "https://raw.githubusercontent.com/Pushkindugam/Artson-Steel-WPI-Prediction-Seasonality/main/WPI_Correlation_Screenshot.png"
}

# Tabs
tabs = st.tabs(["🔮 Prediction", "📈 Seasonality", "🔗 Correlation"])

# Tab 1 – Prediction
with tabs[0]:
    st.subheader("🔮 Forecasting – WPI Steel (2022–2026)")
    st.image(img_urls["Prediction"], caption="WPI Forecast using STL & Trend Extension", use_container_width=True)

# Tab 2 – Seasonality
with tabs[1]:
    st.subheader("📈 Seasonal Patterns – STL Decomposition")
    st.image(img_urls["Seasonality"], caption="Seasonal Patterns in Stainless, Mild Flat & Long Steel", use_container_width=True)

# Tab 3 – Correlation
with tabs[2]:
    st.subheader("🔗 WPI Steel Correlation Heatmap")
    st.image(img_urls["Correlation"], caption="Correlation Matrix Across Steel Categories", use_container_width=True)

# Footer
st.markdown("---")
st.markdown("Developed by **Pushkin Dugam** | [GitHub Repo](https://github.com/Pushkindugam/Artson-Steel-WPI-Prediction-Seasonality)")
