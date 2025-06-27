import streamlit as st

st.set_page_config(
    page_title="WPI Steel Forecast Dashboard",
    layout="wide"
)

# ---------- Sidebar ----------
with st.sidebar:
    st.markdown("## 📘 WPI Steel Guidelines")
    st.markdown("Analysis of steel price indices for procurement decisions.")
    st.markdown("---")
    st.markdown("### 🛠️ Built by Artson SCM Team – 2025")
    st.markdown("*by **Pushkin Dugam***")
    st.image(
        "https://raw.githubusercontent.com/Pushkindugam/Artson-Steel-WPI-Prediction-Seasonality/main/WPI_Correlation_Screenshot.png",
        use_container_width=True,
        caption="Artson Engineering Ltd."
    )

# ---------- Header ----------
st.markdown(
    """
    <div style="text-align: center; margin-top: -40px;">
        <h1>Artson Ltd, A Tata Enterprise</h1>
        <h3 style="color: gray;">WPI Steel Analysis with Forecasting</h3>
    </div>
    """,
    unsafe_allow_html=True
)

# ---------- Screenshot URLs ----------
img_urls = {
    "Prediction": "https://raw.githubusercontent.com/Pushkindugam/Artson-Steel-WPI-Prediction-Seasonality/main/WPI_Prediction_screenshot.png",
    "Seasonality": "https://raw.githubusercontent.com/Pushkindugam/Artson-Steel-WPI-Prediction-Seasonality/main/WPI_Seasonality_screenshot.png",
    "Correlation": "https://raw.githubusercontent.com/Pushkindugam/Artson-Steel-WPI-Prediction-Seasonality/main/WPI_Correlation_Screenshot.png"
}

# ---------- Tabs ----------
tabs = st.tabs(["🔮 Prediction", "📈 Seasonality", "🔗 Correlation"])

def centered_image(img_url, caption):
    left, center, right = st.columns([1, 2, 1])
    with center:
        st.image(img_url, caption=caption, use_container_width=True)

# Tab 1 – Prediction
with tabs[0]:
    st.subheader("🔮 Forecasting – WPI Steel (2022–2026)")
    centered_image(img_urls["Prediction"], "WPI Forecast using STL & Trend Extension")

# Tab 2 – Seasonality
with tabs[1]:
    st.subheader("📈 Seasonal Patterns – STL Decomposition")
    centered_image(img_urls["Seasonality"], "Seasonal Patterns in Stainless, Mild Flat & Long Steel")

# Tab 3 – Correlation
with tabs[2]:
    st.subheader("🔗 WPI Steel Correlation Heatmap")
    centered_image(img_urls["Correlation"], "Correlation Matrix Across Steel Categories")

# ---------- Footer ----------
st.markdown("---")
st.markdown(
    """
    <div style="text-align: center;">
        View the full project on <a href="https://github.com/Pushkindugam/Artson-Steel-WPI-Prediction-Seasonality" target="_blank">GitHub</a>
    </div>
    """,
    unsafe_allow_html=True
)
