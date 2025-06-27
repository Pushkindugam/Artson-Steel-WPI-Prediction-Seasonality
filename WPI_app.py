import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import STL
from sklearn.metrics import mean_squared_error
import numpy as np
import requests

# ---------------- Sidebar ---------------- #
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
    forecasting steel useful for procurement in **EPC industries**.
    """)

    st.markdown("---")
    st.markdown("### 🛠️ Built by Artson SCM Team – 2025")
    st.markdown("*by **Pushkin Dugam***")
    st.markdown("[🔗 GitHub Repository](https://github.com/Pushkindugam/Artson-Steel-WPI-Prediction-Seasonality)")

# ---------------- Page Setup ---------------- #
st.set_page_config(page_title="WPI Steel Dashboard", layout="centered")
st.markdown("<h1 style='text-align: center;'>Artson Ltd, A Tata Enterprise</h1>", unsafe_allow_html=True)
st.markdown("<h2 style='text-align: center;'>WPI Steel Analysis with Forecasting</h2>", unsafe_allow_html=True)

# ---------------- Load Data ---------------- #
@st.cache_data
def load_excel_from_github(url):
    response = requests.get(url)
    return pd.read_excel(response.content)

master_url = "https://github.com/Pushkindugam/Artson-Steel-WPI-Prediction-Seasonality/raw/main/WPI_Master-dataset.xlsx"
forecast_url = "https://github.com/Pushkindugam/Artson-Steel-WPI-Prediction-Seasonality/raw/main/WPI_Steel_jan2022_to_may2026.xlsx"

# ---------------- Tabs ---------------- #
tabs = st.tabs(["📈 Prediction", "📆 Seasonality", "📊 Correlation", "📂 Dataset"])

# ---- Tab 1: Prediction ---- #
with tabs[0]:
    st.header("📈 Forecasting Steel WPI (2022–2026)")
    st.markdown("""
    Steel price forecasting helps predict future trends in WPI, enabling better planning of procurement budgets and contracts.

    These graphs show projected values of WPI for stainless, mild flat, and mild long steel categories from **May 2025 to May 2026**, along with trend analysis of earlier dates.
    """)

    st.image("https://raw.githubusercontent.com/Pushkindugam/Artson-Steel-WPI-Prediction-Seasonality/main/WPI_Prediction_screenshot.png", use_container_width=True)

    df_forecast = load_excel_from_github(forecast_url)
    df_forecast['Date'] = pd.to_datetime(df_forecast['Date'], format='%b-%y')
    df_forecast.set_index('Date', inplace=True)

    fig, ax = plt.subplots(figsize=(10, 4))
    for col in ['WPI (stainless)', 'WPI (mild flat)', 'WPI (mild long)']:
        ax.plot(df_forecast.index, df_forecast[col], label=col)
    ax.set_title("Forecasted WPI Trends")
    ax.set_ylabel("WPI Index")
    ax.grid(True)
    ax.legend()
    st.pyplot(fig)

# ---- Tab 2: Seasonality ---- #
with tabs[1]:
    st.header("📆 Seasonal Patterns of Steel Prices")
    st.markdown("""
    Seasonality analysis reveals repeating patterns in steel prices across months or years.

    Understanding seasonal trends helps procurement teams schedule bulk purchases in **low-price months**, avoiding cost spikes during **peak demand seasons**.
    """)

    st.image("https://raw.githubusercontent.com/Pushkindugam/Artson-Steel-WPI-Prediction-Seasonality/main/WPI_Seasonality_screenshot.png", use_container_width=True)

    steel_types = {
        'WPI (stainless)': 'Stainless',
        'WPI (mild flat)': 'Mild Flat',
        'WPI (mild long)': 'Mild Long'
    }

    for col, label in steel_types.items():
        st.subheader(f"🔁 STL Decomposition: {label}")
        stl = STL(df_forecast[col], period=12).fit()
        seasonal = stl.seasonal

        fig_s, ax = plt.subplots(figsize=(10, 3))
        ax.plot(seasonal, label='Seasonal Component')
        ax.set_title(f"{label} Steel - Seasonal Pattern")
        ax.legend()
        st.pyplot(fig_s)

# ---- Tab 3: Correlation ---- #
with tabs[2]:
    st.header("📊 Correlation of WPI Categories")
    st.markdown("""
    Correlation analysis shows how steel WPI is influenced by other economic and industrial indicators.

    This helps identify key **cost drivers** and supports **data-driven procurement decisions** for project planning and material sourcing.
    """)

    st.image("https://raw.githubusercontent.com/Pushkindugam/Artson-Steel-WPI-Prediction-Seasonality/main/WPI_Correlation_Screenshot.png", use_container_width=True)

    df_master = load_excel_from_github(master_url)
    df_master['Date'] = pd.to_datetime(df_master['Date'])
    df_master.set_index('Date', inplace=True)

    corr = df_master.corr(numeric_only=True)
    st.dataframe(corr.style.background_gradient(cmap='coolwarm').format("{:.2f}"))

    fig_corr, ax = plt.subplots(figsize=(10, 5))
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', ax=ax)
    ax.set_title("WPI Correlation Heatmap")
    st.pyplot(fig_corr)

# ---- Tab 4: Dataset ---- #
with tabs[3]:
    st.header("📂 Master Dataset Overview")
    st.markdown("""
    This dataset contains **monthly data from January 2022 to April 2025**  
    covering key economic and industry indicators that influence **steel WPI**.

    📌 A total of **21 columns** have been curated from **authorized government sources**  
    including commodity prices, economic indices, and input costs relevant to EPC projects.
    """)

    st.subheader("📋 WPI Master Dataset Preview")
    st.dataframe(df_master.head(20), use_container_width=True)

    st.markdown("🔗 [Download Full Excel Dataset](https://github.com/Pushkindugam/Artson-Steel-WPI-Prediction-Seasonality/raw/main/WPI_Master-dataset.xlsx)")



























# import streamlit as st
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# from statsmodels.tsa.seasonal import STL
# from sklearn.metrics import mean_squared_error
# import numpy as np
# import requests

# # ---------------- Sidebar ---------------- #
# with st.sidebar:
#     st.image(
#         "https://raw.githubusercontent.com/Pushkindugam/Artson-Steel-WPI-Prediction-Seasonality/main/artson_logo.png",
#         use_container_width=True,
#         caption="Artson Engineering Ltd."
#     )

#     st.markdown("## 📘 What is WPI?")
#     st.markdown("""
#     WPI stands for **Wholesale Price Index**.  
#     It tracks the price changes of goods at the **wholesale level**,  
#     forecasting steel useful for procurement in **EPC industries**.
#     """)

#     st.markdown("---")
#     st.markdown("### 🛠️ Built by Artson SCM Team – 2025")
#     st.markdown("*by **Pushkin Dugam***")
#     st.markdown("[🔗 GitHub Repository](https://github.com/Pushkindugam/Artson-Steel-WPI-Prediction-Seasonality)")

# # ---------------- Page Setup ---------------- #
# st.set_page_config(page_title="WPI Steel Dashboard", layout="centered")
# st.markdown("<h1 style='text-align: center;'>Artson Ltd, A Tata Enterprise</h1>", unsafe_allow_html=True)
# st.markdown("<h2 style='text-align: center;'>WPI Steel Analysis with Forecasting</h2>", unsafe_allow_html=True)

# # ---------------- Load Data ---------------- #
# @st.cache_data
# def load_excel_from_github(url):
#     response = requests.get(url)
#     return pd.read_excel(response.content)

# master_url = "https://github.com/Pushkindugam/Artson-Steel-WPI-Prediction-Seasonality/raw/main/WPI_Master-dataset.xlsx"
# forecast_url = "https://github.com/Pushkindugam/Artson-Steel-WPI-Prediction-Seasonality/raw/main/WPI_Steel_jan2022_to_may2026.xlsx"

# # ---------------- Tabs ---------------- #
# tabs = st.tabs(["📈 Prediction", "📆 Seasonality", "📊 Correlation", "📂 Dataset"])

# # ---- Tab 1: Prediction ---- #
# with tabs[0]:
#     st.header("📈 Forecasting Steel WPI (2022–2026)")
#     st.image("https://raw.githubusercontent.com/Pushkindugam/Artson-Steel-WPI-Prediction-Seasonality/main/WPI_Prediction_screenshot.png", use_container_width=True)

#     df_forecast = load_excel_from_github(forecast_url)
#     df_forecast['Date'] = pd.to_datetime(df_forecast['Date'], format='%b-%y')
#     df_forecast.set_index('Date', inplace=True)

#     fig, ax = plt.subplots(figsize=(10, 4))
#     for col in ['WPI (stainless)', 'WPI (mild flat)', 'WPI (mild long)']:
#         ax.plot(df_forecast.index, df_forecast[col], label=col)
#     ax.set_title("Forecasted WPI Trends")
#     ax.set_ylabel("WPI Index")
#     ax.grid(True)
#     ax.legend()
#     st.pyplot(fig)

# # ---- Tab 2: Seasonality ---- #
# with tabs[1]:
#     st.header("📆 Seasonal Patterns of Steel Prices")
#     st.image("https://raw.githubusercontent.com/Pushkindugam/Artson-Steel-WPI-Prediction-Seasonality/main/WPI_Seasonality_screenshot.png", use_container_width=True)

#     steel_types = {
#         'WPI (stainless)': 'Stainless',
#         'WPI (mild flat)': 'Mild Flat',
#         'WPI (mild long)': 'Mild Long'
#     }

#     for col, label in steel_types.items():
#         st.subheader(f"🔁 STL Decomposition: {label}")
#         stl = STL(df_forecast[col], period=12).fit()
#         seasonal = stl.seasonal

#         fig_s, ax = plt.subplots(figsize=(10, 3))
#         ax.plot(seasonal, label='Seasonal Component')
#         ax.set_title(f"{label} Steel - Seasonal Pattern")
#         ax.legend()
#         st.pyplot(fig_s)

# # ---- Tab 3: Correlation ---- #
# with tabs[2]:
#     st.header("📊 Correlation of WPI Categories")
#     st.image("https://raw.githubusercontent.com/Pushkindugam/Artson-Steel-WPI-Prediction-Seasonality/main/WPI_Correlation_Screenshot.png", use_container_width=True)

#     df_master = load_excel_from_github(master_url)
#     df_master['Date'] = pd.to_datetime(df_master['Date'])
#     df_master.set_index('Date', inplace=True)

#     corr = df_master.corr(numeric_only=True)
#     st.dataframe(corr.style.background_gradient(cmap='coolwarm').format("{:.2f}"))

#     fig_corr, ax = plt.subplots(figsize=(10, 5))
#     sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', ax=ax)
#     ax.set_title("WPI Correlation Heatmap")
#     st.pyplot(fig_corr)

# # ---- Tab 4: Dataset ---- #
# with tabs[3]:
#     st.header("📂 Master Dataset Overview")
#     st.markdown("""
#     This dataset contains **monthly data from January 2022 to April 2025**  
#     covering key economic and industry indicators that influence **steel WPI**.

#     📌 A total of **21 columns** have been curated from **authorized government sources**  
#     including commodity prices, economic indices, and input costs relevant to EPC projects.
#     """)

#     st.subheader("📋 WPI Master Dataset Preview")
#     st.dataframe(df_master.head(20), use_container_width=True)

#     st.markdown("🔗 [Download Full Excel Dataset](https://github.com/Pushkindugam/Artson-Steel-WPI-Prediction-Seasonality/raw/main/WPI_Master-dataset.xlsx)")
















