import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import STL
from sklearn.metrics import mean_squared_error
import numpy as np
import requests
import openai
from bs4 import BeautifulSoup

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

def get_steel_price(city="Mumbai"):
    url = "https://www.steelmint.com/market-intel/indian-market"
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    price_data = soup.find('div', text=lambda x: x and city in x)
    return price_data.text if price_data else "Price not found"

master_url = "https://github.com/Pushkindugam/Artson-Steel-WPI-Prediction-Seasonality/raw/main/WPI_Master-dataset.xlsx"
forecast_url = "https://github.com/Pushkindugam/Artson-Steel-WPI-Prediction-Seasonality/raw/main/WPI_Steel_jan2022_to_may2026.xlsx"

# ---------------- Tabs ---------------- #
tabs = st.tabs(["📈 Prediction", "📆 Seasonality", "📊 Correlation", "📂 Dataset", "🤖 ML Model", "💬 Ask a Question"])

# ---- Tab 1: Prediction ---- #
with tabs[0]:
    st.header("📈 Forecasting Steel WPI (2022–2026)")

    st.markdown("""
    Steel price forecasting helps predict future trends in WPI,  
    enabling better planning of procurement budgets and contracts.  

    These graphs show projected values of WPI for stainless, mild flat, and mild long steel categories  
    from **May 2025 to May 2026**, along with trend analysis of earlier dates.
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

    Understanding seasonal trends helps procurement teams schedule bulk purchases  
    in **low-price months**, avoiding cost spikes during **peak demand seasons**.
    """)

    st.image("https://raw.githubusercontent.com/Pushkindugam/Artson-Steel-WPI-Prediction-Seasonality/main/WPI_Seasonality_screenshot.png", use_container_width=True)

# ---- Tab 3: Correlation ---- #
with tabs[2]:
    st.header("📊 Correlation of WPI Categories")

    st.markdown("""
    Correlation analysis shows how steel WPI is influenced by other economic and industrial indicators.  

    This helps identify key **cost drivers** and supports **data-driven procurement decisions**  
    for project planning and material sourcing.
    """)

    st.image("https://raw.githubusercontent.com/Pushkindugam/Artson-Steel-WPI-Prediction-Seasonality/main/WPI_Correlation_Screenshot.png", use_container_width=True)

# ---- Tab 4: Dataset ---- #
with tabs[3]:
    st.header("📂 Master Dataset Overview")

    st.markdown("""
    This tab contains the **raw data** collected from government and market sources  
    used for forecasting and correlation analysis.  
    It includes 21 columns across 3+ years, covering commodity prices, fuel rates,  
    construction costs, and WPI categories.
    """)

    df_master = load_excel_from_github(master_url)
    df_master['Date'] = pd.to_datetime(df_master['Date'])
    df_master.set_index('Date', inplace=True)

    st.subheader("📋 WPI Master Dataset Preview")
    st.dataframe(df_master.head(20), use_container_width=True)

    st.markdown("🔗 [Download Full Excel Dataset](https://github.com/Pushkindugam/Artson-Steel-WPI-Prediction-Seasonality/raw/main/WPI_Master-dataset.xlsx)")

# ---- Tab 5: ML Model ---- #
with tabs[4]:
    st.header("🤖 ML Model")

    st.markdown("""
    To forecast the **Wholesale Price Index (WPI)** for stainless, mild flat, and mild long steel categories,  
    a combination of **machine learning** and **statistical time series modeling** was employed.

    The final forecast is an average of two powerful approaches:  
    - 🔸 **XGBoost Regressor** (gradient-boosted decision trees)  
    - 🔸 **SARIMA** (Seasonal AutoRegressive Integrated Moving Average)
    """)

    st.subheader("📌 XGBoost Regressor")
    st.markdown("""
    **XGBoost** is a tree-based machine learning model that captures complex relationships between features.  
    It was trained separately for each steel category using selected economic and industry indicators.

    - **Features used in prediction included:**  
        - Exchange rate (USD/INR)  
        - Inflation rate  
        - Manufacturing PMI  
        - Crude steel and finished steel production  
        - Iron ore, coking coal, and crude petroleum prices  
        - Steel futures indices (MS Futures, SHFE)  
        - Cement WPI, consumption, imports, and exports

    - **Model advantages:**  
        - Captures nonlinear patterns  
        - Handles multivariate dependencies effectively  
        - Performs well even with noisy real-world data
    """)

    st.subheader("📌 SARIMA Time Series Model")
    st.markdown("""
    **SARIMA** is a statistical model suited for forecasting data with trend and seasonality.  
    Separate SARIMA models were built for each WPI category using their historical monthly values.

    - **Model parameters used:**  
        - Order: (1, 1, 1)  
        - Seasonal order: (0, 1, 1, 12)

    - **Purpose:**  
        - Models seasonality and trend patterns in steel price movement  
        - Complements the feature-based XGBoost model by focusing purely on past WPI values
    """)

    st.subheader("🔗 Hybrid Forecasting Strategy")
    st.markdown("""
    The final forecast for each steel type was computed as the **average** of XGBoost and SARIMA predictions.  
    This hybrid approach offers the **strength of both models**:

    - SARIMA handles seasonal cycles and long-term price trends  
    - XGBoost adapts to dynamic changes in economic and industry drivers

    ✅ This ensemble method improves robustness, accuracy, and reliability for procurement planning.
    """)

# ---- Tab 6: Ask a Question ---- #
with tabs[5]:
    st.header("💬 Ask a Question")

    st.markdown("""
    This tab allows you to interact with an AI chatbot trained on this dashboard.  
    You can ask about steel prices, WPI trends, or dataset features.
    """)

    user_question = st.text_input("Ask your question about steel WPI data:")

    if user_question:
        with st.spinner("Thinking..."):
            openai.api_key = st.secrets["openai_api_key"] if "openai_api_key" in st.secrets else ""

            if not openai.api_key:
                st.error("OpenAI API key is missing. Add it in Streamlit secrets to use the chatbot.")
            else:
                context_summary = f"Current Steel Price (Mumbai): {get_steel_price()}\n"
                context_summary += "This dashboard uses XGBoost + SARIMA forecasts based on economic indicators like crude steel, PMI, INR/USD, etc."

                response = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": context_summary},
                        {"role": "user", "content": user_question},
                    ]
                )
                st.success(response['choices'][0]['message']['content'])











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
# tabs = st.tabs(["📈 Prediction", "📆 Seasonality", "📊 Correlation", "📂 Dataset", "🤖 ML Model"])

# # ---- Tab 1: Prediction ---- #
# with tabs[0]:
#     st.header("📈 Forecasting Steel WPI (2022–2026)")
    
#     st.markdown("""
#     Steel price forecasting helps predict future trends in WPI,  
#     enabling better planning of procurement budgets and contracts.  
    
#     These graphs show projected values of WPI for stainless, mild flat, and mild long steel categories  
#     from **May 2025 to May 2026**, along with trend analysis of earlier dates.
#     """)

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

#     st.markdown("""
#     Seasonality analysis reveals repeating patterns in steel prices across months or years.  
    
#     Understanding seasonal trends helps procurement teams schedule bulk purchases  
#     in **low-price months**, avoiding cost spikes during **peak demand seasons**.
#     """)

#     st.image("https://raw.githubusercontent.com/Pushkindugam/Artson-Steel-WPI-Prediction-Seasonality/main/WPI_Seasonality_screenshot.png", use_container_width=True)

# # ---- Tab 3: Correlation ---- #
# with tabs[2]:
#     st.header("📊 Correlation of WPI Categories")

#     st.markdown("""
#     Correlation analysis shows how steel WPI is influenced by other economic and industrial indicators.  
    
#     This helps identify key **cost drivers** and supports **data-driven procurement decisions**  
#     for project planning and material sourcing.
#     """)

#     st.image("https://raw.githubusercontent.com/Pushkindugam/Artson-Steel-WPI-Prediction-Seasonality/main/WPI_Correlation_Screenshot.png", use_container_width=True)

# # ---- Tab 4: Dataset ---- #
# with tabs[3]:
#     st.header("📂 Master Dataset Overview")

#     st.markdown("""
#     This tab contains the **raw data** collected from government and market sources  
#     used for forecasting and correlation analysis.  
#     It includes 21 columns across 3+ years, covering commodity prices, fuel rates,  
#     construction costs, and WPI categories.
#     """)

#     df_master = load_excel_from_github(master_url)
#     df_master['Date'] = pd.to_datetime(df_master['Date'])
#     df_master.set_index('Date', inplace=True)

#     st.subheader("📋 WPI Master Dataset Preview")
#     st.dataframe(df_master.head(20), use_container_width=True)

#     st.markdown("🔗 [Download Full Excel Dataset](https://github.com/Pushkindugam/Artson-Steel-WPI-Prediction-Seasonality/raw/main/WPI_Master-dataset.xlsx)")

# # ---- Tab 5: ML Model ---- #
# with tabs[4]:
#     st.header("🤖 ML Model")

#     st.markdown("""
#     To forecast the **Wholesale Price Index (WPI)** for stainless, mild flat, and mild long steel categories,  
#     a combination of **machine learning** and **statistical time series modeling** was employed.

#     The final forecast is an average of two powerful approaches:  
#     - 🔸 **XGBoost Regressor** (gradient-boosted decision trees)  
#     - 🔸 **SARIMA** (Seasonal AutoRegressive Integrated Moving Average)
#     """)

#     st.subheader("📌 XGBoost Regressor")
#     st.markdown("""
#     **XGBoost** is a tree-based machine learning model that captures complex relationships between features.  
#     It was trained separately for each steel category using selected economic and industry indicators.

#     - **Features used in prediction included:**  
#         - Exchange rate (USD/INR)  
#         - Inflation rate  
#         - Manufacturing PMI  
#         - Crude steel and finished steel production  
#         - Iron ore, coking coal, and crude petroleum prices  
#         - Steel futures indices (MS Futures, SHFE)  
#         - Cement WPI, consumption, imports, and exports

#     - **Model advantages:**  
#         - Captures nonlinear patterns  
#         - Handles multivariate dependencies effectively  
#         - Performs well even with noisy real-world data
#     """)

#     st.subheader("📌 SARIMA Time Series Model")
#     st.markdown("""
#     **SARIMA** is a statistical model suited for forecasting data with trend and seasonality.  
#     Separate SARIMA models were built for each WPI category using their historical monthly values.

#     - **Model parameters used:**  
#         - Order: (1, 1, 1)  
#         - Seasonal order: (0, 1, 1, 12)

#     - **Purpose:**  
#         - Models seasonality and trend patterns in steel price movement  
#         - Complements the feature-based XGBoost model by focusing purely on past WPI values
#     """)

#     st.subheader("🔗 Hybrid Forecasting Strategy")
#     st.markdown("""
#     The final forecast for each steel type was computed as the **average** of XGBoost and SARIMA predictions.  
#     This hybrid approach offers the **strength of both models**:
    
#     - SARIMA handles seasonal cycles and long-term price trends  
#     - XGBoost adapts to dynamic changes in economic and industry drivers

#     ✅ This ensemble method improves robustness, accuracy, and reliability for procurement planning.
#     """)












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
    
#     st.markdown("""
#     Steel price forecasting helps predict future trends in WPI,  
#     enabling better planning of procurement budgets and contracts.  
    
#     These graphs show projected values of WPI for stainless, mild flat, and mild long steel categories  
#     from **May 2025 to May 2026**, along with trend analysis of earlier dates.
#     """)

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

#     st.markdown("""
#     Seasonality analysis reveals repeating patterns in steel prices across months or years.  
    
#     Understanding seasonal trends helps procurement teams schedule bulk purchases  
#     in **low-price months**, avoiding cost spikes during **peak demand seasons**.
#     """)

#     st.image("https://raw.githubusercontent.com/Pushkindugam/Artson-Steel-WPI-Prediction-Seasonality/main/WPI_Seasonality_screenshot.png", use_container_width=True)

# # ---- Tab 3: Correlation ---- #
# with tabs[2]:
#     st.header("📊 Correlation of WPI Categories")

#     st.markdown("""
#     Correlation analysis shows how steel WPI is influenced by other economic and industrial indicators.  
    
#     This helps identify key **cost drivers** and supports **data-driven procurement decisions**  
#     for project planning and material sourcing.
#     """)

#     st.image("https://raw.githubusercontent.com/Pushkindugam/Artson-Steel-WPI-Prediction-Seasonality/main/WPI_Correlation_Screenshot.png", use_container_width=True)

# # ---- Tab 4: Dataset ---- #
# with tabs[3]:
#     st.header("📂 Master Dataset Overview")

#     st.markdown("""
#     This tab contains the **raw data** collected from government and market sources  
#     used for forecasting and correlation analysis.  
#     It includes 21 columns across 3+ years, covering commodity prices, fuel rates,  
#     construction costs, and WPI categories.
#     """)

#     df_master = load_excel_from_github(master_url)
#     df_master['Date'] = pd.to_datetime(df_master['Date'])
#     df_master.set_index('Date', inplace=True)

#     st.subheader("📋 WPI Master Dataset Preview")
#     st.dataframe(df_master.head(20), use_container_width=True)

#     st.markdown("🔗 [Download Full Excel Dataset](https://github.com/Pushkindugam/Artson-Steel-WPI-Prediction-Seasonality/raw/main/WPI_Master-dataset.xlsx)")




































