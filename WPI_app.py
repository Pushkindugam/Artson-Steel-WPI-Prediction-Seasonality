
# --- Imports ---
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import STL
from sklearn.metrics import mean_squared_error
import numpy as np
import requests

# LangChain & NLP
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from bs4 import BeautifulSoup

# --- Sidebar ---
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

# --- Page Setup ---
st.set_page_config(page_title="WPI Steel Dashboard", layout="centered")
st.markdown("<h1 style='text-align: center;'>Artson Ltd, A Tata Enterprise</h1>", unsafe_allow_html=True)
st.markdown("<h2 style='text-align: center;'>WPI Steel Analysis with Forecasting</h2>", unsafe_allow_html=True)

# --- Load Excel from GitHub ---
@st.cache_data
def load_excel_from_github(url):
    response = requests.get(url)
    return pd.read_excel(response.content)

master_url = "https://github.com/Pushkindugam/Artson-Steel-WPI-Prediction-Seasonality/raw/main/WPI_Master-dataset.xlsx"
forecast_url = "https://github.com/Pushkindugam/Artson-Steel-WPI-Prediction-Seasonality/raw/main/WPI_Steel_jan2022_to_may2026.xlsx"

# --- Tabs ---
tabs = st.tabs(["📈 Prediction", "📆 Seasonality", "📊 Correlation", "📂 Dataset", "🤖 ML Model", "🧠 Chatbot"])

# --- Tab 1: Forecasting ---
with tabs[0]:
    st.header("📈 Forecasting Steel WPI (2022–2026)")
    st.markdown("""
    Steel price forecasting helps predict future trends in WPI,  
    enabling better planning of procurement budgets and contracts.  
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

# --- Tab 2: Seasonality ---
with tabs[1]:
    st.header("📆 Seasonal Patterns of Steel Prices")
    st.markdown("""
    Seasonality analysis reveals repeating patterns in steel prices across months or years.  
    Understanding seasonal trends helps procurement teams schedule bulk purchases  
    in **low-price months**, avoiding cost spikes during **peak demand seasons**.
    """)
    st.image("https://raw.githubusercontent.com/Pushkindugam/Artson-Steel-WPI-Prediction-Seasonality/main/WPI_Seasonality_screenshot.png", use_container_width=True)

# --- Tab 3: Correlation ---
with tabs[2]:
    st.header("📊 Correlation of WPI Categories")
    st.markdown("""
    Correlation analysis shows how steel WPI is influenced by other economic and industrial indicators.  
    This helps identify key **cost drivers** and supports **data-driven procurement decisions**.
    """)
    st.image("https://raw.githubusercontent.com/Pushkindugam/Artson-Steel-WPI-Prediction-Seasonality/main/WPI_Correlation_Screenshot.png", use_container_width=True)

# --- Tab 4: Dataset ---
with tabs[3]:
    st.header("📂 Master Dataset Overview")
    st.markdown("""
    This tab contains the **raw data** collected from government and market sources  
    used for forecasting and correlation analysis.
    """)

    df_master = load_excel_from_github(master_url)
    df_master['Date'] = pd.to_datetime(df_master['Date'])
    df_master.set_index('Date', inplace=True)

    st.subheader("📋 WPI Master Dataset Preview")
    st.dataframe(df_master.head(20), use_container_width=True)

    st.markdown("🔗 [Download Full Excel Dataset](https://github.com/Pushkindugam/Artson-Steel-WPI-Prediction-Seasonality/raw/main/WPI_Master-dataset.xlsx)")

# --- Tab 5: ML Model ---
with tabs[4]:
    st.header("🤖 ML Model")
    st.markdown("""
    To forecast the **Wholesale Price Index (WPI)** for stainless, mild flat, and mild long steel categories,  
    a combination of **machine learning** and **statistical time series modeling** was employed.
    """)

    st.subheader("📌 XGBoost Regressor")
    st.markdown("""
    **XGBoost** is a tree-based machine learning model trained on:  
    - Exchange rate, Inflation, PMI  
    - Fuel, Iron ore, Coking coal, Cement WPI  
    - Steel futures, Production, Imports/Exports
    """)

    st.subheader("📌 SARIMA Time Series Model")
    st.markdown("""
    Captures trend & seasonality with parameters:  
    - (1,1,1)(0,1,1,12)  
    Complementing XGBoost by using past WPI values alone.
    """)

    st.subheader("🔗 Hybrid Strategy")
    st.markdown("""
    Forecast is the average of XGBoost & SARIMA predictions — combining nonlinear learning with seasonal trends.
    """)

# --- Tab 6: Chatbot ---
with tabs[5]:
    st.header("🧠 Ask About WPI Trends, Forecasts & Steel Prices")

    st.markdown("""
    Type your question below and the assistant will answer based on:  
    - 📊 Forecast results  
    - 📈 Seasonal trends (e.g., January dip)  
    - 🧠 ML model logic (XGBoost + SARIMA)  
    - 🌐 Live steel prices (JSW, SteelMint)  
    """)

    df = load_excel_from_github(forecast_url)
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)

    # Knowledge summaries
    seasonal_summary = "WPI shows lowest values typically in January, due to reduced construction activity, fiscal tightening, and weather-related slowdowns. Peak prices appear around July–October."
    forecast_summary = f"""
    Stainless steel WPI as of last forecasted month: {df['WPI (stainless)'].iloc[-1]:.2f}  
    Mild flat steel WPI: {df['WPI (mild flat)'].iloc[-1]:.2f}  
    Mild long steel WPI: {df['WPI (mild long)'].iloc[-1]:.2f}  
    Forecast available from {df.index[0].strftime('%b %Y')} to {df.index[-1].strftime('%b %Y')}
    """
    model_summary = """
    Forecast is a hybrid of XGBoost (economic indicators) and SARIMA (seasonal cycles).  
    SARIMA: Order(1,1,1)(0,1,1,12)  
    XGBoost Features: Inflation, FX rate, Cement WPI, Iron Ore, Coal, Futures, Imports/Exports
    """
    correlation_summary = """
    Mild flat and stainless steel WPI are highly correlated.  
    Cement, fuel prices, and production drive costs.
    """

    # Web scraping steel price
    def get_scraped_price():
        try:
            url = "https://www.steelmint.com/market-intel/indian-market"
            response = requests.get(url, timeout=5)
            soup = BeautifulSoup(response.content, "html.parser")
            result = soup.find("div", string=lambda x: x and "HRC Mumbai" in x)
            return result.text.strip() if result else "Live price not found"
        except Exception as e:
            return "Error fetching live price"

    scraped_price = get_scraped_price()

    # Combine all into one string for chatbot
    combined_text = f"""
    {seasonal_summary}

    {forecast_summary}

    {model_summary}

    {correlation_summary}

    Current live price (e.g., HRC Mumbai): {scraped_price}
    """

    # Create chatbot
    @st.cache_resource
    def create_chatbot():
        docs = [Document(page_content=combined_text)]
        splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
        chunks = splitter.split_documents(docs)
        db = FAISS.from_documents(chunks, OpenAIEmbeddings())
        chain = RetrievalQA.from_chain_type(OpenAI(temperature=0), retriever=db.as_retriever())
        return chain

    chatbot = create_chatbot()

    # User Input
    question = st.text_input("💬 Ask a question (e.g., 'Why is Jan lowest?', 'Explain XGBoost')")
    if question:
        with st.spinner("Thinking..."):
            answer = chatbot.run(question)
        st.success("📘 Answer:")
        st.write(answer)
























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




































