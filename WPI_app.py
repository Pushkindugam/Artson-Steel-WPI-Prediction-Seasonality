
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import STL, seasonal_decompose
from sklearn.metrics import mean_squared_error
import numpy as np
import os
import zipfile
import base64

st.set_page_config(page_title="WPI Steel Index Dashboard", layout="wide")
st.title("\U0001F4CA WPI Steel Index Dashboard (2022–2026)")
st.markdown("Explore **Forecasting**, **Seasonality**, and **Correlation** Analysis for Stainless, Mild Flat, and Mild Long Steel")

# Sidebar Uploads
st.sidebar.header("\U0001F4C2 Upload Datasets")
master_file = st.sidebar.file_uploader("Upload `WPI_Master-dataset.xlsx`", type=['xlsx'])
forecast_file = st.sidebar.file_uploader("Upload `WPI_Steel_jan2022_to_may2026.xlsx`", type=['xlsx'])
zip_file = st.sidebar.file_uploader("Upload MA/WMA/EMA ZIP file", type=["zip"])

# Tabs for different analysis sections
tabs = st.tabs(["Correlation", "Forecasting", "Seasonality", "MA/WMA/EMA"])

# ---- Tab 1: Correlation ---- #
with tabs[0]:
    st.subheader("\U0001F4C8 WPI Steel Correlation Matrix")
    st.image("https://github.com/Pushkindugam/Artson-Steel-WPI-Prediction-Seasonality/raw/main/WPI_Correlation_Screenshot.png", use_container_width=True)

# ---- Tab 2: Forecasting ---- #
with tabs[1]:
    if forecast_file:
        df_forecast = pd.read_excel(forecast_file)
        df_forecast['Date'] = pd.to_datetime(df_forecast['Date'], format='%b-%y')
        df_forecast.set_index('Date', inplace=True)

        st.subheader("\U0001F4C8 Forecast Visualization (Early vs Late)")
        df_early = df_forecast[df_forecast.index < '2025-05-01']
        df_late = df_forecast[df_forecast.index >= '2025-05-01']

        fig_trend, ax = plt.subplots(figsize=(14, 6))
        for col, color in zip(['WPI (stainless)', 'WPI (mild flat)', 'WPI (mild long)'], ['blue', 'green', 'orange']):
            ax.plot(df_early.index, df_early[col], label=f'{col} (Early)', color=color)
            ax.plot(df_late.index, df_late[col], '--', label=f'{col} (Late)', color=color)
        ax.set_title('WPI Trends (2022–2026)')
        ax.set_ylabel('WPI Index')
        ax.legend()
        ax.grid(True)
        st.pyplot(fig_trend)

# ---- Tab 3: Seasonality ---- #
with tabs[2]:
    if forecast_file:
        df_forecast = pd.read_excel(forecast_file)
        df_forecast['Date'] = pd.to_datetime(df_forecast['Date'], format='%b-%y')
        df_forecast.set_index('Date', inplace=True)

        steel_types = {
            'WPI (stainless)': 'Stainless',
            'WPI (mild flat)': 'Mild Flat',
            'WPI (mild long)': 'Mild Long'
        }

        for col, label in steel_types.items():
            st.subheader(f"\U0001F4C6 STL Seasonal Decomposition ({label})")
            stl = STL(df_forecast[col], period=12).fit()
            seasonal = stl.seasonal

            seasons = {}
            for y in range(2022, 2026):
                start = f"{y}-05-01"
                end = f"{y+1}-04-01"
                if end in seasonal.index:
                    s = seasonal[start:end]
                    s -= s.mean()
                    seasons[f"{y}-{y+1}"] = s

            fig_season, ax = plt.subplots(figsize=(14, 5))
            markers = ['o', 's', '^', 'x']
            for i, (season_label, series) in enumerate(seasons.items()):
                ax.plot(series.index, series.values, label=season_label, marker=markers[i])
            ax.set_title(f"STL Seasonal Components - {label}")
            ax.set_ylabel("Zero-Mean Seasonal Component")
            ax.legend()
            ax.grid(True)
            st.pyplot(fig_season)

            st.subheader(f"\U0001F501 Season Similarity Metrics - {label}")
            ref = seasons.get("2024-2025")
            target = seasons.get("2025-2026")
            if ref is not None and target is not None:
                mse = mean_squared_error(ref.values, target.values)
                nmse = mse / np.var(ref.values)
                corr = np.corrcoef(ref.values, target.values)[0, 1]

                st.metric("\U0001F501 Correlation", f"{corr:.3f}")
                st.metric("\U0001F4C9 MSE", f"{mse:.2f}")
                st.metric("\u2696\ufe0f Normalized MSE", f"{nmse:.3f}")

            st.subheader(f"\U0001F4C9 Additive Decomposition: Forecast vs Pattern - {label}")
            result = seasonal_decompose(df_forecast[col][:"2025-04-30"], model='additive', period=12)
            pattern = result.seasonal[-12:]
            forecast_part = df_forecast[col]["2025-05-01":]
            future_pattern = np.tile(pattern.values, int(np.ceil(len(forecast_part) / 12)))[:len(forecast_part)]

            fig_align, ax = plt.subplots(figsize=(12, 5))
            ax.plot(forecast_part.index, forecast_part.values, label='Actual Forecast', marker='o', color='gray')
            ax.plot(forecast_part.index, future_pattern, label='Expected Seasonal Pattern', linestyle='--', marker='x', color='blue')
            ax.axvline(pd.to_datetime('2025-05-01'), color='red', linestyle='--', label='Forecast Start')
            ax.set_title(f"{label}: Actual vs Expected Seasonal")
            ax.grid(True)
            ax.legend()
            st.pyplot(fig_align)

# ---- Tab 4: MA/WMA/EMA Forecasts ---- #
with tabs[3]:
    if zip_file:
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extractall("ma_forecasts")
            st.success("Extracted Forecasting ZIP!")

        for file in os.listdir("ma_forecasts"):
            if file.endswith(".xlsx"):
                ma_df = pd.read_excel(os.path.join("ma_forecasts", file))
                break

        st.subheader("\U0001F4C9 MA/WMA/EMA Forecast Table")
        st.dataframe(ma_df.head())

        st.subheader("\U0001F4C8 MA, WMA, EMA Forecast Plot")
        cols = ma_df.columns[1:]
        fig, ax = plt.subplots(figsize=(14, 6))
        for col in cols:
            ax.plot(ma_df.iloc[:, 0], ma_df[col], label=col, marker='o')
        ax.set_title("Forecasting using MA, WMA, EMA")
        ax.set_xlabel("Date")
        ax.set_ylabel("WPI")
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)

# ---- Footer ---- #
st.markdown("---")
st.markdown("Developed by **Pushkin Dugam** | [GitHub](https://github.com/Pushkindugam)")



























# import streamlit as st
# import pandas as pd
# import os

# # Set Streamlit page settings
# st.set_page_config(layout="wide", page_title="WPI Steel Analysis")

# st.title("📊 WPI Steel Forecasting Dashboard")

# # Sidebar Navigation
# page = st.sidebar.radio("Navigate", ["📘 PDF Reports", "📑 Data Tables"])

# # Helper to load Excel file
# def load_excel(file_path):
#     return pd.read_excel(file_path)

# # Helper to read binary file for download
# def get_file_bytes(file_path):
#     with open(file_path, "rb") as f:
#         return f.read()

# # ----------------- PDF Download Tab -----------------
# if page == "📘 PDF Reports":
#     st.header("📥 Download PDF Reports")

#     pdf_files = {
#         "Final Results [WPI Prediction Of Steel].pdf": "📘 Final Results Report",
#         "Correlation_WPI Steel.pdf": "📘 Correlation Report",
#         "Seasonal Pattern Of Steel.pdf": "📘 Seasonal Pattern Report"
#     }

#     for filename, label in pdf_files.items():
#         if os.path.exists(filename):
#             pdf_bytes = get_file_bytes(filename)
#             st.download_button(
#                 label=f"{label} ⬇️",
#                 data=pdf_bytes,
#                 file_name=filename,
#                 mime="application/pdf"
#             )
#         else:
#             st.warning(f"File not found: {filename}")

# # ----------------- Data Table Tab -----------------
# elif page == "📑 Data Tables":
#     st.header("📊 WPI Master Dataset")
#     path1 = "WPI_Master-dataset.xlsx"
#     if os.path.exists(path1):
#         df1 = load_excel(path1)
#         st.dataframe(df1, use_container_width=True)
#     else:
#         st.error("File not found: WPI_Master-dataset.xlsx")

#     st.header("📊 WPI Steel Jan 2022 to May 2026")
#     path2 = "WPI_Steel_jan2022_to_may2026.xlsx"
#     if os.path.exists(path2):
#         df2 = load_excel(path2)
#         st.dataframe(df2, use_container_width=True)
#     else:
#         st.error("File not found: WPI_Steel_jan2022_to_may2026.xlsx")

