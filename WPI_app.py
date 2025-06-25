import streamlit as st
import pandas as pd
import os
import base64

# ----------------- Streamlit Page Config -----------------
st.set_page_config(layout="wide", page_title="WPI Steel Analysis")

st.title("📊 WPI Steel Forecasting Dashboard")

# ----------------- Sidebar Navigation -----------------
page = st.sidebar.radio("Navigate", ["📘 PDF Reports", "📑 Data Tables", "📈 Charts & Trend"])

# ----------------- Helper Functions -----------------
def load_excel(file_path):
    """Load Excel file into a DataFrame."""
    return pd.read_excel(file_path)

# ----------------- PDF Reports Tab -----------------
if page == "📘 PDF Reports":
    st.header("📘 Final Results [WPI Prediction Of Steel]")
    pdf_path1 = "Final Results [WPI Prediction Of Steel].pdf"
    if os.path.exists(pdf_path1):
        with open(pdf_path1, "rb") as f:
            st.download_button("📥 Download Final Results PDF", f, file_name="Final_Results_WPI.pdf")
    else:
        st.error("PDF file not found: Final Results")

    st.header("📘 Correlation_WPI Steel")
    pdf_path2 = "Correlation_WPI Steel.pdf"
    if os.path.exists(pdf_path2):
        with open(pdf_path2, "rb") as f:
            st.download_button("📥 Download Correlation PDF", f, file_name="Correlation_WPI.pdf")
    else:
        st.error("PDF file not found: Correlation")

    st.header("📘 Seasonal Pattern Of Steel")
    pdf_path3 = "Seasonal Pattern Of Steel.pdf"
    if os.path.exists(pdf_path3):
        with open(pdf_path3, "rb") as f:
            st.download_button("📥 Download Seasonal Pattern PDF", f, file_name="Seasonal_Pattern_WPI.pdf")
    else:
        st.error("PDF file not found: Seasonal Pattern")

# ----------------- Data Tables Tab -----------------
elif page == "📑 Data Tables":
    st.header("📊 WPI Master Dataset")
    excel_path1 = "WPI_Master-dataset.xlsx"
    if os.path.exists(excel_path1):
        df1 = load_excel(excel_path1)
        st.dataframe(df1, use_container_width=True)
    else:
        st.error("Excel file not found: WPI_Master-dataset.xlsx")

    st.header("📊 WPI Steel Jan 2022 to May 2026")
    excel_path2 = "WPI_Steel_jan2022_to_may2026.xlsx"
    if os.path.exists(excel_path2):
        df2 = load_excel(excel_path2)
        st.dataframe(df2, use_container_width=True)
    else:
        st.error("Excel file not found: WPI_Steel_jan2022_to_may2026.xlsx")

# ----------------- Charts & Trend Tab -----------------
elif page == "📈 Charts & Trend":
    st.header("📈 WPI Steel Trend (2022–2026)")

    excel_path = "WPI_Steel_jan2022_to_may2026.xlsx"
    if os.path.exists(excel_path):
        df = load_excel(excel_path)

        if "Date" in df.columns and "WPI" in df.columns:
            df["Date"] = pd.to_datetime(df["Date"])
            df = df.sort_values("Date")
            st.line_chart(df.set_index("Date")["WPI"])
        else:
            st.warning("Dataset must include 'Date' and 'WPI' columns.")
    else:
        st.error("Excel file not found for chart.")

        st.line_chart(df.set_index("Date")["WPI"])
    else:
        st.warning("Required columns 'Date' and 'WPI' not found in the dataset.")
