import streamlit as st
import pandas as pd
import os

# Set Streamlit page settings
st.set_page_config(layout="wide", page_title="WPI Steel Analysis")

st.title("📊 WPI Steel Forecasting Dashboard")

# Sidebar Navigation
page = st.sidebar.radio("Navigate", ["📘 PDF Reports", "📑 Data Tables", "📈 Charts & Trend"])

# Helper to load Excel file
def load_excel(file_path):
    return pd.read_excel(file_path)

# Helper to read binary file for download
def get_file_bytes(file_path):
    with open(file_path, "rb") as f:
        return f.read()

# ----------------- PDF Download Tab -----------------
if page == "📘 PDF Reports":
    st.header("📥 Download PDF Reports")

    pdf_files = {
        "Final Results [WPI Prediction Of Steel].pdf": "📘 Final Results Report",
        "Correlation_WPI Steel.pdf": "📘 Correlation Report",
        "Seasonal Pattern Of Steel.pdf": "📘 Seasonal Pattern Report"
    }

    for filename, label in pdf_files.items():
        if os.path.exists(filename):
            pdf_bytes = get_file_bytes(filename)
            st.download_button(
                label=f"{label} ⬇️",
                data=pdf_bytes,
                file_name=filename,
                mime="application/pdf"
            )
        else:
            st.warning(f"File not found: {filename}")

# ----------------- Data Table Tab -----------------
elif page == "📑 Data Tables":
    st.header("📊 WPI Master Dataset")
    path1 = "WPI_Master-dataset.xlsx"
    if os.path.exists(path1):
        df1 = load_excel(path1)
        st.dataframe(df1, use_container_width=True)
    else:
        st.error("File not found: WPI_Master-dataset.xlsx")

    st.header("📊 WPI Steel Jan 2022 to May 2026")
    path2 = "WPI_Steel_jan2022_to_may2026.xlsx"
    if os.path.exists(path2):
        df2 = load_excel(path2)
        st.dataframe(df2, use_container_width=True)
    else:
        st.error("File not found: WPI_Steel_jan2022_to_may2026.xlsx")

