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

def embed_pdf(file_path):
    """Embed a PDF in the Streamlit app using base64 iframe."""
    with open(file_path, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode("utf-8")
        pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="800px" type="application/pdf"></iframe>'
        st.markdown(pdf_display, unsafe_allow_html=True)

# ----------------- PDF Reports Tab -----------------
if page == "📘 PDF Reports":
    st.header("📘 Final Results [WPI Prediction Of Steel]")
    pdf_path1 = "Final Results [WPI Prediction Of Steel].pdf"
    if os.path.exists(pdf_path1):
        embed_pdf(pdf_path1)
    else:
        st.error("PDF file not found: Final Results")

    st.header("📘 Correlation_WPI Steel")
    pdf_path2 = "Correlation_WPI Steel.pdf"
    if os.path.exists(pdf_path2):
        embed_pdf(pdf_path2)
    else:
        st.error("PDF file not found: Correlation")

    st.header("📘 Seasonal Pattern Of Steel")
    pdf_path3 = "Seasonal Pattern Of Steel.pdf"
    if os.path.exists(pdf_path3):
        embed_pdf(pdf_path3)
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


