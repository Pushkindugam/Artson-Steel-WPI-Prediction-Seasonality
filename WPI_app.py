import streamlit as st
import pandas as pd
import base64

st.set_page_config(layout="wide", page_title="WPI Steel Analysis")

st.title("📊 WPI Steel Forecasting Dashboard")

# Sidebar Navigation
page = st.sidebar.radio("Navigate", ["📘 PDF Reports", "📑 Data Tables", "📈 Charts & Trend"])

# -------- Helper Functions --------
def embed_pdf(file_path):
    """Embed PDF in Streamlit using base64 encoding."""
    with open(file_path, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')
        pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="800px" type="application/pdf"></iframe>'
        st.markdown(pdf_display, unsafe_allow_html=True)

def load_excel(file_path):
    """Read Excel file into DataFrame."""
    return pd.read_excel(file_path)

# -------- Page: PDF Reports --------
if page == "📘 PDF Reports":
    st.header("📘 Final Results [WPI Prediction Of Steel]")
    embed_pdf("Final Results [WPI Prediction Of Steel].pdf")

    st.header("📘 Correlation_WPI Steel")
    embed_pdf("Correlation_WPI Steel.pdf")

    st.header("📘 Seasonal Pattern Of Steel")
    embed_pdf("Seasonal Pattern Of Steel.pdf")

# -------- Page: Data Tables --------
elif page == "📑 Data Tables":
    st.header("📊 WPI Master Dataset")
    df1 = load_excel("WPI_Master-dataset.xlsx")
    st.dataframe(df1, use_container_width=True)

    st.header("📊 WPI Steel Jan 2022 to May 2026")
    df2 = load_excel("WPI_Steel_jan2022_to_may2026.xlsx")
    st.dataframe(df2, use_container_width=True)

# -------- Page: Line Chart --------
elif page == "📈 Charts & Trend":
    st.header("📈 WPI Steel Trend (2022–2026)")

    df = load_excel("WPI_Steel_jan2022_to_may2026.xlsx")

    # Ensure Date format
    if "Date" in df.columns and "WPI" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values("Date")
        st.line_chart(df.set_index("Date")["WPI"])
    else:
        st.warning("Required columns 'Date' and 'WPI' not found in the dataset.")
