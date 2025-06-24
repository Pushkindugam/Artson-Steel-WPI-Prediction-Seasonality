import streamlit as st
import urllib.parse

st.set_page_config(page_title="Steel WPI PDF Viewer", layout="wide")

st.title("📄 Steel WPI Analysis Reports")

# Dictionary of PDFs
pdfs = {
    "📘 Correlation WPI Steel": "Correlation_WPI Steel.pdf",
    "📕 Final Results [WPI Prediction Of Steel]": "Final Results [WPI Prediction Of Steel].pdf",
    "📗 Seasonal Pattern Of Steel": "Seasonal Pattern Of Steel.pdf"
}

# Sidebar selection
selected_pdf = st.sidebar.selectbox("Select a PDF to view", list(pdfs.keys()))

# Get the file path
pdf_path = pdfs[selected_pdf]

# Embed the PDF
st.subheader(selected_pdf)
encoded_url = urllib.parse.quote(pdf_path)

pdf_display = f"""
<iframe src="{encoded_url}" width="100%" height="1000px" type="application/pdf"></iframe>
"""

st.markdown(pdf_display, unsafe_allow_html=True)
