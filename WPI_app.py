import streamlit as st
from streamlit.components.v1 import iframe

st.set_page_config(page_title="Steel WPI PDFs", layout="wide")

st.title("📄 Steel WPI Analysis Reports")

pdfs = {
    "📘 Correlation WPI Steel": "Correlation_WPI Steel.pdf",
    "📕 Final Results [WPI Prediction Of Steel]": "Final Results [WPI Prediction Of Steel].pdf",
    "📗 Seasonal Pattern Of Steel": "Seasonal Pattern Of Steel.pdf"
}

selected_pdf = st.sidebar.selectbox("Select PDF to view:", list(pdfs.keys()))

pdf_path = pdfs[selected_pdf]

# Show PDF inside an iframe
st.subheader(selected_pdf)
iframe(f"./{pdf_path}", width=700, height=1000)
