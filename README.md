# 📊 WPI Steel Forecasting Dashboard

This Streamlit app provides an interactive interface to analyze and forecast **Wholesale Price Index (WPI)** data for steel products in India. It allows users to download detailed reports, view structured datasets, and explore the underlying data behind the prediction results.

🌐 **Live App**: [[https://pushkindugam-wpi-steel-forecasting.streamlit.app](https://pushkindugam-wpi-steel-forecasting.streamlit.app)](https://wpiapppy-2jt5zbwrjhis3jd3hht6zz.streamlit.app/)

---

## 🚀 Features

✅ Downloadable PDF reports  
✅ View WPI data tables directly from Excel files  
✅ Yearly Seasonal trend analysis and forecasting modules  
✅ Clean, responsive layout using Streamlit’s wide-mode interface  

---

## 📁 Repository Contents

| File Name                                   | Description |
|---------------------------------------------|-------------|
| `WPI_app.py`                                | 🎯 Main Streamlit application file |
| `WPI_Master-dataset.xlsx`                   | 📊 Master dataset containing historic WPI records |
| `WPI_Steel_jan2022_to_may2026.xlsx`         | 📈 Refined dataset for WPI Steel (2022–2026) |
| `Final Results [WPI Prediction Of Steel].pdf` | 📘 Forecast report with model results |
| `Correlation_WPI Steel.pdf`                 | 📘 Correlation analysis of WPI steel data |
| `Seasonal Pattern Of Steel.pdf`             | 📘 Seasonal decomposition report |
| `requirements.txt`                          | 📦 Python package requirements |

---

## 📘 PDF Report Explanations

| PDF Report | Description | Purpose |
|------------|-------------|---------|
| **Final Results [WPI Prediction Of Steel].pdf** | Contains time series models used, performance evaluation, and steel price forecasts | For stakeholders analyzing steel price trends and making procurement decisions |
| **Correlation_WPI Steel.pdf** | Explores statistical relationships between WPI Steel and other factors like economic indices | Helps uncover driving forces behind price fluctuations |
| **Seasonal Pattern Of Steel.pdf** | Decomposes the steel WPI data into trend, seasonal, and residual components | Useful for identifying repeating patterns and business cycle effects |

---

## 💻 How to Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/Pushkindugam/steel-wpi-forecasting.git
cd steel-wpi-forecasting
