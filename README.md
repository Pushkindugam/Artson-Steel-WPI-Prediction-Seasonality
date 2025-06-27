# 📊 WPI Steel Forecasting Dashboard

An interactive Streamlit dashboard built for **Artson Engineering Ltd.**, a Tata Enterprise. This tool forecasts and analyzes **Wholesale Price Index (WPI)** trends for steel products (stainless, mild flat, and mild long), combining machine learning models with statistical methods.

🌐 **Live App**: [WPI Dashboard](https://wpiapppy-2jt5zbwrjhis3jd3hht6zz.streamlit.app/)

---

## 🚀 Features

✅ Steel WPI Forecasting using XGBoost + SARIMA  
✅ Seasonal trend analysis for bulk purchase planning  
✅ Correlation insights for procurement cost drivers  
✅ Machine Learning model explanation tab  
✅ Dataset viewer with downloadable Excel file  
✅ Clean, centered layout with branding

---

## 🧭 Dashboard Navigation

| Tab                     | Description |
|-------------------------|-------------|
| 📈 **Prediction**        | Forecast WPI for different steel types (May 2025–May 2026) with visual trend plots |
| 📆 **Seasonality**       | Shows recurring seasonal behavior in steel prices to optimize procurement timing |
| 📊 **Correlation**       | Highlights interdependencies between steel WPI and economic indicators |
| 📂 **Dataset**           | Displays master dataset (2022–2025) with key commodity and economic indicators |
| 🤖 **ML Model**          | Explains the forecasting approach using XGBoost Regressor and SARIMA |

---

## 📸 Screenshot

![Dashboard Screenshot](https://raw.githubusercontent.com/Pushkindugam/Artson-Steel-WPI-Prediction-Seasonality/main/WPI_input_screenshot.png)

---

## 📁 Repository Contents

| File Name                                   | Description |
|---------------------------------------------|-------------|
| `WPI_app.py`                                | 🎯 Main Streamlit application |
| `WPI_Master-dataset.xlsx`                   | 📊 Historical dataset of WPI-related variables |
| `WPI_Steel_jan2022_to_may2026.xlsx`         | 📈 Final WPI predictions from hybrid models |
| `Final Results [WPI Prediction Of Steel].pdf` | 📘 Forecast report with charts and model metrics |
| `Correlation_WPI Steel.pdf`                 | 📘 Statistical correlation summary |
| `Seasonal Pattern Of Steel.pdf`             | 📘 Seasonality decomposition of steel WPI |
| `requirements.txt`                          | 📦 Python packages required to run the app |

---

## 📘 PDF Report Summaries

| PDF Report | Purpose |
|------------|---------|
| `Final Results [WPI Prediction Of Steel].pdf` | Contains time-series forecasts, XGBoost and SARIMA model outputs |
| `Correlation_WPI Steel.pdf` | Visualizes correlation matrix of WPI and input indicators |
| `Seasonal Pattern Of Steel.pdf` | Shows yearly seasonal effects to guide inventory planning |

---

## 💡 Technologies Used

- Python
- Streamlit
- Pandas, NumPy
- Matplotlib, Seaborn
- XGBoost
- statsmodels (SARIMA)
- GitHub-hosted datasets and images

---

## 💻 How to Run Locally

### 1. Clone the repository
```bash
git clone https://github.com/Pushkindugam/Artson-Steel-WPI-Prediction-Seasonality.git
cd Artson-Steel-WPI-Prediction-Seasonality
