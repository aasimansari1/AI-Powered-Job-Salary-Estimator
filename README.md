# 💼 AI-Powered Job Salary Estimator

A polished, portfolio-ready data science application that predicts annual salaries using machine learning — no sign-up required, instant predictions.

## ✨ Features

- **Instant salary predictions** based on 11 input factors
- **5 ML models** trained and compared: Ridge Regression, Random Forest, XGBoost, LightGBM, CatBoost
- **Interactive dashboard** with Plotly charts and real-time analytics
- **Career insights**: skill recommendations, career path suggestions, salary growth projections
- **Multi-role comparison** with radar charts
- **Export results** as PDF or CSV
- **No authentication** — fully public, open access

## 🚀 Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Generate data & train models (one-time setup)

```bash
python train.py
```

This will:
- Generate 15,000 synthetic salary records → `data/salary_data.csv`
- Train all 5 ML models and evaluate them
- Save the best model → `models/best_model.pkl`

### 3. Launch the app

```bash
streamlit run app.py
```

The app opens at **http://localhost:8501**

> **First-run note:** If you skip step 2, the app will prompt you to initialize automatically via a one-click button.

## 📂 Project Structure

```
job-salary-estimator/
├── data/
│   └── salary_data.csv          # Generated synthetic dataset (15k rows)
├── models/
│   ├── best_model.pkl           # Saved best model (joblib)
│   └── model_results.json       # Training evaluation metrics
├── app.py                       # Main Streamlit application
├── generate_data.py             # Synthetic salary data generator
├── preprocessing.py             # Feature engineering & encoding utilities
├── train.py                     # Model training, evaluation & saving
├── predict.py                   # Prediction utilities & career suggestions
└── requirements.txt
```

## 📊 Input Features

| Feature | Type | Description |
|---------|------|-------------|
| Job Title | Categorical | 30 roles (Software Engineer, Data Scientist, etc.) |
| Years of Experience | Numerical | 0–35 years |
| Education Level | Ordinal | High School → PhD |
| Skills | Multi-select | 30 technical skills |
| Country | Categorical | 20 countries |
| City | Categorical | Dependent on country |
| Industry | Categorical | 15 industries |
| Company Size | Ordinal | Startup → Enterprise |
| Employment Type | Categorical | Full-time, Part-time, Contract, etc. |
| Work Mode | Categorical | Remote / Hybrid / On-site |
| Certifications | Multi-select | 10 industry certifications |

## 🤖 ML Models

| Model | Notes |
|-------|-------|
| Ridge Regression | Strong linear baseline |
| Random Forest | Ensemble, handles non-linearity |
| XGBoost | Gradient boosting, high accuracy |
| LightGBM | Fast gradient boosting |
| CatBoost | Handles categoricals natively (if installed) |

Evaluation metrics: **R²**, **MAE**, **RMSE**, **5-fold CV R²**

## 🎨 Dashboard Tabs

1. **🎯 Predict** — Salary estimate, range, confidence gauge, factor impacts, growth projection, export
2. **📊 Analytics** — 8 Plotly charts: distributions, experience curves, industry/country comparisons
3. **💡 Insights** — Top-paying skills, career paths, experience band analysis, industry heatmap
4. **🔄 Compare** — Side-by-side comparison of 3 role scenarios with radar chart
5. **🤖 Model Results** — Training metrics, model comparison charts

## 📦 Tech Stack

- **Frontend:** Streamlit with custom CSS
- **ML:** scikit-learn, XGBoost, LightGBM, CatBoost
- **Visualization:** Plotly Express & Graph Objects
- **Data:** Pandas, NumPy
- **Export:** fpdf2 (PDF), Pandas (CSV)
- **Model storage:** Joblib

## ⚠️ Disclaimer

Salary predictions are statistical estimates based on synthetic training data. They are intended for educational and portfolio demonstration purposes, not as financial or career advice.
