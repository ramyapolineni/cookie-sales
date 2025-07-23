# GS Cookies Sales Forecasting Platform

Production-grade ML-powered platform for predicting cookie sales using historical data and troop demographics.

---

## Key Features

**Backend (Flask)**
- ML pipeline with scikit-learn (Ridge, Bayesian Ridge, K-Means, Linear Regression), pandas, numpy, statsmodels
- Generates regression lines, scatter plots, confidence bands, and uncertainty bounds for all predictions
- Automated ETL pipeline from Google Drive → PostgreSQL
- PostgreSQL database for data persistence

**Frontend (React)**
- Interactive dashboard with Recharts visualizations
- Real-time prediction interface for new and returning troop analytics

**Infrastructure**
- Render for deployment
- PostgreSQL for persistent storage
- Google Drive API for raw data fetch

## Setup

```bash
# Adjust frontend API base URL in App.js

# Backend
pip install -r requirements.txt
python app.py

# Frontend
cd frontend
npm install
npm start
```

## Tech Stack

- **Frontend**: React, Recharts
- **Backend**: Python (Flask, scikit-learn, pandas, numpy, statsmodels, SQLAlchemy)
- **Infrastructure**: Render, PostgreSQL, Google Drive APIs 

---

_Built for Girl Scouts of Central Indiana by Krenicki Center for BA & ML._
