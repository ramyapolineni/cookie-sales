# GS Cookies Sales Forecasting Platform

Production-grade ML-powered platform for predicting cookie sales using historical data and troop demographics.

---

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

## Key Features

- **Hybrid ML Model Selection**: Automatically chooses the optimal model (Ridge, Bayesian Ridge, K-Means, Linear Regression) for each troop/cookie
- **Confidence Intervals**: Generates statistical uncertainty bounds for all predictions
- **Analytics**: Regression lines, scatter plots, and confidence bands for transparency and analysis
- **RESTful API**: Endpoints for predictions, analytics, and data retrieval
- **Automated Data Pipeline**: ETL from Google Drive to PostgreSQL, with scheduled retraining and updates.

## Architecture

**Backend (Flask)**
- ML pipeline with scikit-learn (Ridge, Bayesian Ridge, K-Means clustering), pandas, numpy, statsmodels
- Automated ETL pipeline from Google Drive → PostgreSQL
- PostgreSQL database for data persistence

**Frontend (React)**
- Interactive dashboard with Recharts visualizations
- Real-time prediction interface for new and returning troop analytics

**Infrastructure**
- Render for deployment
- PostgreSQL for persistent storage
- Google Drive API for raw data fetch

## Tech Stack

- **Backend**: Python - Flask, scikit-learn, pandas, numpy, statsmodels, SQLAlchemy
- **Frontend**: React, Recharts
- **Infrastructure**: Render, PostgreSQL, Google Drive APIs

---

_Built for Girl Scouts of Central Indiana by Krenicki Center for BA & ML._
