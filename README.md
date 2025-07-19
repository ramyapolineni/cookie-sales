# GS Cookies Sales Forecasting Platform

ML-powered platform for predicting Girl Scout cookie sales using historical data and troop demographics.

## What It Does

- **Sales Forecasting**: Predicts cases sold per cookie type for individual troops using Ridge regression, Bayesian Ridge, and clustering-based ensemble methods
- **Confidence Intervals**: Generates statistical uncertainty bounds for all predictions
- **Analytics**: Provides historical performance analysis and regression insights
- **Real-time API**: RESTful endpoints for predictions and data retrieval

## Architecture

**Backend (Flask)**
- ML pipeline with scikit-learn (Ridge, Bayesian Ridge, K-Means clustering)
- Automated ETL pipeline fetching data from Google Drive
- PostgreSQL database for data persistence

**Frontend (React)**
- Interactive dashboard with Recharts visualizations
- Real-time prediction interface

## Data Pipeline
1. Fetch raw sales/participation data from Google Drive
2. Clean and standardize data
3. Apply cookie type mapping and normalization
4. Merge with historical data (2020-2024)
5. Generate ML-ready datasets

## Setup

```bash
# Backend
pip install -r requirements.txt
python app.py

# Frontend
cd frontend
npm install
npm start
```

## Key API Endpoints

- `POST /api/predict`: Generate sales forecasts for troops
- `GET /api/history/<troop_id>`: Historical performance data
- `GET /api/regression/<troop_id>`: Regression analysis
- `GET /api/cookie_breakdown/<troop_id>`: Product mix analysis

## Tech Stack

- **Backend**: Python, Flask, scikit-learn, pandas, numpy, statsmodels
- **Frontend**: React, Recharts
- **Infrastructure**: Render, PostgreSQL, Google Drive API 
