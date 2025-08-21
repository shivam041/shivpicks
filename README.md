# NBA Player Points Predictions App

A Streamlit application that predicts NBA player performance using machine learning models including Random Forest, XGBoost, Neural Networks, and Monte Carlo simulations.

## Features

- **Multiple ML Models**: Random Forest, XGBoost, and Neural Network regression
- **Monte Carlo Simulation**: Statistical simulation for performance prediction
- **Real-time Data**: Uses NBA API for current and historical data
- **Interactive UI**: Streamlit-based interface with team and player selection
- **Performance Metrics**: Model evaluation with MSE, RMSE, MAE, and R² scores

## Dependencies

The app uses only essential, lightweight libraries:
- `streamlit` - Web interface
- `pandas` - Data manipulation
- `numpy` - Numerical computations
- `scikit-learn` - Machine learning models
- `xgboost` - Gradient boosting
- `altair` - Data visualization
- `nba_api` - NBA statistics
- `requests` - HTTP requests

## Deployment

This app is optimized for Streamlit Cloud deployment with a minimal requirements.txt that avoids compatibility issues.

## Usage

1. Select a home team and away team
2. Choose your preferred ML model
3. Adjust the rolling window size for data analysis
4. Generate predictions and view results

## Technical Details

- Uses modern Streamlit caching (`@st.cache_data`, `@st.cache_resource`)
- Implements concurrent data fetching for performance
- Handles API timeouts and connection errors gracefully
- Responsive design with progress bars and spinners
