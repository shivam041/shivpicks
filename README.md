# 🏀 NBA Player Points Predictor

A fast, accurate, and user-friendly Streamlit application that predicts NBA player points against specific teams using advanced statistical analysis.

## 🎯 Features

- **Fast Analysis**: Optimized for speed with intelligent caching
- **Accurate Predictions**: Uses multiple statistical factors with weighted algorithms
- **User-Friendly Interface**: Clean, modern Streamlit UI
- **Real-time Data**: Live data from NBA API
- **Comprehensive Stats**: Career averages, recent form, home/away splits, and opponent history

## 📊 Prediction Factors

The app uses a sophisticated algorithm that combines multiple factors:

- **Career Average** (30% weight): Long-term performance baseline
- **Recent Form** (30% weight): Last 10 games performance
- **Home/Away** (20% weight): Venue-specific performance differences
- **Opponent History** (25% weight): Performance against specific teams
- **Rolling Average** (25% weight): Last 5 games trends

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the App

```bash
cd MODEL
streamlit run app.py
```

### 3. Use the App

1. Enter the player's full name (e.g., "LeBron James", "Stephen Curry")
2. Select the opponent team from the dropdown
3. Choose whether the player is playing at home or away
4. Click "Analyze Player" to get predictions

## 💡 How It Works

1. **Data Collection**: Fetches recent game data from NBA API
2. **Statistical Analysis**: Calculates multiple performance metrics
3. **Weighted Algorithm**: Combines factors using sophisticated weighting
4. **Confidence Scoring**: Provides confidence level for each prediction
5. **Detailed Results**: Shows comprehensive breakdown of all factors

## 🔧 Technical Details

- **Framework**: Streamlit
- **Data Source**: NBA API (swar/nba_api)
- **Caching**: Intelligent caching for performance optimization
- **Error Handling**: Robust error handling for network issues
- **Responsive Design**: Works on desktop and mobile devices

## 📈 Example Output

The app provides:
- **Predicted Points**: Main prediction for the upcoming game
- **Career Average**: Player's long-term scoring average
- **Recent Form**: Performance in last 10 games
- **Home/Away Splits**: Venue-specific performance
- **Opponent History**: Past performance against the specific team
- **Confidence Score**: Reliability indicator for the prediction
- **Detailed Statistics**: Comprehensive breakdown of all metrics

## 🎯 Use Cases

- **Fantasy Sports**: Make informed decisions for fantasy basketball
- **Sports Betting**: Analyze player performance trends
- **Game Analysis**: Understand player matchups and historical performance
- **Player Research**: Study player statistics and trends

## ⚡ Performance

- **First Run**: May take 10-15 seconds to fetch initial data
- **Subsequent Runs**: Much faster due to intelligent caching
- **Data Freshness**: Cached data expires after 1 hour for accuracy
- **Network Optimization**: Handles API timeouts and retries gracefully

## 🛠️ Requirements

- Python 3.8+
- Streamlit 1.40.0+
- NBA API 1.1.0+
- Pandas 2.0.0+
- NumPy 1.24.0+
- Scikit-learn 1.3.0+

## 🔒 Data Privacy

- No personal data is collected or stored
- All data comes from public NBA API
- Caching is local and temporary
- No external data sharing

## 🆘 Troubleshooting

### Common Issues

1. **"Player not found"**: Check spelling and use full name
2. **"No game data available"**: Player may not have recent games
3. **Slow loading**: First run takes longer, subsequent runs are faster
4. **API errors**: Network issues, try again in a few minutes

### Performance Tips

- Use exact player names for best results
- Ensure stable internet connection
- Clear cache if experiencing issues
- Restart app if it becomes unresponsive

## 📝 License

This project is open source and available under the MIT License.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues and pull requests.

---

**🏀 Powered by NBA API • 📊 Advanced Statistical Analysis • ⚡ Fast & Accurate Predictions**
