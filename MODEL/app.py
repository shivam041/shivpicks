import streamlit as st
import pandas as pd
import numpy as np
from nba_api.stats.static import players
from nba_api.stats.endpoints import playergamelog
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import time
from requests.exceptions import ReadTimeout, ConnectionError
from concurrent.futures import ThreadPoolExecutor, as_completed
import altair as alt
import xgboost as xgb
import warnings
from team_database import get_team_roster, get_all_team_abbreviations, get_team_name

warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="NBA Player Points Predictions",
    page_icon="🏀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced CSS styling
st.markdown("""
    <style>
    .main-header {
        font-size: 48px !important;
        font-weight: bold;
        text-align: center;
        color: #FF6B35;
        margin-bottom: 30px;
    }
    .sub-header {
        font-size: 24px;
        color: #2E86AB;
        margin-bottom: 20px;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .stProgress .st-bo {
        background-color: #FF6B35;
    }
    .model-card {
        background-color: #f8f9fa;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #FF6B35;
        margin: 10px 0;
    }
    </style>
    """, unsafe_allow_html=True)

# Title
st.markdown('<p class="main-header">NBA Player Points Predictions 🏀</p>', unsafe_allow_html=True)

# Constants
SEASON_CURRENT = '2024-25'
SEASON_PREVIOUS = '2023-24'
MAX_RETRIES = 2
CACHE_TTL = 7200
API_TIMEOUT = 15
MAX_WORKERS = 5

# Helper Functions
@st.cache_data(ttl=CACHE_TTL)
def get_player_id(player_name):
    """Fetch player ID with error handling."""
    try:
        player_dict = players.find_players_by_full_name(player_name)
        if not player_dict:
            return None
        return player_dict[0]['id']
    except:
        return None

@st.cache_data(ttl=CACHE_TTL)
def fetch_player_gamelog(player_id, season):
    """Fetch player game log with optimized timeout."""
    for attempt in range(MAX_RETRIES):
        try:
            gamelog = playergamelog.PlayerGameLog(
                player_id=player_id, 
                season=season, 
                timeout=API_TIMEOUT
            ).get_data_frames()[0]
            return gamelog
        except:
            if attempt < MAX_RETRIES - 1:
                time.sleep(0.5)
                continue
            return None

def get_player_data(player_name):
    """Get player data for current and previous season."""
    player_id = get_player_id(player_name)
    if not player_id:
        return None

    gamelogs = []
    for season in [SEASON_CURRENT, SEASON_PREVIOUS]:
        gamelog = fetch_player_gamelog(player_id, season)
        if gamelog is not None and not gamelog.empty:
            gamelogs.append(gamelog)
    
    if gamelogs:
        combined_data = pd.concat(gamelogs, ignore_index=True)
        return combined_data
    return None

def fetch_all_player_data(roster):
    """Fetch data for all players in roster."""
    player_data = {}
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_player = {executor.submit(get_player_data, player): player for player in roster}
        for future in as_completed(future_to_player):
            player = future_to_player[future]
            try:
                data = future.result()
                if data is not None and not data.empty:
                    player_data[player] = data
            except:
                continue
    return player_data

def preprocess_game_log(game_log, rolling_window=5):
    """Preprocess game log data."""
    try:
        game_log = game_log.copy()
        game_log['GAME_DATE'] = pd.to_datetime(game_log['GAME_DATE'])    
        game_log['HOME_AWAY'] = np.where(game_log['MATCHUP'].str.contains('@'), 'Away', 'Home')
        
        # Convert essential columns
        essential_cols = ['PTS', 'REB', 'AST', 'BLK', 'STL', 'FGM', 'FGA', 'FG_PCT', 
                         'FG3M', 'FG3A', 'FG3_PCT', 'FTM', 'FTA', 'FT_PCT', 
                         'OREB', 'DREB', 'TOV', 'PF', 'PLUS_MINUS']
        for col in essential_cols:
            if col in game_log.columns:
                game_log[col] = pd.to_numeric(game_log[col], errors='coerce')

        game_log = game_log.sort_values('GAME_DATE', ascending=True)
        game_log.fillna(method='ffill', inplace=True)
        
        # Calculate rolling averages
        for stat in ['PTS', 'REB', 'AST', 'BLK', 'STL', 'FGM', 'FGA', 'FTM', 'OREB', 'DREB']:
            if stat in game_log.columns:
                game_log[f'AVG_{stat}'] = game_log[stat].rolling(window=rolling_window, min_periods=1).mean()

        game_log.dropna(inplace=True)
        return game_log
    except:
        return None

# ML Model Functions
@st.cache_resource
def train_random_forest_model(game_log):
    """Train Random Forest model."""
    try:
        features = game_log[['AVG_PTS', 'AVG_REB', 'AVG_AST', 'AVG_BLK', 'AVG_STL', 
                            'FGM', 'FGA', 'FG_PCT', 'FG3M', 'FG3A', 'FG3_PCT', 
                            'FTM', 'FTA', 'FT_PCT', 'OREB', 'DREB', 'TOV', 'PF', 'PLUS_MINUS']]
        target = game_log['PTS']
        
        if len(features) < 5:
            return None, None
            
        X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
        model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        metrics = {'MSE': mse, 'RMSE': rmse, 'MAE': mae, 'R2': r2}
        return model, metrics
    except:
        return None, None

@st.cache_resource
def train_xgboost_model(game_log):
    """Train XGBoost model."""
    try:
        features = game_log[['AVG_PTS', 'AVG_REB', 'AVG_AST', 'AVG_BLK', 'AVG_STL', 
                            'FGM', 'FGA', 'FG_PCT', 'FG3M', 'FG3A', 'FG3_PCT', 
                            'FTM', 'FTA', 'FT_PCT', 'OREB', 'DREB', 'TOV', 'PF', 'PLUS_MINUS']]
        target = game_log['PTS']
        
        if len(features) < 5:
            return None, None
            
        X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
        model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42, verbosity=0)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        metrics = {'MSE': mse, 'RMSE': rmse, 'MAE': mae, 'R2': r2}
        return model, metrics
    except:
        return None, None

@st.cache_resource
def train_neural_network_model(game_log):
    """Train Neural Network model."""
    try:
        features = game_log[['AVG_PTS', 'AVG_REB', 'AVG_AST', 'AVG_BLK', 'AVG_STL', 
                            'FGM', 'FGA', 'FG_PCT', 'FG3M', 'FG3A', 'FG3_PCT', 
                            'FTM', 'FTA', 'FT_PCT', 'OREB', 'DREB', 'TOV', 'PF', 'PLUS_MINUS']]
        target = game_log['PTS']
        
        if len(features) < 5:
            return None, None
            
        X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
        model = MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        metrics = {'MSE': mse, 'RMSE': rmse, 'MAE': mae, 'R2': r2}
        return model, metrics
    except:
        return None, None

def predict_with_model(model, average_stats, model_type='rf'):
    """Predict player performance using trained model."""
    try:
        feature_order = ['AVG_PTS', 'AVG_REB', 'AVG_AST', 'AVG_BLK', 'AVG_STL', 
                        'FGM', 'FGA', 'FG_PCT', 'FG3M', 'FG3A', 'FG3_PCT', 
                        'FTM', 'FTA', 'FT_PCT', 'OREB', 'DREB', 'TOV', 'PF', 'PLUS_MINUS']
        features = np.array([average_stats.get(feature, 0) for feature in feature_order]).reshape(1, -1)
        prediction = model.predict(features)[0]
        return max(0, prediction)
    except:
        return None

def monte_carlo_simulation(game_log, opponent_team, num_simulations=1000):
    """Monte Carlo simulation for predictions."""
    try:
        opponent_games = game_log[game_log['MATCHUP'].str.contains(opponent_team)]
        
        if opponent_games.empty:
            return None

        points = opponent_games['PTS'].values
        if len(points) < 2:
            return None
            
        mean_val = points.mean()
        std_val = points.std(ddof=1)
        
        simulated_values = np.random.normal(loc=mean_val, scale=std_val, size=num_simulations)
        simulated_values = np.clip(simulated_values, a_min=0, a_max=None)
        
        return {
            "mean": simulated_values.mean(),
            "median": np.median(simulated_values),
            "std": simulated_values.std(),
            "ci_lower": np.percentile(simulated_values, 2.5),
            "ci_upper": np.percentile(simulated_values, 97.5)
        }
    except:
        return None

def run_model_predictions(home_team, away_team, model_type, rolling_window):
    """Run predictions for a specific model type."""
    results = {}
    
    for team, opponent in [(home_team, away_team), (away_team, home_team)]:
        st.subheader(f"📊 {team} vs {opponent}")
        
        # Get roster from local database (instant!)
        roster = get_team_roster(team)
        if not roster:
            st.error(f"❌ No roster found for {team}")
            continue
            
        st.success(f"✅ Found {len(roster)} players in {team}")
        
        # Fetch player data
        with st.spinner(f"⏳ Fetching data for {len(roster)} players..."):
            player_data = fetch_all_player_data(roster)
        
        if not player_data:
            st.warning(f"⚠️ No player data available for {team}")
            continue
            
        st.info(f"📈 Analyzing {len(player_data)} players with {model_type} model...")
        
        predictions = {}
        model_metrics = []
        
        # Progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        total_players = len(player_data)
        for idx, (player_name, game_log) in enumerate(player_data.items()):
            status_text.text(f"Processing {player_name}... ({idx+1}/{total_players})")
            
            try:
                processed_log = preprocess_game_log(game_log, rolling_window)
                if processed_log is None or len(processed_log) < 3:
                    predictions[player_name] = "N/A - Not enough data"
                    continue

                # Train model based on type
                if model_type == "Random Forest":
                    model, metrics = train_random_forest_model(processed_log)
                elif model_type == "XGBoost":
                    model, metrics = train_xgboost_model(processed_log)
                elif model_type == "Neural Network":
                    model, metrics = train_neural_network_model(processed_log)
                elif model_type == "Monte Carlo":
                    model, metrics = None, None
                
                if model is not None:
                    model_metrics.append(metrics)
                    
                    # Prepare features for prediction
                    opponent_games = processed_log[processed_log['MATCHUP'].str.contains(opponent)]
                    if opponent_games.empty:
                        average_stats = processed_log.iloc[-1][['AVG_PTS', 'AVG_REB', 'AVG_AST', 'AVG_BLK', 'AVG_STL', 
                                                            'FGM', 'FGA', 'FG_PCT', 'FG3M', 'FG3A', 'FG3_PCT', 
                                                            'FTM', 'FTA', 'FT_PCT', 'OREB', 'DREB', 'TOV', 'PF', 'PLUS_MINUS']].to_dict()
                    else:
                        average_stats = opponent_games[['AVG_PTS', 'AVG_REB', 'AVG_AST', 'AVG_BLK', 'AVG_STL', 
                                                       'FGM', 'FGA', 'FG_PCT', 'FG3M', 'FG3A', 'FG3_PCT', 
                                                       'FTM', 'FTA', 'FT_PCT', 'OREB', 'DREB', 'TOV', 'PF', 'PLUS_MINUS']].mean().to_dict()
                    
                    prediction = predict_with_model(model, average_stats, model_type.lower())
                    if prediction is not None:
                        predictions[player_name] = f"{prediction:.1f}"
                    else:
                        predictions[player_name] = "N/A - Prediction failed"
                        
                elif model_type == "Monte Carlo":
                    mc_result = monte_carlo_simulation(processed_log, opponent)
                    if mc_result is not None:
                        predictions[player_name] = f"{mc_result['mean']:.1f} ± {mc_result['std']:.1f}"
                    else:
                        predictions[player_name] = "N/A - Not enough data"
                else:
                    predictions[player_name] = "N/A - Model training failed"
                    
            except Exception as e:
                predictions[player_name] = "N/A - Error processing"
            
            progress_bar.progress((idx + 1) / total_players)
        
        # Clear progress indicators
        progress_bar.empty()
        status_text.empty()
        
        # Display results
        if predictions:
            st.success(f"✅ Analysis complete for {team}")
            
            # Filter out N/A predictions for metrics
            valid_predictions = {k: v for k, v in predictions.items() if not v.startswith("N/A")}
            
            if valid_predictions:
                # Convert to numeric for calculations
                numeric_predictions = {}
                for player, pred in valid_predictions.items():
                    try:
                        if "±" in pred:  # Monte Carlo format
                            numeric_predictions[player] = float(pred.split("±")[0].strip())
                        else:
                            numeric_predictions[player] = float(pred)
                    except:
                        continue
                
                if numeric_predictions:
                    sorted_preds = sorted(numeric_predictions.items(), key=lambda x: x[1], reverse=True)
                    
                    # Display metrics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("🏆 Top Scorer", sorted_preds[0][0], f"{sorted_preds[0][1]:.1f} pts")
                    with col2:
                        avg_pts = np.mean(list(numeric_predictions.values()))
                        st.metric("📈 Team Average", f"{avg_pts:.1f} pts", f"{len(numeric_predictions)} players")
                    with col3:
                        total_pts = sum(numeric_predictions.values())
                        st.metric("🎯 Projected Total", f"{total_pts:.0f} pts", "from analyzed players")
                    
                    # Create chart
                    df_preds = pd.DataFrame(list(numeric_predictions.items()), columns=['Player', 'Predicted Points'])
                    df_preds = df_preds.sort_values('Predicted Points', ascending=False)
                    
                    chart = alt.Chart(df_preds).mark_bar(color='#FF6B35').encode(
                        x=alt.X('Player:N', sort='-y', title='Players'),
                        y=alt.Y('Predicted Points:Q', title='Predicted Points'),
                        tooltip=['Player:N', 'Predicted Points:Q']
                    ).properties(
                        title=f'{model_type}: {team} vs {opponent}',
                        width=600,
                        height=400
                    )
                    
                    st.altair_chart(chart, use_container_width=True)
            
            # Show all predictions in a table
            st.markdown("### 📋 All Player Predictions")
            df_all = pd.DataFrame(list(predictions.items()), columns=['Player', 'Prediction'])
            st.dataframe(df_all, use_container_width=True)
            
            # Show model metrics if available
            if model_metrics and model_type != "Monte Carlo":
                with st.expander("📈 Model Performance Metrics"):
                    avg_rmse = np.mean([m['RMSE'] for m in model_metrics if m])
                    avg_r2 = np.mean([m['R2'] for m in model_metrics if m])
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Average RMSE", f"{avg_rmse:.2f}")
                    with col2:
                        st.metric("Average R² Score", f"{avg_r2:.3f}")
            
            results[team] = predictions
        else:
            st.warning(f"⚠️ No predictions available for {team}")
        
        st.markdown("---")
    
    return results

def main():
    """Main application with all ML models."""
    
    # Sidebar
    with st.sidebar:
        st.title("🏀 NBA Predictions")
        st.markdown("---")
        
        # Model selection
        st.subheader("🤖 Select ML Model")
        model_type = st.selectbox(
            "Choose Model:",
            ["Random Forest", "XGBoost", "Neural Network", "Monte Carlo"],
            help="Different ML approaches for predictions"
        )
        
        st.markdown("---")
        
        # Team selection
        st.subheader("🏟️ Select Teams")
        teams_list = get_all_team_abbreviations()
        
        home_team = st.selectbox("Home Team", teams_list, index=teams_list.index('LAL') if 'LAL' in teams_list else 0)
        away_teams = [t for t in teams_list if t != home_team]
        away_team = st.selectbox("Away Team", away_teams)
        
        st.markdown("---")
        
        # Settings
        st.subheader("⚙️ Settings")
        rolling_window = st.slider(
            "Rolling Window Size", 
            min_value=3, 
            max_value=15, 
            value=5,
            help="Number of recent games to average"
        )
        
        st.markdown("---")
        
        # Model info
        st.markdown("### 📊 Model Info")
        if model_type == "Random Forest":
            st.info("🌳 Ensemble of decision trees for robust predictions")
        elif model_type == "XGBoost":
            st.info("🚀 Gradient boosting for high-accuracy predictions")
        elif model_type == "Neural Network":
            st.info("🧠 Deep learning approach for complex patterns")
        elif model_type == "Monte Carlo":
            st.info("🎲 Statistical simulation for probability distributions")
        
        st.markdown("---")
        
        # Run button
        run_analysis = st.button("🚀 Run Analysis", type="primary", use_container_width=True)
        
        if run_analysis:
            st.success("Analysis started!")
    
    # Main content
    if run_analysis:
        st.markdown(f"### 🎯 {model_type} Predictions")
        
        with st.spinner(f"🔄 Running {model_type} predictions..."):
            results = run_model_predictions(home_team, away_team, model_type, rolling_window)
        
        if results:
            st.success("✅ Analysis complete!")
            
            # Summary comparison
            if len(results) == 2:
                st.markdown("### 📊 Team Comparison")
                col1, col2 = st.columns(2)
                
                teams = list(results.keys())
                for i, team in enumerate(teams):
                    valid_predictions = {k: v for k, v in results[team].items() if not v.startswith("N/A")}
                    if valid_predictions:
                        numeric_values = []
                        for pred in valid_predictions.values():
                            try:
                                if "±" in pred:
                                    numeric_values.append(float(pred.split("±")[0].strip()))
                                else:
                                    numeric_values.append(float(pred))
                            except:
                                continue
                        
                        if numeric_values:
                            team_total = sum(numeric_values)
                            avg_pts = np.mean(numeric_values)
                            
                            with col1 if i == 0 else col2:
                                st.metric(f"{team} Projected", f"{team_total:.0f} pts", 
                                         f"Avg: {avg_pts:.1f} pts ({len(numeric_values)} players)")
        else:
            st.error("❌ Could not generate predictions. Please try again.")
    
    else:
        # Landing page
        st.markdown("### 🎮 Welcome to NBA ML Predictions!")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("""
            **🎯 Features:**
            - Multiple ML models
            - Instant team rosters
            - All players analyzed
            - Interactive charts
            """)
        
        with col2:
            st.markdown("""
            **⚡ Performance:**
            - Local database
            - Fast predictions
            - Smart caching
            - Responsive UI
            """)
        
        with col3:
            st.markdown("""
            **📊 Models:**
            - Random Forest
            - XGBoost
            - Neural Network
            - Monte Carlo
            """)
        
        st.markdown("---")
        st.info("👈 Use the sidebar to select your model and teams!")

if __name__ == "__main__":
    main()
