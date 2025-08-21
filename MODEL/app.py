import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import altair as alt
import xgboost as xgb
import warnings
from team_database import get_team_roster, get_all_team_abbreviations, get_team_name
from optimized_api import nba_client, get_team_predictions_fast, get_players_data_batch_fast

warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="NBA Fast Predictions",
    page_icon="⚡",
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
    .speed-indicator {
        background-color: #4CAF50;
        color: white;
        padding: 10px;
        border-radius: 5px;
        text-align: center;
        font-weight: bold;
        margin: 10px 0;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    </style>
    """, unsafe_allow_html=True)

# Title
st.markdown('<p class="main-header">⚡ NBA Fast Predictions 🏀</p>', unsafe_allow_html=True)

# Speed indicator
st.markdown('<div class="speed-indicator">🚀 Optimized for 30-second predictions!</div>', unsafe_allow_html=True)

# Constants
SEASON_CURRENT = '2024-25'
MAX_WORKERS = 3
BATCH_SIZE = 5

def preprocess_game_log_fast(game_log, rolling_window=5):
    """Fast preprocessing with minimal calculations."""
    try:
        game_log = game_log.copy()
        game_log['GAME_DATE'] = pd.to_datetime(game_log['GAME_DATE'])
        
        # Convert only essential columns
        essential_cols = ['PTS', 'REB', 'AST', 'FGM', 'FGA', 'FG_PCT', 'FTM', 'FTA', 'FT_PCT']
        for col in essential_cols:
            if col in game_log.columns:
                game_log[col] = pd.to_numeric(game_log[col], errors='coerce')

        game_log = game_log.sort_values('GAME_DATE', ascending=True)
        game_log.fillna(method='ffill', inplace=True)
        
        # Calculate only essential rolling averages
        for stat in ['PTS', 'REB', 'AST']:
            if stat in game_log.columns:
                game_log[f'AVG_{stat}'] = game_log[stat].rolling(window=rolling_window, min_periods=1).mean()

        game_log.dropna(inplace=True)
        return game_log
    except:
        return None

def train_model_fast(game_log, model_type='rf'):
    """Fast model training with minimal features."""
    try:
        # Use only essential features for speed
        feature_cols = ['AVG_PTS', 'AVG_REB', 'AVG_AST', 'FGM', 'FGA', 'FG_PCT', 'FTM', 'FTA', 'FT_PCT']
        available_features = [col for col in feature_cols if col in game_log.columns]
        
        if len(available_features) < 3 or len(game_log) < 5:
            return None, None
            
        features = game_log[available_features]
        target = game_log['PTS']
        
        X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
        
        if model_type == 'rf':
            model = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
        elif model_type == 'xgb':
            model = xgb.XGBRegressor(n_estimators=50, learning_rate=0.1, random_state=42, verbosity=0)
        elif model_type == 'nn':
            model = MLPRegressor(hidden_layer_sizes=(50, 25), max_iter=200, random_state=42)
        else:
            return None, None
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        metrics = {'RMSE': rmse, 'R2': r2}
        return model, metrics
    except:
        return None, None

def predict_fast(model, game_log, opponent_team):
    """Fast prediction using minimal data."""
    try:
        if game_log is None or len(game_log) < 3:
            return None
            
        # Use recent averages for prediction
        recent_stats = game_log.tail(3)[['AVG_PTS', 'AVG_REB', 'AVG_AST', 'FGM', 'FGA', 'FG_PCT', 'FTM', 'FTA', 'FT_PCT']].mean()
        
        # Check which features the model expects
        feature_cols = ['AVG_PTS', 'AVG_REB', 'AVG_AST', 'FGM', 'FGA', 'FG_PCT', 'FTM', 'FTA', 'FT_PCT']
        available_features = [col for col in feature_cols if col in recent_stats.index]
        
        if len(available_features) < 3:
            return None
            
        features = np.array([recent_stats[col] for col in available_features]).reshape(1, -1)
        prediction = model.predict(features)[0]
        return max(0, prediction)
    except:
        return None

def run_ultra_fast_predictions(home_team, away_team, model_type, rolling_window):
    """Ultra-fast predictions using optimized methods."""
    results = {}
    start_time = time.time()
    
    for team, opponent in [(home_team, away_team), (away_team, home_team)]:
        st.subheader(f"📊 {team} vs {opponent}")
        
        # Get roster instantly from local database
        roster = get_team_roster(team)
        if not roster:
            st.error(f"❌ No roster found for {team}")
            continue
            
        st.success(f"✅ Found {len(roster)} players in {team}")
        
        # Try ultra-fast method first
        st.info("⚡ Using ultra-fast prediction method...")
        
        try:
            # Get quick predictions using team-level data
            quick_predictions = get_team_predictions_fast(roster, opponent)
            
            if quick_predictions and any(v > 0 for v in quick_predictions.values()):
                st.success("🚀 Ultra-fast predictions generated!")
                
                # Display results immediately
                valid_predictions = {k: v for k, v in quick_predictions.items() if v > 0}
                
                if valid_predictions:
                    sorted_preds = sorted(valid_predictions.items(), key=lambda x: x[1], reverse=True)
                    
                    # Quick metrics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("🏆 Top Scorer", sorted_preds[0][0], f"{sorted_preds[0][1]:.1f} pts")
                    with col2:
                        avg_pts = np.mean(list(valid_predictions.values()))
                        st.metric("📈 Team Average", f"{avg_pts:.1f} pts", f"{len(valid_predictions)} players")
                    with col3:
                        total_pts = sum(valid_predictions.values())
                        st.metric("🎯 Projected Total", f"{total_pts:.0f} pts", "from analyzed players")
                    
                    # Quick chart
                    df_preds = pd.DataFrame(list(valid_predictions.items()), columns=['Player', 'Predicted Points'])
                    df_preds = df_preds.sort_values('Predicted Points', ascending=False)
                    
                    chart = alt.Chart(df_preds).mark_bar(color='#4CAF50').encode(
                        x=alt.X('Player:N', sort='-y', title='Players'),
                        y=alt.Y('Predicted Points:Q', title='Predicted Points'),
                        tooltip=['Player:N', 'Predicted Points:Q']
                    ).properties(
                        title=f'⚡ Fast Predictions: {team} vs {opponent}',
                        width=600,
                        height=400
                    )
                    
                    st.altair_chart(chart, use_container_width=True)
                    
                    results[team] = valid_predictions
                    
                    # Show all predictions
                    st.markdown("### 📋 All Player Predictions")
                    df_all = pd.DataFrame(list(quick_predictions.items()), columns=['Player', 'Prediction'])
                    st.dataframe(df_all, use_container_width=True)
                    
                    continue  # Skip to next team
                    
        except Exception as e:
            st.warning(f"⚠️ Ultra-fast method failed, falling back to standard method...")
        
        # Fallback to standard method if ultra-fast fails
        st.info(f"📈 Fetching detailed data for {len(roster)} players...")
        
        # Fetch player data in batches
        player_data = get_players_data_batch_fast(roster)
        
        if not player_data:
            st.warning(f"⚠️ No player data available for {team}")
            continue
            
        st.info(f"🤖 Training {model_type} models...")
        
        predictions = {}
        model_metrics = []
        
        # Progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        total_players = len(player_data)
        for idx, (player_name, game_log) in enumerate(player_data.items()):
            status_text.text(f"Processing {player_name}... ({idx+1}/{total_players})")
            
            try:
                processed_log = preprocess_game_log_fast(game_log, rolling_window)
                if processed_log is None or len(processed_log) < 3:
                    predictions[player_name] = "N/A - Not enough data"
                    continue

                # Train model based on type
                model_type_short = 'rf' if 'Random Forest' in model_type else 'xgb' if 'XGBoost' in model_type else 'nn'
                model, metrics = train_model_fast(processed_log, model_type_short)
                
                if model is not None:
                    model_metrics.append(metrics)
                    prediction = predict_fast(model, processed_log, opponent)
                    
                    if prediction is not None:
                        predictions[player_name] = f"{prediction:.1f}"
                    else:
                        predictions[player_name] = "N/A - Prediction failed"
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
            
            results[team] = predictions
        else:
            st.warning(f"⚠️ No predictions available for {team}")
        
        st.markdown("---")
    
    # Show timing
    elapsed_time = time.time() - start_time
    st.success(f"⏱️ Total analysis time: {elapsed_time:.1f} seconds")
    
    if elapsed_time <= 30:
        st.balloons()
        st.success("🎉 Target achieved! Predictions generated within 30 seconds!")
    else:
        st.warning(f"⚠️ Target missed by {elapsed_time - 30:.1f} seconds")
    
    return results

def main():
    """Main application optimized for speed."""
    
    # Sidebar
    with st.sidebar:
        st.title("⚡ Fast NBA Predictions")
        st.markdown("---")
        
        # Model selection
        st.subheader("🤖 Select ML Model")
        model_type = st.selectbox(
            "Choose Model:",
            ["Random Forest", "XGBoost", "Neural Network"],
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
            max_value=10, 
            value=5,
            help="Number of recent games to average (smaller = faster)"
        )
        
        st.markdown("---")
        
        # Speed info
        st.markdown("### 🚀 Speed Features")
        st.info("""
        - **Local team database** - Instant roster loading
        - **Batch API calls** - Multiple players at once
        - **Smart caching** - 24-hour data retention
        - **Rate limiting** - Prevents API blocks
        - **Fallback methods** - Quick predictions when possible
        """)
        
        st.markdown("---")
        
        # Run button
        run_analysis = st.button("⚡ Run Ultra-Fast Analysis", type="primary", use_container_width=True)
        
        if run_analysis:
            st.success("🚀 Ultra-fast analysis started!")
    
    # Main content
    if run_analysis:
        st.markdown(f"### 🎯 {model_type} Predictions")
        
        with st.spinner(f"⚡ Running ultra-fast {model_type} predictions..."):
            results = run_ultra_fast_predictions(home_team, away_team, model_type, rolling_window)
        
        if results:
            st.success("✅ Ultra-fast analysis complete!")
            
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
        st.markdown("### 🎮 Welcome to NBA Ultra-Fast Predictions!")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("""
            **⚡ Speed Features:**
            - Local team database
            - Batch API processing
            - Smart caching
            - Rate limiting
            """)
        
        with col2:
            st.markdown("""
            **🤖 ML Models:**
            - Random Forest
            - XGBoost
            - Neural Network
            - Optimized training
            """)
        
        with col3:
            st.markdown("""
            **🎯 Performance:**
            - Target: 30 seconds
            - All players analyzed
            - Interactive charts
            - Complete results
            """)
        
        st.markdown("---")
        st.info("👈 Use the sidebar to start your ultra-fast analysis!")

if __name__ == "__main__":
    main()
