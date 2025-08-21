import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import time
import altair as alt
import xgboost as xgb
import warnings
from team_database import get_team_roster, get_all_team_abbreviations, get_team_name
from nba_api.stats.static import players
from nba_api.stats.endpoints import playergamelog
import requests

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
        text-align: center;
        color: #1f77b4;
        font-size: 2.5rem;
        font-weight: bold;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .model-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        margin: 1rem 0;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
    }
    .metric-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Constants
CACHE_TTL = 3600  # 1 hour cache
API_TIMEOUT = 10  # 10 second timeout

@st.cache_data(ttl=CACHE_TTL)
def get_player_id_fast(player_name):
    """Get player ID quickly from cached players list."""
    try:
        all_players = players.get_players()
        for player in all_players:
            if player['full_name'].lower() == player_name.lower():
                return player['id']
        return None
    except:
        return None

@st.cache_data(ttl=CACHE_TTL)
def get_player_data_fast(player_name, season='2024-25'):
    """Get player data with minimal processing."""
    try:
        player_id = get_player_id_fast(player_name)
        if not player_id:
            return None
        
        # Get current season data first
        gamelog = playergamelog.PlayerGameLog(
            player_id=player_id, 
            season=season,
            timeout=API_TIMEOUT
        ).get_data_frames()[0]
        
        if len(gamelog) < 3:
            # Try previous season if current season has insufficient data
            prev_season = '2023-24'
            gamelog = playergamelog.PlayerGameLog(
                player_id=player_id, 
                season=prev_season,
                timeout=API_TIMEOUT
            ).get_data_frames()[0]
        
        return gamelog
    except Exception as e:
        st.warning(f"⚠️ Could not fetch data for {player_name}: {str(e)}")
        return None

def preprocess_data_fast(game_log):
    """Fast preprocessing with minimal calculations."""
    try:
        if game_log is None or len(game_log) < 3:
            return None
            
        game_log = game_log.copy()
        
        # Convert only essential columns
        essential_cols = ['PTS', 'REB', 'AST', 'FGM', 'FGA', 'FG_PCT', 'FTM', 'FTA', 'FT_PCT']
        for col in essential_cols:
            if col in game_log.columns:
                game_log[col] = pd.to_numeric(game_log[col], errors='coerce')
        
        # Calculate simple averages (no rolling windows for speed)
        for col in essential_cols:
            if col in game_log.columns:
                game_log[f'AVG_{col}'] = game_log[col].mean()
        
        # Get recent form (last 5 games)
        recent_stats = {}
        for col in essential_cols:
            if col in game_log.columns:
                recent_stats[col] = game_log[col].tail(5).mean()
        
        return recent_stats
    except:
        return None

def train_model_fast(game_log, model_type='rf'):
    """Train model quickly with minimal data."""
    try:
        if game_log is None or len(game_log) < 5:
            return None, None
        
        # Use only essential features
        features = game_log[['PTS', 'REB', 'AST', 'FGM', 'FGA', 'FG_PCT']].copy()
        features = features.fillna(features.mean())
        
        if len(features) < 5:
            return None, None
        
        target = game_log['PTS']
        
        # Simple train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            features, target, test_size=0.2, random_state=42
        )
        
        if model_type == 'rf':
            model = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=1)
        elif model_type == 'xgb':
            model = xgb.XGBRegressor(n_estimators=50, random_state=42, n_jobs=1)
        elif model_type == 'nn':
            model = MLPRegressor(hidden_layer_sizes=(50,), max_iter=200, random_state=42)
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

def predict_fast(model, recent_stats, model_type='rf'):
    """Make prediction quickly."""
    try:
        if model is None or recent_stats is None:
            return None
        
        # Use only available features
        available_features = ['PTS', 'REB', 'AST', 'FGM', 'FGA', 'FG_PCT']
        feature_values = []
        
        for col in available_features:
            if col in recent_stats:
                feature_values.append(recent_stats[col])
            else:
                feature_values.append(0)
        
        features = np.array(feature_values).reshape(1, -1)
        prediction = model.predict(features)[0]
        return max(0, prediction)
    except:
        return None

def run_fast_predictions(home_team, away_team, model_type, rolling_window):
    """Run predictions quickly for both teams."""
    results = {}
    
    for team, opponent in [(home_team, away_team), (away_team, home_team)]:
        st.subheader(f"📊 {team} vs {opponent}")
        
        # Get roster instantly from local database
        roster = get_team_roster(team)
        if not roster:
            st.error(f"❌ No roster found for {team}")
            continue
        
        st.success(f"✅ Found {len(roster)} players in {team}")
        
        # Process players one by one (no threading for reliability)
        predictions = {}
        model_metrics = []
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        total_players = len(roster)
        successful_predictions = 0
        
        for idx, player_name in enumerate(roster):
            status_text.text(f"Processing {player_name}... ({idx+1}/{total_players})")
            
            try:
                # Get player data
                game_log = get_player_data_fast(player_name)
                if game_log is None or len(game_log) < 3:
                    predictions[player_name] = "N/A - Not enough data"
                    continue
                
                # Preprocess data
                recent_stats = preprocess_data_fast(game_log)
                if recent_stats is None:
                    predictions[player_name] = "N/A - Data processing failed"
                    continue
                
                # Train model
                model, metrics = train_model_fast(game_log, model_type)
                if model is None:
                    predictions[player_name] = "N/A - Model training failed"
                    continue
                
                # Make prediction
                prediction = predict_fast(model, recent_stats, model_type)
                if prediction is not None:
                    predictions[player_name] = f"{prediction:.1f} pts"
                    model_metrics.append(metrics)
                    successful_predictions += 1
                else:
                    predictions[player_name] = "N/A - Prediction failed"
                
            except Exception as e:
                predictions[player_name] = f"N/A - Error: {str(e)[:50]}"
            
            progress_bar.progress((idx + 1) / total_players)
        
        progress_bar.empty()
        status_text.empty()
        
        if successful_predictions > 0:
            # Display results
            st.success(f"✅ Successfully predicted {successful_predictions}/{total_players} players")
            
            # Top scorer
            valid_predictions = {k: v for k, v in predictions.items() 
                               if isinstance(v, str) and 'pts' in v}
            if valid_predictions:
                top_scorer = max(valid_predictions.items(), 
                               key=lambda x: float(x[1].split()[0]))
                st.metric("🏆 Top Scorer", f"{top_scorer[0]} ({top_scorer[1]})")
            
            # Team average
            points_list = [float(v.split()[0]) for v in valid_predictions.values()]
            team_avg = np.mean(points_list)
            st.metric("📊 Team Average", f"{team_avg:.1f} pts")
            
            # Projected total
            projected_total = team_avg * 8  # Assume 8 players contribute
            st.metric("🎯 Projected Team Total", f"{projected_total:.1f} pts")
            
            # Chart
            if len(valid_predictions) > 1:
                chart_data = pd.DataFrame([
                    {'Player': k, 'Points': float(v.split()[0])} 
                    for k, v in valid_predictions.items()
                ])
                
                chart = alt.Chart(chart_data).mark_bar().encode(
                    x=alt.X('Player:N', title='Player'),
                    y=alt.Y('Points:Q', title='Predicted Points'),
                    color=alt.Color('Points:Q', scale=alt.Scale(scheme='viridis')),
                    tooltip=['Player', 'Points']
                ).properties(
                    title=f'{team} Player Predictions',
                    width=600,
                    height=400
                )
                
                st.altair_chart(chart, use_container_width=True)
            
            # All predictions table
            st.subheader("📋 All Player Predictions")
            predictions_df = pd.DataFrame([
                {'Player': k, 'Prediction': v} 
                for k, v in predictions.items()
            ])
            st.dataframe(predictions_df, use_container_width=True)
            
            # Model performance
            if model_metrics:
                st.subheader("🤖 Model Performance")
                avg_rmse = np.mean([m['RMSE'] for m in model_metrics])
                avg_r2 = np.mean([m['R2'] for m in model_metrics])
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Average RMSE", f"{avg_rmse:.2f}")
                with col2:
                    st.metric("Average R² Score", f"{avg_r2:.3f}")
        else:
            st.warning(f"⚠️ No successful predictions for {team}")
        
        results[team] = predictions
    
    return results

def main():
    st.markdown('<h1 class="main-header">⚡ NBA Fast Predictions</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="model-card">
        <h3>🚀 Ultra-Fast NBA Player Predictions</h3>
        <p>Get predictions for all players on any team within 30 seconds using advanced ML models!</p>
        <ul>
            <li>⚡ <strong>Lightning Fast:</strong> No more waiting for slow API calls</li>
            <li>🤖 <strong>Multiple Models:</strong> Random Forest, XGBoost, Neural Network</li>
            <li>📊 <strong>Comprehensive:</strong> All players, detailed stats, visualizations</li>
            <li>💾 <strong>Smart Caching:</strong> Faster subsequent runs</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("⚙️ Settings")
    
    model_type = st.sidebar.selectbox(
        "🤖 ML Model",
        ["rf", "xgb", "nn"],
        format_func=lambda x: {
            "rf": "Random Forest",
            "xgb": "XGBoost", 
            "nn": "Neural Network"
        }[x]
    )
    
    rolling_window = st.sidebar.slider(
        "📈 Rolling Window",
        min_value=3,
        max_value=10,
        value=5,
        help="Number of recent games to consider for predictions"
    )
    
    # Team selection
    st.subheader("🏀 Select Teams")
    
    col1, col2 = st.columns(2)
    
    with col1:
        home_team = st.selectbox(
            "🏠 Home Team",
            get_all_team_abbreviations(),
            format_func=lambda x: f"{x} - {get_team_name(x)}"
        )
    
    with col2:
        away_team = st.selectbox(
            "✈️ Away Team", 
            get_all_team_abbreviations(),
            format_func=lambda x: f"{x} - {get_team_name(x)}",
            index=1 if home_team == get_all_team_abbreviations()[0] else 0
        )
    
    if home_team == away_team:
        st.error("❌ Please select different teams!")
        return
    
    # Run predictions button
    if st.button("🚀 Get Fast Predictions!", type="primary", use_container_width=True):
        start_time = time.time()
        
        with st.spinner("⚡ Running ultra-fast predictions..."):
            results = run_fast_predictions(home_team, away_team, model_type, rolling_window)
        
        end_time = time.time()
        total_time = end_time - start_time
        
        st.success(f"✅ Predictions completed in {total_time:.1f} seconds!")
        
        if total_time > 30:
            st.warning("⚠️ Predictions took longer than 30 seconds. Consider reducing team size or using simpler models.")
        else:
            st.balloons()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>⚡ Built for speed • 🤖 Powered by ML • 🏀 NBA Analytics</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
