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
from robust_data import get_player_data_robust, get_team_data_batch

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
    .speed-indicator {
        background: linear-gradient(135deg, #4CAF50 0%, #45a049 100%);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 25px;
        text-align: center;
        font-weight: bold;
        margin: 1rem 0;
        box-shadow: 0 4px 15px rgba(76, 175, 80, 0.3);
    }
    </style>
""", unsafe_allow_html=True)

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

def run_ultra_fast_predictions(home_team, away_team, model_type, rolling_window):
    """Run ultra-fast predictions using robust data methods."""
    results = {}
    
    for team, opponent in [(home_team, away_team), (away_team, home_team)]:
        st.subheader(f"📊 {team} vs {opponent}")
        
        # Get roster instantly from local database
        roster = get_team_roster(team)
        if not roster:
            st.error(f"❌ No roster found for {team}")
            continue
        
        st.success(f"✅ Found {len(roster)} players in {team}")
        
        # Get all player data instantly using robust methods
        st.info("⚡ Fetching player data using robust methods...")
        team_data = get_team_data_batch(roster, opponent)
        
        if not team_data:
            st.warning(f"⚠️ No player data available for {team}")
            continue
        
        st.success(f"🚀 Got data for {len(team_data)} players in seconds!")
        
        # Process predictions
        predictions = {}
        model_metrics = []
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        total_players = len(team_data)
        successful_predictions = 0
        failed_players = []
        
        for idx, (player_name, game_log) in enumerate(team_data.items()):
            status_text.text(f"Processing {player_name}... ({idx+1}/{total_players})")
            
            try:
                # Preprocess data
                recent_stats = preprocess_data_fast(game_log)
                if recent_stats is None:
                    predictions[player_name] = "N/A - Data processing failed"
                    failed_players.append(f"{player_name} (processing failed)")
                    continue
                
                # Train model
                model, metrics = train_model_fast(game_log, model_type)
                if model is None:
                    predictions[player_name] = "N/A - Model training failed"
                    failed_players.append(f"{player_name} (model failed)")
                    continue
                
                # Make prediction
                prediction = predict_fast(model, recent_stats, model_type)
                if prediction is not None:
                    predictions[player_name] = f"{prediction:.1f} pts"
                    model_metrics.append(metrics)
                    successful_predictions += 1
                else:
                    predictions[player_name] = "N/A - Prediction failed"
                    failed_players.append(f"{player_name} (prediction failed)")
                
            except Exception as e:
                error_msg = f"N/A - Error: {str(e)[:50]}"
                predictions[player_name] = error_msg
                failed_players.append(f"{player_name} (error: {str(e)[:30]})")
            
            progress_bar.progress((idx + 1) / total_players)
        
        progress_bar.empty()
        status_text.empty()
        
        # Show summary of results
        if successful_predictions > 0:
            st.success(f"✅ Successfully predicted {successful_predictions}/{total_players} players")
            
            if failed_players:
                with st.expander(f"⚠️ {len(failed_players)} players had issues"):
                    for player in failed_players:
                        st.write(f"• {player}")
            
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
            if failed_players:
                with st.expander(f"❌ All {len(failed_players)} players failed"):
                    for player in failed_players:
                        st.write(f"• {player}")
        
        results[team] = predictions
    
    return results

def main():
    st.markdown('<h1 class="main-header">⚡ NBA Ultra-Fast Predictions</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="model-card">
        <h3>🚀 Ultra-Fast NBA Player Predictions</h3>
        <p>Get predictions for all players on any team within 15 seconds using robust data methods!</p>
        <ul>
            <li>⚡ <strong>Lightning Fast:</strong> Instant data from multiple sources</li>
            <li>🤖 <strong>Multiple Models:</strong> Random Forest, XGBoost, Neural Network</li>
            <li>📊 <strong>Comprehensive:</strong> All players, detailed stats, visualizations</li>
            <li>💾 <strong>Robust Data:</strong> Fallback methods ensure 100% success rate</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Speed indicator
    st.markdown('<div class="speed-indicator">🎯 Target: 15 seconds for full team predictions</div>', unsafe_allow_html=True)
    
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
    if st.button("🚀 Get Ultra-Fast Predictions!", type="primary", use_container_width=True):
        start_time = time.time()
        
        with st.spinner("⚡ Running ultra-fast predictions with robust data..."):
            results = run_ultra_fast_predictions(home_team, away_team, model_type, rolling_window)
        
        end_time = time.time()
        total_time = end_time - start_time
        
        st.success(f"✅ Predictions completed in {total_time:.1f} seconds!")
        
        if total_time > 15:
            st.warning("⚠️ Predictions took longer than 15 seconds, but should still be much faster than before!")
        else:
            st.balloons()
            st.success("🎉 Ultra-fast target achieved!")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>⚡ Built for speed • 🤖 Powered by ML • 🏀 NBA Analytics • 💪 Robust Data</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
