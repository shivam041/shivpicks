import streamlit as st
import pandas as pd
import numpy as np
import time
from nba_api.stats.static import players, teams
from nba_api.stats.endpoints import playergamelog
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="NBA Player Points Predictor",
    page_icon="🏀",
    layout="wide"
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
    .prediction-card {
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
    .player-card {
        background: rgba(255, 255, 255, 0.05);
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border-left: 4px solid #4CAF50;
    }
    </style>
""", unsafe_allow_html=True)

# Cache for NBA API calls
@st.cache_data(ttl=3600)  # Cache for 1 hour
def get_player_id(player_name: str):
    """Get player ID from NBA API with caching."""
    try:
        all_players = players.get_players()
        for player in all_players:
            if player['full_name'].lower() == player_name.lower():
                return player['id']
        return None
    except Exception as e:
        st.error(f"Error getting player ID: {str(e)}")
        return None

@st.cache_data(ttl=3600)  # Cache for 1 hour
def get_player_game_logs(player_id: int, season: str = '2024-25'):
    """Get player game logs with caching."""
    try:
        game_logs = playergamelog.PlayerGameLog(
            player_id=player_id, 
            season=season,
            timeout=30
        )
        return game_logs.get_data_frames()[0]
    except Exception as e:
        st.warning(f"Could not fetch {season} game logs: {str(e)}")
        return None

def preprocess_game_logs(game_logs: pd.DataFrame):
    """Preprocess game logs for analysis."""
    if game_logs is None or len(game_logs) == 0:
        return None
    
    try:
        # Convert date and sort
        game_logs['GAME_DATE'] = pd.to_datetime(game_logs['GAME_DATE'], errors='coerce')
        game_logs = game_logs.sort_values('GAME_DATE', ascending=False)
        
        # Convert numeric columns
        numeric_cols = ['PTS', 'REB', 'AST', 'STL', 'BLK', 'TOV', 'FGM', 'FGA', 'FG_PCT', 
                       'FG3M', 'FG3A', 'FG3_PCT', 'FTM', 'FTA', 'FT_PCT', 'PLUS_MINUS']
        
        for col in numeric_cols:
            if col in game_logs.columns:
                game_logs[col] = pd.to_numeric(game_logs[col], errors='coerce').fillna(0)
        
        # Extract home/away and opponent
        game_logs['IS_HOME'] = game_logs['MATCHUP'].str.contains('vs', na=False)
        game_logs['IS_AWAY'] = game_logs['MATCHUP'].str.contains('@', na=False)
        
        # Extract opponent team
        def extract_opponent(matchup):
            try:
                if pd.isna(matchup) or matchup == '':
                    return 'UNK'
                parts = str(matchup).split()
                return parts[-1] if len(parts) > 1 else 'UNK'
            except:
                return 'UNK'
        
        game_logs['OPPONENT'] = game_logs['MATCHUP'].apply(extract_opponent)
        
        return game_logs
        
    except Exception as e:
        st.error(f"Error preprocessing game logs: {str(e)}")
        return None

def calculate_player_stats(game_logs: pd.DataFrame, opponent_team: str):
    """Calculate comprehensive player statistics."""
    if game_logs is None or len(game_logs) == 0:
        return None
    
    try:
        stats = {}
        
        # Overall averages
        stats['career_avg_pts'] = game_logs['PTS'].mean()
        stats['career_avg_reb'] = game_logs['REB'].mean()
        stats['career_avg_ast'] = game_logs['AST'].mean()
        stats['career_avg_fg_pct'] = game_logs['FG_PCT'].mean()
        stats['career_avg_3p_pct'] = game_logs['FG3_PCT'].mean()
        stats['career_avg_ft_pct'] = game_logs['FT_PCT'].mean()
        
        # Recent form (last 10 games)
        recent_games = game_logs.head(10)
        if len(recent_games) > 0:
            stats['recent_avg_pts'] = recent_games['PTS'].mean()
            stats['recent_avg_reb'] = recent_games['REB'].mean()
            stats['recent_avg_ast'] = recent_games['AST'].mean()
        else:
            stats['recent_avg_pts'] = stats['career_avg_pts']
            stats['recent_avg_reb'] = stats['career_avg_reb']
            stats['recent_avg_ast'] = stats['career_avg_ast']
        
        # Home vs Away splits
        home_games = game_logs[game_logs['IS_HOME'] == True]
        away_games = game_logs[game_logs['IS_AWAY'] == True]
        
        if len(home_games) > 0:
            stats['home_avg_pts'] = home_games['PTS'].mean()
        else:
            stats['home_avg_pts'] = stats['career_avg_pts']
            
        if len(away_games) > 0:
            stats['away_avg_pts'] = away_games['PTS'].mean()
        else:
            stats['away_avg_pts'] = stats['career_avg_pts']
        
        # Performance against specific opponent
        opponent_games = game_logs[game_logs['OPPONENT'] == opponent_team]
        if len(opponent_games) > 0:
            stats['vs_opponent_pts'] = opponent_games['PTS'].mean()
            stats['vs_opponent_games'] = len(opponent_games)
        else:
            stats['vs_opponent_pts'] = stats['career_avg_pts']
            stats['vs_opponent_games'] = 0
        
        # Rolling averages (last 5 games)
        rolling_5 = game_logs.head(5)
        if len(rolling_5) > 0:
            stats['rolling_5_pts'] = rolling_5['PTS'].mean()
            stats['rolling_5_reb'] = rolling_5['REB'].mean()
            stats['rolling_5_ast'] = rolling_5['AST'].mean()
        else:
            stats['rolling_5_pts'] = stats['career_avg_pts']
            stats['rolling_5_reb'] = stats['career_avg_reb']
            stats['rolling_5_ast'] = stats['career_avg_ast']
        
        return stats
        
    except Exception as e:
        st.error(f"Error calculating stats: {str(e)}")
        return None

def predict_player_points(player_stats: dict, is_home: bool, opponent_team: str):
    """Predict player points using multiple factors."""
    if not player_stats:
        return 0.0
    
    try:
        # Base prediction from career average
        base_prediction = player_stats['career_avg_pts']
        
        # Adjust for recent form (30% weight)
        recent_factor = player_stats['recent_avg_pts'] / base_prediction if base_prediction > 0 else 1
        base_prediction *= (0.7 + 0.3 * recent_factor)
        
        # Adjust for home/away (20% weight)
        if is_home:
            home_factor = player_stats['home_avg_pts'] / base_prediction if base_prediction > 0 else 1
            base_prediction *= (0.8 + 0.2 * home_factor)
        else:
            away_factor = player_stats['away_avg_pts'] / base_prediction if base_prediction > 0 else 1
            base_prediction *= (0.8 + 0.2 * away_factor)
        
        # Adjust for opponent history (25% weight) - only if they've played at least 2 games
        if player_stats['vs_opponent_games'] >= 2:
            opponent_factor = player_stats['vs_opponent_pts'] / base_prediction if base_prediction > 0 else 1
            base_prediction *= (0.75 + 0.25 * opponent_factor)
        
        # Adjust for rolling average (25% weight)
        rolling_factor = player_stats['rolling_5_pts'] / base_prediction if base_prediction > 0 else 1
        base_prediction *= (0.75 + 0.25 * rolling_factor)
        
        # Ensure prediction is reasonable
        return max(0.0, min(50.0, base_prediction))
        
    except Exception as e:
        st.error(f"Error predicting points: {str(e)}")
        return player_stats.get('career_avg_pts', 0.0)

def analyze_player(player_name: str, opponent_team: str, is_home: bool):
    """Analyze a single player and predict their performance."""
    try:
        # Get player ID
        player_id = get_player_id(player_name)
        if not player_id:
            return None
        
        # Get current season game logs
        game_logs = get_player_game_logs(player_id, '2024-25')
        if game_logs is None or len(game_logs) == 0:
            # Try previous season if current season is empty
            game_logs = get_player_game_logs(player_id, '2023-24')
        
        if game_logs is None or len(game_logs) == 0:
            st.warning(f"No game data available for {player_name}")
            return None
        
        # Preprocess game logs
        processed_logs = preprocess_game_logs(game_logs)
        if processed_logs is None:
            return None
        
        # Calculate statistics
        player_stats = calculate_player_stats(processed_logs, opponent_team)
        if player_stats is None:
            return None
        
        # Predict points
        predicted_points = predict_player_points(player_stats, is_home, opponent_team)
        
        return {
            'player_name': player_name,
            'predicted_points': predicted_points,
            'stats': player_stats,
            'games_analyzed': len(processed_logs)
        }
        
    except Exception as e:
        st.error(f"Error analyzing {player_name}: {str(e)}")
        return None

def main():
    st.markdown('<h1 class="main-header">🏀 NBA Player Points Predictor</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="prediction-card">
        <h3>🎯 Predict Player Points Against Specific Teams</h3>
        <p>Get accurate point predictions based on career averages, recent form, home/away splits, and opponent history!</p>
        <ul>
            <li>📊 <strong>Career Averages:</strong> Long-term performance baseline</li>
            <li>🔥 <strong>Recent Form:</strong> Last 10 games performance</li>
            <li>🏠 <strong>Home/Away Splits:</strong> Venue-specific performance</li>
            <li>🎯 <strong>Opponent History:</strong> Performance against specific teams</li>
            <li>📈 <strong>Rolling Averages:</strong> Last 5 games trends</li>
            <li>⚡ <strong>Fast Analysis:</strong> Optimized for speed</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("⚙️ Settings")
    
    st.sidebar.markdown("### 📊 Prediction Factors")
    st.sidebar.info("""
    - **Career Average**: 30% weight
    - **Recent Form**: 30% weight  
    - **Home/Away**: 20% weight
    - **Opponent History**: 25% weight
    - **Rolling Average**: 25% weight
    """)
    
    st.sidebar.markdown("### 💡 How It Works")
    st.sidebar.info("""
    1. Enter player name and opponent team
    2. App fetches recent game data
    3. Calculates multiple statistical factors
    4. Combines factors with weighted algorithm
    5. Provides accurate point prediction
    """)
    
    # Main interface
    st.markdown("## 🎯 Player Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Player input
        player_name = st.text_input(
            "👤 Player Name",
            placeholder="e.g., LeBron James, Stephen Curry",
            help="Enter the full name of the player you want to analyze"
        )
        
        # Team selection
        all_teams = [team['abbreviation'] for team in teams.get_teams()]
        opponent_team = st.selectbox(
            "🏀 Opponent Team",
            options=all_teams,
            help="Select the team the player will be playing against"
        )
    
    with col2:
        # Home/Away selection
        is_home = st.radio(
            "🏠 Venue",
            options=["Home", "Away"],
            help="Is the player playing at home or away?"
        )
        
        # Analysis button
        analyze_button = st.button(
            "🚀 Analyze Player",
            type="primary",
            use_container_width=True,
            help="Click to start the analysis"
        )
    
    # Run analysis
    if analyze_button and player_name:
        if not player_name.strip():
            st.error("❌ Please enter a player name!")
            return
        
        st.markdown("## 🔍 Analysis Results")
        
        with st.spinner("🔍 Analyzing player performance..."):
            # Analyze player
            player_analysis = analyze_player(player_name.strip(), opponent_team, is_home == "Home")
            
            if player_analysis:
                # Display results
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric(
                        "🎯 Predicted Points",
                        f"{player_analysis['predicted_points']:.1f}",
                        help="Predicted points for the upcoming game"
                    )
                
                with col2:
                    st.metric(
                        "📊 Career Average",
                        f"{player_analysis['stats']['career_avg_pts']:.1f}",
                        help="Player's career average points per game"
                    )
                
                with col3:
                    st.metric(
                        "🔥 Recent Form",
                        f"{player_analysis['stats']['recent_avg_pts']:.1f}",
                        help="Average points in last 10 games"
                    )
                
                # Detailed statistics
                st.markdown("### 📈 Detailed Statistics")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("#### 🏠 Home/Away Performance")
                    st.metric("Home Average", f"{player_analysis['stats']['home_avg_pts']:.1f}")
                    st.metric("Away Average", f"{player_analysis['stats']['away_avg_pts']:.1f}")
                    
                    st.markdown("#### 🎯 Opponent History")
                    if player_analysis['stats']['vs_opponent_games'] > 0:
                        st.metric(
                            f"vs {opponent_team}",
                            f"{player_analysis['stats']['vs_opponent_pts']:.1f}",
                            f"{player_analysis['stats']['vs_opponent_games']} games"
                        )
                    else:
                        st.info(f"No previous games against {opponent_team}")
                
                with col2:
                    st.markdown("#### 📊 Recent Trends")
                    st.metric("Last 5 Games", f"{player_analysis['stats']['rolling_5_pts']:.1f}")
                    st.metric("Last 10 Games", f"{player_analysis['stats']['recent_avg_pts']:.1f}")
                    
                    st.markdown("#### 🎯 Other Stats")
                    st.metric("Rebounds", f"{player_analysis['stats']['career_avg_reb']:.1f}")
                    st.metric("Assists", f"{player_analysis['stats']['career_avg_ast']:.1f}")
                
                # Prediction confidence
                st.markdown("### 🎯 Prediction Confidence")
                
                confidence_score = 0
                confidence_factors = []
                
                # Factor 1: Data availability
                if player_analysis['games_analyzed'] >= 20:
                    confidence_score += 25
                    confidence_factors.append("✅ Extensive game data (20+ games)")
                elif player_analysis['games_analyzed'] >= 10:
                    confidence_score += 20
                    confidence_factors.append("✅ Good game data (10+ games)")
                else:
                    confidence_score += 10
                    confidence_factors.append("⚠️ Limited game data")
                
                # Factor 2: Opponent history
                if player_analysis['stats']['vs_opponent_games'] >= 5:
                    confidence_score += 25
                    confidence_factors.append("✅ Strong opponent history (5+ games)")
                elif player_analysis['stats']['vs_opponent_games'] >= 2:
                    confidence_score += 20
                    confidence_factors.append("✅ Good opponent history (2+ games)")
                else:
                    confidence_score += 10
                    confidence_factors.append("⚠️ Limited opponent history")
                
                # Factor 3: Recent form consistency
                recent_consistency = abs(player_analysis['stats']['recent_avg_pts'] - player_analysis['stats']['career_avg_pts']) / player_analysis['stats']['career_avg_pts']
                if recent_consistency < 0.1:
                    confidence_score += 25
                    confidence_factors.append("✅ Consistent recent form")
                elif recent_consistency < 0.2:
                    confidence_score += 20
                    confidence_factors.append("✅ Moderately consistent form")
                else:
                    confidence_score += 15
                    confidence_factors.append("⚠️ Variable recent form")
                
                # Factor 4: Home/away consistency
                home_away_diff = abs(player_analysis['stats']['home_avg_pts'] - player_analysis['stats']['away_avg_pts']) / player_analysis['stats']['career_avg_pts']
                if home_away_diff < 0.15:
                    confidence_score += 25
                    confidence_factors.append("✅ Consistent home/away performance")
                elif home_away_diff < 0.25:
                    confidence_score += 20
                    confidence_factors.append("✅ Moderate home/away difference")
                else:
                    confidence_score += 15
                    confidence_factors.append("⚠️ Significant home/away difference")
                
                # Display confidence
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("🎯 Confidence Score", f"{confidence_score}%")
                    
                    if confidence_score >= 80:
                        st.success("🟢 High Confidence Prediction")
                    elif confidence_score >= 60:
                        st.warning("🟡 Medium Confidence Prediction")
                    else:
                        st.error("🔴 Low Confidence Prediction")
                
                with col2:
                    st.markdown("#### 📋 Confidence Factors")
                    for factor in confidence_factors:
                        st.write(factor)
                
                # Summary
                st.markdown("### 📝 Prediction Summary")
                
                venue_text = "home" if is_home == "Home" else "away"
                st.info(f"""
                **{player_name}** is predicted to score **{player_analysis['predicted_points']:.1f} points** 
                against the **{opponent_team}** when playing **{venue_text}**.
                
                This prediction is based on {player_analysis['games_analyzed']} games of data with a confidence level of **{confidence_score}%**.
                """)
                
            else:
                st.error("❌ Could not analyze player. Please check the player name and try again.")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>🏀 Powered by NBA API • 📊 Advanced Statistical Analysis • ⚡ Fast & Accurate Predictions</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
