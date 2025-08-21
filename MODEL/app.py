import streamlit as st
import pandas as pd
import numpy as np
from nba_api.stats.static import players, teams
from nba_api.stats.endpoints import playergamelog, commonteamroster
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
    </style>
    """, unsafe_allow_html=True)

# Title
st.markdown('<p class="main-header">NBA Player Points Predictions 🏀</p>', unsafe_allow_html=True)

# Optimized Constants
SEASON_CURRENT = '2024-25'
SEASON_PREVIOUS = '2023-24'
MAX_RETRIES = 2
CACHE_TTL = 7200
API_TIMEOUT = 10
MAX_WORKERS = 3

# Helper Functions with Better Error Handling
@st.cache_data(ttl=CACHE_TTL)
def get_team_abbreviations():
    """Retrieve a list of team abbreviations."""
    return [team['abbreviation'] for team in teams.get_teams()]

@st.cache_data(ttl=CACHE_TTL)
def get_team_roster(team_abbreviation):
    """Fetch team roster with timeout handling."""
    for attempt in range(MAX_RETRIES):
        try:
            team_info = teams.find_team_by_abbreviation(team_abbreviation)
            if not team_info:
                return []

            team_id = team_info['id']
            roster = commonteamroster.CommonTeamRoster(
                team_id=team_id, 
                timeout=API_TIMEOUT
            ).get_data_frames()[0]
            return roster['PLAYER'].tolist()
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                time.sleep(1)
                continue
            return []

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

def fetch_key_players_data(roster, max_players=8):
    """Fetch data for key players only for faster response."""
    # Take first max_players (usually starters + key bench players)
    key_players = roster[:max_players]
    player_data = {}
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_player = {executor.submit(get_player_data, player): player for player in key_players}
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
    """Optimized preprocessing with fewer calculations."""
    try:
        game_log = game_log.copy()
        game_log['GAME_DATE'] = pd.to_datetime(game_log['GAME_DATE'])    
        game_log['HOME_AWAY'] = np.where(game_log['MATCHUP'].str.contains('@'), 'Away', 'Home')
        
        # Convert only essential columns
        essential_cols = ['PTS', 'REB', 'AST', 'FGM', 'FGA', 'FG_PCT', 'FTM', 'FTA', 'FT_PCT']
        for col in essential_cols:
            game_log[col] = pd.to_numeric(game_log[col], errors='coerce')

        game_log = game_log.sort_values('GAME_DATE', ascending=True)
        game_log.fillna(method='ffill', inplace=True)
        
        # Calculate rolling averages for key stats only
        for stat in ['PTS', 'REB', 'AST']:
            game_log[f'AVG_{stat}'] = game_log[stat].rolling(window=rolling_window, min_periods=1).mean()

        game_log.dropna(inplace=True)
        return game_log
    except:
        return None

@st.cache_resource
def train_simple_model(game_log):
    """Simplified model training for faster performance."""
    try:
        features = game_log[['AVG_PTS', 'AVG_REB', 'AVG_AST', 'FGM', 'FGA', 'FG_PCT']]
        target = game_log['PTS']
        
        if len(features) < 5:
            return None, None
            
        X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
        model = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)  # Reduced trees for speed
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        
        return model, rmse
    except:
        return None, None

def predict_points(model, game_log, opponent_team):
    """Simple prediction function."""
    try:
        opponent_games = game_log[game_log['MATCHUP'].str.contains(opponent_team)]
        
        if opponent_games.empty:
            # Use recent averages if no opponent history
            recent_stats = game_log.tail(5)[['AVG_PTS', 'AVG_REB', 'AVG_AST', 'FGM', 'FGA', 'FG_PCT']].mean()
        else:
            recent_stats = opponent_games[['AVG_PTS', 'AVG_REB', 'AVG_AST', 'FGM', 'FGA', 'FG_PCT']].mean()
        
        features = np.array(recent_stats).reshape(1, -1)
        prediction = model.predict(features)[0]
        return max(0, prediction)  # Ensure non-negative
    except:
        return None

def create_prediction_chart(predictions, team, opponent):
    """Create optimized chart."""
    df = pd.DataFrame(list(predictions.items()), columns=['Player', 'Predicted Points'])
    df = df.sort_values('Predicted Points', ascending=False)
    
    chart = alt.Chart(df).mark_bar(color='#FF6B35').encode(
        x=alt.X('Player:N', sort='-y', title='Players'),
        y=alt.Y('Predicted Points:Q', title='Predicted Points'),
        tooltip=['Player:N', 'Predicted Points:Q']
    ).properties(
        title=f'{team} vs {opponent} - Point Predictions',
        width=600,
        height=400
    )
    
    return chart

def run_predictions(home_team, away_team, max_players):
    """Main prediction function with optimized flow."""
    results = {}
    
    for team, opponent in [(home_team, away_team), (away_team, home_team)]:
        st.subheader(f"📊 {team} vs {opponent}")
        
        # Progress tracking
        progress_placeholder = st.empty()
        
        with progress_placeholder.container():
            st.info(f"⏳ Fetching roster for {team}...")
            roster = get_team_roster(team)
            
            if not roster:
                st.error(f"❌ Could not fetch roster for {team}")
                continue
            
            st.success(f"✅ Found {len(roster)} players")
            st.info(f"⏳ Analyzing top {min(max_players, len(roster))} players...")
            
            player_data = fetch_key_players_data(roster, max_players)
            
            if not player_data:
                st.warning(f"⚠️ No player data available for {team}")
                continue
            
            predictions = {}
            progress_bar = st.progress(0)
            
            total_players = len(player_data)
            for idx, (player_name, game_log) in enumerate(player_data.items()):
                processed_log = preprocess_game_log(game_log)
                if processed_log is not None and len(processed_log) >= 3:
                    model, rmse = train_simple_model(processed_log)
                    if model is not None:
                        pred = predict_points(model, processed_log, opponent)
                        if pred is not None:
                            predictions[player_name] = pred
                
                progress_bar.progress((idx + 1) / total_players)
        
        # Clear progress and show results
        progress_placeholder.empty()
        
        if predictions:
            st.success(f"✅ Analysis complete for {team}")
            
            # Display metrics
            sorted_preds = sorted(predictions.items(), key=lambda x: x[1], reverse=True)
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("🏆 Top Scorer", sorted_preds[0][0], f"{sorted_preds[0][1]:.1f} pts")
            with col2:
                avg_pts = np.mean(list(predictions.values()))
                st.metric("📈 Team Average", f"{avg_pts:.1f} pts", f"{len(predictions)} players")
            with col3:
                total_pts = sum(predictions.values())
                st.metric("🎯 Projected Total", f"{total_pts:.0f} pts", "from analyzed players")
            
            # Show chart
            chart = create_prediction_chart(predictions, team, opponent)
            st.altair_chart(chart, use_container_width=True)
            
            results[team] = predictions
        else:
            st.warning(f"⚠️ No predictions available for {team}")
        
        st.markdown("---")
    
    return results

def main():
    """Main application with improved UI."""
    
    # Sidebar
    with st.sidebar:
        st.title("🏀 Controls")
        st.markdown("---")
        
        # Team selection
        st.subheader("🏟️ Select Teams")
        teams_list = get_team_abbreviations()
        
        home_team = st.selectbox("Home Team", teams_list, index=teams_list.index('LAL') if 'LAL' in teams_list else 0)
        away_teams = [t for t in teams_list if t != home_team]
        away_team = st.selectbox("Away Team", away_teams)
        
        st.markdown("---")
        
        # Settings
        st.subheader("⚙️ Settings")
        max_players = st.slider(
            "Players to Analyze", 
            min_value=5, 
            max_value=12, 
            value=8,
            help="More players = slower but more complete analysis"
        )
        
        st.markdown("---")
        
        # Run button
        run_analysis = st.button("🚀 Run Analysis", type="primary", use_container_width=True)
        
        if run_analysis:
            st.success("Analysis started!")
        
        st.markdown("---")
        st.markdown("### ℹ️ About")
        st.info("Fast NBA point predictions using Random Forest ML model with optimized data fetching.")
    
    # Main content
    if run_analysis:
        st.markdown("### 🎯 Prediction Results")
        
        with st.spinner("🔄 Running predictions..."):
            results = run_predictions(home_team, away_team, max_players)
        
        if results:
            st.success("✅ Analysis complete!")
            
            # Summary comparison
            if len(results) == 2:
                st.markdown("### 📊 Team Comparison")
                col1, col2 = st.columns(2)
                
                teams = list(results.keys())
                with col1:
                    team1_total = sum(results[teams[0]].values())
                    st.metric(f"{teams[0]} Projected", f"{team1_total:.0f} pts", 
                             f"{len(results[teams[0]])} players analyzed")
                
                with col2:
                    team2_total = sum(results[teams[1]].values())
                    st.metric(f"{teams[1]} Projected", f"{team2_total:.0f} pts", 
                             f"{len(results[teams[1]])} players analyzed")
        else:
            st.error("❌ Could not generate predictions. Please try again with different teams.")
    
    else:
        # Landing page content
        st.markdown("### 🎮 Welcome to NBA Predictions!")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("""
            **🎯 Features:**
            - Fast ML predictions
            - Real NBA data
            - Interactive charts
            """)
        
        with col2:
            st.markdown("""
            **⚡ Optimized:**
            - Quick API calls
            - Smart caching
            - Responsive UI
            """)
        
        with col3:
            st.markdown("""
            **📊 Analytics:**
            - Player comparisons
            - Team projections
            - Performance metrics
            """)
        
        st.markdown("---")
        st.info("👈 Use the sidebar to select teams and start your analysis!")

if __name__ == "__main__":
    main()
