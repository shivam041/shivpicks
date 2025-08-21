import streamlit as st
import pandas as pd
import numpy as np
from nba_api.stats.static import players, teams
from nba_api.stats.endpoints import playergamelog, commonteamroster
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import time
from requests.exceptions import ReadTimeout, ConnectionError
from concurrent.futures import ThreadPoolExecutor

# Set page config
st.set_page_config(
    page_title="NBA Game Predictions",
    page_icon="🏀",
    layout="wide"
)

# Add CSS styling
st.markdown("""
    <style>
    .big-font {
        font-size:30px !important;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)

# Title
st.markdown('<p class="big-font">NBA Game Predictions 🏀</p>', unsafe_allow_html=True)

# Common Functions
def get_team_roster(team_abbreviation):
    try:
        team_info = teams.find_team_by_abbreviation(team_abbreviation)
        if not team_info:
            st.error(f"Team '{team_abbreviation}' not found.")
            return []
        
        team_id = team_info['id']
        roster = commonteamroster.CommonTeamRoster(team_id=team_id).get_data_frames()[0]
        return roster['PLAYER'].tolist()
    except Exception as e:
        st.error(f"Error getting roster: {e}")
        return []
    
@st.cache_data
def get_player_data(player_name, max_retries=3):
    for attempt in range(max_retries):
        try:
            player_dict = players.find_players_by_full_name(player_name)
            if not player_dict:
                st.error(f"Player '{player_name}' not found.")
                return None
            
            player_id = player_dict[0]['id']
            
            # Get data for both seasons
            current_season = playergamelog.PlayerGameLog(
                player_id=player_id, 
                season='2024-25',
                timeout=60
            ).get_data_frames()[0]
            
            previous_season = playergamelog.PlayerGameLog(
                player_id=player_id, 
                season='2023-24',
                timeout=60
            ).get_data_frames()[0]
            
            # Combine the data
            combined_data = pd.concat([current_season, previous_season], ignore_index=True)
            return combined_data
            
        except (ReadTimeout, ConnectionError) as e:
            if attempt < max_retries - 1:
                time.sleep(2)
            else:
                st.error(f"Failed to fetch data for {player_name} after {max_retries} attempts")
                return None
    pass

def fetch_all_player_data(roster):
    with ThreadPoolExecutor() as executor:
        player_data = list(executor.map(get_player_data, roster))
    return player_data

def preprocess_game_log(game_log):
    game_log['GAME_DATE'] = pd.to_datetime(game_log['GAME_DATE'])    
    game_log['HOME_AWAY'] = np.where(game_log['MATCHUP'].str.contains('@'), 'Away', 'Home')
    
    for col in ['PTS', 'REB', 'AST', 'BLK', 'STL', 'FGM', 'FGA', 'FG_PCT', 
                'FG3M', 'FG3A', 'FG3_PCT', 'FTM', 'FTA', 'FT_PCT', 
                'OREB', 'DREB', 'TOV', 'PF', 'PLUS_MINUS']:
        game_log[col] = game_log[col].astype(float)

    game_log = game_log.sort_values('GAME_DATE', ascending=True)

    rolling_window = 5
    for stat in ['PTS', 'REB', 'AST', 'BLK', 'STL', 'FGM', 'FGA', 'FTM', 'OREB', 'DREB']:
        game_log[f'AVG_{stat}'] = game_log[stat].rolling(window=rolling_window).mean()

    game_log.dropna(inplace=True)

    return game_log

# Linear Regression Functions
def train_model(game_log):
    features = game_log[['AVG_PTS', 'AVG_REB', 'AVG_AST', 'AVG_BLK', 'AVG_STL', 
                         'FGM', 'FGA', 'FG_PCT', 'FG3M', 'FG3A', 'FG3_PCT', 
                         'FTM', 'FTA', 'FT_PCT', 'OREB', 'DREB', 'TOV', 'PF', 'PLUS_MINUS']]
    target = game_log['PTS']
    
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    st.write(f'Test MSE: {mse:.2f} (± {np.sqrt(mse):.2f})')  # Display MSE with ± notation

    return model, mse  # Return model and MSE

def predict_performance_against_team(model, game_log, opponent_team):
    opponent_games = game_log[game_log['MATCHUP'].str.contains(opponent_team)]
    
    if opponent_games.empty:
        st.warning(f"No previous games found against team: {opponent_team}.")
        return None

    avg_stats = opponent_games[['PTS', 'REB', 'AST', 'BLK', 'STL', 
                               'FGM', 'FGA', 'FG_PCT', 'FG3M', 'FG3A', 'FG3_PCT', 
                               'FTM', 'FTA', 'FT_PCT', 'OREB', 'DREB', 'TOV', 
                               'PF', 'PLUS_MINUS']].mean()

    features = pd.DataFrame({col: [avg_stats[col.replace('AVG_', '')] if 'AVG_' in col else avg_stats[col]] 
                           for col in ['AVG_PTS', 'AVG_REB', 'AVG_AST', 'AVG_BLK', 'AVG_STL', 
                                     'FGM', 'FGA', 'FG_PCT', 'FG3M', 'FG3A', 'FG3_PCT', 
                                     'FTM', 'FTA', 'FT_PCT', 'OREB', 'DREB', 'TOV', 'PF', 'PLUS_MINUS']})
    
    return model.predict(features)[0]

# Monte Carlo Functions
def monte_carlo_simulation(game_log, opponent_team, num_simulations=10000, confidence_level=0.95):
    opponent_games = game_log[game_log['MATCHUP'].str.contains(opponent_team)]
    
    if opponent_games.empty:
        st.warning(f"No previous games found against team: {opponent_team}.")
        return None

    stats = {}
    stat_mapping = {
        'PTS': 'points',
        'REB': 'rebounds',
        'AST': 'assists'
    }
    
    for stat_key, stat_name in stat_mapping.items():
        values = opponent_games[stat_key].values
        mean_val = values.mean()
        std_val = values.std() if len(values) > 1 else 0
        
        simulated_values = np.random.normal(loc=mean_val, scale=std_val, size=num_simulations)
        
        stats[stat_name] = {
            "mean": simulated_values.mean(),
            "median": np.median(simulated_values),
            "std": simulated_values.std(),
            "ci_lower": np.percentile(simulated_values, (1-confidence_level) / 2 * 100),
            "ci_upper": np.percentile(simulated_values, (1 + confidence_level) / 2 * 100)
        }
    
    return stats

def main():
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Select a page:", ["Rolling Averages", "Linear Regression", "Monte Carlo Simulation"])

    if page == "Rolling Averages":
        st.header("Rolling Averages Predictions")
        home_team = st.selectbox("Select Home Team", options=[team['abbreviation'] for team in teams.get_teams()])
        away_team = st.selectbox("Select Away Team", options=[team['abbreviation'] for team in teams.get_teams()])

        if st.button("Generate Predictions"):
            with st.spinner("Generating predictions..."):
                teams_to_analyze = [(home_team, away_team), (away_team, home_team)]
                for team, opponent in teams_to_analyze:
                    roster = get_team_roster(team)
                    st.subheader(f"Analyzing {len(roster)} players from {team} against {opponent}...")
                    
                    predictions = {}
                    for player_name in roster:
                        game_log = get_player_data(player_name)
                        if game_log is None or len(game_log) < 5:
                            st.warning(f"Insufficient data for {player_name}")
                            continue
                            
                        try:
                            processed_log = preprocess_game_log(game_log)
                            rolling_avg = processed_log['PTS'].mean()  # Example of rolling average
                            predictions[player_name] = rolling_avg
                        except Exception as e:
                            st.error(f"Error processing {player_name}: {e}")
                            continue
                    
                    # Display results
                    st.write(f"\n{'='*50}")
                    st.write(f"Predictions for {team} against {opponent}:")
                    st.write(f"{'='*50}")
                    
                    for player_name, avg_points in predictions.items():
                        st.write(f"{player_name}: {avg_points:.1f} points")

    elif page == "Linear Regression":
        st.header("Linear Regression Predictions")
        home_team = st.selectbox("Select Home Team", options=[team['abbreviation'] for team in teams.get_teams()])
        away_team = st.selectbox("Select Away Team", options=[team['abbreviation'] for team in teams.get_teams()])

        if st.button("Generate Predictions"):
            with st.spinner("Generating predictions..."):
                teams_to_analyze = [(home_team, away_team), (away_team, home_team)]
                for team, opponent in teams_to_analyze:
                    roster = get_team_roster(team)
                    st.subheader(f"Analyzing {len(roster)} players from {team} against {opponent}...")
                    
                    linear_predictions = {}
                    for player_name in roster:
                        game_log = get_player_data(player_name)
                        if game_log is None or len(game_log) < 5:
                            st.warning(f"Insufficient data for {player_name}")
                            continue
                            
                        try:
                            processed_log = preprocess_game_log(game_log)
                            model, mse = train_model(processed_log)  # Get model and MSE
                            linear_pred = predict_performance_against_team(model, processed_log, opponent)
                            if linear_pred is not None:
                                linear_predictions[player_name] = linear_pred
                        except Exception as e:
                            st.error(f"Error processing {player_name}: {e}")
                            continue
                    
                    # Display results
                    st.write(f"\n{'='*50}")
                    st.write(f"Predictions for {team} against {opponent}:")
                    st.write(f"{'='*50}")
                    
                    for player_name, linear_points in linear_predictions.items():
                        st.write(f"{player_name}: {linear_points:.1f} points (± {np.sqrt(mse):.2f})")  # Display prediction with ± MSE

    elif page == "Monte Carlo Simulation":
        st.header("Monte Carlo Simulation Predictions")
        home_team = st.selectbox("Select Home Team", options=[team['abbreviation'] for team in teams.get_teams()])
        away_team = st.selectbox("Select Away Team", options=[team['abbreviation'] for team in teams.get_teams()])

        if st.button("Generate Predictions"):
            with st.spinner("Generating predictions..."):
                teams_to_analyze = [(home_team, away_team), (away_team, home_team)]
                for team, opponent in teams_to_analyze:
                    roster = get_team_roster(team)
                    st.subheader(f"Analyzing {len(roster)} players from {team} against {opponent}...")
                    
                    monte_carlo_predictions = {}
                    for player_name in roster:
                        game_log = get_player_data(player_name)
                        if game_log is None or len(game_log) < 5:
                            st.warning(f"Insufficient data for {player_name}")
                            continue
                            
                        try:
                            processed_log = preprocess_game_log(game_log)
                            mc_pred = monte_carlo_simulation(processed_log, opponent)
                            if mc_pred is not None:
                                monte_carlo_predictions[player_name] = mc_pred
                        except Exception as e:
                            st.error(f"Error processing {player_name}: {e}")
                            continue
                    
                    # Display results
                    st.write(f"\n{'='*50}")
                    st.write(f"Predictions for {team} against {opponent}:")
                    st.write(f"{'='*50}")
                    
                    for player_name, mc_stats in monte_carlo_predictions.items():
                        st.write(f"{player_name}: {mc_stats['points']['mean']:.1f} points (± {mc_stats['points']['std']:.1f})")

if __name__ == "__main__":
    main()