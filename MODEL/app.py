import streamlit as st
import pandas as pd
import numpy as np
from nba_api.stats.static import players, teams
from nba_api.stats.endpoints import playergamelog, commonteamroster
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import time
from requests.exceptions import ReadTimeout, ConnectionError
from concurrent.futures import ThreadPoolExecutor, as_completed
import altair as alt
import xgboost as xgb

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
st.markdown('<p class="big-font">NBA Player Points Predictions 🏀</p>', unsafe_allow_html=True)

# Constants
SEASON_CURRENT = '2024-25'
SEASON_PREVIOUS = '2023-24'
MAX_RETRIES = 3
CACHE_TTL = 3600  # Cache TTL in seconds

# Helper Functions

def get_team_abbreviations():
    """Retrieve a list of team abbreviations."""
    return [team['abbreviation'] for team in teams.get_teams()]

def get_team_by_abbreviation(abbreviation):
    """Retrieve team information by its abbreviation."""
    return teams.find_team_by_abbreviation(abbreviation)

@st.cache_data(ttl=CACHE_TTL)
def get_team_roster(team_abbreviation):
    """
    Fetch the roster for a given team.

    Parameters:
        team_abbreviation (str): The abbreviation of the team.

    Returns:
        list: List of player full names.
    """
    try:
        team_info = get_team_by_abbreviation(team_abbreviation)
        if not team_info:
            st.error(f"Team '{team_abbreviation}' not found.")
            return []

        team_id = team_info['id']
        roster = commonteamroster.CommonTeamRoster(team_id=team_id).get_data_frames()[0]
        player_names = roster['PLAYER'].tolist()
        return player_names
    except Exception as e:
        st.error(f"Error fetching roster for team '{team_abbreviation}': {e}")
        return []

@st.cache_data(ttl=CACHE_TTL)
def get_player_id(player_name):
    """
    Fetch the player ID given the full name.

    Parameters:
        player_name (str): Full name of the player.

    Returns:
        int or None: Player ID if found, else None.
    """
    try:
        player_dict = players.find_players_by_full_name(player_name)
        if not player_dict:
            st.warning(f"Player '{player_name}' not found.")
            return None
        return player_dict[0]['id']
    except Exception as e:
        st.error(f"Error fetching player ID for '{player_name}': {e}")
        return None

@st.cache_data(ttl=CACHE_TTL)
def fetch_player_gamelog(player_id, season):
    """
    Fetch the game log for a player for a given season.

    Parameters:
        player_id (int): The player's ID.
        season (str): The season string (e.g., '2023-24').

    Returns:
        pd.DataFrame or None: Game log dataframe or None if error.
    """
    try:
        gamelog = playergamelog.PlayerGameLog(player_id=player_id, season=season, timeout=60).get_data_frames()[0]
        return gamelog
    except Exception as e:
        st.error(f"Error fetching game log for player ID '{player_id}' in season '{season}': {e}")
        return None

def get_player_data(player_name, max_retries=MAX_RETRIES):
    """
    Retrieve and combine game logs for current and previous seasons for a player.

    Parameters:
        player_name (str): Full name of the player.
        max_retries (int): Maximum number of retries for API calls.

    Returns:
        pd.DataFrame or None: Combined game log dataframe or None if failed.
    """
    player_id = get_player_id(player_name)
    if not player_id:
        return None

    gamelogs = []
    seasons = [SEASON_CURRENT, SEASON_PREVIOUS]

    for season in seasons:
        for attempt in range(max_retries):
            try:
                gamelog = fetch_player_gamelog(player_id, season)
                if gamelog is not None:
                    gamelogs.append(gamelog)
                break  # Break out of retry loop if successful
            except (ReadTimeout, ConnectionError) as e:
                if attempt < max_retries - 1:
                    time.sleep(2)
                else:
                    st.error(f"Failed to fetch data for {player_name} in season {season} after {max_retries} attempts.")
                    return None
    if gamelogs:
        combined_data = pd.concat(gamelogs, ignore_index=True)
        return combined_data
    return None

def fetch_all_player_data(roster):
    """
    Fetch game data for all players in the roster concurrently.

    Parameters:
        roster (list): List of player names.

    Returns:
        dict: Dictionary mapping player names to their game logs.
    """
    player_data = {}
    with ThreadPoolExecutor(max_workers=10) as executor:
        future_to_player = {executor.submit(get_player_data, player): player for player in roster}
        for future in as_completed(future_to_player):
            player = future_to_player[future]
            try:
                data = future.result()
                if data is not None and not data.empty:
                    player_data[player] = data
                else:
                    st.warning(f"No data available for player: {player}")
            except Exception as e:
                st.error(f"Error fetching data for player '{player}': {e}")
    return player_data

def preprocess_game_log(game_log, rolling_window):
    """
    Preprocess the game log data by converting dates, identifying home/away games, 
    handling missing data, and calculating rolling averages.

    Parameters:
        game_log (pd.DataFrame): Raw game log data.
        rolling_window (int): Number of games to include in rolling average.

    Returns:
        pd.DataFrame: Preprocessed game log data with rolling averages.
    """
    game_log = game_log.copy()
    game_log['GAME_DATE'] = pd.to_datetime(game_log['GAME_DATE'])    
    game_log['HOME_AWAY'] = np.where(game_log['MATCHUP'].str.contains('@'), 'Away', 'Home')

    numeric_cols = ['PTS', 'REB', 'AST', 'BLK', 'STL', 'FGM', 'FGA', 'FG_PCT', 
                   'FG3M', 'FG3A', 'FG3_PCT', 'FTM', 'FTA', 'FT_PCT', 
                   'OREB', 'DREB', 'TOV', 'PF', 'PLUS_MINUS']

    for col in numeric_cols:
        game_log[col] = pd.to_numeric(game_log[col], errors='coerce')

    game_log = game_log.sort_values('GAME_DATE', ascending=True)

    # Handle missing data with forward fill
    game_log.fillna(method='ffill', inplace=True)
    game_log.dropna(inplace=True)

    # Calculate rolling averages
    for stat in ['PTS', 'REB', 'AST', 'BLK', 'STL', 'FGM', 'FGA', 'FTM', 'OREB', 'DREB']:
        game_log[f'AVG_{stat}'] = game_log[stat].rolling(window=rolling_window).mean()

    game_log.dropna(inplace=True)

    return game_log

@st.cache_resource
def train_random_forest_model(game_log):
    """
    Train a Random Forest Regressor model.

    Parameters:
        game_log (pd.DataFrame): Preprocessed game log data.

    Returns:
        tuple: Trained model and its evaluation metrics.
    """
    features = game_log[['AVG_PTS', 'AVG_REB', 'AVG_AST', 'AVG_BLK', 'AVG_STL', 
                         'FGM', 'FGA', 'FG_PCT', 'FG3M', 'FG3A', 'FG3_PCT', 
                         'FTM', 'FTA', 'FT_PCT', 'OREB', 'DREB', 'TOV', 'PF', 'PLUS_MINUS']]
    target = game_log['PTS']

    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    metrics = {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2
    }

    return model, metrics

@st.cache_resource
def train_xgboost_model(game_log):
    """
    Train an XGBoost Regressor model.

    Parameters:
        game_log (pd.DataFrame): Preprocessed game log data.

    Returns:
        tuple: Trained model and its evaluation metrics.
    """
    features = game_log[['AVG_PTS', 'AVG_REB', 'AVG_AST', 'AVG_BLK', 'AVG_STL', 
                         'FGM', 'FGA', 'FG_PCT', 'FG3M', 'FG3A', 'FG3_PCT', 
                         'FTM', 'FTA', 'FT_PCT', 'OREB', 'DREB', 'TOV', 'PF', 'PLUS_MINUS']]
    target = game_log['PTS']

    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
    model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42, verbosity=0)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    metrics = {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2
    }

    return model, metrics

@st.cache_resource
def train_neural_network_model(game_log):
    """
    Train a Neural Network Regressor (MLPRegressor) model.

    Parameters:
        game_log (pd.DataFrame): Preprocessed game log data.

    Returns:
        tuple: Trained model and its evaluation metrics.
    """
    features = game_log[['AVG_PTS', 'AVG_REB', 'AVG_AST', 'AVG_BLK', 'AVG_STL', 
                         'FGM', 'FGA', 'FG_PCT', 'FG3M', 'FG3A', 'FG3_PCT', 
                         'FTM', 'FTA', 'FT_PCT', 'OREB', 'DREB', 'TOV', 'PF', 'PLUS_MINUS']]
    target = game_log['PTS']

    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
    model = MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    metrics = {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2
    }

    return model, metrics

def predict_with_model(model, average_stats, model_type='rf'):
    """
    Predict player performance using the trained model.

    Parameters:
        model (sklearn estimator): Trained regression model.
        average_stats (dict): Dictionary of average stats.
        model_type (str): Type of the model ('rf', 'xgb', 'nn').

    Returns:
        float: Predicted points.
    """
    feature_order = ['AVG_PTS', 'AVG_REB', 'AVG_AST', 'AVG_BLK', 'AVG_STL', 
                     'FGM', 'FGA', 'FG_PCT', 'FG3M', 'FG3A', 'FG3_PCT', 
                     'FTM', 'FTA', 'FT_PCT', 'OREB', 'DREB', 'TOV', 'PF', 'PLUS_MINUS']
    features = np.array([average_stats.get(feature, 0) for feature in feature_order]).reshape(1, -1)
    prediction = model.predict(features)[0]
    return prediction

def monte_carlo_simulation(game_log, opponent_team, num_simulations=10000, confidence_level=0.95):
    """
    Perform Monte Carlo simulation to predict player performance against a specific team.

    Parameters:
        game_log (pd.DataFrame): Preprocessed game log data.
        opponent_team (str): Abbreviation of the opponent team.
        num_simulations (int): Number of simulations to run.
        confidence_level (float): Confidence level for intervals.

    Returns:
        dict: Simulation statistics for points, rebounds, and assists.
    """
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
        std_val = values.std(ddof=1) if len(values) > 1 else 0

        # Using normal distribution for simulation; consider other distributions if needed
        simulated_values = np.random.normal(loc=mean_val, scale=std_val, size=num_simulations)
        simulated_values = np.clip(simulated_values, a_min=0, a_max=None)  # Ensure no negative values

        stats[stat_name] = {
            "mean": simulated_values.mean(),
            "median": np.median(simulated_values),
            "std": simulated_values.std(),
            "ci_lower": np.percentile(simulated_values, (1 - confidence_level) / 2 * 100),
            "ci_upper": np.percentile(simulated_values, (1 + confidence_level) / 2 * 100)
        }

    return stats

def visualize_predictions(predictions, title, stat='points'):
    """
    Visualize prediction statistics using Altair.

    Parameters:
        predictions (dict): Dictionary containing prediction stats for each player.
        title (str): Title of the chart.
        stat (str): The statistic to visualize ('points', 'rebounds', 'assists').
    """
    data = []
    for player, stats in predictions.items():
        stat_data = stats.get(stat, {})
        data.append({
            'Player': player,
            'Mean': stat_data.get('mean', 0),
            'Std Dev': stat_data.get('std', 0),
            'CI Lower': stat_data.get('ci_lower', 0),
            'CI Upper': stat_data.get('ci_upper', 0)
        })
    
    df = pd.DataFrame(data)
    
    chart = alt.Chart(df).mark_bar().encode(
        x=alt.X('Player', sort='-y'),
        y=alt.Y('Mean', title=f'Average {stat.capitalize()}'),
        tooltip=['Player', 'Mean', 'Std Dev', 'CI Lower', 'CI Upper']
    ).properties(
        title=title,
        width=800,
        height=400
    )
    
    st.altair_chart(chart, use_container_width=True)

def visualize_model_metrics(metrics, model_name):
    """
    Display model evaluation metrics.

    Parameters:
        metrics (dict): Dictionary of evaluation metrics.
        model_name (str): Name of the model.
    """
    st.write(f"**{model_name} Evaluation Metrics:**")
    st.write(f"- Mean Squared Error (MSE): {metrics['MSE']:.2f}")
    st.write(f"- Root Mean Squared Error (RMSE): {metrics['RMSE']:.2f}")
    st.write(f"- Mean Absolute Error (MAE): {metrics['MAE']:.2f}")
    st.write(f"- R² Score: {metrics['R2']:.2f}")

# Main Application Functions

def random_forest_page(rolling_window):
    """
    Handle the Random Forest Regressor prediction page.

    Parameters:
        rolling_window (int): Number of games to include in rolling average.
    """
    st.header("Random Forest Regressor Predictions")
    home_team = st.selectbox("Select Home Team", options=get_team_abbreviations(), key='home_rf')
    away_team = st.selectbox("Select Away Team", options=[abbr for abbr in get_team_abbreviations() if abbr != home_team], key='away_rf')

    if st.button("Generate Random Forest Predictions"):
        with st.spinner("Fetching and processing data..."):
            teams_to_analyze = [(home_team, away_team), (away_team, home_team)]
            for team, opponent in teams_to_analyze:
                roster = get_team_roster(team)
                st.subheader(f"Analyzing {len(roster)} players from {team} against {opponent}...")

                player_data = fetch_all_player_data(roster)
                predictions = {}
                model_metrics = []

                progress_bar = st.progress(0)
                total_players = len(player_data)
                for idx, (player_name, game_log) in enumerate(player_data.items(), 1):
                    try:
                        processed_log = preprocess_game_log(game_log, rolling_window)
                        if len(processed_log) < 10:
                            st.warning(f"Insufficient data for {player_name}")
                            continue

                        model, metrics = train_random_forest_model(processed_log)
                        model_metrics.append(metrics)

                        # Prepare average stats against opponent
                        opponent_games = processed_log[processed_log['MATCHUP'].str.contains(opponent)]
                        if opponent_games.empty:
                            st.warning(f"No previous games against {opponent} for player {player_name}. Using overall averages.")
                            average_stats = processed_log.iloc[-1][['AVG_PTS', 'AVG_REB', 'AVG_AST', 'AVG_BLK', 'AVG_STL', 
                                                                    'FGM', 'FGA', 'FG_PCT', 'FG3M', 'FG3A', 'FG3_PCT', 
                                                                    'FTM', 'FTA', 'FT_PCT', 'OREB', 'DREB', 'TOV', 'PF', 'PLUS_MINUS']].to_dict()
                        else:
                            average_stats = opponent_games[['AVG_PTS', 'AVG_REB', 'AVG_AST', 'AVG_BLK', 'AVG_STL', 
                                                           'FGM', 'FGA', 'FG_PCT', 'FG3M', 'FG3A', 'FG3_PCT', 
                                                           'FTM', 'FTA', 'FT_PCT', 'OREB', 'DREB', 'TOV', 'PF', 'PLUS_MINUS']].mean().to_dict()

                        prediction = predict_with_model(model, average_stats, model_type='rf')
                        predictions[player_name] = prediction
                    except Exception as e:
                        st.error(f"Error processing {player_name}: {e}")
                        continue
                    progress_bar.progress(idx / total_players)
                
                progress_bar.empty()

                # Aggregate evaluation metrics
                if model_metrics:
                    avg_mse = np.mean([m['MSE'] for m in model_metrics])
                    avg_rmse = np.mean([m['RMSE'] for m in model_metrics])
                    avg_mae = np.mean([m['MAE'] for m in model_metrics])
                    avg_r2 = np.mean([m['R2'] for m in model_metrics])
                    avg_metrics = {
                        'MSE': avg_mse,
                        'RMSE': avg_rmse,
                        'MAE': avg_mae,
                        'R2': avg_r2
                    }
                    visualize_model_metrics(avg_metrics, "Random Forest Regressor")

                # Display predictions using visualization
                if predictions:
                    df_preds = pd.DataFrame.from_dict(predictions, orient='index', columns=['Predicted Points']).reset_index()
                    df_preds.rename(columns={'index': 'Player'}, inplace=True)

                    chart = alt.Chart(df_preds).mark_bar().encode(
                        x=alt.X('Player', sort='-y'),
                        y=alt.Y('Predicted Points', title='Predicted Points'),
                        tooltip=['Player', 'Predicted Points']
                    ).properties(
                        title=f'Random Forest Predicted Points for {team} against {opponent}',
                        width=800,
                        height=400
                    )

                    st.altair_chart(chart, use_container_width=True)
                else:
                    st.info(f"No predictions available for team {team} against {opponent}.")

def xgboost_page(rolling_window):
    """
    Handle the XGBoost Regressor prediction page.

    Parameters:
        rolling_window (int): Number of games to include in rolling average.
    """
    st.header("XGBoost Regressor Predictions")
    home_team = st.selectbox("Select Home Team", options=get_team_abbreviations(), key='home_xgb')
    away_team = st.selectbox("Select Away Team", options=[abbr for abbr in get_team_abbreviations() if abbr != home_team], key='away_xgb')

    if st.button("Generate XGBoost Predictions"):
        with st.spinner("Fetching and processing data..."):
            teams_to_analyze = [(home_team, away_team), (away_team, home_team)]
            for team, opponent in teams_to_analyze:
                roster = get_team_roster(team)
                st.subheader(f"Analyzing {len(roster)} players from {team} against {opponent}...")

                player_data = fetch_all_player_data(roster)
                predictions = {}
                model_metrics = []

                progress_bar = st.progress(0)
                total_players = len(player_data)
                for idx, (player_name, game_log) in enumerate(player_data.items(), 1):
                    try:
                        processed_log = preprocess_game_log(game_log, rolling_window)
                        if len(processed_log) < 10:
                            st.warning(f"Insufficient data for {player_name}")
                            continue

                        model, metrics = train_xgboost_model(processed_log)
                        model_metrics.append(metrics)

                        # Prepare average stats against opponent
                        opponent_games = processed_log[processed_log['MATCHUP'].str.contains(opponent)]
                        if opponent_games.empty:
                            st.warning(f"No previous games against {opponent} for player {player_name}. Using overall averages.")
                            average_stats = processed_log.iloc[-1][['AVG_PTS', 'AVG_REB', 'AVG_AST', 'AVG_BLK', 'AVG_STL', 
                                                                    'FGM', 'FGA', 'FG_PCT', 'FG3M', 'FG3A', 'FG3_PCT', 
                                                                    'FTM', 'FTA', 'FT_PCT', 'OREB', 'DREB', 'TOV', 'PF', 'PLUS_MINUS']].to_dict()
                        else:
                            average_stats = opponent_games[['AVG_PTS', 'AVG_REB', 'AVG_AST', 'AVG_BLK', 'AVG_STL', 
                                                           'FGM', 'FGA', 'FG_PCT', 'FG3M', 'FG3A', 'FG3_PCT', 
                                                           'FTM', 'FTA', 'FT_PCT', 'OREB', 'DREB', 'TOV', 'PF', 'PLUS_MINUS']].mean().to_dict()

                        prediction = predict_with_model(model, average_stats, model_type='xgb')
                        predictions[player_name] = prediction
                    except Exception as e:
                        st.error(f"Error processing {player_name}: {e}")
                        continue
                    progress_bar.progress(idx / total_players)
                
                progress_bar.empty()

                # Aggregate evaluation metrics
                if model_metrics:
                    avg_mse = np.mean([m['MSE'] for m in model_metrics])
                    avg_rmse = np.mean([m['RMSE'] for m in model_metrics])
                    avg_mae = np.mean([m['MAE'] for m in model_metrics])
                    avg_r2 = np.mean([m['R2'] for m in model_metrics])
                    avg_metrics = {
                        'MSE': avg_mse,
                        'RMSE': avg_rmse,
                        'MAE': avg_mae,
                        'R2': avg_r2
                    }
                    visualize_model_metrics(avg_metrics, "XGBoost Regressor")

                # Display predictions using visualization
                if predictions:
                    df_preds = pd.DataFrame.from_dict(predictions, orient='index', columns=['Predicted Points']).reset_index()
                    df_preds.rename(columns={'index': 'Player'}, inplace=True)

                    chart = alt.Chart(df_preds).mark_bar().encode(
                        x=alt.X('Player', sort='-y'),
                        y=alt.Y('Predicted Points', title='Predicted Points'),
                        tooltip=['Player', 'Predicted Points']
                    ).properties(
                        title=f'XGBoost Predicted Points for {team} against {opponent}',
                        width=800,
                        height=400
                    )

                    st.altair_chart(chart, use_container_width=True)
                else:
                    st.info(f"No predictions available for team {team} against {opponent}.")

def neural_network_page(rolling_window):
    """
    Handle the Neural Network Regressor prediction page.

    Parameters:
        rolling_window (int): Number of games to include in rolling average.
    """
    st.header("Neural Network Regressor Predictions")
    home_team = st.selectbox("Select Home Team", options=get_team_abbreviations(), key='home_nn')
    away_team = st.selectbox("Select Away Team", options=[abbr for abbr in get_team_abbreviations() if abbr != home_team], key='away_nn')

    if st.button("Generate Neural Network Predictions"):
        with st.spinner("Fetching and processing data..."):
            teams_to_analyze = [(home_team, away_team), (away_team, home_team)]
            for team, opponent in teams_to_analyze:
                roster = get_team_roster(team)
                st.subheader(f"Analyzing {len(roster)} players from {team} against {opponent}...")

                player_data = fetch_all_player_data(roster)
                predictions = {}
                model_metrics = []

                progress_bar = st.progress(0)
                total_players = len(player_data)
                for idx, (player_name, game_log) in enumerate(player_data.items(), 1):
                    try:
                        processed_log = preprocess_game_log(game_log, rolling_window)
                        if len(processed_log) < 10:
                            st.warning(f"Insufficient data for {player_name}")
                            continue

                        model, metrics = train_neural_network_model(processed_log)
                        model_metrics.append(metrics)

                        # Prepare average stats against opponent
                        opponent_games = processed_log[processed_log['MATCHUP'].str.contains(opponent)]
                        if opponent_games.empty:
                            st.warning(f"No previous games against {opponent} for player {player_name}. Using overall averages.")
                            average_stats = processed_log.iloc[-1][['AVG_PTS', 'AVG_REB', 'AVG_AST', 'AVG_BLK', 'AVG_STL', 
                                                                    'FGM', 'FGA', 'FG_PCT', 'FG3M', 'FG3A', 'FG3_PCT', 
                                                                    'FTM', 'FTA', 'FT_PCT', 'OREB', 'DREB', 'TOV', 'PF', 'PLUS_MINUS']].to_dict()
                        else:
                            average_stats = opponent_games[['AVG_PTS', 'AVG_REB', 'AVG_AST', 'AVG_BLK', 'AVG_STL', 
                                                           'FGM', 'FGA', 'FG_PCT', 'FG3M', 'FG3A', 'FG3_PCT', 
                                                           'FTM', 'FTA', 'FT_PCT', 'OREB', 'DREB', 'TOV', 'PF', 'PLUS_MINUS']].mean().to_dict()

                        prediction = predict_with_model(model, average_stats, model_type='nn')
                        predictions[player_name] = prediction
                    except Exception as e:
                        st.error(f"Error processing {player_name}: {e}")
                        continue
                    progress_bar.progress(idx / total_players)
                
                progress_bar.empty()

                # Aggregate evaluation metrics
                if model_metrics:
                    avg_mse = np.mean([m['MSE'] for m in model_metrics])
                    avg_rmse = np.mean([m['RMSE'] for m in model_metrics])
                    avg_mae = np.mean([m['MAE'] for m in model_metrics])
                    avg_r2 = np.mean([m['R2'] for m in model_metrics])
                    avg_metrics = {
                        'MSE': avg_mse,
                        'RMSE': avg_rmse,
                        'MAE': avg_mae,
                        'R2': avg_r2
                    }
                    visualize_model_metrics(avg_metrics, "Neural Network Regressor")

                # Display predictions using visualization
                if predictions:
                    df_preds = pd.DataFrame.from_dict(predictions, orient='index', columns=['Predicted Points']).reset_index()
                    df_preds.rename(columns={'index': 'Player'}, inplace=True)

                    chart = alt.Chart(df_preds).mark_bar().encode(
                        x=alt.X('Player', sort='-y'),
                        y=alt.Y('Predicted Points', title='Predicted Points'),
                        tooltip=['Player', 'Predicted Points']
                    ).properties(
                        title=f'Neural Network Predicted Points for {team} against {opponent}',
                        width=800,
                        height=400
                    )

                    st.altair_chart(chart, use_container_width=True)
                else:
                    st.info(f"No predictions available for team {team} against {opponent}.")

def monte_carlo_simulation_page(rolling_window):
    """
    Handle the Monte Carlo Simulation prediction page.

    Parameters:
        rolling_window (int): Number of games to include in rolling average.
    """
    st.header("Monte Carlo Simulation Predictions")
    home_team = st.selectbox("Select Home Team", options=get_team_abbreviations(), key='home_mc_new')
    away_team = st.selectbox("Select Away Team", options=[abbr for abbr in get_team_abbreviations() if abbr != home_team], key='away_mc_new')

    if st.button("Generate Monte Carlo Simulation Predictions"):
        with st.spinner("Fetching, processing, and simulating data..."):
            teams_to_analyze = [(home_team, away_team), (away_team, home_team)]
            for team, opponent in teams_to_analyze:
                roster = get_team_roster(team)
                st.subheader(f"Analyzing {len(roster)} players from {team} against {opponent}...")

                player_data = fetch_all_player_data(roster)
                predictions = {}

                progress_bar = st.progress(0)
                total_players = len(player_data)
                for idx, (player_name, game_log) in enumerate(player_data.items(), 1):
                    try:
                        processed_log = preprocess_game_log(game_log, rolling_window)
                        if len(processed_log) < 10:
                            st.warning(f"Insufficient data for {player_name}")
                            continue

                        mc_pred = monte_carlo_simulation(processed_log, opponent)
                        if mc_pred is not None:
                            predictions[player_name] = mc_pred
                    except Exception as e:
                        st.error(f"Error processing {player_name}: {e}")
                        continue
                    progress_bar.progress(idx / total_players)
                
                progress_bar.empty()

                # Display predictions using visualization
                if predictions:
                    visualize_predictions(predictions, f'Monte Carlo Simulation for {team} against {opponent}', stat='points')
                else:
                    st.info(f"No simulations available for team {team} against {opponent}.")

def main():
    """Main function to run the Streamlit application."""
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    
    # Adjustable Rolling Window Size
    rolling_window = st.sidebar.number_input("Rolling Window Size", min_value=1, max_value=20, value=5, step=1)
    
    page = st.sidebar.radio("Select a page:", ["Random Forest", "XGBoost", "Neural Network", "Monte Carlo Simulation"])
    
    if page == "Random Forest":
        random_forest_page(rolling_window)
    elif page == "XGBoost":
        xgboost_page(rolling_window)
    elif page == "Neural Network":
        neural_network_page(rolling_window)
    elif page == "Monte Carlo Simulation":
        monte_carlo_simulation_page(rolling_window)

if __name__ == "__main__":
    main()