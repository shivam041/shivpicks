from nba_api.stats.endpoints import leaguegamefinder

def fetch_nba_data(season):
    game_finder = leaguegamefinder.LeagueGameFinder(season_nullable=season)
    games = game_finder.get_data_frames()[0]
    # Extract relevant features and labels from the retrieved data
    features = games[['TEAM_ID', 'GAME_DATE', 'PTS', 'AST', 'REB']]  
    labels = games['WL'].apply(lambda x: 1 if x == 'W' else 0)  # Convert 'W'/'L' to binary labels
    return features, labels

# Example: Fetch NBA data for the 2021-2022 season
season = '2021-22'
features, labels = fetch_nba_data(season)
print(features)
print(labels)