"""
Given the following inputs:
- <game_data> is a list of dictionaries, with each dictionary representing a player's shot attempts in a game. The list can be empty, but any dictionary in the list will include the following keys: gameID, playerID, gameDate, fieldGoal2Attempted, fieldGoal2Made, fieldGoal3Attempted, fieldGoal3Made, freeThrowAttempted, freeThrowMade. All values in this dictionary are ints, except for gameDate which is of type str in the format 'MM/DD/YYYY'
- <true_shooting_cutoff> is the minimum True Shooting percentage value for a player to qualify in a game. It will be an int value >= 0.
- <player_count> is the number of players that need to meet the <true_shooting_cutoff> in order for a gameID to qualify. It will be an int value >= 0.

Implement find_qualified_games to return a list of unique qualified gameIDs in which at least <player_count> players have a True Shooting percentage >= <true_shooting_cutoff>, ordered from most to least recent game.
"""
from datetime import datetime

def find_qualified_games(game_data: list[dict], true_shooting_cutoff: int, player_count: int) -> list[int]:
	# Replace the line below with your code

    qualified_games = {}
    
    # Process each player game data
    for player in game_data:
        game_id = player['gameID']
        ts_percentage = calculate_true_shooting_percentage(player)
        
        if ts_percentage >= true_shooting_cutoff:
            # Count how many players qualify for each game
            if game_id not in qualified_games:
                qualified_games[game_id] = {
                    'date': player['gameDate'],
                    'qualified_players': 0
                }
            qualified_games[game_id]['qualified_players'] += 1
    
    # Find games where the number of qualified players meets the player_count
    result = []
    for game_id, game_info in qualified_games.items():
        if game_info['qualified_players'] >= player_count:
            result.append((game_info['date'], game_id))
    
    # Sort by date (most recent first)
    result.sort(key=lambda x: datetime.strptime(x[0], '%m/%d/%Y'), reverse=True)
    
    # Return just the gameIDs in the correct order
    return [game_id for _, game_id in result]

def calculate_true_shooting_percentage(player):
    points = (2 * player['fieldGoal2Made'] +
              3 * player['fieldGoal3Made'] +
              player['freeThrowMade'])
    fga = (player['fieldGoal2Attempted'] +
           player['fieldGoal3Attempted'] +
           0.44 * player['freeThrowAttempted'])
    # Avoid division by zero
    if fga == 0:
        return 0
    return (points / (2 * fga)) * 100