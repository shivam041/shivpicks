"""
Optimized NBA API Module
Uses batch calls, better caching, and static data to dramatically improve performance.
"""

import pandas as pd
import numpy as np
from nba_api.stats.static import players
from nba_api.stats.endpoints import playergamelog, leaguegamelog
from nba_api.stats.library.parameters import Season
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import streamlit as st
from typing import Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')

# Cache TTL - Much longer for better performance
CACHE_TTL = 86400  # 24 hours
BATCH_SIZE = 5  # Process players in batches
MAX_WORKERS = 3  # Reduced to avoid rate limiting

class OptimizedNBAClient:
    """Optimized NBA API client with batch processing and smart caching."""
    
    def __init__(self):
        self.players_cache = {}
        self.gamelog_cache = {}
        self.last_request_time = 0
        self.min_request_interval = 0.1  # 100ms between requests
        
    def _rate_limit(self):
        """Implement rate limiting to avoid API blocks."""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.min_request_interval:
            time.sleep(self.min_request_interval - time_since_last)
        self.last_request_time = time.time()
    
    @st.cache_data(ttl=CACHE_TTL)
    def get_all_players_dict(self) -> Dict[str, int]:
        """Get all players once and cache them."""
        try:
            all_players = players.get_players()
            return {player['full_name']: player['id'] for player in all_players}
        except:
            return {}
    
    def get_player_id_batch(self, player_names: List[str]) -> Dict[str, int]:
        """Get player IDs for multiple players at once."""
        players_dict = self.get_all_players_dict()
        return {name: players_dict.get(name, None) for name in player_names}
    
    @st.cache_data(ttl=CACHE_TTL)
    def get_team_gamelog_batch(self, team_abbreviation: str, season: str = '2024-25') -> pd.DataFrame:
        """Get all games for a team in one API call instead of individual players."""
        try:
            self._rate_limit()
            # Use league game finder to get team games
            from nba_api.stats.endpoints import leaguegamefinder
            from nba_api.stats.library.parameters import TeamID
            
            # Get team ID first
            from nba_api.stats.static import teams
            team_info = teams.find_team_by_abbreviation(team_abbreviation)
            if not team_info:
                return pd.DataFrame()
            
            team_id = team_info['id']
            
            # Get all games for the team
            gamefinder = leaguegamefinder.LeagueGameFinder(
                team_id_nullable=team_id,
                season_nullable=season,
                timeout=10
            )
            
            games_df = gamefinder.get_data_frames()[0]
            return games_df
            
        except Exception as e:
            st.warning(f"Could not fetch team games for {team_abbreviation}: {str(e)}")
            return pd.DataFrame()
    
    def get_player_gamelog_optimized(self, player_name: str, season: str = '2024-25') -> Optional[pd.DataFrame]:
        """Get player gamelog with optimized caching."""
        cache_key = f"{player_name}_{season}"
        
        if cache_key in self.gamelog_cache:
            return self.gamelog_cache[cache_key]
        
        try:
            # Get player ID from cached dictionary
            players_dict = self.get_all_players_dict()
            player_id = players_dict.get(player_name)
            
            if not player_id:
                return None
            
            self._rate_limit()
            
            # Fetch gamelog with shorter timeout
            gamelog = playergamelog.PlayerGameLog(
                player_id=player_id, 
                season=season, 
                timeout=8  # Reduced timeout
            ).get_data_frames()[0]
            
            # Cache the result
            self.gamelog_cache[cache_key] = gamelog
            return gamelog
            
        except Exception as e:
            return None
    
    def get_players_data_batch(self, player_names: List[str], seasons: List[str] = None) -> Dict[str, pd.DataFrame]:
        """Get data for multiple players efficiently."""
        if seasons is None:
            seasons = ['2024-25', '2023-24']
        
        results = {}
        
        # Process in smaller batches to avoid overwhelming the API
        for i in range(0, len(player_names), BATCH_SIZE):
            batch = player_names[i:i + BATCH_SIZE]
            
            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                future_to_player = {
                    executor.submit(self._get_player_data_multi_season, player, seasons): player 
                    for player in batch
                }
                
                for future in as_completed(future_to_player):
                    player = future_to_player[future]
                    try:
                        data = future.result()
                        if data is not None and not data.empty:
                            results[player] = data
                    except:
                        continue
            
            # Small delay between batches
            time.sleep(0.2)
        
        return results
    
    def _get_player_data_multi_season(self, player_name: str, seasons: List[str]) -> Optional[pd.DataFrame]:
        """Get player data for multiple seasons."""
        gamelogs = []
        
        for season in seasons:
            gamelog = self.get_player_gamelog_optimized(player_name, season)
            if gamelog is not None and not gamelog.empty:
                gamelogs.append(gamelog)
        
        if gamelogs:
            combined_data = pd.concat(gamelogs, ignore_index=True)
            return combined_data
        
        return None
    
    def get_quick_predictions(self, team_roster: List[str], opponent_team: str) -> Dict[str, float]:
        """Get quick predictions using historical averages and minimal API calls."""
        predictions = {}
        
        # Get team games data once
        team_games = self.get_team_gamelog_batch(opponent_team)
        
        if team_games.empty:
            # Fallback to individual player data
            return self._fallback_predictions(team_roster, opponent_team)
        
        # Use team data to estimate player performance
        for player_name in team_roster:
            try:
                # Quick prediction based on team averages vs opponent
                player_data = self.get_player_gamelog_optimized(player_name, '2024-25')
                
                if player_data is not None and not player_data.empty:
                    # Simple prediction: recent average + opponent adjustment
                    recent_avg = player_data['PTS'].tail(5).mean()
                    
                    # Get opponent defensive rating from team games
                    opponent_defense = team_games['PTS'].mean() if not team_games.empty else 110
                    
                    # Simple adjustment factor
                    if opponent_defense < 105:  # Good defense
                        prediction = recent_avg * 0.9
                    elif opponent_defense > 115:  # Poor defense
                        prediction = recent_avg * 1.1
                    else:
                        prediction = recent_avg
                    
                    predictions[player_name] = max(0, prediction)
                else:
                    predictions[player_name] = 0.0
                    
            except:
                predictions[player_name] = 0.0
        
        return predictions
    
    def _fallback_predictions(self, team_roster: List[str], opponent_team: str) -> Dict[str, float]:
        """Fallback method using minimal data."""
        predictions = {}
        
        for player_name in team_roster:
            try:
                # Get only current season data for speed
                player_data = self.get_player_gamelog_optimized(player_name, '2024-25')
                
                if player_data is not None and not player_data.empty:
                    # Use season average as prediction
                    season_avg = player_data['PTS'].mean()
                    predictions[player_name] = season_avg
                else:
                    predictions[player_name] = 0.0
                    
            except:
                predictions[player_name] = 0.0
        
        return predictions

# Global instance
nba_client = OptimizedNBAClient()

# Convenience functions
def get_player_data_fast(player_name: str) -> Optional[pd.DataFrame]:
    """Fast player data retrieval."""
    return nba_client.get_player_gamelog_optimized(player_name)

def get_team_predictions_fast(team_roster: List[str], opponent_team: str) -> Dict[str, float]:
    """Get team predictions quickly."""
    return nba_client.get_quick_predictions(team_roster, opponent_team)

def get_players_data_batch_fast(player_names: List[str]) -> Dict[str, pd.DataFrame]:
    """Get data for multiple players efficiently."""
    return nba_client.get_players_data_batch(player_names)
