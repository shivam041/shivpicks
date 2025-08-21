"""
Comprehensive NBA Data Analysis Module
Uses NBA API to analyze 5 years of historical data, home/away splits, and advanced stats.
"""

import pandas as pd
import numpy as np
import streamlit as st
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# NBA API imports
from nba_api.stats.static import players, teams
from nba_api.stats.endpoints import playergamelog, playercareerstats, commonplayerinfo
from nba_api.stats.library.parameters import Season

class ComprehensiveNBAAnalyzer:
    """Comprehensive NBA data analyzer using 5 years of historical data."""
    
    def __init__(self):
        self.seasons = ['2024-25', '2023-24', '2022-23', '2021-22', '2020-21']
        self.cache = {}
    
    def get_player_id(self, player_name: str) -> Optional[int]:
        """Get player ID from NBA API."""
        try:
            # Check cache first
            cache_key = f"player_id_{player_name.lower()}"
            if cache_key in self.cache:
                return self.cache[cache_key]
            
            all_players = players.get_players()
            for player in all_players:
                if player['full_name'].lower() == player_name.lower():
                    # Cache the result
                    self.cache[cache_key] = player['id']
                    return player['id']
            return None
        except Exception as e:
            st.error(f"Error getting player ID for {player_name}: {str(e)}")
            return None
    
    def get_player_career_stats(self, player_id: int) -> Optional[pd.DataFrame]:
        """Get comprehensive career stats for a player."""
        try:
            # Check cache first
            cache_key = f"career_stats_{player_id}"
            if cache_key in self.cache:
                return self.cache[cache_key]
            
            career_stats = playercareerstats.PlayerCareerStats(player_id=player_id)
            result = career_stats.get_data_frames()[0]
            
            # Cache the result
            self.cache[cache_key] = result
            return result
        except Exception as e:
            st.warning(f"Could not fetch career stats: {str(e)}")
            return None
    
    def get_player_game_logs(self, player_id: int, season: str) -> Optional[pd.DataFrame]:
        """Get game logs for a specific season."""
        try:
            # Check cache first
            cache_key = f"game_logs_{player_id}_{season}"
            if cache_key in self.cache:
                return self.cache[cache_key]
            
            game_logs = playergamelog.PlayerGameLog(
                player_id=player_id, 
                season=season,
                timeout=30
            )
            result = game_logs.get_data_frames()[0]
            
            # Cache the result
            self.cache[cache_key] = result
            return result
        except Exception as e:
            st.warning(f"Could not fetch {season} game logs: {str(e)}")
            return None
    
    def analyze_player_performance(self, player_name: str, opponent_team: str) -> Dict:
        """Analyze player performance with comprehensive historical data."""
        
        # Get player ID
        player_id = self.get_player_id(player_name)
        if not player_id:
            return None
        
        # Get career stats
        career_stats = self.get_player_career_stats(player_id)
        
        # Collect 5 years of game logs
        all_game_logs = []
        for season in self.seasons:
            game_logs = self.get_player_game_logs(player_id, season)
            if game_logs is not None and len(game_logs) > 0:
                game_logs['SEASON'] = season
                all_game_logs.append(game_logs)
        
        if not all_game_logs:
            return None
        
        # Combine all game logs
        combined_logs = pd.concat(all_game_logs, ignore_index=True)
        
        # Clean and process data
        processed_data = self._process_game_logs(combined_logs, opponent_team)
        
        return processed_data
    
    def clear_cache(self):
        """Clear the internal cache."""
        self.cache.clear()
        st.success("🗑️ Cache cleared successfully!")
    
    def get_cache_info(self):
        """Get information about the current cache."""
        return {
            'total_items': len(self.cache),
            'cache_keys': list(self.cache.keys())
        }
    
    def _safe_numeric_conversion(self, value):
        """Safely convert value to numeric, handling edge cases."""
        try:
            if pd.isna(value) or value == '' or value is None:
                return 0.0
            return float(value)
        except (ValueError, TypeError):
            return 0.0
    
    def _process_game_logs(self, game_logs: pd.DataFrame, opponent_team: str) -> Dict:
        """Process game logs to extract key statistical factors."""
        
        try:
            # Convert date and clean data
            game_logs['GAME_DATE'] = pd.to_datetime(game_logs['GAME_DATE'], errors='coerce')
            game_logs = game_logs.sort_values('GAME_DATE', ascending=False)
            
            # Convert numeric columns safely
            numeric_cols = ['PTS', 'REB', 'AST', 'STL', 'BLK', 'TOV', 'FGM', 'FGA', 'FG_PCT', 
                           'FG3M', 'FG3A', 'FG3_PCT', 'FTM', 'FTA', 'FT_PCT', 'PLUS_MINUS']
            
            for col in numeric_cols:
                if col in game_logs.columns:
                    game_logs[col] = game_logs[col].apply(self._safe_numeric_conversion)
            
            # Extract home/away information safely
            game_logs['IS_HOME'] = game_logs['MATCHUP'].str.contains('vs', na=False)
            game_logs['IS_AWAY'] = game_logs['MATCHUP'].str.contains('@', na=False)
            
            # Extract opponent team safely
            def extract_opponent(matchup):
                try:
                    if pd.isna(matchup) or matchup == '':
                        return 'UNK'
                    parts = str(matchup).split()
                    return parts[-1] if len(parts) > 1 else 'UNK'
                except:
                    return 'UNK'
            
            game_logs['OPPONENT'] = game_logs['MATCHUP'].apply(extract_opponent)
            
            # Calculate comprehensive averages
            analysis = {}
            
            # Overall career averages
            analysis['career_averages'] = {
                'PTS': game_logs['PTS'].mean(),
                'REB': game_logs['REB'].mean(),
                'AST': game_logs['AST'].mean(),
                'FG_PCT': game_logs['FG_PCT'].mean(),
                'FG3_PCT': game_logs['FG3_PCT'].mean(),
                'FT_PCT': game_logs['FT_PCT'].mean(),
                'TOV': game_logs['TOV'].mean(),
                'PLUS_MINUS': game_logs['PLUS_MINUS'].mean()
            }
            
            # Last 5 years averages
            analysis['recent_averages'] = {}
            for season in self.seasons:
                season_data = game_logs[game_logs['SEASON'] == season]
                if len(season_data) > 0:
                    analysis['recent_averages'][season] = {
                        'PTS': season_data['PTS'].mean(),
                        'REB': season_data['REB'].mean(),
                        'AST': season_data['AST'].mean(),
                        'FG_PCT': season_data['FG_PCT'].mean(),
                        'FG3_PCT': season_data['FG3_PCT'].mean(),
                        'FT_PCT': season_data['FT_PCT'].mean(),
                        'TOV': season_data['TOV'].mean(),
                        'PLUS_MINUS': season_data['PLUS_MINUS'].mean()
                    }
            
            # Home vs Away splits
            home_games = game_logs[game_logs['IS_HOME'] == True]
            away_games = game_logs[game_logs['IS_AWAY'] == True]
            
            if len(home_games) > 0:
                analysis['home_averages'] = {
                    'PTS': home_games['PTS'].mean(),
                    'REB': home_games['REB'].mean(),
                    'AST': home_games['AST'].mean(),
                    'FG_PCT': home_games['FG_PCT'].mean(),
                    'FG3_PCT': home_games['FG3_PCT'].mean(),
                    'FT_PCT': home_games['FT_PCT'].mean(),
                    'TOV': home_games['TOV'].mean(),
                    'PLUS_MINUS': home_games['PLUS_MINUS'].mean()
                }
            
            if len(away_games) > 0:
                analysis['away_averages'] = {
                    'PTS': away_games['PTS'].mean(),
                    'REB': away_games['REB'].mean(),
                    'AST': away_games['AST'].mean(),
                    'FG_PCT': away_games['FG_PCT'].mean(),
                    'FG3_PCT': away_games['FG3_PCT'].mean(),
                    'FT_PCT': away_games['FT_PCT'].mean(),
                    'TOV': away_games['TOV'].mean(),
                    'PLUS_MINUS': away_games['PLUS_MINUS'].mean()
                }
            
            # Recent form (last 10 games)
            recent_games = game_logs.head(10)
            if len(recent_games) > 0:
                analysis['recent_form'] = {
                    'PTS': recent_games['PTS'].mean(),
                    'REB': recent_games['REB'].mean(),
                    'AST': recent_games['AST'].mean(),
                    'FG_PCT': recent_games['FG_PCT'].mean(),
                    'FG3_PCT': recent_games['FG3_PCT'].mean(),
                    'FT_PCT': recent_games['FT_PCT'].mean(),
                    'TOV': recent_games['TOV'].mean(),
                    'PLUS_MINUS': recent_games['PLUS_MINUS'].mean()
                }
            
            # Performance against specific opponent
            opponent_games = game_logs[game_logs['OPPONENT'] == opponent_team]
            if len(opponent_games) > 0:
                analysis['vs_opponent'] = {
                    'PTS': opponent_games['PTS'].mean(),
                    'REB': opponent_games['REB'].mean(),
                    'AST': opponent_games['AST'].mean(),
                    'FG_PCT': opponent_games['FG_PCT'].mean(),
                    'FG3_PCT': opponent_games['FG3_PCT'].mean(),
                    'FT_PCT': opponent_games['FT_PCT'].mean(),
                    'TOV': opponent_games['TOV'].mean(),
                    'PLUS_MINUS': opponent_games['PLUS_MINUS'].mean(),
                    'games_played': len(opponent_games)
                }
            
            # Season-by-season trends
            analysis['season_trends'] = {}
            for season in self.seasons:
                season_data = game_logs[game_logs['SEASON'] == season]
                if len(season_data) > 0:
                    analysis['season_trends'][season] = {
                        'games_played': len(season_data),
                        'avg_pts': season_data['PTS'].mean(),
                        'pts_std': season_data['PTS'].std(),
                        'consistency_score': 1 / (1 + season_data['PTS'].std() / season_data['PTS'].mean()) if season_data['PTS'].mean() > 0 else 0
                    }
            
            # Advanced metrics
            analysis['advanced_metrics'] = {
                'efficiency': (game_logs['FG_PCT'].mean() + game_logs['FG3_PCT'].mean() + game_logs['FT_PCT'].mean()) / 3,
                'versatility': game_logs[['PTS', 'REB', 'AST', 'STL', 'BLK']].std().mean(),
                'consistency': 1 / (1 + game_logs['PTS'].std() / game_logs['PTS'].mean()) if game_logs['PTS'].mean() > 0 else 0,
                'clutch_factor': game_logs['PLUS_MINUS'].mean() / (game_logs['PTS'].mean() + 1)
            }
            
            return analysis
            
        except Exception as e:
            st.error(f"Error processing game logs: {str(e)}")
            return None
    
    def predict_player_points(self, player_analysis: Dict, is_home: bool, opponent_team: str) -> float:
        """Predict player points using comprehensive analysis."""
        
        if not player_analysis:
            return 0.0
        
        # Base prediction from career average
        base_prediction = player_analysis['career_averages']['PTS']
        
        # Adjust for recent form (30% weight)
        if 'recent_form' in player_analysis:
            recent_factor = player_analysis['recent_form']['PTS'] / base_prediction
            base_prediction *= (0.7 + 0.3 * recent_factor)
        
        # Adjust for home/away (20% weight)
        if is_home and 'home_averages' in player_analysis:
            home_factor = player_analysis['home_averages']['PTS'] / base_prediction
            base_prediction *= (0.8 + 0.2 * home_factor)
        elif not is_home and 'away_averages' in player_analysis:
            away_factor = player_analysis['away_averages']['PTS'] / base_prediction
            base_prediction *= (0.8 + 0.2 * away_factor)
        
        # Adjust for opponent history (25% weight)
        if 'vs_opponent' in player_analysis and player_analysis['vs_opponent']['games_played'] >= 3:
            opponent_factor = player_analysis['vs_opponent']['PTS'] / base_prediction
            base_prediction *= (0.75 + 0.25 * opponent_factor)
        
        # Adjust for season trends (15% weight)
        if 'season_trends' in player_analysis:
            recent_seasons = list(player_analysis['season_trends'].keys())[:2]  # Last 2 seasons
            if recent_seasons:
                recent_avg = np.mean([player_analysis['season_trends'][s]['avg_pts'] for s in recent_seasons])
                trend_factor = recent_avg / base_prediction
                base_prediction *= (0.85 + 0.15 * trend_factor)
        
        # Apply consistency adjustment
        if 'advanced_metrics' in player_analysis:
            consistency = player_analysis['advanced_metrics']['consistency']
            base_prediction *= (0.9 + 0.1 * consistency)
        
        # Ensure prediction is reasonable
        return max(0.0, min(50.0, base_prediction))
    
    def get_team_analysis(self, roster: List[str], opponent_team: str, is_home: bool) -> Dict:
        """Analyze entire team roster against opponent."""
        
        team_analysis = {}
        total_predicted_points = 0
        
        st.info(f"🔍 Analyzing {len(roster)} players against {opponent_team}...")
        
        # Create progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, player_name in enumerate(roster):
            status_text.text(f"📊 Analyzing {player_name}... ({i+1}/{len(roster)})")
            
            try:
                # Analyze player
                player_analysis = self.analyze_player_performance(player_name, opponent_team)
                
                if player_analysis:
                    # Predict points
                    predicted_points = self.predict_player_points(player_analysis, is_home, opponent_team)
                    
                    team_analysis[player_name] = {
                        'analysis': player_analysis,
                        'predicted_points': predicted_points,
                        'career_avg': player_analysis['career_averages']['PTS'],
                        'recent_form': player_analysis.get('recent_form', {}).get('PTS', 0),
                        'home_away_avg': player_analysis.get('home_averages' if is_home else 'away_averages', {}).get('PTS', 0),
                        'vs_opponent_avg': player_analysis.get('vs_opponent', {}).get('PTS', 0)
                    }
                    
                    total_predicted_points += predicted_points
                    st.success(f"✅ {player_name}: {predicted_points:.1f} pts predicted")
                else:
                    st.warning(f"⚠️ Could not analyze {player_name}")
                    team_analysis[player_name] = {
                        'analysis': None,
                        'predicted_points': 0.0,
                        'career_avg': 0.0,
                        'recent_form': 0.0,
                        'home_away_avg': 0.0,
                        'vs_opponent_avg': 0.0
                    }
                    
            except Exception as e:
                st.error(f"❌ Error analyzing {player_name}: {str(e)}")
                team_analysis[player_name] = {
                    'analysis': None,
                    'predicted_points': 0.0,
                    'career_avg': 0.0,
                    'recent_form': 0.0,
                    'home_away_avg': 0.0,
                    'vs_opponent_avg': 0.0
                }
            
            # Update progress
            progress_bar.progress((i + 1) / len(roster))
        
        # Clear progress indicators
        progress_bar.empty()
        status_text.empty()
        
        team_analysis['team_total'] = total_predicted_points
        team_analysis['team_average'] = total_predicted_points / len(roster)
        
        return team_analysis

# Global instance
nba_analyzer = ComprehensiveNBAAnalyzer()
