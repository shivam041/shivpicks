import streamlit as st
import pandas as pd
import numpy as np
import time
import altair as alt
import warnings
from team_database import get_team_roster, get_all_team_abbreviations, get_team_name
from comprehensive_nba import nba_analyzer
from typing import Dict

warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="NBA Comprehensive Predictions",
    page_icon="🏀",
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
    .analysis-card {
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
    .stats-container {
        background: rgba(255, 255, 255, 0.1);
        padding: 1rem;
        border-radius: 10px;
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

def display_player_analysis(player_name: str, player_data: Dict):
    """Display detailed player analysis."""
    
    with st.expander(f"📊 Detailed Analysis: {player_name}", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 🎯 Prediction Summary")
            st.metric("Predicted Points", f"{player_data['predicted_points']:.1f}")
            st.metric("Career Average", f"{player_data['career_avg']:.1f}")
            st.metric("Recent Form", f"{player_data['recent_form']:.1f}")
            st.metric("Home/Away Avg", f"{player_data['home_away_avg']:.1f}")
            st.metric("vs Opponent", f"{player_data['vs_opponent_avg']:.1f}")
        
        with col2:
            if player_data['analysis']:
                analysis = player_data['analysis']
                
                st.markdown("### 📈 Advanced Metrics")
                
                if 'advanced_metrics' in analysis:
                    metrics = analysis['advanced_metrics']
                    st.metric("Efficiency", f"{metrics['efficiency']:.3f}")
                    st.metric("Versatility", f"{metrics['versatility']:.2f}")
                    st.metric("Consistency", f"{metrics['consistency']:.3f}")
                    st.metric("Clutch Factor", f"{metrics['clutch_factor']:.3f}")
                
                if 'season_trends' in analysis:
                    st.markdown("### 📅 Season Trends")
                    for season, data in analysis['season_trends'].items():
                        st.write(f"**{season}**: {data['avg_pts']:.1f} pts ({data['games_played']} games)")

def display_team_comparison(home_analysis: Dict, away_analysis: Dict):
    """Display team comparison analysis."""
    
    st.markdown("## 🏆 Team Comparison")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### 🏠 Home Team")
        st.metric("Total Predicted", f"{home_analysis['team_total']:.1f} pts")
        st.metric("Team Average", f"{home_analysis['team_average']:.1f} pts")
        st.metric("Players Analyzed", len([k for k in home_analysis.keys() if k not in ['team_total', 'team_average']]))
    
    with col2:
        st.markdown("### ✈️ Away Team")
        st.metric("Total Predicted", f"{away_analysis['team_total']:.1f} pts")
        st.metric("Team Average", f"{away_analysis['team_average']:.1f} pts")
        st.metric("Players Analyzed", len([k for k in away_analysis.keys() if k not in ['team_total', 'team_average']]))
    
    with col3:
        st.markdown("### 🎯 Game Prediction")
        home_total = home_analysis['team_total']
        away_total = away_analysis['team_total']
        predicted_winner = "Home Team" if home_total > away_total else "Away Team"
        margin = abs(home_total - away_total)
        
        st.metric("Predicted Winner", predicted_winner)
        st.metric("Predicted Margin", f"{margin:.1f} pts")
        st.metric("Total Points", f"{home_total + away_total:.1f} pts")

def create_prediction_charts(home_analysis: Dict, away_analysis: Dict):
    """Create visualization charts for predictions."""
    
    # Player predictions chart
    home_players = {k: v for k, v in home_analysis.items() if k not in ['team_total', 'team_average']}
    away_players = {k: v for k, v in away_analysis.items() if k not in ['team_total', 'team_average']}
    
    # Prepare data for charts
    chart_data = []
    
    for player_name, data in home_players.items():
        chart_data.append({
            'Player': player_name,
            'Team': 'Home',
            'Predicted Points': data['predicted_points'],
            'Career Average': data['career_avg']
        })
    
    for player_name, data in away_players.items():
        chart_data.append({
            'Player': player_name,
            'Team': 'Away',
            'Predicted Points': data['predicted_points'],
            'Career Average': data['career_avg']
        })
    
    df = pd.DataFrame(chart_data)
    
    # Top scorers chart
    top_scorers = df.nlargest(10, 'Predicted Points')
    
    chart1 = alt.Chart(top_scorers).mark_bar().encode(
        x=alt.X('Player:N', title='Player', sort='-y'),
        y=alt.Y('Predicted Points:Q', title='Predicted Points'),
        color=alt.Color('Team:N', scale=alt.Scale(range=['#4CAF50', '#FF6B6B'])),
        tooltip=['Player', 'Team', 'Predicted Points', 'Career Average']
    ).properties(
        title='Top 10 Predicted Scorers',
        width=700,
        height=400
    )
    
    st.altair_chart(chart1, use_container_width=True)
    
    # Team comparison chart
    team_totals = pd.DataFrame([
        {'Team': 'Home', 'Total Points': home_analysis['team_total']},
        {'Team': 'Away', 'Total Points': away_analysis['team_total']}
    ])
    
    chart2 = alt.Chart(team_totals).mark_bar().encode(
        x='Team:N',
        y='Total Points:Q',
        color=alt.Color('Team:N', scale=alt.Scale(range=['#4CAF50', '#FF6B6B'])),
        tooltip=['Team', 'Total Points']
    ).properties(
        title='Team Total Predictions',
        width=400,
        height=300
    )
    
    col1, col2 = st.columns(2)
    with col1:
        st.altair_chart(chart2, use_container_width=True)
    
    with col2:
        # Prediction accuracy indicators
        st.markdown("### 📊 Prediction Confidence")
        
        # Calculate confidence based on data availability
        home_confidence = len([k for k in home_analysis.keys() if k not in ['team_total', 'team_average'] and home_analysis[k]['analysis'] is not None]) / len(home_analysis.keys()) * 100
        away_confidence = len([k for k in away_analysis.keys() if k not in ['team_total', 'team_average'] and away_analysis[k]['analysis'] is not None]) / len(away_analysis.keys()) * 100
        
        st.metric("Home Team Confidence", f"{home_confidence:.1f}%")
        st.metric("Away Team Confidence", f"{away_confidence:.1f}%")

def run_comprehensive_analysis(home_team: str, away_team: str):
    """Run comprehensive NBA analysis for both teams."""
    
    st.markdown("## 🔍 Starting Comprehensive Analysis")
    st.info("This analysis will examine 5 years of historical data, home/away splits, and advanced statistical factors for each player.")
    
    start_time = time.time()
    
    # Get team rosters
    home_roster = get_team_roster(home_team)
    away_roster = get_team_roster(away_team)
    
    if not home_roster or not away_roster:
        st.error("❌ Could not retrieve team rosters")
        return None, None
    
    st.success(f"✅ Home Team ({home_team}): {len(home_roster)} players")
    st.success(f"✅ Away Team ({away_team}): {len(away_roster)} players")
    
    # Analyze home team
    st.markdown(f"### 🏠 Analyzing {home_team} (Home)")
    home_analysis = nba_analyzer.get_team_analysis(home_roster, away_team, is_home=True)
    
    # Analyze away team
    st.markdown(f"### ✈️ Analyzing {away_team} (Away)")
    away_analysis = nba_analyzer.get_team_analysis(away_roster, home_team, is_home=False)
    
    end_time = time.time()
    analysis_time = end_time - start_time
    
    st.success(f"🎉 Analysis completed in {analysis_time:.1f} seconds!")
    
    return home_analysis, away_analysis

def main():
    st.markdown('<h1 class="main-header">🏀 NBA Comprehensive Predictions</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="analysis-card">
        <h3>🔍 Comprehensive NBA Player Analysis & Predictions</h3>
        <p>Get detailed predictions for every player using 5 years of historical data, home/away splits, and advanced statistical analysis!</p>
        <ul>
            <li>📊 <strong>5 Years of Data:</strong> Comprehensive historical analysis from 2020-2025</li>
            <li>🏠 <strong>Home/Away Splits:</strong> Performance differences in different venues</li>
            <li>🎯 <strong>Opponent History:</strong> How players perform against specific teams</li>
            <li>📈 <strong>Recent Form:</strong> Last 10 games performance trends</li>
            <li>🤖 <strong>Advanced Metrics:</strong> Efficiency, consistency, and clutch factors</li>
            <li>⚡ <strong>Real-time Analysis:</strong> Live data from NBA API</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("⚙️ Analysis Settings")
    
    st.sidebar.markdown("### 📊 Data Sources")
    st.sidebar.info("""
    - **NBA API**: Official statistics and game logs
    - **5 Seasons**: 2020-21 through 2024-25
    - **Real-time**: Live data fetching with caching
    - **Comprehensive**: All available player statistics
    """)
    
    st.sidebar.markdown("### 🎯 Prediction Factors")
    st.sidebar.info("""
    - **Career Averages**: 30% weight
    - **Recent Form**: 30% weight  
    - **Home/Away**: 20% weight
    - **Opponent History**: 25% weight
    - **Season Trends**: 15% weight
    - **Consistency**: 10% weight
    """)
    
    # Team selection
    st.markdown("## 🏀 Select Teams for Analysis")
    
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
    
    # Run analysis button
    if st.button("🚀 Start Comprehensive Analysis", type="primary", use_container_width=True):
        
        # Run analysis
        home_analysis, away_analysis = run_comprehensive_analysis(home_team, away_team)
        
        if home_analysis and away_analysis:
            # Display results
            st.markdown("## 📊 Analysis Results")
            
            # Team comparison
            display_team_comparison(home_analysis, away_analysis)
            
            # Charts
            create_prediction_charts(home_analysis, away_analysis)
            
            # Detailed player analysis
            st.markdown("## 👥 Player-by-Player Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"### 🏠 {home_team} Players")
                for player_name, player_data in home_analysis.items():
                    if player_name not in ['team_total', 'team_average']:
                        display_player_analysis(player_name, player_data)
            
            with col2:
                st.markdown(f"### ✈️ {away_team} Players")
                for player_name, player_data in away_analysis.items():
                    if player_name not in ['team_total', 'team_average']:
                        display_player_analysis(player_name, player_data)
            
            # Summary
            st.markdown("## 🎯 Game Prediction Summary")
            
            home_total = home_analysis['team_total']
            away_total = away_analysis['team_total']
            winner = home_team if home_total > away_total else away_team
            margin = abs(home_total - away_total)
            
            st.success(f"🏆 **Predicted Winner**: {winner}")
            st.info(f"📊 **Predicted Score**: {home_team} {home_total:.0f} - {away_team} {away_total:.0f}")
            st.info(f"🎯 **Predicted Margin**: {margin:.1f} points")
            st.info(f"⚡ **Total Points**: {home_total + away_total:.0f} points")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>🏀 Powered by NBA API • 📊 5 Years of Historical Data • 🤖 Advanced Statistical Analysis • ⚡ Real-time Predictions</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
