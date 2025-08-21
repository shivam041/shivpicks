"""
Local Team Database for NBA App
This eliminates the need for API calls to get team rosters, making the app much faster.
"""

import json
import os
from typing import Dict, List, Optional

# NBA Team Rosters Database (as of 2024-25 season)
TEAM_ROSTERS = {
    "ATL": [
        "Trae Young", "Dejounte Murray", "Bogdan Bogdanovic", "De'Andre Hunter", 
        "Clint Capela", "Onyeka Okongwu", "Saddiq Bey", "Jalen Johnson",
        "Kobe Bufkin", "AJ Griffin", "Wesley Matthews", "Garrison Mathews"
    ],
    "BOS": [
        "Jayson Tatum", "Jaylen Brown", "Kristaps Porzingis", "Derrick White",
        "Jrue Holiday", "Al Horford", "Payton Pritchard", "Sam Hauser",
        "Luke Kornet", "Oshae Brissett", "Dalano Banton", "Jordan Walsh"
    ],
    "BKN": [
        "Mikal Bridges", "Cameron Johnson", "Spencer Dinwiddie", "Dorian Finney-Smith",
        "Nic Claxton", "Ben Simmons", "Royce O'Neale", "Lonnie Walker IV",
        "Dennis Smith Jr.", "Trendon Watford", "Day'Ron Sharpe", "Noah Clowney"
    ],
    "CHA": [
        "LaMelo Ball", "Terry Rozier", "Gordon Hayward", "P.J. Washington",
        "Mark Williams", "Brandon Miller", "Cody Martin", "JT Thor",
        "Nick Richards", "Bryce McGowens", "James Bouknight", "Leaky Black"
    ],
    "CHI": [
        "Zach LaVine", "DeMar DeRozan", "Nikola Vucevic", "Coby White",
        "Patrick Williams", "Ayo Dosunmu", "Andre Drummond", "Alex Caruso",
        "Jevon Carter", "Torrey Craig", "Julian Phillips", "Dalen Terry"
    ],
    "CLE": [
        "Donovan Mitchell", "Darius Garland", "Evan Mobley", "Jarrett Allen",
        "Max Strus", "Caris LeVert", "Isaac Okoro", "Georges Niang",
        "Tristan Thompson", "Sam Merrill", "Craig Porter Jr.", "Emoni Bates"
    ],
    "DAL": [
        "Luka Doncic", "Kyrie Irving", "Tim Hardaway Jr.", "Grant Williams",
        "Dereck Lively II", "Josh Green", "Dante Exum", "Dwight Powell",
        "Seth Curry", "Maxi Kleber", "Jaden Hardy", "Olivier-Maxence Prosper"
    ],
    "DEN": [
        "Nikola Jokic", "Jamal Murray", "Michael Porter Jr.", "Aaron Gordon",
        "Kentavious Caldwell-Pope", "Reggie Jackson", "Christian Braun", "Peyton Watson",
        "Zeke Nnaji", "Julian Strawther", "Hunter Tyson", "Braxton Key"
    ],
    "DET": [
        "Cade Cunningham", "Jaden Ivey", "Ausar Thompson", "Isaiah Stewart",
        "Jalen Duren", "Bojan Bogdanovic", "Alec Burks", "Marcus Sasser",
        "James Wiseman", "Killian Hayes", "Marvin Bagley III", "Isaiah Livers"
    ],
    "GSW": [
        "Stephen Curry", "Klay Thompson", "Draymond Green", "Andrew Wiggins",
        "Kevon Looney", "Chris Paul", "Gary Payton II", "Jonathan Kuminga",
        "Moses Moody", "Brandin Podziemski", "Trayce Jackson-Davis", "Cory Joseph"
    ],
    "HOU": [
        "Fred VanVleet", "Jalen Green", "Dillon Brooks", "Jabari Smith Jr.",
        "Alperen Sengun", "Amen Thompson", "Cam Whitmore", "Tari Eason",
        "Jeff Green", "Aaron Holiday", "Jock Landale", "Reggie Bullock Jr."
    ],
    "IND": [
        "Tyrese Haliburton", "Bruce Brown", "Bennedict Mathurin", "Obi Toppin",
        "Myles Turner", "Buddy Hield", "TJ McConnell", "Aaron Nesmith",
        "Jalen Smith", "Isaiah Jackson", "Andrew Nembhard", "Jordan Nwora"
    ],
    "LAC": [
        "Kawhi Leonard", "Paul George", "Russell Westbrook", "Ivica Zubac",
        "Terance Mann", "Norman Powell", "Mason Plumlee", "Bones Hyland",
        "Amir Coffey", "Brandon Boston Jr.", "Kobe Brown", "Moussa Diabate"
    ],
    "LAL": [
        "LeBron James", "Anthony Davis", "Austin Reaves", "D'Angelo Russell",
        "Taurean Prince", "Rui Hachimura", "Jarred Vanderbilt", "Gabe Vincent",
        "Christian Wood", "Jaxson Hayes", "Max Christie", "Cam Reddish"
    ],
    "MEM": [
        "Ja Morant", "Desmond Bane", "Jaren Jackson Jr.", "Marcus Smart",
        "Steven Adams", "Luke Kennard", "Derrick Rose", "Ziaire Williams",
        "Santi Aldama", "Xavier Tillman", "David Roddy", "Kennedy Chandler"
    ],
    "MIA": [
        "Jimmy Butler", "Bam Adebayo", "Tyler Herro", "Kyle Lowry",
        "Duncan Robinson", "Caleb Martin", "Haywood Highsmith", "Jaime Jaquez Jr.",
        "Orlando Robinson", "Nikola Jovic", "R.J. Hampton", "Cole Swider"
    ],
    "MIL": [
        "Giannis Antetokounmpo", "Damian Lillard", "Khris Middleton", "Brook Lopez",
        "Malik Beasley", "Bobby Portis", "Pat Connaughton", "MarJon Beauchamp",
        "Andre Jackson Jr.", "AJ Green", "Chris Livingston", "Lindell Wigginton"
    ],
    "MIN": [
        "Anthony Edwards", "Karl-Anthony Towns", "Rudy Gobert", "Mike Conley",
        "Jaden McDaniels", "Naz Reid", "Nickeil Alexander-Walker", "Kyle Anderson",
        "Shake Milton", "Troy Brown Jr.", "Josh Minott", "Luka Garza"
    ],
    "NOP": [
        "Zion Williamson", "Brandon Ingram", "CJ McCollum", "Jonas Valanciunas",
        "Herbert Jones", "Trey Murphy III", "Dyson Daniels", "Larry Nance Jr.",
        "Jose Alvarado", "Jordan Hawkins", "Naji Marshall", "E.J. Liddell"
    ],
    "NYK": [
        "Jalen Brunson", "Julius Randle", "RJ Barrett", "Mitchell Robinson",
        "Donte DiVincenzo", "Immanuel Quickley", "Josh Hart", "Isaiah Hartenstein",
        "Quentin Grimes", "Evan Fournier", "Jericho Sims", "Miles McBride"
    ],
    "OKC": [
        "Shai Gilgeous-Alexander", "Chet Holmgren", "Jalen Williams", "Josh Giddey",
        "Luguentz Dort", "Isaiah Joe", "Cason Wallace", "Kenrich Williams",
        "Jaylin Williams", "Ousmane Dieng", "Aaron Wiggins", "Tre Mann"
    ],
    "ORL": [
        "Paolo Banchero", "Franz Wagner", "Wendell Carter Jr.", "Markelle Fultz",
        "Cole Anthony", "Jalen Suggs", "Jonathan Isaac", "Moritz Wagner",
        "Gary Harris", "Chuma Okeke", "Anthony Black", "Jett Howard"
    ],
    "PHI": [
        "Joel Embiid", "Tyrese Maxey", "Tobias Harris", "De'Anthony Melton",
        "Kelly Oubre Jr.", "Marcus Morris Sr.", "Robert Covington", "Patrick Beverley",
        "Paul Reed", "Furkan Korkmaz", "Danuel House Jr.", "Jaden Springer"
    ],
    "PHX": [
        "Kevin Durant", "Devin Booker", "Bradley Beal", "Jusuf Nurkic",
        "Eric Gordon", "Grayson Allen", "Drew Eubanks", "Josh Okogie",
        "Keita Bates-Diop", "Yuta Watanabe", "Bol Bol", "Saben Lee"
    ],
    "POR": [
        "Scoot Henderson", "Anfernee Simons", "Jerami Grant", "Deandre Ayton",
        "Malcolm Brogdon", "Matisse Thybulle", "Robert Williams III", "Shaedon Sharpe",
        "Toumani Camara", "Kris Murray", "Jabari Walker", "Duop Reath"
    ],
    "SAC": [
        "De'Aaron Fox", "Domantas Sabonis", "Keegan Murray", "Harrison Barnes",
        "Kevin Huerter", "Malik Monk", "Trey Lyles", "Davion Mitchell",
        "Chris Duarte", "Sasha Vezenkov", "JaVale McGee", "Kessler Edwards"
    ],
    "SAS": [
        "Victor Wembanyama", "Devin Vassell", "Jeremy Sochan", "Keldon Johnson",
        "Zach Collins", "Tre Jones", "Malaki Branham", "Cedi Osman",
        "Doug McDermott", "Sandro Mamukelashvili", "Blake Wesley", "Sidy Cissoko"
    ],
    "TOR": [
        "Scottie Barnes", "Pascal Siakam", "OG Anunoby", "Dennis Schroder",
        "Jakob Poeltl", "Gary Trent Jr.", "Chris Boucher", "Malachi Flynn",
        "Precious Achiuwa", "Jalen McDaniels", "Gradey Dick", "Garrett Temple"
    ],
    "UTA": [
        "Lauri Markkanen", "Jordan Clarkson", "Collin Sexton", "John Collins",
        "Walker Kessler", "Talen Horton-Tucker", "Kelly Olynyk", "Ochai Agbaji",
        "Keyonte George", "Taylor Hendricks", "Brice Sensabaugh", "Luka Samanic"
    ],
    "WAS": [
        "Kyle Kuzma", "Jordan Poole", "Tyus Jones", "Deni Avdija",
        "Daniel Gafford", "Bilal Coulibaly", "Corey Kispert", "Landry Shamet",
        "Johnny Davis", "Anthony Gill", "Delon Wright", "Mike Muscala"
    ]
}

def get_team_roster(team_abbreviation: str) -> List[str]:
    """
    Get team roster from local database.
    
    Args:
        team_abbreviation (str): Team abbreviation (e.g., 'LAL', 'GSW')
        
    Returns:
        List[str]: List of player names on the team
    """
    return TEAM_ROSTERS.get(team_abbreviation.upper(), [])

def get_all_team_abbreviations() -> List[str]:
    """
    Get all available team abbreviations.
    
    Returns:
        List[str]: List of all team abbreviations
    """
    return list(TEAM_ROSTERS.keys())

def get_team_name(team_abbreviation: str) -> str:
    """
    Get full team name from abbreviation.
    
    Args:
        team_abbreviation (str): Team abbreviation
        
    Returns:
        str: Full team name
    """
    team_names = {
        "ATL": "Atlanta Hawks", "BOS": "Boston Celtics", "BKN": "Brooklyn Nets",
        "CHA": "Charlotte Hornets", "CHI": "Chicago Bulls", "CLE": "Cleveland Cavaliers",
        "DAL": "Dallas Mavericks", "DEN": "Denver Nuggets", "DET": "Detroit Pistons",
        "GSW": "Golden State Warriors", "HOU": "Houston Rockets", "IND": "Indiana Pacers",
        "LAC": "Los Angeles Clippers", "LAL": "Los Angeles Lakers", "MEM": "Memphis Grizzlies",
        "MIA": "Miami Heat", "MIL": "Milwaukee Bucks", "MIN": "Minnesota Timberwolves",
        "NOP": "New Orleans Pelicans", "NYK": "New York Knicks", "OKC": "Oklahoma City Thunder",
        "ORL": "Orlando Magic", "PHI": "Philadelphia 76ers", "PHX": "Phoenix Suns",
        "POR": "Portland Trail Blazers", "SAC": "Sacramento Kings", "SAS": "San Antonio Spurs",
        "TOR": "Toronto Raptors", "UTA": "Utah Jazz", "WAS": "Washington Wizards"
    }
    return team_names.get(team_abbreviation.upper(), team_abbreviation)

def is_valid_team(team_abbreviation: str) -> bool:
    """
    Check if team abbreviation is valid.
    
    Args:
        team_abbreviation (str): Team abbreviation to check
        
    Returns:
        bool: True if valid, False otherwise
    """
    return team_abbreviation.upper() in TEAM_ROSTERS

def get_player_count(team_abbreviation: str) -> int:
    """
    Get number of players on a team.
    
    Args:
        team_abbreviation (str): Team abbreviation
        
    Returns:
        int: Number of players on the team
    """
    roster = get_team_roster(team_abbreviation)
    return len(roster) if roster else 0
