from nba_api.stats.endpoints import playergamelog
from nba_api.stats.static import teams
from nba_api.stats.endpoints import commonteamroster
import pandas as pd

def transform_matchup(matchup):
    # Split the string on ' ' (space)
    teams = matchup.split(' ')
    # Determine if the game is home ('vs.') or away ('@')
    if '@' in teams:
        # For away games, format is '@Opponent'
        return f"@{teams[2]}"
    else:
        # For home games, format is 'Opponent'
        return teams[2]


nuggets = [team for team in teams.get_teams() if team['nickname'] == 'Nuggets'][0]
nuggets_id = nuggets['id']

suns = [team for team in teams.get_teams() if team['nickname'] == 'Suns'][0]
suns_id = suns['id']

roster_nuggets = commonteamroster.CommonTeamRoster(team_id=nuggets_id).get_data_frames()[0]
roster = commonteamroster.CommonTeamRoster(team_id=suns_id).get_data_frames()[0]

all_players_game_logs = []

# Loop through each player on the roster
for index, player in roster.iterrows():
    player_id = player['PLAYER_ID']

    # Fetch the game log for the player for the 2023-24 season
    gamelog = playergamelog.PlayerGameLog(player_id=player_id, season='2023-24').get_data_frames()[0]

    # Add a 'PLAYER_NAME' column to the DataFrame
    gamelog['PLAYER_NAME'] = player['PLAYER']

    # Append the DataFrame to our list
    all_players_game_logs.append(gamelog)

# Concatenate all the individual player game logs into a single DataFrame
#all_nuggets_game_logs = pd.concat(all_players_game_logs, ignore_index=True)
all_suns_game_logs = pd.concat(all_players_game_logs, ignore_index=True)

# Columns to be removed
columns_to_remove = ['SEASON_ID', 'GAME_DATE','Player_ID', 'Game_ID', 'VIDEO_AVAILABLE']

# Remove the specified columns
#all_nuggets_game_logs_cleaned = all_nuggets_game_logs.drop(columns=columns_to_remove)
#all_nuggets_game_logs_cleaned['MATCHUP'] = all_nuggets_game_logs_cleaned['MATCHUP'].apply(transform_matchup)


all_suns_game_logs_cleaned = all_suns_game_logs.drop(columns=columns_to_remove)
all_suns_game_logs_cleaned['MATCHUP'] = all_suns_game_logs_cleaned['MATCHUP'].apply(transform_matchup)


# Specify the filename
# csv_filename = 'nuggets_game_logs_2023_24.csv'

# Save the DataFrame to a CSV file
# all_nuggets_game_logs_cleaned.to_csv(csv_filename, index=False)

# print(f"Saved DataFrame to '{csv_filename}'.")

csv_filename = 'suns_game_logs_2023_24.csv'

# Save the DataFrame to a CSV file
all_suns_game_logs_cleaned.to_csv(csv_filename, index=False)

print(f"Saved DataFrame to '{csv_filename}'.")


