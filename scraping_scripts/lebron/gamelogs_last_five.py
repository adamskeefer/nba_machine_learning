from nba_api.stats.endpoints import playergamelog
from nba_api.stats.static import players
import pandas as pd
import os

lebron = players.find_players_by_full_name("LeBron James")[0]
lebron_id = lebron['id']

seasons = ['2020-21', '2021-22', '2022-23', '2023-24', '2024-25']
all_gamelogs = pd.DataFrame()

for season in seasons:
    gamelogs = playergamelog.PlayerGameLog(player_id=lebron_id, season=season)
    gamelogs_df = gamelogs.get_data_frames()[0]
    gamelogs_df['SEASON'] = season
    all_gamelogs = pd.concat([all_gamelogs, gamelogs_df], ignore_index=True)

data_directory = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data', 'lebron')
output_file = os.path.join(data_directory, 'lebron_gamelogs_last_five.csv')

all_gamelogs.to_csv(output_file, index=False)
print("Data saved!")

print(all_gamelogs.head())