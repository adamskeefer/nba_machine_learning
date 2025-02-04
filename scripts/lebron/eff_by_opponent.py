## THIS SCRIPT FETCHES LEBRON'S EFFICIENCY OVER THE LAST FIVE SEASONS BY OPPONENT ##
from nba_api.stats.endpoints import playergamelog
from nba_api.stats.static import players
import pandas as pd
import os

def calculate_efficiency(row):
    ts = (row['PTS'] / (2 * (row['FGA'] + 0.44 * row['FTA']))) * 100
    per = (row['PTS'] + row['REB'] + row['AST'] + row['STL'] + row['BLK']
           - row['FGA'] - row['FTA'] - row['TOV'])
    
    return pd.Series({
        'TS%': ts,
        'PER': per
    })

lebron = players.find_players_by_full_name("LeBron James")[0]
lebron_id = lebron['id']

seasons = ['2014-15', '2015-16', '2016-17', '2017-18', '2018-19', '2019-20', '2020-21', '2021-22', '2022-23', '2024-25']

gamelogs = pd.DataFrame()

for season in seasons:
    season_gamelogs = playergamelog.PlayerGameLog(player_id=lebron_id, season=season)
    gamelogs_df = season_gamelogs.get_data_frames()[0]
    gamelogs_df['SEASON'] = season
    gamelogs = pd.concat([gamelogs, gamelogs_df], ignore_index=True)

efficiency_df = gamelogs.apply(calculate_efficiency, axis=1)
efficiency_df = pd.concat([gamelogs[['MATCHUP', 'PTS', 'REB', 'AST', 'STL', 'BLK', 'FGA', 'FTA', 'TOV', 'SEASON']], efficiency_df], axis=1)

efficiency_df['OPPONENT'] = efficiency_df['MATCHUP'].apply(lambda x: x.split()[-1])

opp_efficiency = efficiency_df.groupby(['OPPONENT', 'SEASON']).agg({
    'TS%': 'mean',
    'PER': 'mean',
    'PTS': 'mean',
    'REB': 'mean',
    'AST': 'mean',
    'STL': 'mean',
    'BLK': 'mean'
}).reset_index()

data_directory = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data', 'lebron')
os.makedirs(data_directory, exist_ok=True)
data_file = os.path.join(data_directory, 'efficiency_by_opp_last_ten.csv')
opp_efficiency.to_csv(data_file, index=False)
print("Data saved!")

print(opp_efficiency.head())