import pandas as pd
from nba_api.stats.endpoints import leaguedashteamstats, leaguestandings
import os

team_stats = leaguedashteamstats.LeagueDashTeamStats(season='2019-20', measure_type_detailed_defense='Advanced').get_data_frames()[0]
standings = leaguestandings.LeagueStandings(season='2019-20').get_data_frames()[0]
playoff_data = standings[['TeamID', 'ClinchedPlayoffBirth']]
merged_data = pd.merge(team_stats, playoff_data, left_on='TEAM_ID', right_on='TeamID')
merged_data.loc[
    merged_data['TEAM_NAME'].isin(['Portland Trail Blazers']),
    'ClinchedPlayoffBirth'
] = 1

team_stats_tuah = leaguedashteamstats.LeagueDashTeamStats(season='2019-20').get_data_frames()[0]
merged_data_tuah = pd.merge(merged_data, team_stats_tuah, left_on="TEAM_ID", right_on='TEAM_ID')
merged_data = merged_data_tuah

data_directory = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data', 'teams')
os.makedirs(data_directory, exist_ok=True)
data_file = os.path.join(data_directory, 'team_stats_19_20.csv')
merged_data.to_csv(data_file, index=False)
print("Data saved!")

print(team_stats.head())