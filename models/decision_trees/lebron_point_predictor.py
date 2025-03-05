import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import os

data_directory = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data', 'lebron')
data_file = os.path.join(data_directory, 'lebron_gamelogs_last_five.csv')
data = pd.read_csv(data_file)


data['GAME_DATE'] = pd.to_datetime(data['GAME_DATE'], format='%b %d, %Y')

data['WIN'] = data['WL'].apply(lambda x: 1 if x == 'W' else 0)
data['HOME'] = data['MATCHUP'].apply(lambda x: 1 if "vs" in x else 0)

data.sort_values(['SEASON_ID', 'GAME_DATE'], inplace=True)
print(data.head())
def calculate_avg_pts_last_5(season_df):
    season_df['AVG_PTS_LAST_5'] = season_df['PTS'].rolling(window=5, min_periods=5).mean()
    return season_df.iloc[5:]


data = (
    data.groupby('SEASON_ID', group_keys=False, as_index=False)
    .apply(lambda df: calculate_avg_pts_last_5(df))
    .reset_index(drop=True)
)
data.dropna(subset=['AVG_PTS_LAST_5'], inplace=True)
data['Game_Date_Diff'] = data.groupby('SEASON_ID')['GAME_DATE'].diff().dt.days
data['BACK_TO_BACK'] = data['Game_Date_Diff'].apply(lambda x: 1 if x == 1 else 0)
data['OPPONENT'] = data['MATCHUP'].apply(lambda x: x.split()[-1])

data['AVG_PTS_VS_OPPONENT'] = data.groupby('OPPONENT')['PTS'].transform(lambda x: x.expanding().mean())

data.drop(columns=['Game_Date_Diff', 'OPPONENT'], inplace=True)

X = data[['HOME', 'AVG_PTS_LAST_5', 'BACK_TO_BACK', 'AVG_PTS_VS_OPPONENT']]
y = data['PTS'] 
trainX, testX, trainY, testY = train_test_split(X, y, test_size=0.2, random_state=42)
model = GradientBoostingRegressor(random_state=42)
model.fit(trainX, trainY)

predY = model.predict(testX)
mae = mean_absolute_error(testY, predY)
rmse = mean_squared_error(testY, predY)

print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
