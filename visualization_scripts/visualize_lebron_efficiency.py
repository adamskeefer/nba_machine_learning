import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

input_file = os.path.join('data', 'lebron', 'efficiency_by_opp_last_ten.csv')
efficiency_df = pd.read_csv(input_file)

plt.figure(figsize=(14,8))
sns.barplot(x='OPPONENT', y='TS%', hue='SEASON', data=efficiency_df, palette='viridis')
plt.title("LeSunshine's True Shooting Percentage (TS%) by Opp (Last Ten Seasons)")
plt.xlabel('Opp')
plt.ylabel('True Shooting Percentage (TS%)')
plt.xticks(rotation=45)
plt.legend(title='Season', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

plt.figure(figsize=(14,8))
sns.barplot(x='OPPONENT', y='PER', hue='SEASON', data=efficiency_df, palette='magma')
plt.title("The King's Player Efficiency Rating (PER) by Opp (Last Ten Seasons)")
plt.xlabel('Opp')
plt.ylabel('Player Efficiency Rating (PER)')
plt.xticks(rotation=45)
plt.legend(title='Season', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()