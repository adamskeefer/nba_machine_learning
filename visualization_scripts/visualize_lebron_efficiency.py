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
plt.savefig(os.path.join('visualizations', 'lebron', 'ts_by_opp_barplot.png'))
plt.show()

plt.figure(figsize=(14,8))
sns.barplot(x='OPPONENT', y='PER', hue='SEASON', data=efficiency_df, palette='magma')
plt.title("The King's Player Efficiency Rating (PER) by Opp (Last Ten Seasons)")
plt.xlabel('Opp')
plt.ylabel('Player Efficiency Rating (PER)')
plt.xticks(rotation=45)
plt.legend(title='Season', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig(os.path.join('visualizations', 'lebron', 'per_by_opp_barplot.png'))
plt.show()

plt.figure(figsize=(16,10))
heatmap_data = efficiency_df.pivot_table(
    index="OPPONENT", 
    columns="SEASON", 
    values="TS%",
    aggfunc="mean"
)

heatmap_data['Average_TS%'] = heatmap_data.mean(axis=1)
heatmap_data = heatmap_data.sort_values(by="Average_TS%", ascending=False)
heatmap_data = heatmap_data.drop(columns=['Average_TS%'])

sns.heatmap(
    heatmap_data,
    cmap="coolwarm",
    annot=True,
    fmt=".2%",
    linewidth=0.5,
    cbar_kws={'label': 'True Shooting Percentage (TS%)'}
)

plt.title("Bron Bron's TS% By Opp", fontsize=16)
plt.xlabel("Season", fontsize=14)
plt.ylabel("Opp", fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join('visualizations', 'lebron', 'ts_by_opp_heat.png'))
plt.show()