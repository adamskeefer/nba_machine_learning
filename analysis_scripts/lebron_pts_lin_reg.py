import pandas as pd
import os
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from scipy.stats import zscore
import matplotlib.pyplot as plt
import numpy as np

input_file = os.path.join('data', 'lebron', 'lebron_gamelogs_last_five.csv')
data = pd.read_csv(input_file)

z_scores = zscore(data['PTS'])
abs_z_scores = abs(z_scores)
filtered_entries = (abs_z_scores < 3)
data = data[filtered_entries]

features = ['MIN', 'FGA', 'FTA', 'FG3A']
target = 'PTS'

data = data.dropna(subset=features + [target])

x = data[features]
y = data[target]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(x_train, y_train)

y_pred = model.predict(x_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
print(f"R2 Score: {r2}")

errors = np.abs(y_test - y_pred)
plt.figure(figsize=(8,6))
scatter = plt.scatter(y_test, y_pred, c=errors, cmap='viridis', s=50, alpha=0.7, edgecolor='k')

plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--', linewidth=2)
cbar = plt.colorbar(scatter)
cbar.set_label('Error Magnitude')

plt.xlabel("Actual PTS")
plt.ylabel("Predicted PTS")
plt.title("Actual vs Predicted PTS with Error Magnitude")
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()

plt.savefig(os.path.join('visualizations', 'lebron', 'actual_vs_predicted_pts.png'))
plt.close()