import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import os
import numpy as np
import matplotlib.pyplot as plt

data_directory = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data', 'teams')
data_file = os.path.join(data_directory, 'team_stats_23_24.csv')
data = pd.read_csv(data_file)
features = ['E_DEF_RATING', 'TS_PCT']
X = data[features]
y = data['ClinchedPlayoffBirth']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

svm_model = SVC(kernel='linear')
svm_model.fit(X_train, y_train)
train_accuracy = svm_model.score(X_train, y_train)
test_accuracy = svm_model.score(X_test, y_test)

print(f"Training accuracy: {train_accuracy:.3f}")
print(f"Testing accuracy: {test_accuracy:.3f}")

mins = X_train.min(axis=0) - 0.1
maxs = X_train.max(axis=0) + 0.1
x = np.arange(mins[0], maxs[0], 0.01)
y = np.arange(mins[1], maxs[1], 0.01)
X_grid, Y_grid = np.meshgrid(x, y)
coordinates = np.array([X_grid.ravel(), Y_grid.ravel()]).T
Z = svm_model.predict(coordinates)
Z = Z.reshape(X_grid.shape)
plt.contourf(X_grid, Y_grid, Z, cmap='coolwarm', alpha=0.6)
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='coolwarm', edgecolors='k')
plt.title('SVM Decision Boundary')
plt.xlabel('E_DEF_RATING')
plt.ylabel('TS_PCT')
plt.show()