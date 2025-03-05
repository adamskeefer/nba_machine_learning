import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier as KNN
from matplotlib.colors import ListedColormap
from sklearn.preprocessing import StandardScaler
import os

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

mins = X_train.min(axis=0) - 0.1
maxs = X_train.max(axis=0) + 0.1
x = np.arange(mins[0], maxs[0], 0.01)
y = np.arange(mins[1], maxs[1], 0.01)
X_grid, Y_grid = np.meshgrid(x, y)
coordinates = np.array([X_grid.ravel(), Y_grid.ravel()]).T

color = ('aquamarine', 'bisque', 'lightgrey')
cmap = ListedColormap(color)

K_vals = [1, 3, 9]

fig, axs = plt.subplots(2, 2, figsize=(10, 8), dpi=150, sharex=True, sharey=True)
fig.tight_layout()

for ax, K in zip(axs.ravel(), K_vals):
    knn = KNN(n_neighbors=K)
    knn.fit(X_train, y_train)

    Z = knn.predict(coordinates)
    Z = Z.reshape(X_grid.shape)

    # Plot the decision regions
    ax.pcolormesh(X_grid, Y_grid, Z, cmap=cmap, shading='nearest')
    ax.contour(X_grid, Y_grid, Z, colors='black', linewidths=0.5)

    scatter = ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='coolwarm', s=40, edgecolors='k')

    ax.set_title(f'{K}-NN Decision Regions', fontsize=12)
    ax.tick_params(axis='both', labelsize=10)

    train_accuracy = knn.score(X_train, y_train)
    test_accuracy = knn.score(X_test, y_test)

    print('The accuracy for K={} on the train data is {:.3f}'.format(K, test_accuracy))
    print('The accuracy for K={} on the test data is {:.3f}'.format(K, test_accuracy))
    ax.text(0.05, 0.95, f'Train: {train_accuracy:.3f}\nTest: {test_accuracy:.3f}',
            transform=ax.transAxes, fontsize=10, verticalalignment='top', horizontalalignment='left')


visualization_directory = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'visualizations')
save_file = os.path.join(visualization_directory, 'knn_practice.png')

plt.suptitle('Decision Boundaries and Accuracy for k-NN with Different k Values', fontsize=16)
plt.xlabel('E_DEF_RATING', fontsize=14)
plt.ylabel('TS_PCT', fontsize=14)
plt.xticks(rotation=45)

plt.tight_layout()
plt.subplots_adjust(top=0.9)
plt.savefig(save_file)
plt.show()