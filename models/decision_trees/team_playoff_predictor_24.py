import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_graphviz, export_text, plot_tree
import matplotlib.pyplot as plt
import numpy as np

data_directory = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data', 'teams')
data_file = os.path.join(data_directory, 'team_stats_23_24.csv')
data = pd.read_csv(data_file)

data['RPG'] = data['REB'] / 82
features = ['E_DEF_RATING', 'TS_PCT', 'EFG_PCT']
X = data[features]
y = data['ClinchedPlayoffBirth']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=30)
model = DecisionTreeClassifier(criterion='entropy', max_depth=4, random_state=30)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = np.mean(y_pred == y_test)

print(f"Accuracy: {accuracy * 100:.2f}%")
print("Feature Importances:", model.feature_importances_)

tree_rules = export_text(model, feature_names=features)
print(tree_rules)
plt.figure(figsize=(10, 6))
plot_tree(model, feature_names=features, class_names=["No Playoff", "Playoff"], filled=True, rounded=True)
plt.title("Decision Tree")
data_directory = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'visualizations')
data_file = os.path.join(data_directory, 'decision_tree_bpg_spg.png')
plt.savefig(data_file, dpi=300, bbox_inches="tight")
