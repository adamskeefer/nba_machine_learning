import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import seaborn as sns

data_directory = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data', 'teams')
csv_files = [f for f in os.listdir(data_directory) if f.endswith('.csv')]

visualization_directory = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'visualizations')
os.makedirs(visualization_directory, exist_ok=True)

accuracies = []
feature_importances_list = []
seasons = []


for csv_file in csv_files:
    data_file = os.path.join(data_directory, csv_file)
    data = pd.read_csv(data_file)
    
    data['PPG'] = data['PTS'] / 82
    features = ['E_DEF_RATING', 'TS_PCT', 'EFG_PCT', 'PPG']
    X = data[features]
    y = data['ClinchedPlayoffBirth']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=30)
    
    model = DecisionTreeClassifier(criterion='entropy', max_depth=6, random_state=30)
    model.fit(X_train, y_train)
    

    y_pred = model.predict(X_test)
    accuracy = np.mean(y_pred == y_test)

    feature_importances = model.feature_importances_
    
    accuracies.append(accuracy * 100)
    feature_importances_list.append(feature_importances)
    seasons.append(csv_file.split('.')[0])
    
  
    plt.figure(figsize=(10, 6))
    plot_tree(model, feature_names=features, class_names=["No Playoff", "Playoff"], filled=True, rounded=True)
    plt.title(f"Decision Tree for {csv_file}")
    tree_image_file = os.path.join(visualization_directory, f'decision_tree_{csv_file.split(".")[0]}.png')
    plt.savefig(tree_image_file, dpi=300, bbox_inches="tight")
    plt.close()


sns.set_theme(style="whitegrid")
plt.figure(figsize=(10, 6))
plt.plot(seasons, accuracies, marker='o', color='teal', linestyle='-', markersize=8, label='Accuracy')
for i, txt in enumerate(accuracies):
    plt.annotate(f'{txt:.2f}%', (seasons[i], accuracies[i]), textcoords="offset points", xytext=(0, 10), ha='center', fontsize=10)
plt.title('Model Accuracy Across NBA Seasons', fontsize=14, fontweight='bold')
plt.xlabel('Season', fontsize=12)
plt.ylabel('Accuracy (%)', fontsize=12)
plt.xticks(rotation=45, ha='right', fontsize=10)
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.tight_layout()
accuracy_plot_file = os.path.join(visualization_directory, 'accuracy_over_seasons_new.png')
plt.savefig(accuracy_plot_file, dpi=300)
plt.close()


seasons = ['2023-24', '2022-23', '2021-22', '2020-21', '2019-20']  # Example seasons
plt.figure(figsize=(12, 8))
colors = sns.color_palette("Set2", n_colors=len(features))
markers = ['o', 's', 'D', '^', 'x']
for i, feature in enumerate(features):
    importance_values = [imp[i] for imp in feature_importances_list]
    plt.scatter(seasons, importance_values, label=feature, color=colors[i], marker=markers[i], s=100)

# Add labels and title
plt.title('Feature Importances Across NBA Seasons', fontsize=16, fontweight='bold')
plt.xlabel('Season', fontsize=14)
plt.ylabel('Feature Importance', fontsize=14)
plt.xticks(rotation=45)
plt.legend(title='Features', loc='upper left')
plt.grid(True, axis='y', linestyle='--', alpha=0.6)
plt.tight_layout()
feature_importance_plot_file = os.path.join(visualization_directory, 'feature_importances_scatter_new.png')
plt.savefig(feature_importance_plot_file, dpi=300)
plt.close()


print("Model performance and visualizations saved!")

results_df = pd.DataFrame({
    'season': seasons,
    'accuracy': accuracies,
    'average_feature_importances': feature_importances_list
})
results_summary_file = os.path.join(visualization_directory, 'model_results_summary.csv')
results_df.to_csv(results_summary_file, index=False)