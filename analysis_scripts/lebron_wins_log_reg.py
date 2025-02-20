import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

input_file = os.path.join('data', 'lebron', 'lebron_gamelogs_last_five.csv')
data = pd.read_csv(input_file)

data['WIN'] = data['WL'].apply(lambda x: 1 if x == 'W' else 0)
data['HOME'] = data['MATCHUP'].apply(lambda x: 0 if '@' in x else 1)

features = ['PTS', 'REB', 'AST', 'STL', 'BLK', 'TOV', 'FGA', 'FTA', 'FG3A', 'HOME']
target = 'WIN'

data = data.dropna(subset=features + [target])

x = data[features]
y = data[target]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

model = LogisticRegression()
model.fit(x_train, y_train)
y_pred = model.predict(x_test)

accuracy = accuracy_score(y_test, y_pred)
classification = classification_report(y_test, y_pred)
confusion = confusion_matrix(y_test, y_pred)

print(f"Accuracy: {accuracy}")
print("Classification Report:")
print(classification)
print("Confusion Matrix:")
print(confusion)

sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.savefig(os.path.join('visualizations', 'lebron', 'win_logistic_confusion.png'))
plt.close()