import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
import os

data_directory = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data', 'lebron')
data_file = os.path.join(data_directory, 'lebron_gamelogs_last_five.csv')
data = pd.read_csv(data_file)

data['WIN'] = data['WL'].apply(lambda x: 1 if x == 'W' else 0)
features = ['PTS', 'FG_PCT', 'FG3_PCT', 'AST', 'MIN']
X = data[features]
y = data['WIN']

trainX, testX, trainY, testY = train_test_split(X, y, test_size=0.2, random_state=50)

base_estimator = DecisionTreeClassifier(
    max_depth=6,
    random_state=None
)

ada = AdaBoostClassifier(
    estimator=base_estimator,
    n_estimators=50,
    learning_rate=1.0,
    random_state=42
)
ada.fit(trainX, trainY)

train_accuracy = ada.score(trainX, trainY)
test_accuracy = ada.score(testX, testY)

print(f"Training Accuracy: {train_accuracy:.2f}")
print(f"Testing Accuracy: {test_accuracy:.2f}")

y_pred = ada.predict(testX)
print("\nClassification Report:")
print(classification_report(testY, y_pred))