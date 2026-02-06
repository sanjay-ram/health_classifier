import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn import tree
import matplotlib.pyplot as plt

df = pd.read_csv('daten.csv')
print(df.head())

X = df[['Blutdruck_sys', 'Cholesterin']]
y = df['Diagnose']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
classifier = DecisionTreeClassifier(random_state=42)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')
plt.figure(figsize=(12,8))
tree.plot_tree(classifier, feature_names=['Blutdruck_sys', 'Cholesterin'], class_names=['gesund', 'krank'], filled=True)
plt.show()