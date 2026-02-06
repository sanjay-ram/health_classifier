import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score
from sklearn import tree
import matplotlib.pyplot as plt
from dash import Dash, html, dcc
import plotly.express as px
import base64
import os

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



app = Dash(__name__)


fig_scatter = px.scatter(
    df,
    x='Blutdruck_sys',
    y='Cholesterin',
    color='Diagnose',
    title='Blutdruck vs. Cholesterin',
    labels={'Blutdruck_sys':'Blutdruck', 'Cholesterin':'Cholesterin'}
)


app.layout = html.Div([
    html.H1("Patienten Dashboard"),
    
    html.P(f"Decision Tree Accuracy: {accuracy*100:.2f}%"),


    dcc.Graph(figure=fig_scatter)
])

if __name__ == "__main__":
    app.run(debug=True)