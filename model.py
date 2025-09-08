# model.py

from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import joblib

# Load the Iris dataset
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = pd.Series(iris.target)

# Train the model
model = RandomForestClassifier()
model.fit(X, y)

# Save the model and feature names
joblib.dump(model, 'iris_model.pkl')
joblib.dump(iris.feature_names, 'iris_features.pkl')

print("Model training complete. Files saved: iris_model.pkl, iris_features.pkl")
