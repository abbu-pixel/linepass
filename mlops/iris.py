# predict_iris.py

import joblib
import json
import pandas as pd

# Load model and features
model = joblib.load("iris_model.pkl")
features = joblib.load("iris_features.pkl")

# Load sample input (from JSON)
with open("iris_input.json", "r") as f:
    input_data = json.load(f)

# Convert to DataFrame
X_new = pd.DataFrame([input_data], columns=features)

# Predict
prediction = model.predict(X_new)[0]

# Iris target mapping
target_map = {0: "setosa", 1: "versicolor", 2: "virginica"}

print(f"Predicted class: {target_map[prediction]}")
