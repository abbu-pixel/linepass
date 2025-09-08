# app.py

from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

# Load model and features
model = joblib.load("iris_model.pkl")
features = joblib.load("iris_features.pkl")
target_map = {0: "Setosa", 1: "Versicolor", 2: "Virginica"}

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    if request.method == "POST":
        # Get input from form
        input_data = [float(request.form[feature]) for feature in features]
        df = pd.DataFrame([input_data], columns=features)

        # Predict
        pred_class = model.predict(df)[0]
        prediction = target_map[pred_class]

    return render_template("index.html", features=features, prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
