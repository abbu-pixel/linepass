# app.py
from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load trained model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# Define class names
class_names = ["Setosa", "Versicolor", "Virginica"]

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get input values from form
        features = [float(x) for x in request.form.values()]
        features = np.array([features])
        prediction = model.predict(features)[0]
        predicted_class = class_names[prediction]
        return render_template("index.html", result=f"Predicted Species: {predicted_class}")
    except Exception as e:
        return render_template("index.html", result=f"Error: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)
