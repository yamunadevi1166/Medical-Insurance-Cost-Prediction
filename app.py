
from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# Load the trained model
model = joblib.load("insurance_model.joblib")


@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None

    if request.method == "POST":
        age = int(request.form["age"])
        sex = request.form["sex"]
        bmi = float(request.form["bmi"])
        children = int(request.form["children"])
        smoker = request.form["smoker"]
        region = request.form["region"]

        # Prepare input data
        user_data = pd.DataFrame([{
            "age": age,
            "sex": sex,
            "bmi": bmi,
            "children": children,
            "smoker": smoker,
            "region": region
        }])

        # Predict insurance cost
        prediction = round(float(model.predict(user_data)[0]), 2)

    return render_template("index.html", prediction=prediction)


if __name__ == "__main__":
    app.run(debug=True)
