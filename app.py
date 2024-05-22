from flask import Flask, render_template, request, redirect, jsonify
import json
import pandas as pd
import numpy as np

import joblib
import pickle
encoder = joblib.load('encoder.pkl')
loaded_model = pickle.load(open("trained_model.pkl", "rb"))

df = pd.read_csv("mylaptopdata.csv")  # Replace with your data path
columnlist = ["Company", "TypeName", "Inches", "ScreenResolution", "Cpu", "Ram", "Memory", "Gpu", "OpSys"] #, "Weight_Category"]
data=df[columnlist]
features = list(data.columns)
#FLASK
app = Flask(__name__)

# Load available feature names and their unique values
feature_options = {}
for feature in features:
    feature_options[feature] = data[feature].unique().tolist()

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "n_features" in request.form:  # Check for feature selection form
            n = int(request.form["n_features"])
            return render_template("index.html", feature_options=feature_options, n=n)
        else:
            n = len([key for key in request.form if key.startswith("feature_")])
            if n > 0:
                try:
                    n = int(n)
                except ValueError:
                    return "Invalid value for n", 400
            user_input = []
            print("User input:", user_input)
            for i in range(n):
                feature_name = request.form[f"feature_{i}"]
                feature_value = request.form[f"value_{i}"]
                user_input.append((feature_name, feature_value))
            user_input_json = json.dumps(user_input)  # Convert to JSON
            return redirect(f"/prediction?input={user_input_json}")
    else:
        return render_template("index.html", feature_options=feature_options)

@app.route("/get_feature_values")
def get_feature_values():
    feature = request.args.get("feature")
    if feature:
        try:
            unique_values = data[feature].unique().tolist()  # Get unique values from the DataFrame
            return jsonify(unique_values)  # Return as JSON response
        except KeyError:
            return jsonify([]), 400  # Return empty list if feature not found
    else:
        return jsonify([]), 400  # Return empty list if feature not provided

@app.route("/prediction")
def prediction():
    user_input = request.args.get("input")
    filtered_input = user_input
    if user_input:
        try:
            user_input = [tuple(item) for item in json.loads(user_input)]
            n = len(user_input)
            # Handle missing features
            # Handle missing features
            for feature in features:
                found = False
                for item in user_input:
                    if item[0] == feature:  # Check the first element of the tuple
                        found = True
                        break
                if not found:
                    user_input.append((feature, np.nan))
            print("User input wut:", user_input)
            X_new = pd.DataFrame(columns=features)
            feature_values = {}
            for feature, value in user_input:
                if value != np.nan:  # Only store non-NaN values
                    feature_values[feature] = value
            for feature in features:
                X_new.loc[0, feature] = feature_values.get(feature, np.nan)  # Use .get() for default NaN
            print("X_new", X_new)
            X_new.info()
            # Encode categorical features in user input
            X_new_encoded = encoder.transform(X_new)
            prediction = loaded_model.predict(X_new_encoded)[0]
            print("Prediction",prediction)
            prediction_str = str(prediction)
            return render_template("prediction.html", user_input=filtered_input, prediction=prediction_str)
        except ValueError:
            return "Error", 400
    else:
        return "Invalid input" , 400

@app.route("/table")
def show_table():
    return render_template("table.html", data=df.to_html(), features=features, feature_options=feature_options)

@app.route("/search", methods=["POST"])
def search():
    sf = request.form[f"feature"]
    sv = request.form[f"value"]
    # Filter data based on query
    filtered_data = df[df[sf].str.contains(sv, case=False)]  # Example
    return render_template("table.html", data=filtered_data.to_html(), feature_options=feature_options)

if __name__ == "__main__":
    app.run(debug=True)