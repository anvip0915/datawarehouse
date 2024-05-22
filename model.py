import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder

encoder = OneHotEncoder(handle_unknown='ignore')  # Ignore unseen categories during prediction
# Load data and perform preprocessing (if needed)
data = pd.read_csv("mylaptopdata.csv")  # Replace with your data path
data = data.dropna(subset=["Price_VND"])

# Feature and target variables
features = ["Company", "TypeName", "Inches", "ScreenResolution", "Cpu", "Ram", "Memory", "Gpu", "OpSys"] #, "Weight_Category"]
target = "Price_VND"

# Split data into training and testing sets
X = data[features]  # Get features from the original DataFrame
X_encoded = encoder.fit_transform(X)

y = data[target]

"""category_map = {
    "<30M": 0,
    "30M-60M": 1,
    ">60M": 2
}
y = y.map(category_map)"""
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

def accuracy_scoring(estimator, X_test, y_true):
    y_pred = estimator.predict(X_test)
    return accuracy_score(y_true, y_pred)

# Create and train the model
model = RandomForestClassifier()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
A_S = accuracy_scoring(model, X_test, y_test)
print(y_pred)
print("Acuracy Scoring:", A_S)

# Tuning (optional)
param_grid = {
    "n_estimators": [100, 200, 300],
    "max_depth": [5, 10, 15],
    "min_samples_split": [2, 5, 10],
}
grid_search = GridSearchCV(estimator=RandomForestClassifier(), param_grid=param_grid, cv=5, scoring=accuracy_scoring)
grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_

y_pred_best = best_model.predict(X_test)
best_model_score = accuracy_scoring(best_model, X_test, y_test)
print("Acuracy Scoring of best_model:", best_model_score)
# Save the trained model to a file
import pickle
pickle.dump(model, open("trained_model.pkl", "wb"))

import joblib
joblib.dump(encoder, 'encoder.pkl')