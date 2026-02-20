import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_diabetes
import joblib


# Load a simple dataset (Diabetes progression)
data = load_diabetes(scaled=False)
X = pd.DataFrame(data.data, columns=data.feature_names)[['bmi']] # Use BMI as feature
y = data.target

# Train a real model
model = LinearRegression()
model.fit(X, y)

# Save the artifact
joblib.dump(model, "model.joblib")
print("Model trained and saved as model.joblib")
