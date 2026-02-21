import joblib

import pandas as pd


model = joblib.load("model.joblib")

BMI_TO_PREDICT = 25.5

input_data = pd.DataFrame([[BMI_TO_PREDICT]], columns=['bmi'])

prediction = model.predict(input_data)

rounded_prediction = round(prediction[0], 2)

print("\n-----------------------RESULT-----------------------")
print(f"| Predicted 1-year diabetes progression | {rounded_prediction}   |")
print("----------------------------------------------------\n")
