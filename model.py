import pickle 
import numpy as np
import pandas as pd
from sklearn.calibration import LabelEncoder
from sklearn.preprocessing import StandardScaler

df = pd.read_csv(r"C:\Users\USER\Desktop\Scrap Probability Prediction\test_cat.csv")

with open(r"C:\Users\USER\Desktop\Scrap Probability Prediction\Scrap_Probability_Prediction.pkl", 'rb') as file:
    model = pickle.load(file)

cat_features = df.select_dtypes(include=['object']).columns
num_features = df.select_dtypes(include=['int64', 'float64']).columns

# Initialize label encoders for each categorical feature
label_encoders = {col: LabelEncoder() for col in cat_features}

new_df = df.copy()

# Encode the categorical features
for col in cat_features:
    new_df[col] = label_encoders[col].fit_transform(new_df[col])

# Combine the numerical and encoded categorical features
dataset_preprocessed = pd.concat([new_df[cat_features], new_df[num_features]], axis=1)

# feature scaling
# Scale the features
scaler = StandardScaler()
X_test = scaler.fit_transform(dataset_preprocessed)

y_pred = model.predict(X_test)


# Convert predictions to percentage and round to 2 decimal places
y_pred = np.round(y_pred * 100, 2)

# Convert the predictions to a pandas Series for easier string formatting and add '%' sign
y_pred_series = pd.Series(y_pred)
y_pred_formatted = y_pred_series.apply(lambda x: f"{x}%")

# Append the formatted predictions to the original DataFrame
df['Scrap_Prob_Results'] = y_pred_formatted
print(df)
