# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

# Load the dataset (replace 'your_dataset.csv' with the actual file path)
df = pd.read_csv('drug_sales_prediction_dataset.csv')

# Display the first few rows of the dataset
print(df.head())

# Step 1: Data Preprocessing
# Check for missing values
print(df.isnull().sum())

# Drop rows or handle missing values if necessary (e.g., filling with mean/median)
# df.fillna(method='ffill', inplace=True)  # Forward fill (example, customize as needed)

# One-hot encode the 'Season' column
df = pd.get_dummies(df, columns=['Season'], drop_first=True)

# Step 2: Define Features (X) and Target (y)
# Updated to use actual columns from one-hot encoding
# Assuming the original 'Season' column has categories 'Spring', 'Summer', 'Winter'
X = df[['MarketingSpend', 'CompetitorPrice', 'AveragePrice', 'EconomicIndex', 'DemographicFactor', 'PrescriptionRate', 'Season_Spring', 'Season_Summer', 'Season_Winter']]  
y = df['SalesVolume']

# Optional: Scaling the numerical features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 3: Split the dataset into Training and Testing sets (80/20 split)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Step 4: Train the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 5: Make Predictions on the Test Set
y_pred = model.predict(X_test)

# Step 6: Model Evaluation
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"Mean Absolute Error (MAE): {mae}")
print(f"Mean Squared Error (MSE): {mse}")
print(f"Root Mean Squared Error (RMSE): {rmse}")
print(f"R-squared (R²): {r2}")

# Step 7: Display Coefficients (for interpretability)
coefficients = pd.DataFrame(model.coef_, X.columns, columns=['Coefficient'])
print(coefficients)

from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import seaborn as sns

# Step 4: Train the Random Forest Regression model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Step 5: Make Predictions on the Test Set
y_pred = model.predict(X_test)

# Step 6: Model Evaluation
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"Mean Absolute Error (MAE): {mae}")
print(f"Mean Squared Error (MSE): {mse}")
print(f"Root Mean Squared Error (RMSE): {rmse}")
print(f"R-squared (R²): {r2}")

# Step 7: Extract Feature Importance
importances = model.feature_importances_
feature_names = X.columns
# Create a DataFrame for better visualization
feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})

# Sort features by importance
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

# Print the most important features
print("Feature Importance:\n", feature_importance_df)

# Step 8: Visualize Feature Importance
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance_df)
plt.title('Feature Importance in Random Forest')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.show()