import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load the dataset
file_path = r"C:\Users\itski\Downloads\MOR\pH_dataset.csv"
df = pd.read_csv(file_path)

# Fix column names (remove any extra formatting issues)
df.columns = df.columns.str.replace("**", "").str.strip()

# Convert Timestamp to datetime format
df['Timestamp'] = pd.to_datetime(df['Timestamp'])

# Convert timestamps to numeric values (hours since the first timestamp)
df['Time_Hours'] = (df['Timestamp'] - df['Timestamp'].min()).dt.total_seconds() / 3600

# Define features (X) and target variables (y)
X = df[['Time_Hours']]
y = df[['Butterhead Lettuce (pH)', 'Cherry Tomato (pH)']]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Calculate Mean Squared Error
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse:.4f}")

# Sort values before plotting to avoid zig-zag lines
sorted_indices = X_test['Time_Hours'].argsort()
X_test_sorted = X_test.iloc[sorted_indices]
y_test_sorted = y_test.iloc[sorted_indices]
y_pred_sorted = y_pred[sorted_indices]

# Create a figure with two subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

# Plot for Butterhead Lettuce
ax1.plot(X_test_sorted['Time_Hours'], y_test_sorted['Butterhead Lettuce (pH)'], color='blue', label='Actual pH (Lettuce)', linestyle='-')
ax1.plot(X_test_sorted['Time_Hours'], y_pred_sorted[:, 0], color='red', label='Predicted pH (Lettuce)', linestyle='dashed')
ax1.set_xlabel("Time (Hours)")
ax1.set_ylabel("pH Level")
ax1.set_title("Actual vs Predicted pH (Butterhead Lettuce)")
ax1.legend()
ax1.grid(True)

# Plot for Cherry Tomato
ax2.plot(X_test_sorted['Time_Hours'], y_test_sorted['Cherry Tomato (pH)'], color='green', label='Actual pH (Tomato)', linestyle='-')
ax2.plot(X_test_sorted['Time_Hours'], y_pred_sorted[:, 1], color='orange', label='Predicted pH (Tomato)', linestyle='dashed')
ax2.set_xlabel("Time (Hours)")
ax2.set_ylabel("pH Level")
ax2.set_title("Actual vs Predicted pH (Cherry Tomato)")
ax2.legend()
ax2.grid(True)

# Adjust layout and display the figure
plt.tight_layout()
plt.show()