import serial
import time
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import numpy as np
import warnings

# Suppress scikit-learn warnings
warnings.filterwarnings("ignore", category=UserWarning)

file_path = r"C:\Users\itski\Downloads\MOR\pH_dataset.csv"
df = pd.read_csv(file_path)

df.columns = df.columns.str.replace("**", "").str.strip()

df['Timestamp'] = pd.to_datetime(df['Timestamp'])

df['Time_Hours'] = (df['Timestamp'] - df['Timestamp'].min()).dt.total_seconds() / 3600

scaler = StandardScaler()
X = scaler.fit_transform(df[['Time_Hours']])
y = df[['Butterhead Lettuce (pH)', 'Cherry Tomato (pH)']]

model = RandomForestRegressor(n_estimators=100, random_state=42)

model.fit(X, y)

arduino_port = 'COM3'
baud_rate = 9600
ser = serial.Serial(arduino_port, baud_rate, timeout=1)
time.sleep(2)

timestamps = []
lettuce_pH_values = []
tomato_pH_values = []

lettuce_pH_predictions = []
tomato_pH_predictions = []

def read_arduino_data():
    if ser.in_waiting > 0:
        line = ser.readline().decode('utf-8').strip()
        if line:
            try:
                lettuce_pH, tomato_pH = map(float, line.split(','))
                return lettuce_pH, tomato_pH
            except ValueError:
                print(f"Invalid data received: {line}")
                return None, None
    return None, None

def update_plot(frame):
    global df

    lettuce_pH, tomato_pH = read_arduino_data()
    if lettuce_pH is not None and tomato_pH is not None:
        timestamps.append(time.time())
        lettuce_pH_values.append(lettuce_pH)
        tomato_pH_values.append(tomato_pH)

        print(f"New Data: Lettuce pH = {lettuce_pH}, Tomato pH = {tomato_pH}")

        if len(timestamps) % 5 == 0:
            new_data = pd.DataFrame({
                'Time_Hours': (np.array(timestamps) - timestamps[0]) / 3600,
                'Butterhead Lettuce (pH)': lettuce_pH_values,
                'Cherry Tomato (pH)': tomato_pH_values
            })
            df = pd.concat([df, new_data], ignore_index=True)

            X = scaler.fit_transform(df[['Time_Hours']])
            y = df[['Butterhead Lettuce (pH)', 'Cherry Tomato (pH)']]

            model.fit(X, y)

        current_time_hours = (timestamps[-1] - timestamps[0]) / 3600  # Convert to hours
        current_time_normalized = scaler.transform([[current_time_hours]])
        prediction = model.predict(current_time_normalized)
        lettuce_pred, tomato_pred = prediction[0]

        lettuce_pH_predictions.append(lettuce_pred)
        tomato_pH_predictions.append(tomato_pred)

    ax1.clear()
    ax2.clear()

    ax1.plot(timestamps, lettuce_pH_values, color='blue', label='Actual pH (Lettuce)', linestyle='-')
    ax1.plot(timestamps, lettuce_pH_predictions, color='red', label='Predicted pH (Lettuce)', linestyle='dashed')
    ax1.set_xlabel("Time (Seconds)")
    ax1.set_ylabel("pH Level")
    ax1.set_title("Real-Time pH (Butterhead Lettuce)")
    ax1.legend()
    ax1.grid(True)

    ax2.plot(timestamps, tomato_pH_values, color='green', label='Actual pH (Tomato)', linestyle='-')
    ax2.plot(timestamps, tomato_pH_predictions, color='orange', label='Predicted pH (Tomato)', linestyle='dashed')
    ax2.set_xlabel("Time (Seconds)")
    ax2.set_ylabel("pH Level")
    ax2.set_title("Real-Time pH (Cherry Tomato)")
    ax2.legend()
    ax2.grid(True)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))

ani = animation.FuncAnimation(fig, update_plot, interval=1000)  # Update every 1 second
plt.show()
