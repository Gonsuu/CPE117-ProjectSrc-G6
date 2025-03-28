import serial
import time
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

file_path = r"C:\Users\itski\Downloads\MOR\pH_dataset.csv"
df = pd.read_csv(file_path)

df.columns = df.columns.str.replace("**", "").str.strip()
df['Timestamp'] = pd.to_datetime(df['Timestamp'])
df['Time_Hours'] = (df['Timestamp'] - df['Timestamp'].min()).dt.total_seconds() / 3600

scaler = StandardScaler()
X = scaler.fit_transform(df[['Time_Hours']])
y = df[['Butterhead Lettuce (pH)', 'Cherry Tomato (pH)']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
mse_lettuce = mean_squared_error(y_test['Butterhead Lettuce (pH)'], y_pred[:, 0])
mse_tomato = mean_squared_error(y_test['Cherry Tomato (pH)'], y_pred[:, 1])
print(f"Initial MSE for Butterhead Lettuce: {mse_lettuce:.4f}")
print(f"Initial MSE for Cherry Tomato: {mse_tomato:.4f}")

arduino_port = 'COM3'
baud_rate = 9600
ser = serial.Serial(arduino_port, baud_rate, timeout=1)
time.sleep(2)

timestamps = []
lettuce_pH_values = []
tomato_pH_values = []
lettuce_pH_predictions = []
tomato_pH_predictions = []
lettuce_mse_values = []
tomato_mse_values = []

last_mse_update_time = time.time()

def read_arduino_data():
    if ser.in_waiting > 0:
        line = ser.readline().decode('utf-8').strip()
        if line:
            try:
                lettuce_pH, tomato_pH = map(float, line.split(','))
                return lettuce_pH, tomato_pH
            except ValueError:
                return None, None
    return None, None

def get_nutrient_recommendation(pH_lettuce, pH_tomato):
    recommendations = []
    if pH_lettuce < 5.5:
        recommendations.append("Butterhead Lettuce: pH is too low (acidic). Add pH Up solution.")
    elif pH_lettuce > 6.5:
        recommendations.append("Butterhead Lettuce: pH is too high (alkaline). Add pH Down solution.")
    else:
        recommendations.append("Butterhead Lettuce: pH is optimal. No action needed.")
    if pH_tomato < 5.5:
        recommendations.append("Cherry Tomato: pH is too low (acidic). Add pH Up solution.")
    elif pH_tomato > 6.5:
        recommendations.append("Cherry Tomato: pH is too high (alkaline). Add pH Down solution.")
    else:
        recommendations.append("Cherry Tomato: pH is optimal. No action needed.")
    return recommendations

def update_plot(frame):
    global df, last_mse_update_time

    lettuce_pH, tomato_pH = read_arduino_data()
    if lettuce_pH is not None and tomato_pH is not None:
        timestamps.append(time.time())
        lettuce_pH_values.append(lettuce_pH)
        tomato_pH_values.append(tomato_pH)

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

        current_time_hours = (timestamps[-1] - timestamps[0]) / 3600
        current_time_normalized = scaler.transform([[current_time_hours]])
        prediction = model.predict(current_time_normalized)
        lettuce_pred, tomato_pred = prediction[0]
        lettuce_pH_predictions.append(lettuce_pred)
        tomato_pH_predictions.append(tomato_pred)

        current_time = time.time()
        if current_time - last_mse_update_time >= 5:
            if len(lettuce_pH_values) > 10:
                lettuce_mse = mean_squared_error(lettuce_pH_values[-10:], lettuce_pH_predictions[-10:])
                tomato_mse = mean_squared_error(tomato_pH_values[-10:], tomato_pH_predictions[-10:])
                lettuce_mse_values.append(lettuce_mse)
                tomato_mse_values.append(tomato_mse)
                print(f"Updated MSE for Butterhead Lettuce: {lettuce_mse:.4f}")
                print(f"Updated MSE for Cherry Tomato: {tomato_mse:.4f}")
            last_mse_update_time = current_time

        recommendations = get_nutrient_recommendation(lettuce_pH, tomato_pH)

    ax1.clear()
    ax2.clear()
    ax3.clear()

    ax1.plot(timestamps, lettuce_pH_values, color='blue', label='Actual pH (Lettuce)', linestyle='-')
    ax1.plot(timestamps, lettuce_pH_predictions, color='red', label='Predicted pH (Lettuce)', linestyle='dashed')
    ax1.set_xlabel("Time (Seconds)")
    ax1.set_ylabel("pH Level")
    ax1.set_title("Real-Time pH (Butterhead Lettuce)")
    ax1.legend(loc='upper left')
    ax1.grid(True)

    ax2.plot(timestamps, tomato_pH_values, color='green', label='Actual pH (Tomato)', linestyle='-')
    ax2.plot(timestamps, tomato_pH_predictions, color='orange', label='Predicted pH (Tomato)', linestyle='dashed')
    ax2.set_xlabel("Time (Seconds)")
    ax2.set_ylabel("pH Level")
    ax2.set_title("Real-Time pH (Cherry Tomato)")
    ax2.legend(loc='upper left')
    ax2.grid(True)

    if lettuce_pH is not None and tomato_pH is not None:
        recommendation_text = "\n".join(recommendations)
        ax3.text(
            0.1, 0.5, recommendation_text,
            ha="left", va="center", fontsize=10, bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5),
            wrap=True
        )
        ax3.axis('off')

    plt.tight_layout()

fig = plt.figure(figsize=(14, 10))
ax1 = plt.subplot2grid((2, 2), (0, 0))
ax2 = plt.subplot2grid((2, 2), (1, 0))
ax3 = plt.subplot2grid((2, 2), (0, 1), rowspan=2)

ani = animation.FuncAnimation(fig, update_plot, interval=1000)
plt.show()
