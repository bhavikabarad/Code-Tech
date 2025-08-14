# weather_api_visualization.py

import requests
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Replace with your actual API key
API_KEY = '5473cdc0912e0fe13b99f7a6bc25c5e8' # 👈 Paste your OpenWeatherMap API key here
CITY = 'Surat'  # You can change to any city you like
UNITS = 'metric'  # For Celsius
URL = f"http://api.openweathermap.org/data/2.5/forecast?q={CITY}&appid={API_KEY}&units={UNITS}"

# Fetch data from API
response = requests.get(URL)
data = response.json()

# Check if data fetch was successful
if response.status_code != 200:
    print("Failed to fetch data:", data.get("message", "Unknown error"))
    exit()

# Process weather forecast data
forecast_list = data['list']
weather_data = {
    'datetime': [entry['dt_txt'] for entry in forecast_list],
    'temperature': [entry['main']['temp'] for entry in forecast_list],
    'humidity': [entry['main']['humidity'] for entry in forecast_list]
}

# Convert to DataFrame
df = pd.DataFrame(weather_data)
df['datetime'] = pd.to_datetime(df['datetime'])

# Print first few rows
print("Sample Weather Data:")
print(df.head())

# --------------------------------------
# Step 4: Create Visualization Dashboard
# --------------------------------------

# Set Seaborn style
sns.set(style="whitegrid")

# 1. Line Plot: Temperature over time
plt.figure(figsize=(12, 6))
sns.lineplot(x='datetime', y='temperature', data=df, marker='o', color='red')
plt.title(f'Temperature Forecast for {CITY}', fontsize=16)
plt.xlabel('Date Time')
plt.ylabel('Temperature (°C)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 2. Bar Plot: Humidity over time
plt.figure(figsize=(12, 6))
sns.barplot(x='datetime', y='humidity', data=df, color='skyblue')
plt.title(f'Humidity Forecast for {CITY}', fontsize=16)
plt.xlabel('Date Time')
plt.ylabel('Humidity (%)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
