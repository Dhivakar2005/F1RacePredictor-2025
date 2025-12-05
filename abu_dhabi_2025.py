import fastf1
import pandas as pd
import numpy as np
import requests
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer

fastf1.Cache.enable_cache("f1_cache")

# load the 2024 Monaco session data
session_2024 = fastf1.get_session(2024, 8, "R")
session_2024.load()
laps_2024 = session_2024.laps[["Driver", "LapTime", "Sector1Time", "Sector2Time", "Sector3Time"]].copy()
laps_2024.dropna(inplace=True)

# convert lap and sector times to seconds
for col in ["LapTime", "Sector1Time", "Sector2Time", "Sector3Time"]:
    laps_2024[f"{col} (s)"] = laps_2024[col].dt.total_seconds()

# aggregate sector times by driver
sector_times_2024 = laps_2024.groupby("Driver").agg({
    "Sector1Time (s)": "mean",
    "Sector2Time (s)": "mean",
    "Sector3Time (s)": "mean"
}).reset_index()

sector_times_2024["TotalSectorTime (s)"] = (
    sector_times_2024["Sector1Time (s)"] +
    sector_times_2024["Sector2Time (s)"] +
    sector_times_2024["Sector3Time (s)"]
)

clean_air_race_pace = {
    "VER": 93.191067, "HAM": 94.020622, "LEC": 93.418667, "NOR": 93.428600, "ALO": 94.784333,
    "PIA": 93.232111, "RUS": 93.833378, "SAI": 94.497444, "STR": 95.318250, "HUL": 95.345455,
    "OCO": 95.682128
}

# quali data from Monaco GP
qualifying_2025 = pd.DataFrame({
    "Driver": ["NOR","PIA","SAI","HUL","VER","GAS","RUS","ALO",
               "BOT","PER","TSU","LAW","STR","MAG","ZHO","HAM",
               "COL","DOO","ALB"],
    "QualifyingTime (s)": [
        82.595,   # NOR 1:22.595
        82.804,   # PIA 1:22.804
        82.824,   # SAI 1:22.824
        82.886,   # HUL 1:22.886
        82.945,   # VER 1:22.945
        82.984,   # GAS 1:22.984
        83.132,   # RUS 1:23.132
        83.196,   # ALO 1:23.196
        83.204,   # BOT 1:23.204
        83.264,   # PER 1:23.264
        83.419,   # TSU 1:23.419
        83.472,   # LAW 1:23.472
        83.784,   # STR 1:23.784
        83.877,   # MAG 1:23.877
        83.880,   # ZHO 1:23.880
        83.887,   # HAM 1:23.887
        83.912,   # COL 1:23.912
        84.105,   # DOO 1:24.105
        83.821    # ALB 1:23.821
    ]
})

qualifying_2025["CleanAirRacePace (s)"] = qualifying_2025["Driver"].map(clean_air_race_pace)

API_KEY = "your_openweathermap_api_key"
weather_url = f"http://api.openweathermap.org/data/2.5/forecast?lat=43.7384&lon=7.4246&appid={API_KEY}&units=metric"
response = requests.get(weather_url)
weather_data = response.json()
forecast_time = "2025-05-25 13:00:00"  
forecast_data = next((f for f in weather_data["list"] if f["dt_txt"] == forecast_time), None)

rain_probability = forecast_data["pop"] if forecast_data else 0
temperature = forecast_data["main"]["temp"] if forecast_data else 20

# adjust qualifying time based on weather conditions
if rain_probability >= 0.75:
    qualifying_2025["QualifyingTime"] = qualifying_2025["QualifyingTime (s)"] * qualifying_2025["WetPerformanceFactor"]
else:
    qualifying_2025["QualifyingTime"] = qualifying_2025["QualifyingTime (s)"]

# add constructor's data
team_points = {
    "McLaren": 800,
    "Mercedes": 459,
    "Red Bull": 426,
    "Ferrari": 382,
    "Williams": 137,
    "Racing Bulls": 92,
    "Aston Martin": 80,
    "Haas": 73,
    "Kick Sauber": 68,
    "Alpine": 22
}

max_points = max(team_points.values())
team_performance_score = {team: points / max_points for team, points in team_points.items()}

driver_to_team = {
    "NOR": "McLaren",
    "VER": "Red Bull Racing",
    "PIA": "McLaren",
    "RUS": "Mercedes AMG Motorsport",
    "LEC": "Ferrari",
    "HAM": "Ferrari",
    "ANT": "Mercedes AMG Motorsport",     
    "ALB": "Williams",
    "SAI": "Williams",                     
    "HAD": "Racing Bulls",                 
    "HUL": "Sauber",
    "ALO": "Aston Martin F1 Team",
    "BEA": "Haas F1 Team",                 
    "LAW": "Racing Bulls",                 
    "TSU": "Red Bull Racing",
    "OCO": "Haas F1 Team",
    "STR": "Aston Martin F1 Team",
    "GAS": "Alpine F1 Team",
    "BOR": "Sauber",                       
    "COL": "Alpine F1 Team",               
    "DOO": "Alpine F1 Team"                
}


qualifying_2025["Team"] = qualifying_2025["Driver"].map(driver_to_team)
qualifying_2025["TeamPerformanceScore"] = qualifying_2025["Team"].map(team_performance_score)

average_position_change_abudhabi = {
    "VER": 4,
    "NOR": 0,
    "PIA": 3,
    "RUS": 2,
    "SAI": -3,
    "ALB": 0,
    "LEC": -1,
    "OCO": None,   
    "HAM": -2,
    "STR": 0,
    "GAS": -1,
    "ALO": -1,
    "HUL": -1
}

qualifying_2025["AveragePositionChange"] = qualifying_2025["Driver"].map(average_position_change_abudhabi)

# merge qualifying and sector times data
merged_data = qualifying_2025.merge(sector_times_2024[["Driver", "TotalSectorTime (s)"]], on="Driver", how="left")
merged_data["RainProbability"] = rain_probability
merged_data["Temperature"] = temperature
merged_data["QualifyingTime"] = merged_data["QualifyingTime"]


valid_drivers = merged_data["Driver"].isin(laps_2024["Driver"].unique())
merged_data = merged_data[valid_drivers]

# define features (X) and target (y)
X = merged_data[[
    "QualifyingTime", "RainProbability", "Temperature", "TeamPerformanceScore", 
    "CleanAirRacePace (s)", "AveragePositionChange"
]]
y = laps_2024.groupby("Driver")["LapTime (s)"].mean().reindex(merged_data["Driver"])

# impute missing values for features
imputer = SimpleImputer(strategy="median")
X_imputed = imputer.fit_transform(X)

# train-test split
X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.3, random_state=37)

# train gradient boosting model
model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.7, max_depth=3, random_state=37)
model.fit(X_train, y_train)
merged_data["PredictedRaceTime (s)"] = model.predict(X_imputed)

# sort the results to find the predicted winner
final_results = merged_data.sort_values("PredictedRaceTime (s)")
print("\n🏁 Predicted 2025 Abu Dhabi GP Winner 🏁\n")
print(final_results[["Driver", "PredictedRaceTime (s)"]])
y_pred = model.predict(X_test)
print(f"Model Error (MAE): {mean_absolute_error(y_test, y_pred):.2f} seconds")

# plot effect of clean air race pace
plt.figure(figsize=(12, 8))
plt.scatter(final_results["CleanAirRacePace (s)"], final_results["PredictedRaceTime (s)"])
for i, driver in enumerate(final_results["Driver"]):
    plt.annotate(driver, (final_results["CleanAirRacePace (s)"].iloc[i], final_results["PredictedRaceTime (s)"].iloc[i]),
                 xytext=(5, 5), textcoords='offset points')
plt.xlabel("clean air race pace (s)")
plt.ylabel("predicted race time (s)")
plt.title("effect of clean air race pace on predicted race results")
plt.tight_layout()
plt.show()

# Plot feature importances
feature_importance = model.feature_importances_
features = X.columns

plt.figure(figsize=(8,5))
plt.barh(features, feature_importance, color='skyblue')
plt.xlabel("Importance")
plt.title("Feature Importance in Race Time Prediction")
plt.tight_layout()
plt.show()

# sort results and get top 3
final_results = merged_data.sort_values("PredictedRaceTime (s)").reset_index(drop=True)
podium = final_results.loc[:2, ["Driver", "PredictedRaceTime (s)"]]

print("\n🏆 Predicted in the Top 3 🏆")
print(f"🥇 P1: {podium.iloc[0]['Driver']}")
print(f"🥈 P2: {podium.iloc[1]['Driver']}") 
print(f"🥉 P3: {podium.iloc[2]['Driver']}")