# 🏎️ F1 Abu Dhabi GP 2025 Predictions - Machine Learning Model

Welcome to the Abu Dhabi GP 2025 prediction module! This project uses machine learning, FastF1 API data, historical F1 results, and qualifying data to predict race outcomes.

---

## 🚀 Project Overview

This script implements a **Gradient Boosting Regressor** to predict the 2025 Abu Dhabi GP race results based on:

* FastF1 API data for the 2024 Monaco GP
* 2025 qualifying session times for Abu Dhabi
* Driver clean air race pace
* Team performance scores
* Weather conditions (rain probability, temperature)
* Historical position change trends

It generates predicted lap times for each driver and ranks them to determine the likely podium finishers.

---

## 📊 Data Sources

* **FastF1 API**: Lap times, sector times, and race data
* **2025 Qualifying Data**: Abu Dhabi GP qualifying session
* **Historical Race Results**: From 2024 Monaco GP
* **Weather API**: OpenWeatherMap forecast for Abu Dhabi

---

## 🏁 How It Works

1. **Data Collection:** Fetches session data from FastF1 and weather forecast.
2. **Preprocessing & Feature Engineering:** Converts lap/sector times to seconds, normalizes drivers and teams, and calculates derived features like `TotalSectorTime`, `TeamPerformanceScore`, and `AveragePositionChange`.
3. **Model Training:** Gradient Boosting Regressor is trained on 2024 lap times.
4. **Prediction:** The model predicts race times for the Abu Dhabi GP 2025.
5. **Evaluation:** Model performance is measured using Mean Absolute Error (MAE).
6. **Visualization:** Generates plots showing the effect of clean air pace and feature importance.

---

## 🔧 Usage

Run the prediction script:

```bash
py abu_dhabi_2025.py
```

Expected output:

```
🏁 Predicted 2025 Abu Dhabi GP Winner 🏁
Driver: NOR, Predicted Race Time: 93.42s

🏆 Predicted in the Top 3 🏆
🥇 P1: LEC
🥈 P2: NOR
🥉 P3: PIA

```

Plots include:

* Effect of clean air race pace on predicted race time
* Feature importance in race time prediction

---

## 📈 Model Performance

The **Mean Absolute Error (MAE)** is used to evaluate predictions. Lower values indicate more accurate predictions.

---

## 📌 Future Improvements

* Include pit stop strategy and tire degradation
* Use deep learning for sequential lap prediction
* Incorporate live weather and track temperature dynamically
* Extend predictions to all 2025 GP races

---

## 🛠 Dependencies

* `fastf1`
* `pandas`
* `numpy`
* `scikit-learn`
* `matplotlib`
* `requests`

---

## 📜 License

MIT License

---

🏎️ Start predicting F1 races like a data scientist! 🚀
