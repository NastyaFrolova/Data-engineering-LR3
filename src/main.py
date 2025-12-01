from fastapi import FastAPI
import pandas as pd
import numpy as np
import joblib
from clearml import Model
from datetime import datetime, timedelta
import requests
import os

app = FastAPI(title="Weather Forecast API")

# Словарь с городами и координатами
geoloc = {"Москва": {"lat": 55.7558, "lon": 37.6173}}

# Загрузка всех моделей при старте
model_ids = {"Москва": '4d0aa1a615154fdf80e3a66873e855f1'}
models = {}
target = 'temperature_2m_max'

print("Starting model loading...")
for city, model_id in model_ids.items():
    print(f"Loading model for {city}...")
    try:
        model_filename = f'weather_model_{city.lower()}.pkl'
        models[city] = joblib.load(model_filename)
        print(f"Model for {city} loaded successfully.")
    except Exception as e:
        print(f"Error for {city}: {e}")
print("Model loading finished.")


# Вспомогательные функции
def get_weather_data(city_name, days=1460):
    if city_name not in geoloc:
        print(f"Error for '{city_name}'.")
        return None
    coordintations = geoloc[city_name]
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {"latitude": coordintations["lat"], "longitude": coordintations["lon"], "start_date": start_date,
              "end_date": end_date, "daily": [target, "temperature_2m_min", "temperature_2m_mean", "precipitation_sum"],
              "timezone": "Europe/Moscow"}
    response = requests.get(url, params=params)
    response.raise_for_status()
    df = pd.DataFrame(response.json()['daily'])
    df['date'] = pd.to_datetime(df['time'])
    df.rename(columns={target: 'temperature_2m_max'}, inplace=True)
    return df[['date', 'temperature_2m_max', 'temperature_2m_min', 'temperature_2m_mean', 'precipitation_sum']]


# Создаем признаки
def create_features(df, target_variable):
    df = df.copy()

    # Календарные признаки
    df['dayofweek'] = df['date'].dt.dayofweek
    df['dayofyear'] = df['date'].dt.dayofyear

    # Циклические признаки
    df['dow_sin'] = np.sin(df['dayofweek'] * (2 * np.pi / 7))
    df['dow_cos'] = np.cos(df['dayofweek'] * (2 * np.pi / 7))
    df['doy_sin'] = np.sin(df['dayofyear'] * (2 * np.pi / 365))
    df['doy_cos'] = np.cos(df['dayofyear'] * (2 * np.pi / 365))

    # Лаги
    for lag in [1, 7, 14]:
        df[f'{target_variable}_lag_{lag}'] = df[target_variable].shift(lag)

    # Скользящие средние
    for window in [7, 30]:
        df[f'{target_variable}_rolling_mean_{window}'] = (
            df[target_variable].shift(1).rolling(window).mean()
        )
        df[f'{target_variable}_rolling_std_{window}'] = (
            df[target_variable].shift(1).rolling(window).std()
        )

    return df


def predict_for_city(city_name, trained_model):
    history_df = get_weather_data(city_name)
    if history_df is None or history_df.empty:
        return []
    forecast_list = []
    current_df = history_df.copy()
    prediction_ot = 1.5
    for _ in range(7):
        # Определяем дату для следующего прогноза
        last_date = current_df['date'].iloc[-1]
        next_date = last_date + timedelta(days=1)

        # Создаем времененную для будущей даты, чтобы считать календарные признаки
        future_row = pd.DataFrame({'date': [next_date]})
        future_row[target] = np.nan  # Целевую переменную пока не знаем

        # Временно добавляем заглушку к нашим данным
        extended_df = pd.concat([current_df, future_row], ignore_index=True)

        # Создаем признаки для всего расширенного датафрейма
        featured_df = create_features(extended_df, target)

        # Извлекаем признаки только для последней строки
        features_to_predict = featured_df.drop(columns=['date', target]).iloc[[-1]]

        features_to_predict = features_to_predict[models[city].feature_name_]

        # Проверяем, хватает ли данных для создания признаков
        if features_to_predict.isnull().all().all():
            print(f"Not enough for {next_date.strftime('%Y-%m-%d')}.")
            break

        # Сам прозноз
        prediction = trained_model.predict(features_to_predict)[0]

        # Форматируем результат для ответа API
        forecast_point = {
            "date": next_date.strftime("%Y-%m-%d"),
            "predicted_temperature_max": round(prediction, 2),
            "confidence_interval_low": round(prediction - 1.96 * prediction_ot, 2),
            "confidence_interval_high": round(prediction + 1.96 * prediction_ot, 2)
        }
        forecast_list.append(forecast_point)

        # Добавляем полученный прогноз обратно в датафрейм
        predicted_row = pd.DataFrame({'date': [next_date], target: [prediction]})

        # Обновляем
        current_df = pd.concat([current_df, predicted_row], ignore_index=True)

    return forecast_list


@app.post("/predict")
def predict(request: dict):
    all_results = []
    cities_to_predict = request.get("cities", [])

    if not cities_to_predict:
        return {"Error": "Please provide a list of cities in the 'cities' field."}

    for city0 in cities_to_predict:
        # Проверяем, есть ли координаты для города
        if city0 not in geoloc:
            print(f"Skipping {city0}, coordinates not found in 'geoloc'.")
            continue

        # Проверяем, была ли модель для этого города загружена
        if city0 not in models:
            print(f"Skipping {city0}, model not found in 'models'.")
            continue

        # Далее берем модель из кэша
        city_model = models[city0]

        print(f"Generating forecast for {city0}...")
        city_forecast = predict_for_city(city0, city_model)

        all_results.append({"city": city0, "forecast": city_forecast})

    return {"results": all_results}


@app.get("/")
def read_root():
    return {"message": "Weather API is running. Use /predict."}
