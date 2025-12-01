import requests
import pandas as pd
from datetime import datetime

# Основывается на ЛР1
# Города и координаты (широта + долгота)
Geoloc = {"Москва": {"lat": 55.7558, "lon": 37.6173},
          "Самара": {"lat": 53.1955, "lon": 50.1018},
          "Санкт-Петербург": {"lat": 59.9343, "lon": 30.3351}}


# Функция для получения погодных данных городов
def get_historical_forecast(city_name, start_date, end_date, daily_variables):
    if city_name not in Geoloc:
        print(f"Coordinates {city_name} not found.")
        return None

    coords = Geoloc[city_name]

    # Используем url open meteo из первой лабораторной
    url = "https://archive-api.open-meteo.com/v1/archive"

    params = {
        "latitude": coords["lat"],  # Широта
        "longitude": coords["lon"],  # Долгота
        "start_date": start_date, # Начальная дата
        "end_date": end_date, # Конечная
        "daily": daily_variables, # Параметры для рассмотрения
        "timezone": "Europe/Moscow" # Часовой пояс
    }

    try:
        # Пробуем сделать запрос
        print(f"For {city_name}: {start_date}-:-{end_date}...")
        response = requests.get(url, params=params)
        response.raise_for_status()  # Проверяем на наличие ошибок
        return response.json()

    except requests.exceptions.RequestException as e:
        print(f"Error for {city_name}: {e}") # Формат: ошибка для такого-то города - сама ошибка
        return None


if __name__ == "__main__":
    # Указываем города, начинаем с одного
    our_cities = ["Москва"]  # Можно добавить еще и скачать несколько, поддерживает три города

    # Период рассмотрения
    start = "2020-01-01" # Нужно, чтобы было более 3 лет
    end = datetime.now().strftime("%Y-%m-%d") # Конечная по сегодняшней дате

    # Целевые переменные
    daily_vars = "temperature_2m_max,temperature_2m_min,temperature_2m_mean,precipitation_sum" # Как указано в задании, берем пока их, но можно добавить еще

    # Цикл по городам, так как можно добавить несколько
    for city in our_cities:
        # Вызываем нашу функцию для получения данных
        weather_data_json = get_historical_forecast(city, start, end, daily_vars)

        # Обработка и сохранение данных, если успех
        if weather_data_json:
            print(f"Data for {city} - success. Processing...")

            # Извлекаем дневные данные из ответа
            daily_data = weather_data_json.get('daily', {})

            # Создаем DataFrame
            df = pd.DataFrame(daily_data)

            # Переименуем колонку с временем для удобства
            if 'time' in df.columns:
                df.rename(columns={'time': 'date'}, inplace=True)
                df['date'] = pd.to_datetime(df['date'])

            # Сохраняем в отдельный файл для каждого города, если таковые есть
            output_filename = f"{city.lower()}_weather_history.csv"
            df.to_csv(output_filename, index=False)

            print(f"Data for {city} was saved in: {output_filename}")
        else:
            print(f"Error for {city}. Skip.")

        print("-" * 30)  # Разделитель для удобства
