import pandas as pd
import numpy as np
from clearml import Task, Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import lightgbm as lgb
import joblib  # Для сохранения модели
import os

# Инициализация задачи для ClearML
task = Task.init(project_name="Weather Forecast", task_name="LightGBM Training")

# Подключение гиперпараметров
params = {
    "city": "Москва",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "n_estimators": 100,
    "feature_fraction": 0.8,
    "random_state": 42,
}
task.connect(params, name="hyperparams") # Логируем параметры

# Загрузка данных из датасета
dataset_id = '5597380c5f2b4d21b7df7f9c83922be4' # Добавляем сюда название датасета после database.py
dataset = Dataset.get(dataset_id=dataset_id)
dataset_path = dataset.get_local_copy() # Путь датасета

# Ищем нужный CSV-файл в загруженной папке
city_name = params['city']
csv_filename = f"{city_name.lower()}_weather_history.csv"
csv_path = os.path.join(dataset_path, csv_filename)

if not os.path.exists(csv_path):
    raise FileNotFoundError(f"File {csv_filename} not found in {dataset_id}")

df = pd.read_csv(csv_path)
df['date'] = pd.to_datetime(df['date'])

# Генерация признаков
def create_features(df, target_variable):
    df = df.copy()
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
        df[f'{target_variable}_rolling_mean_{window}'] = df[target_variable].shift(1).rolling(window).mean()
        df[f'{target_variable}_rolling_std_{window}'] = df[target_variable].shift(1).rolling(window).std()

    df.dropna(inplace=True)
    return df


# Целевой параметр
target = 'temperature_2m_max'
df_featured = create_features(df, target)

# Разделение данных
X = df_featured.drop(columns=['date', target])
y = df_featured[target]

# Разбиваем на обучающие и тестовые
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Обучение модели
print(f"Training model for {city_name}...")
valid_params = lgb.LGBMRegressor().get_params().keys()
model_params = {k: v for k, v in params.items() if k in valid_params}

model = lgb.LGBMRegressor(**model_params)
model.fit(X_train, y_train, eval_set=[(X_test, y_test)], eval_metric='mae', callbacks=[lgb.log_evaluation(0)])

# Оценка и логирование
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"MAE: {mae:.4f}, RMSE: {rmse:.4f}")

logger = task.get_logger()
logger.report_scalar(title="MAE", series="final_mae", iteration=1, value=mae)
logger.report_single_value("RMSE", rmse)

# Сохраняем модель
model_filename = f'weather_model_{city_name.lower()}.pkl'
joblib.dump(model, model_filename)
task.upload_artifact(name='model', artifact_object=model_filename)

print("Training complete!")