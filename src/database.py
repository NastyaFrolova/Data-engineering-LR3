from clearml import Dataset
import os


def main():
    # Создаем новый датасет
    dataset = Dataset.create(
        dataset_name="Weather Forecast for Cities",
        dataset_project="Weather Forecast",
        dataset_version="1.0.0"
    )

    # Находим все файлы созданные api
    csv_files = [f for f in os.listdir('.') if f.endswith('_weather_history.csv')]

    if not csv_files:
        print("Error: files not found. Start api.py")
        return

    # Добавляем каждый файл в датасет
    for csv_file in csv_files:
        print(f"Add file {csv_file} in dataset...")
        dataset.add_files(path=csv_file)

    # Загружаем датасет на сервер ClearML
    dataset.upload()
    dataset.finalize()

    print(f"Success! Dataaset ID: {dataset.id}")


if __name__ == "__main__":
    main()