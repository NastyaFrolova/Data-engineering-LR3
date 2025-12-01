# Используем легковесный образ Python
FROM python:3.9-slim

# Устанавливаем рабочую директорию
WORKDIR /app

# Копируем файл с зависимостями и устанавливаем их
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Потребовалась еще эта зависимость
RUN apt-get update && apt-get install -y libgomp1

# Копируем папку кодом
COPY . .

# Открываем порт для FastAPI
EXPOSE 8000

# Команда для запуска приложения.
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]