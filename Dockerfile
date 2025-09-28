# Используем официальный Python образ
FROM python:3.10-slim

# Устанавливаем рабочую директорию
WORKDIR /app

# Устанавливаем системные зависимости
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Копируем файл зависимостей
COPY requirements.txt .

# Устанавливаем Python зависимости
RUN pip install --no-cache-dir -r requirements.txt

# Копируем код приложения
COPY main.py .

# Копируем модель (убедитесь, что файл находится в той же директории)
COPY best_model_crf_all.pt .

# Открываем порт
EXPOSE 8002

# Запускаем сервер
CMD ["python", "-u", "main.py"]