# Используем образ с уже установленными ML библиотеками
FROM python:3.10

# Устанавливаем рабочую директорию
WORKDIR /app

# Устанавливаем только нужные системные зависимости
RUN apt-get update && apt-get install -y \
    git \
    git-lfs \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && git lfs install

# Копируем файл зависимостей
COPY requirements.txt .

# Устанавливаем Python зависимости
RUN pip install --no-cache-dir -r requirements.txt

# Копируем весь код приложения
COPY . .

# Создаем непривилегированного пользователя
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Открываем порт
EXPOSE 8002

# Запускаем сервер
CMD ["python", "-u", "main.py"]