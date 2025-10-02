# NER Service

Сервис для извлечения именованных сущностей (тип товара, бренд, объем, процент) из поисковых запросов пользователей торговой сети Пятерочка на основе BERT + CRF.

## Системные требования

- **ОС**: Ubuntu 22.04 LTS или выше
- **CPU**: минимум 4 ядра
- **RAM**: минимум 6 GB
- **Диск**: минимум 45 GB NVMe
- **Сеть**: доступ в интернет
- **ПО**: Docker 20.10+, Docker Compose 2.0+, Git, Git LFS

## Быстрый запуск (локально)

```bash
git clone https://github.com/templeJJs/x5_lct_v3.git
cd x5_lct_v3
docker-compose up -d
```

Сервис доступен на http://localhost:8002

## Развертывание на удаленном сервере

### 1. Подключение к серверу

```bash
ssh username@your-server-ip
```

### 2. Установка зависимостей

```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y docker.io docker-compose git-lfs
git lfs install
```

### 3. Клонирование и запуск

```bash
git clone https://github.com/templeJJs/x5_lct_v3.git
cd x5_lct_v3
docker-compose up -d
```

### 4. Открытие порта (если нужен доступ извне)

```bash
sudo ufw allow 8002/tcp
sudo ufw reload
```

### 5. Проверка работы

```bash
docker-compose ps
docker-compose logs -f
curl http://localhost:8002/health
curl -X POST "http://localhost:8002/api/predict" -H "Content-Type: application/json" -d '{"input": "Водка Белуга 0.5л 40%"}'
```

## Управление сервисом

```bash
# Остановка
docker-compose down

# Перезапуск
docker-compose restart

# Пересборка
docker-compose build --no-cache
docker-compose up -d
```

## API Endpoints

**GET /health** - проверка статуса

**POST /api/predict** - предсказание для одного текста

```json
{"input": "Молоко 3.2%"}
```

**POST /api/predict_batch** - батчевое предсказание

## Пример ответа

```json
[
  {"start_index": 0, "end_index": 6, "entity": "B-TYPE"},
  {"start_index": 7, "end_index": 11, "entity": "B-PERCENT"}
]
```

## Устранение проблем

```bash
# Если модель не загрузилась
git lfs pull

# Просмотр логов
docker-compose logs ner-service
```

## Архитектура

- **Модель**: RuBERT + CRF
- **Backend**: FastAPI
- **Развертывание**: Docker + Docker Compose

Документация API: http://localhost:8002/docs