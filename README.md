# NER Service для распознавания алкогольных напитков

Сервис для извлечения именованных сущностей (тип, бренд, объем, процент) из названий алкогольных напитков на основе BERT + CRF.

## Быстрый запуск

```bash
git clone https://github.com/templeJJs/x5_lct_v3.git
cd x5_lct_v3
docker-compose up -d
```

**Готово!** Сервис доступен на http://localhost:8002

## Проверка работы

```bash
# Проверка статуса
curl http://localhost:8002/health

# Тестовый запрос
curl -X POST "http://localhost:8002/api/predict" \
  -H "Content-Type: application/json" \
  -d '{"input": "Водка Белуга 0.5л 40%"}'
```

## Пример ответа

```json
[
  {"start_index": 0, "end_index": 5, "entity": "B-TYPE"},
  {"start_index": 6, "end_index": 12, "entity": "B-BRAND"},
  {"start_index": 13, "end_index": 17, "entity": "B-VOLUME"},
  {"start_index": 18, "end_index": 21, "entity": "B-PERCENT"}
]
```

## API

- `GET /health` - статус сервиса
- `POST /api/predict` - предсказание для одного текста
- `POST /api/predict_batch` - батчевое предсказание

## Требования

- Docker
- Docker Compose
- Git LFS (для загрузки модели)

## Архитектура

- **Модель**: RuBERT + CRF для NER
- **API**: FastAPI с автоматической документацией
- **Развертывание**: Docker + Docker Compose