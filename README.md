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

# NER Service для Пятёрочки

Сервис извлечения именованных сущностей из поисковых запросов на основе RuBERT + CRF.

## Распознаваемые сущности

- **TYPE** — категория товара (молоко, хлеб, вода)
- **BRAND** — бренд (Coca-Cola, Простоквашино)
- **VOLUME** — объём/вес (0.5л, 200г, 10шт)
- **PERCENT** — процент жирности/крепости (2.5%, 40%)

## Системные требования

- Ubuntu 22.04+
- CPU: 4+ ядра
- RAM: 6+ GB
- Диск: 45+ GB NVMe
- Docker 20.10+, Docker Compose 2.0+
- Git с Git LFS

## Быстрый старт

### Локально

```bash
git clone https://github.com/templeJJs/x5_lct_v3.git
cd x5_lct_v3
docker-compose up -d
```

Сервис доступен на `http://localhost:8002`

### На удалённом сервере

```bash
# Подключение
ssh username@server-ip

# Установка зависимостей
sudo apt update && sudo apt upgrade -y
sudo apt install -y docker.io docker-compose git-lfs
git lfs install

# Клонирование и запуск
git clone https://github.com/templeJJs/x5_lct_v3.git
cd x5_lct_v3
docker-compose up -d

# Открытие порта (опционально)
sudo ufw allow 8002/tcp
sudo ufw reload
```

### Проверка работы

```bash
# Статус контейнеров
docker-compose ps

# Логи
docker-compose logs -f

# Проверка health
curl http://localhost:8002/health

# Тестовый запрос
curl -X POST "http://localhost:8002/api/predict" \
  -H "Content-Type: application/json" \
  -d '{"input": "Водка Белуга 0.5л 40%"}'
```

## API

### Эндпоинты

**`GET /health`** — статус сервиса

**`POST /api/predict`** — предсказание для одного запроса

Запрос:
```json
{
  "input": "Молоко Простоквашино 3.2% 1л"
}
```

Ответ:
```json
[
  {"start_index": 0, "end_index": 6, "entity": "B-TYPE"},
  {"start_index": 7, "end_index": 19, "entity": "B-BRAND"},
  {"start_index": 20, "end_index": 24, "entity": "B-PERCENT"},
  {"start_index": 25, "end_index": 27, "entity": "B-VOLUME"}
]
```

**`POST /api/predict_batch`** — батчевая обработка

Документация: `http://localhost:8002/docs`

## Управление

```bash
# Остановка
docker-compose down

# Перезапуск
docker-compose restart

# Пересборка
docker-compose build --no-cache
docker-compose up -d

# Просмотр логов
docker-compose logs ner-service
```

## Производительность

Метрики на тестовой выборке (27k примеров):

| Класс   | Precision | Recall | F1-Score | Примеров |
|---------|-----------|--------|----------|----------|
| BRAND   | 0.98      | 0.98   | 0.98     | 6213     |
| PERCENT | 0.68      | 0.80   | 0.73     | 84       |
| TYPE    | 0.99      | 0.99   | 0.99     | 21046    |
| VOLUME  | 0.95      | 0.91   | 0.93     | 245      |
| **Weighted avg** | **0.98** | **0.99** | **0.98** | **27588** |

**Submission score:** 0.93 F1

## Архитектура

- **Модель:** RuBERT (DeepPavlov) + CRF layer
- **Backend:** FastAPI + Uvicorn
- **Контейнеризация:** Docker + Docker Compose
- **NER framework:** Transformers + PyTorch

## Решение проблемы дисбаланса классов

### Проблема

Критический дисбаланс в датасете

### Подход

Минимальная целевая аугментация оказалась эффективнее агрессивного расширения:

1. **Добавлено 150 синтетических примеров** для PERCENT через LLM-генерацию
2. **Результат:** F1 для PERCENT вырос с <50% до 73%
3. **Отказ от:**
   - Агрессивной аугментации (500+ примеров) — привела к переобучению
   - Hard negative mining — создала новые проблемы с precision
   - Multi-task learning — минимальный эффект (+2% F1)

## Известные ограничения

- PERCENT имеет precision 68% из-за путаницы с цифрами в названиях товаров
- VOLUME показывает recall 91% — пропускает редкие форматы объёмов
- Модель чувствительна к опечаткам в редких классах


# NER модель для извлечения сущностей из поисковых запросов

## Задача
Автоматическое извлечение ключевых сущностей из пользовательских поисковых запросов в приложении «Пятёрочка» для улучшения качества поиска товаров.

**Распознаваемые сущности:**
- TYPE — категория товара (молоко, хлеб, вода, чипсы)
- BRAND — бренд (Coca-Cola, Простоквашино, Lays)
- VOLUME — объём/вес/количество (0.5 л, 1 л, 200 г, 10 шт)
- PERCENT — процент (2.5%, 15%)

## Проблема
Критический дисбаланс классов в датасете (27000 примеров):
- TYPE: 21046 примеров (76%)
- BRAND: 6213 примеров (22%)
- VOLUME: 245 примеров (0.9%)
- PERCENT: 84 примера (0.3%)

## Решение и эксперименты

### Этап 1: Baseline
**Подход:** Добавил ~150 синтетических примеров для PERCENT через LLM-генерацию запросов с процентами жирности.

**Результат:** 
- Общий F1 вырос до 93% (weighted)
- PERCENT F1: 73% (было <50%)
- Модель начала хотя бы видеть редкий класс

### Этап 2: Масштабирование аугментации
**Подход:** Расширил датасет до 500+ примеров через Claude API:
- Генерация запросов для товаров с процентами (молоко, творог, сметана)
- Побочно увеличились примеры VOLUME (молоко 2.5% 1л)

**Результат:**
- PERCENT recall: 68% → 80% 
- Проблема: precision упал до 50% — модель начала видеть проценты везде

### Этап 3: Hard Negative Mining
**Подход:** Создание контрпримеров — товары с цифрами, которые НЕ являются процентами:
1. Извлёк все TYPE-сущности из датасета
2. Нормализовал названия через DeepSeek API (выбор из-за цены)
3. Отфильтровал товары с цифрами: "хлеб 5 злаков", "шоколад 3 бита", "чипсы 3D"
4. Сгенерировал по 10 вариантов с опечатками для каждого

**Результат:** Precision улучшился, но recall упал обратно до 70%. Модель перестала доверять цифрам.

### Этап 4: Multi-Task Learning
**Подход:** Добавил вспомогательную голову для бинарной классификации "есть ли процент в запросе?" для регуляризации.

**Результат:** Минимальное улучшение (+2% F1), основная проблема не решена.

### Этап 5: Другие эксперименты
- **Class Weights:** взвешивание loss пропорционально редкости класса — эффект <3%
- **Focal Loss:** фокус на сложных примерах — без значимого улучшения
- **DeBERTa вместо BERT:** схожие результаты

## Финальная модель
Вернулся к конфигурации после Этапа 1 (baseline + 150 примеров), как наиболее стабильной.

**Метрики на тренировочной выборке:**
```
              precision    recall  f1-score   support
BRAND           0.98        0.98      0.98      6213
PERCENT         0.68        0.80      0.73        84
TYPE            0.99        0.99      0.99     21046
VOLUME          0.95        0.91      0.93       245

weighted avg    0.98        0.99      0.98     27588
macro avg       0.90        0.92      0.91     27588
```

**Submission score:** 0.93 F1 (weighted)

## Ключевые выводы

1. **Минимальная аугментация работает лучше агрессивной** — 150 примеров дали стабильный результат, 500+ привели к переобучению

2. **Дисбаланс 1:250 критичен для BERT** — при 0.3% примеров стандартные методы балансировки не работают

3. **Контрпримеры создают новые проблемы** — попытка научить модель НЕ размечать цифры привела к недоверию к настоящим процентам

## Что можно улучшить

- Rule-based post-processing с регулярками для PERCENT/VOLUME
- Порог confidence для редких классов
- Ансамбль с голосованием
- Двухэтапная классификация: сначала есть/нет сущность, потом какая
- Active learning на реальных запросах пользователей
- Отдельная специализированная модель для PERCENT/VOLUME

Ссылка на google collab - https://colab.research.google.com/drive/1Zyo-Zoqi0V0ojoVPXu-rq-U5AhEUbMml?usp=sharing
Датасеты можно найти в папке датасет