import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from torchcrf import CRF
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict
import uvicorn
import logging
from logging.handlers import QueueHandler, QueueListener
from queue import Queue
from datetime import datetime
import time
import asyncio
from asyncio import Queue as AsyncQueue
from collections import deque
import threading

# Настройка асинхронного логирования
log_queue = Queue()
queue_handler = QueueHandler(log_queue)

# Обработчик который пишет в stdout
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(
    logging.Formatter('%(asctime)s.%(msecs)03d - %(name)s - %(levelname)s - %(message)s',
                      datefmt='%Y-%m-%d %H:%M:%S')
)

# Слушатель очереди (работает в отдельном потоке)
queue_listener = QueueListener(log_queue, stream_handler)
queue_listener.start()

# Настраиваем логгер
logger = logging.getLogger(__name__)
logger.addHandler(queue_handler)
logger.setLevel(logging.INFO)

# Флаг для включения/выключения детального логирования
DETAILED_LOGGING = False  # Поставь True если нужна отладка

# ============ Конфигурация ============
# ВАЖНО: отключаем CUDA для 4-ядерного CPU
device = torch.device('cpu')
# Устанавливаем количество потоков для CPU
torch.set_num_threads(4)  # Используем все 4 ядра
logger.info(f"Используем устройство: {device}, потоков: {torch.get_num_threads()}")

# Параметры батчинга
BATCH_SIZE = 10  # Размер батча для обработки
BATCH_TIMEOUT = 0.05  # Таймаут в секундах для накопления батча

# Маппинг меток
label_to_id = {
    "O": 0,
    "B-TYPE": 1,
    "I-TYPE": 2,
    "B-BRAND": 3,
    "I-BRAND": 4,
    "B-VOLUME": 5,
    "I-VOLUME": 6,
    "B-PERCENT": 7,
    "I-PERCENT": 8
}
id_to_label = {v: k for k, v in label_to_id.items()}

# Параметры модели
MODEL_NAME = "DeepPavlov/rubert-base-cased"
MAX_LENGTH = 20
MODEL_PATH = "best_model_crf_all.pt"


# ============ Модель BertCRF ============
class BertCRFForNER(nn.Module):
    def __init__(self, model_name, num_labels, dropout_rate=0.1):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)
        self.crf = CRF(num_labels, batch_first=True)

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        sequence_output = self.dropout(sequence_output)
        emissions = self.classifier(sequence_output)

        if labels is not None:
            crf_mask = attention_mask.bool()
            crf_labels = labels.clone()
            crf_labels[labels == -100] = 0
            loss = -self.crf(emissions, crf_labels, mask=crf_mask, reduction='mean')
            return loss, emissions
        else:
            predictions = self.crf.decode(emissions, mask=attention_mask.bool())
            return predictions


# ============ Pydantic модели ============
class PredictRequest(BaseModel):
    input: str
    include_o: bool = True


class EntityResponse(BaseModel):
    start_index: int
    end_index: int
    entity: str


# ============ Инициализация приложения ============
app = FastAPI(title="NER Prediction Service", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def load_model():
    """Загружает модель и токенайзер"""
    start_time = time.time()

    # Токенайзер
    logger.info("Загружаем токенайзер...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # Загружаем checkpoint
    logger.info(f"Загружаем модель из {MODEL_PATH}...")
    checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)

    # Достаём модель
    model = checkpoint['model_state_dict']

    model.to(device)
    model.eval()

    # Включаем режим inference для ускорения
    torch.set_grad_enabled(False)

    load_time = time.time() - start_time
    logger.info(f"Модель загружена за {load_time:.2f} секунд")

    return model, tokenizer


# Глобальные переменные для модели
model = None
tokenizer = None
batch_queue = None
batch_processor_task = None


# ============ Батч процессор ============
class BatchProcessor:
    def __init__(self, model, tokenizer, batch_size=8, timeout=0.05):
        self.model = model
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.timeout = timeout
        self.queue = AsyncQueue()

    async def process_batches(self):
        """Основной цикл обработки батчей"""
        while True:
            batch = []
            futures = []

            # Накапливаем батч
            try:
                # Ждём первый элемент
                item = await self.queue.get()
                batch.append(item)
                futures.append(item['future'])

                # Пытаемся добрать до batch_size или по таймауту
                deadline = time.time() + self.timeout
                while len(batch) < self.batch_size and time.time() < deadline:
                    try:
                        remaining_time = max(0, deadline - time.time())
                        item = await asyncio.wait_for(self.queue.get(), timeout=remaining_time)
                        batch.append(item)
                        futures.append(item['future'])
                    except asyncio.TimeoutError:
                        break

            except Exception as e:
                logger.error(f"Ошибка в batch processor: {e}")
                continue

            if batch:
                try:
                    # Обрабатываем батч
                    results = self._process_batch(batch)

                    # Возвращаем результаты
                    for future, result in zip(futures, results):
                        if not future.done():
                            future.set_result(result)

                except Exception as e:
                    logger.error(f"Ошибка обработки батча: {e}")
                    for future in futures:
                        if not future.done():
                            future.set_exception(e)

    def _process_batch(self, batch_items):
        """Обработка батча текстов"""
        texts = [item['text'] for item in batch_items]
        include_o_flags = [item['include_o'] for item in batch_items]

        # Токенизация батча
        encodings = self.tokenizer(
            texts,
            truncation=True,
            max_length=MAX_LENGTH,
            return_offsets_mapping=True,
            return_tensors='pt',
            padding=True  # Включаем паддинг для батча
        )

        input_ids = encodings['input_ids']
        attention_mask = encodings['attention_mask']

        # Предсказание для батча
        with torch.no_grad():
            predictions_batch = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=None
            )

        # Обработка результатов для каждого текста
        results = []
        for idx, (text, include_o) in enumerate(zip(texts, include_o_flags)):
            predictions = predictions_batch[idx]
            offset_mapping = encodings['offset_mapping'][idx].tolist()

            # Собираем предсказания для каждого токена
            token_predictions = []
            for j, (pred_id, (start, end)) in enumerate(zip(predictions, offset_mapping)):
                if start == 0 and end == 0:  # Специальные токены
                    continue
                if j >= len(predictions):
                    break

                pred_label = id_to_label[pred_id]
                token_predictions.append({
                    'start': start,
                    'end': end,
                    'label': pred_label
                })

            # Группируем по словам и создаём спаны
            entities = []
            current_pos = 0
            words = text.split()
            previous_entity_type = None

            for word_idx, word in enumerate(words):
                # Находим позицию слова в тексте
                word_start = text.find(word, current_pos)
                if word_start == -1:
                    continue
                word_end = word_start + len(word)

                # Находим все токены, которые попадают в диапазон этого слова
                word_tokens = [
                    t for t in token_predictions
                    if t['start'] >= word_start and t['start'] < word_end
                ]

                if word_tokens:
                    # Берём метку первого токена слова
                    word_label = word_tokens[0]['label']

                    if word_label != 'O':
                        entity_type = word_label[2:] if word_label.startswith(('B-', 'I-')) else word_label

                        # Корректируем B-/I- теги на основе контекста
                        if previous_entity_type and previous_entity_type == entity_type:
                            final_label = f'I-{entity_type}'
                        else:
                            final_label = f'B-{entity_type}'

                        entities.append({
                            'start_index': word_start,
                            'end_index': word_end,
                            'entity': final_label
                        })
                        previous_entity_type = entity_type
                    else:
                        # Добавляем O метку если нужно
                        if include_o:
                            entities.append({
                                'start_index': word_start,
                                'end_index': word_end,
                                'entity': 'O'
                            })
                        previous_entity_type = None
                else:
                    # Нет токенов для слова - добавляем O если нужно
                    if include_o:
                        entities.append({
                            'start_index': word_start,
                            'end_index': word_end,
                            'entity': 'O'
                        })
                    previous_entity_type = None

                current_pos = word_end

            results.append(entities)

        return results

    async def predict(self, text: str, include_o: bool = True):
        """Добавляет текст в очередь и ждёт результат"""
        future = asyncio.get_event_loop().create_future()
        await self.queue.put({
            'text': text,
            'include_o': include_o,
            'future': future
        })
        return await future


@app.on_event("startup")
async def startup_event():
    """Загружает модель при старте сервера"""
    global model, tokenizer, batch_queue, batch_processor_task
    model, tokenizer = load_model()

    # Создаём батч процессор
    batch_processor = BatchProcessor(model, tokenizer, BATCH_SIZE, BATCH_TIMEOUT)
    batch_queue = batch_processor

    # Запускаем фоновую задачу обработки батчей
    batch_processor_task = asyncio.create_task(batch_processor.process_batches())

    logger.info(f"Сервер готов к работе. Батчинг: размер={BATCH_SIZE}, таймаут={BATCH_TIMEOUT}с")


@app.on_event("shutdown")
async def shutdown_event():
    """Останавливаем батч процессор при выключении"""
    global batch_processor_task
    if batch_processor_task:
        batch_processor_task.cancel()


# ============ API эндпоинты ============
@app.get("/")
async def root():
    """Проверка работоспособности сервера"""
    return {"message": "NER Prediction Service is running", "status": "ok"}


@app.get("/health")
async def health_check():
    """Health check эндпоинт"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "device": str(device),
        "batch_size": BATCH_SIZE,
        "batch_timeout": BATCH_TIMEOUT
    }


@app.post("/api/predict", response_model=List[EntityResponse])
async def predict(request: PredictRequest):
    """
    Предсказание NER меток для текста (с батчингом)
    """
    if DETAILED_LOGGING:
        request_start = time.time()
        request_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
        logger.info(f"[{request_time}] Запрос получен: text='{request.input[:50]}...', include_o={request.include_o}")

    try:
        if model is None or batch_queue is None:
            raise HTTPException(status_code=503, detail="Модель не загружена")

        # Получаем предсказания через батч процессор
        entities = await batch_queue.predict(request.input, request.include_o)

        # Преобразуем в формат ответа
        response = [
            EntityResponse(
                start_index=entity['start_index'],
                end_index=entity['end_index'],
                entity=entity['entity']
            )
            for entity in entities
        ]

        if DETAILED_LOGGING:
            request_end = time.time()
            duration_ms = (request_end - request_start) * 1000
            response_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
            logger.info(
                f"[{response_time}] Ответ отправлен: найдено {len(entities)} сущностей, время обработки: {duration_ms:.1f}ms")

        return response

    except Exception as e:
        logger.error(f"Ошибка при предсказании: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============ Запуск сервера ============
if __name__ == "__main__":
    try:
        uvicorn.run(
            "main:app",
            host="0.0.0.0",
            port=8002,
            reload=False,
            log_level="warning",
            access_log=False,
            use_colors=False,
            workers=1
        )
    finally:
        # Останавливаем listener при выходе
        queue_listener.stop()