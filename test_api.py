import requests
import json

# URL сервера
BASE_URL = "http://217.114.3.51:8002"


def test_predict():
    """Тестирует основной эндпоинт предсказания"""

    # Тестовые примеры
    test_cases = [
        {"input": "сгущенное молоко"},
        {"input": "Coca-Cola 0.5л"},
        {"input": "молоко Домик в деревне 3.2% 1л"},
        {"input": ""},  # Пустая строка
        {"input": "хлеб"},
        {"input": "вода Святой источник 1.5л"},
    ]

    print("=" * 60)
    print("Тестирование /api/predict")
    print("=" * 60)

    for i, test_case in enumerate(test_cases, 1):
        print(f"\nТест {i}: {test_case['input']!r}")

        try:
            response = requests.post(
                f"{BASE_URL}/api/predict",
                json=test_case,
                headers={"Content-Type": "application/json"}
            )

            if response.status_code == 200:
                result = response.json()
                print(f"Статус: ✅ OK")
                print(f"Результат: {json.dumps(result, ensure_ascii=False, indent=2)}")

                # Проверка формата ответа
                if isinstance(result, list):
                    for entity in result:
                        if not all(k in entity for k in ['start_index', 'end_index', 'entity']):
                            print("⚠️  Неверный формат сущности!")
                            break
                    else:
                        # Визуализация сущностей
                        if result and test_case['input']:
                            text = test_case['input']
                            print("\nВизуализация:")
                            for entity in result:
                                start = entity['start_index']
                                end = entity['end_index']
                                chunk = text[start:end]
                                print(f"  [{chunk}] -> {entity['entity']}")
            else:
                print(f"❌ Ошибка: {response.status_code}")
                print(f"Детали: {response.text}")

        except Exception as e:
            print(f"❌ Исключение: {e}")


def test_batch_predict():
    """Тестирует батчевый эндпоинт"""

    texts = [
        "молоко 1л",
        "пиво Балтика 7 0.5л",
        "сыр Российский"
    ]

    print("\n" + "=" * 60)
    print("Тестирование /api/predict_batch")
    print("=" * 60)

    try:
        response = requests.post(
            f"{BASE_URL}/api/predict_batch",
            json=texts,
            headers={"Content-Type": "application/json"}
        )

        if response.status_code == 200:
            results = response.json()
            print(f"Статус: ✅ OK")

            for i, (text, result) in enumerate(zip(texts, results), 1):
                print(f"\nТекст {i}: {text}")
                print(f"Сущности: {json.dumps(result, ensure_ascii=False, indent=2)}")
        else:
            print(f"❌ Ошибка: {response.status_code}")
            print(f"Детали: {response.text}")

    except Exception as e:
        print(f"❌ Исключение: {e}")


def test_health():
    """Проверяет health check"""

    print("\n" + "=" * 60)
    print("Проверка здоровья сервера")
    print("=" * 60)

    try:
        response = requests.get(f"{BASE_URL}/health")

        if response.status_code == 200:
            result = response.json()
            print(f"Статус: ✅ OK")
            print(f"Результат: {json.dumps(result, indent=2)}")
        else:
            print(f"❌ Ошибка: {response.status_code}")

    except Exception as e:
        print(f"❌ Не удалось подключиться к серверу: {e}")
        print("Убедитесь, что сервер запущен на порту 8002")
        return False

    return True


if __name__ == "__main__":
    # Сначала проверяем, что сервер работает
    if test_health():
        # Запускаем тесты
        test_predict()
        test_batch_predict()
    else:
        print("\n⚠️  Сервер не доступен. Запустите его командой:")
        print("python main.py")