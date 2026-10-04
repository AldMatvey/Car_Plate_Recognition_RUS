from ultralytics import YOLO
import multiprocessing
import os


def main():
    DATA_YAML = "../dataset/data.yaml"

    if not os.path.exists(DATA_YAML):
        print(f"ОШИБКА: Файл {DATA_YAML} не найден!")
        return

    model = YOLO("../src/yolov8n.pt")

    results = model.train(
        data=DATA_YAML,
        epochs=100,  # Количество эпох
        imgsz=640,  # Размер изображений
        batch=8,  # Размер батча
        device=0,  # GPU
        patience=30,
        workers=2,
        name="plate_detection",  # Имя папки для результатов
        exist_ok=True,
        pretrained=True,  # Использовать предобученные веса
        optimizer="auto",  # Автоматический выбор оптимизатора
        verbose=True,  # Подробный вывод
        seed=42,  # Для воспроизводимости
        hsv_h=0.015,  # Оттенок
        hsv_s=0.7,  # Насыщенность
        hsv_v=0.4,  # Яркость
        degrees=0.0,  # Поворот
        translate=0.1,  # Сдвиг
        scale=0.5,  # Масштаб
        fliplr=0.5,  # Отражение по горизонтали
        mosaic=1.0,  # Mosaic-аугментация
        mixup=0.0,  # Mixup
    )

    print("\n=== Обучение завершено! ===")
    print(f"Лучшая модель: runs/detect/plate_detection/weights/best.pt")
    print(f"Последняя модель: runs/detect/plate_detection/weights/last.pt")


if __name__ == "__main__":
    multiprocessing.freeze_support()  # Важно для Windows
    main()