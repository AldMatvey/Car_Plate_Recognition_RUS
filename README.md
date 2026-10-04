# Car Plate Recognition (Russian)

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-orange)](https://github.com/ultralytics/ultralytics)
[![PaddleOCR](https://img.shields.io/badge/PaddleOCR-3.3.3-green)](https://github.com/PaddlePaddle/PaddleOCR)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)
[![Platform](https://img.shields.io/badge/Platform-Windows%20%7C%20Linux-lightgrey)]()

Система распознавания **российских автомобильных номеров** на edge-устройствах.  
Детекция номеров через **YOLOv8**, распознавание символов через **PaddleOCR** с постобработкой под формат РФ.

---

## Возможности

- **Детекция номера** - YOLOv8, обученная на датасете российских номеров (mAP50 = **0.988**).
- **Распознавание символов** - PaddleOCR с кириллической моделью.
- **Постобработка** - приведение к формату `А123AA 77` с учётом позиций (буква-цифра-буква) и заменой похожих символов (`O`→`0`, `B`→`8`, `T`→`1`).
- **Работа на NPU** - поддержка Qualcomm QCS6490 (Hexagon) через TFLite + QNN Delegate.
- **MQTT-интеграция** - публикация распознанных номеров (в разработке).

---

## Результаты

| Метрика | Значение |
|---------|----------|
| Детекция (mAP50) | 0.988 |
| Детекция (mAP50-95) | 0.802 |
| Точность OCR (100 изображений) | ~80% |
| Скорость детекции (GPU) | ~10 мс |
| Скорость OCR (CPU) | ~50 мс |

 **Пример работы:**
![Вход](docs/images/1_raw.jpg)
![Выход](docs/images/1_res.jpg)

---
## Архитектура
![Диаграмма](docs/images/diagram.png)
