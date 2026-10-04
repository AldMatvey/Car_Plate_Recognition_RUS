"""
Car Plate Recognition — главный скрипт.

Запускает полный пайплайн:
    1. Детекция номера (YOLOv8)
    2. Предобработка (OpenCV)
    3. OCR (PaddleOCR)
    4. Постобработка (формат РФ)

Использование:
    python main.py
"""

import os
import re
import csv
import cv2

from detect import PlateDetector
from preprocess import preprocess
from ocr_engine import PlateOCR


# === НАСТРОЙКИ ===
MODEL_PATH = "./runs/detect/plate_detection/weights/best.pt"
INPUT_DIR = "./benchmark_images"
OUTPUT_CSV = "benchmark_results.csv"
DEBUG_DIR = "./debug_benchmark"


def main():
    os.makedirs(DEBUG_DIR, exist_ok=True)
    detector = PlateDetector(MODEL_PATH)
    ocr = PlateOCR()

    results = []
    images = sorted(os.listdir(INPUT_DIR))

    for idx, image_name in enumerate(images, 1):
        image_path = os.path.join(INPUT_DIR, image_name)

        try:
            img, detections = detector.detect(image_path)

            if not detections:
                results.append({
                    'image': image_name,
                    'detected': 0,
                    'plate': '',
                    'status': 'no_detection'
                })
                print(f"[{idx}/{len(images)}] {image_name}: нет номера")
                continue

            for det_idx, detection in enumerate(detections):
                crop = detection[0]
                bbox = detection[1]
                conf = detection[2]

                # Сохраняем вырезанный номер
                crop_path = os.path.join(DEBUG_DIR, f"{idx:03d}_{det_idx}_crop.jpg")
                cv2.imwrite(crop_path, crop)

                # --- Предобработка ---
                try:
                    processed = preprocess(crop)
                    proc_path = os.path.join(DEBUG_DIR, f"{idx:03d}_{det_idx}_proc.jpg")
                    cv2.imwrite(proc_path, processed)
                except Exception as e:
                    print(f"    ОШИБКА preprocess: {e}")
                    processed = crop

                # --- OCR ---
                try:
                    raw_text = ocr.read(processed)
                except Exception as e:
                    print(f"    ОШИБКА OCR: {e}")
                    raw_text = ""

                # --- Постобработка ---
                plate, main_text, region_text = ocr.parse(raw_text)
                status = 'ok' if plate and '?' not in plate else 'partial'
                plate = plate or ""

                # --- Логирование ---
                log_path = os.path.join(DEBUG_DIR, f"{idx:03d}_{det_idx}_log.txt")
                with open(log_path, "w", encoding="utf-8") as f:
                    f.write(f"image: {image_name}\n")
                    f.write(f"detection: {det_idx}\n")
                    f.write(f"bbox: {bbox}\n")
                    f.write(f"det_conf: {conf:.3f}\n")
                    f.write(f"\n--- OCR ---\n")
                    f.write(f"raw_text:    '{raw_text}'\n")
                    f.write(f"main_text:   '{main_text}'\n")
                    f.write(f"region_text: '{region_text}'\n")
                    f.write(f"plate:       '{plate}'\n")

                results.append({
                    'image': image_name,
                    'detected': 1,
                    'plate': plate,
                    'status': status
                })

                print(f"[{idx}/{len(images)}] {image_name} "
                      f"(det {det_idx}): raw='{raw_text}' → '{plate}'")

        except Exception as e:
            print(f"[{idx}/{len(images)}] {image_name}: ОШИБКА — {e}")
            results.append({
                'image': image_name,
                'detected': -1,
                'plate': '',
                'status': f'error: {e}'
            })

    # --- Сохранение CSV ---
    with open(OUTPUT_CSV, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['image', 'detected', 'plate', 'status'])
        writer.writeheader()
        writer.writerows(results)

    # --- Статистика ---
    total = len(results)
    ok = sum(1 for r in results if r['status'] == 'ok')
    partial = sum(1 for r in results if r['status'] == 'partial')
    no_det = sum(1 for r in results if r['status'] == 'no_detection')
    errors = sum(1 for r in results if r['status'].startswith('error'))

    print(f"\n=== Статистика ===")
    print(f"Всего записей: {total}")
    print(f"OK:            {ok} ({ok/total*100:.1f}%)")
    print(f"Partial:       {partial} ({partial/total*100:.1f}%)")
    print(f"No detection:  {no_det} ({no_det/total*100:.1f}%)")
    print(f"Errors:        {errors} ({errors/total*100:.1f}%)")
    print(f"\nCSV:              {OUTPUT_CSV}")
    print(f"Отладочные файлы: {DEBUG_DIR}/")


if __name__ == "__main__":
    main()