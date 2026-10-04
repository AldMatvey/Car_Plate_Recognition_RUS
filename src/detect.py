import cv2
import numpy as np
from ultralytics import YOLO


class PlateDetector:
    def __init__(self, model_path: str):
        self.model = YOLO(model_path)

    def detect(self, image_path: str, padding: int = 5):
        """
        Возвращает список вырезанных номеров и координат.

        Returns:
            List[Tuple[np.ndarray, Tuple[int, int, int, int], float]]
            [(crop, (x1, y1, x2, y2), confidence), ...]
        """
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"Не удалось загрузить {image_path}")

        results = self.model(image_path)
        boxes = results[0].boxes.xyxy.cpu().numpy()
        confs = results[0].boxes.conf.cpu().numpy()

        detections = []
        h_img, w_img = img.shape[:2]

        for box, conf in zip(boxes, confs):
            x1, y1, x2, y2 = map(int, box[:4])
            x1 = max(0, x1 - padding)
            y1 = max(0, y1 - padding)
            x2 = min(w_img, x2 + padding)
            y2 = min(h_img, y2 + padding)

            crop = img[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            detections.append((crop, (x1, y1, x2, y2), float(conf)))

        return img, detections