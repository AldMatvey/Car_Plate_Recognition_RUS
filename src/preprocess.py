import cv2
import numpy as np


def preprocess(crop: np.ndarray, target_h: int = 80, use_sharpen: bool = True) -> np.ndarray:
    """
    Предобработка для PaddleOCR:
    - увеличение, если номер маленький
    - grayscale
    - CLAHE (контраст)
    - Unsharp Masking (резкость)
    """
    # Увеличение
    h, w = crop.shape[:2]
    if h < target_h:
        scale = target_h / h
        crop = cv2.resize(crop, (int(w * scale), target_h),
                          interpolation=cv2.INTER_CUBIC)

    # Grayscale
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    # CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # Unsharp Masking — повышение резкости
    if use_sharpen:
        # Gaussian blur с маленьким ядром
        blurred = cv2.GaussianBlur(enhanced, (0, 0), sigmaX=1.5)
        # Формула Unsharp: sharpened = original + amount * (original - blurred)
        amount = 1.5  # сила повышения резкости (можно настроить)
        enhanced = cv2.addWeighted(enhanced, 1 + amount, blurred, -amount, 0)

    # PaddleOCR ожидает 3-канальное BGR
    result = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
    return result