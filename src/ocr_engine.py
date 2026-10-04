import re
import numpy as np
from paddleocr import PaddleOCR

# === РАЗРЕШЁННЫЕ СИМВОЛЫ ===
RUS_LETTERS = "АВЕКМНОРСТУХ"
DIGITS = "0123456789"

# === ЗАМЕНЫ ЦИФРА -> БУКВА (для позиций букв) ===
DIGIT_TO_LETTER = {
    '0': 'О',
    '1': 'Т',
    '3': 'З',  # не в RUS_LETTERS, но может пригодиться
    '4': 'А',
    '6': 'Б',  # не в RUS_LETTERS
    '8': 'В',
    '9': 'Р',
}

# === ЗАМЕНЫ БУКВА -> ЦИФРА (для позиций цифр) ===
LETTER_TO_DIGIT = {
    'О': '0',
    'З': '3',
    'Б': '6',
    'В': '8',
    'Р': '9',
    'Т': '7',
    'А': '4',
    'С': '0',
    'Е': '6',
    'У': '4',
}

# === НОРМАЛИЗАЦИЯ ЛАТИНИЦЫ В КИРИЛЛИЦУ ===
LAT_TO_RUS = {
    'A': 'А', 'B': 'В', 'E': 'Е', 'K': 'К', 'M': 'М',
    'H': 'Н', 'O': 'О', 'P': 'Р', 'C': 'С', 'T': 'Т',
    'Y': 'У', 'X': 'Х', 'D': 'В',
}

FALLBACK_LETTER = '?'
FALLBACK_DIGIT = '?'


def normalize_to_cyrillic(text: str) -> str:
    """Приводит латинские буквы к кириллическим аналогам."""
    return ''.join(LAT_TO_RUS.get(ch.upper(), ch.upper()) for ch in text)


class PlateOCR:
    def __init__(self, lang='ru'):
        self.ocr = PaddleOCR(
            lang=lang,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            enable_mkldnn=False,
        )

    def read(self, image: np.ndarray) -> str:
        if image is None or image.size == 0:
            return ""
        result = self.ocr.predict(image)
        if not result:
            return ""
        try:
            texts = result[0].get('rec_texts', [])
            return "".join(texts).upper().strip()
        except (IndexError, AttributeError, TypeError):
            try:
                lines = result[0]
                return "".join([line[1][0] for line in lines]).upper().strip()
            except Exception:
                return ""

    def correct_main(self, text: str) -> str:
        """
        Основная часть номера: [БУКВА][3 ЦИФРЫ][2 БУКВЫ].
        Берёт первые 6 символов и принудительно корректирует по позициям.
        """
        # Нормализация латиницы
        text = normalize_to_cyrillic(text)

        # Убираем всё, кроме букв и цифр
        clean = re.sub(r'[^А-Я0-9]', '', text)

        if len(clean) < 6:
            return ""

        clean = clean[:6]  # только main-часть

        out = []
        for i, ch in enumerate(clean):
            if i in (0, 4, 5):  # позиции букв
                if ch in RUS_LETTERS:
                    out.append(ch)
                elif ch in DIGIT_TO_LETTER:
                    out.append(DIGIT_TO_LETTER[ch])
                else:
                    out.append(FALLBACK_LETTER)
            else:  # позиции 1, 2, 3 — цифры
                if ch in DIGITS:
                    out.append(ch)
                elif ch in LETTER_TO_DIGIT:
                    out.append(LETTER_TO_DIGIT[ch])
                else:
                    out.append(FALLBACK_DIGIT)

        return ''.join(out)

    def correct_region(self, text: str, main_text: str = "") -> str:
        """
        Регион: 2–3 цифры после main-части.
        """
        text = normalize_to_cyrillic(text)
        clean = re.sub(r'[^0-9А-Я]', '', text)

        out = []
        for ch in clean:
            if ch in DIGITS:
                out.append(ch)
            elif ch in LETTER_TO_DIGIT:
                out.append(LETTER_TO_DIGIT[ch])

        # Регион обычно 2–3 цифры
        return ''.join(out)[:3]

    def parse(self, raw_text: str):
        """
        Полный парсинг: main + region.

        Returns:
            (plate, main_text, region_text) или (None, "", "") если не удалось.
        """
        main_text = self.correct_main(raw_text)
        if not main_text or '?' in main_text:
            return None, main_text, ""

        # Регион — всё, что идёт после первых 6 символов main-части
        text_norm = normalize_to_cyrillic(raw_text)
        clean = re.sub(r'[^А-Я0-9]', '', text_norm)
        region_raw = clean[6:] if len(clean) > 6 else ""
        region_text = self.correct_region(region_raw, main_text)

        plate = f"{main_text} {region_text}".strip() if region_text else main_text
        return plate, main_text, region_text