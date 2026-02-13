# ocr_modules/base_modules/reliability.py

def is_reliable(confidence, engine):
    thresholds = {
        "east": 0.6,
        "tesseract": 0.5,
        "easyocr": 0.5,
        "paddleocr": 0.5
    }
    return confidence >= thresholds.get(engine, 0.5)