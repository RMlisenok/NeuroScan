import cv2
import numpy as np

MIN_SIZE = 128
LAPLACIAN_THRESHOLD = 40.0
BRIGHTNESS_MIN = 20
BRIGHTNESS_MAX = 240
CONTRAST_MIN = 15.0
BRAIN_RATIO_MIN = 0.15
BRAIN_RATIO_MAX = 0.95


class QCResult:
    def __init__(self, ok, reasons, metrics):
        self.ok = ok
        self.reasons = reasons
        self.metrics = metrics

    def to_dict(self):
        return {"ok": self.ok, "reasons": self.reasons, "metrics": self.metrics}


def _is_mri_like(gray):
    h, w = gray.shape
    cy, cx = h // 2, w // 2
    ch, cw = h // 4, w // 4
    center = gray[cy - ch:cy + ch, cx - cw:cx + cw]
    border = np.concatenate([
        gray[:10, :].ravel(), gray[-10:, :].ravel(),
        gray[:, :10].ravel(), gray[:, -10:].ravel()
    ])
    return center.mean() > border.mean() + 10


def check_image(img_bgr):
    reasons, metrics = [], {}
    if img_bgr is None or img_bgr.size == 0:
        return QCResult(False, ["Файл повреждён или пустой"], metrics)

    h, w = img_bgr.shape[:2]
    metrics["size"] = f"{w}x{h}"

    if min(h, w) < MIN_SIZE:
        reasons.append(f"Слишком маленькое разрешение ({w}x{h}). Минимум {MIN_SIZE}px.")

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    brightness = float(gray.mean())
    metrics["brightness"] = round(brightness, 1)
    if brightness < BRIGHTNESS_MIN:
        reasons.append(f"Слишком тёмное изображение (яркость {brightness:.0f}).")
    elif brightness > BRIGHTNESS_MAX:
        reasons.append(f"Пересвеченное изображение (яркость {brightness:.0f}).")

    contrast = float(gray.std())
    metrics["contrast"] = round(contrast, 1)
    if contrast < CONTRAST_MIN:
        reasons.append(f"Низкий контраст (std {contrast:.1f}).")

    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    metrics["sharpness"] = round(lap_var, 1)
    if lap_var < LAPLACIAN_THRESHOLD:
        reasons.append(f"Изображение размыто (резкость {lap_var:.0f}).")

    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    brain_ratio = float((thresh > 0).mean())
    metrics["brain_ratio"] = round(brain_ratio, 3)
    if brain_ratio < BRAIN_RATIO_MIN:
        reasons.append("Не похоже на МРТ головного мозга (мало яркой области).")
    elif brain_ratio > BRAIN_RATIO_MAX:
        reasons.append("Не похоже на МРТ головного мозга (нет тёмного фона).")

    if not _is_mri_like(gray):
        reasons.append("Структура изображения не соответствует МРТ.")

    b, g, r = cv2.split(img_bgr.astype(np.float32))
    color_diff = float(np.mean(np.abs(r - g)) + np.mean(np.abs(g - b)))
    metrics["color_diff"] = round(color_diff, 1)
    if color_diff > 8.0:
        reasons.append("Изображение цветное — МРТ обычно в оттенках серого.")

    return QCResult(len(reasons) == 0, reasons, metrics)