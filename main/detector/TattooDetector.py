import cv2
import numpy as np
from ultralytics import YOLO
from .Detector import Detector
from typing import Optional, Tuple

class TattooDetector(Detector): #TODO ни разу не тестировалась - найти подходящие фото для анализа
    def __init__(self, model_path: str):
        self.model = YOLO(model_path)

    def detect(self, image_bgr: np.ndarray, roi_mask: Optional[np.ndarray] = None, **kwargs) -> Tuple[list, list]:
        """
        Возвращает (masks, confidences).
        Если передан roi_mask, ищет только внутри него (вырезает ROI - коробку).
        """
        if roi_mask is not None and cv2.countNonZero(roi_mask) > 0:
            ys, xs = np.where(roi_mask > 0)
            x1, y1, x2, y2 = xs.min(), ys.min(), xs.max(), ys.max()
            # Расширяем ROI на 10% для запаса
            pad_x = int((x2 - x1) * 0.1)
            pad_y = int((y2 - y1) * 0.1)
            x1 = max(0, x1 - pad_x); y1 = max(0, y1 - pad_y)
            x2 = min(image_bgr.shape[1], x2 + pad_x)
            y2 = min(image_bgr.shape[0], y2 + pad_y)
            crop = image_bgr[y1:y2, x1:x2]
        else:
            crop = image_bgr
            x1, y1 = 0, 0

        results = self.model.predict(crop, conf=0.25, imgsz=640, verbose=False)
        masks = []
        confs = []
        if results and results[0].masks is not None:
            raw_masks = results[0].masks.data.cpu().numpy()
            raw_confs = results[0].boxes.conf.cpu().numpy() if results[0].boxes is not None else []
            for i, m in enumerate(raw_masks):
                # Возвращаем маску в координатах оригинала
                full_mask = np.zeros(image_bgr.shape[:2], dtype=np.uint8)
                resized = cv2.resize(m, (x2 - x1, y2 - y1), interpolation=cv2.INTER_NEAREST)
                full_mask[y1:y2, x1:x2] = (resized > 0.5).astype(np.uint8) * 255
                masks.append(full_mask)
                confs.append(float(raw_confs[i]) if i < len(raw_confs) else 0.0)
        return masks, confs